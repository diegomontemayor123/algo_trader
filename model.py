import os
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from features import FTR_FUNC
from compute_features import *
from torch.optim.lr_scheduler import _LRScheduler
from loadconfig import load_config
import matplotlib.pyplot as plt
from data_prep import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "trained_model_new.pth"
LOAD_MODEL = False
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class TransformerTrader(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout, seq_len, tickers, feature_attention_enabled):
        super().__init__()
        self.seq_len = seq_len
        self.feature_attention_enabled = feature_attention_enabled
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, input_dim))
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(tickers))
        )
    def forward(self, x):
        x = x * self.feature_weights * self.feature_attention_enabled
        x = x + self.pos_embedding
        encoded = self.transformer_encoder(x)
        last_hidden = encoded[:, -1, :]
        weights = self.mlp_head(last_hidden)
        return weights

def calculate_heads(input_dim, max_heads):
    if input_dim % max_heads != 0:
        for heads in range(max_heads, 0, -1):
            if input_dim % heads == 0:
                return heads
    else:
        return max_heads

def create_model(input_dimension, config):
    heads = calculate_heads(input_dimension, config["MAX_HEADS"])
    print(f"[Model] Creating TransformerTrader with input_dim={input_dimension}, heads={heads}, device={DEVICE}")
    return TransformerTrader(
        input_dim=input_dimension,
        num_heads=heads,
        num_layers=config["LAYER_COUNT"],
        dropout=config["DROPOUT"],
        seq_len=config["LOOKBACK"],
        tickers=config["TICKERS"],
        feature_attention_enabled=config["FEATURE_ATTENTION_ENABLED"]
    ).to(DEVICE, non_blocking=True)

def split_train_validation(sequences, targets, validation_ratio):
    total_samples = len(sequences)
    val_size = int(total_samples * validation_ratio)
    train_size = total_samples - val_size
    return (sequences[:train_size], targets[:train_size],
            sequences[train_size:], targets[train_size:])

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, l2_lambda=1e-4, loss_min_mean=0.0, loss_return_penalty=0.0, l2_penalty_enabled=False, return_penalty_enabled=False):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.loss_min_mean = loss_min_mean
        self.loss_return_penalty = loss_return_penalty
        self.l2_penalty_enabled = l2_penalty_enabled
        self.return_penalty_enabled = return_penalty_enabled
    def forward(self, portfolio_weights, target_returns, model=None):
        returns = (portfolio_weights * target_returns).sum(dim=1)
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + 1e-6
        sharpe_ratio = mean_return / (std_return + 1e-6)
        low_return_penalty = torch.clamp(self.loss_min_mean - mean_return, min=0.0)
        loss = -sharpe_ratio + self.loss_return_penalty * low_return_penalty * self.return_penalty_enabled
        if self.l2_penalty_enabled and model is not None:
            l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += self.l2_lambda * l2_penalty
        return loss

class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=50, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.d_model ** -0.5
        lr = scale * min(step ** (-0.5), step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.optimizer.param_groups]

def train_model_with_validation(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config["DECAY"])
    total_steps = config["EPOCHS"] * len(train_loader)
    learning_warmup_steps = int(total_steps * config["WARMUP_FRAC"])
    print(f"[Scheduler] Total steps: {total_steps}, LEARNING_WARMUP steps: {learning_warmup_steps}")
    learning_scheduler = TransformerLRScheduler(optimizer, d_model=model.mlp_head[0].in_features, warmup_steps=learning_warmup_steps)
    loss_function = DifferentiableSharpeLoss(l2_lambda=1e-4,loss_min_mean=config["LOSS_MIN_MEAN"],loss_return_penalty=config["LOSS_RETURN_PENALTY"],l2_penalty_enabled=config["L2_PENALTY_ENABLED"],return_penalty_enabled=config["RETURN_PENALTY_ENABLED"])
    best_val_loss = float('inf')
    patience_counter = 0
    lrs = []
    for epoch in range(config["EPOCHS"]):
        print(f"[Training] Epoch {epoch + 1}/{config['EPOCHS']}")
        model.train()
        train_losses = []
        for batch_features, batch_returns in train_loader:
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            batch_returns = batch_returns.to(DEVICE, non_blocking=True)
            raw_weights = model(batch_features)
            abs_sum = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
            scaling_factor = torch.clamp(config["MAX_LEVERAGE"] / abs_sum, max=1.0)
            normalized_weights = raw_weights * scaling_factor
            loss = loss_function(normalized_weights, batch_returns, model)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[Training][Error] Loss is NaN or Inf during training step")
            optimizer.zero_grad()
            loss.backward()
            nan_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[Training][Error] NaN or Inf detected in gradients of {name}")
                    nan_grads = True
            if nan_grads:
                print("[Training][Error] Stopping training due to invalid gradients")
                break
            optimizer.step()
            learning_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        model.eval()
        val_portfolio_returns = []
        with torch.no_grad():
            for batch_features, batch_returns in val_loader:
                batch_features = batch_features.to(DEVICE, non_blocking=True)
                batch_returns = batch_returns.to(DEVICE, non_blocking=True)
                raw_weights = model(batch_features)
                weight_sum = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
                scaling_factor = torch.clamp(config["MAX_LEVERAGE"] / weight_sum, max=1.0)
                normalized_weights = raw_weights * scaling_factor
                portfolio_returns = (normalized_weights * batch_returns).sum(dim=1)
                val_portfolio_returns.extend(portfolio_returns.cpu().numpy())
        val_returns_array = np.array(val_portfolio_returns)
        mean_ret = val_returns_array.mean()
        std_ret = val_returns_array.std() + 1e-6
        avg_val_loss = -(mean_ret / std_ret)
        print(f"[Training] Train Loss: {avg_train_loss:.4f} | Validation: {abs(avg_val_loss):.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("[Training] Improvement; continuing...")
        else:
            patience_counter += 1
            if patience_counter >= config["EARLY_STOP_PATIENCE"]:
                print("[Training] Early stopping due to val loss plateau")
                break
    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule During Training')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('learning_rate_schedule.png')
    plt.close()
    print("[Training] Saved learning rate schedule plot as 'learning_rate_schedule.png'")
    return model

def train_main_model(config):
    features, returns = compute_features(config["TICKERS"], config["START_DATE"], config["END_DATE"], config["FEATURES"])
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns, config)
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4)
    model = create_model(train_dataset[0][0].shape[1], config)
    trained_model = train_model_with_validation(model, train_loader, val_loader, config)
    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"[Model] Trained model saved to {MODEL_PATH}")
    return trained_model

def load_trained_model(input_dimension, config, path=MODEL_PATH):
    model = create_model(input_dimension, config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded trained model from {path}")
    return model

if __name__ == "__main__":
    from backtest import run_backtest
    config = load_config()
    features, returns = compute_features(config["TICKERS"], config["START_DATE"], config["END_DATE"], config["FEATURES"])
    _, _, test_dataset = prepare_main_datasets(features, returns, config)
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        trained_model = load_trained_model(test_dataset[0][0].shape[1], config)
    else:
        trained_model = train_main_model(config)
    
    # Run backtest and capture strategy and benchmark metrics
    results = run_backtest(
        device=DEVICE,
        initial_capital=config["INITIAL_CAPITAL"],
        split_date=config["SPLIT_DATE"],
        lookback=config["LOOKBACK"],
        max_leverage=config["MAX_LEVERAGE"],
        compute_features=compute_features,
        normalize_features=normalize_features,
        tickers=config["TICKERS"],
        start_date=config["START_DATE"],
        end_date=config["END_DATE"],
        features=config["FEATURES"],
        test_chunk_months=config["TEST_CHUNK_MONTHS"],
        model=trained_model,
        plot=True,
        config=config,
        retrain=config["RETRAIN"]
    )

    # === DEBUG ADDITION ===
    def debug_metric(name, val):
        if val is None:
            print(f"[Debug] {name} is None")
        elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            print(f"[Debug] {name} is NaN or Inf")
        else:
            print(f"[Debug] {name} = {val}")

    debug_metric("strategy_sharpe", results.get("strategy_sharpe"))
    debug_metric("strategy_max_drawdown", results.get("strategy_max_drawdown"))
    debug_metric("benchmark_sharpe", results.get("benchmark_sharpe"))
    debug_metric("benchmark_max_drawdown", results.get("benchmark_max_drawdown"))

    if "strategy_returns" in results:
        strat_returns = results["strategy_returns"]
        print(f"[Debug] strategy_returns length: {len(strat_returns)}")
        print(f"[Debug] strategy_returns sample: {strat_returns[:5]}")
        if np.any(np.isnan(strat_returns)):
            print("[Debug] strategy_returns contains NaNs")
        if np.any(np.isinf(strat_returns)):
            print("[Debug] strategy_returns contains Infs")

    if "strategy_equity_curve" in results:
        equity = results["strategy_equity_curve"]
        print(f"[Debug] strategy_equity_curve length: {len(equity)}")
        print(f"[Debug] strategy_equity_curve sample: {equity[:5]}")
        if np.any(np.isnan(equity)):
            print("[Debug] strategy_equity_curve contains NaNs")
        if np.any(np.isinf(equity)):
            print("[Debug] strategy_equity_curve contains Infs")
    # === End Debug Addition ===

    sharpe_ratio = results.get("strategy_sharpe", float('nan'))
    max_drawdown = results.get("strategy_max_drawdown", float('nan'))
    benchmark_sharpe = results.get("benchmark_sharpe", float('nan'))
    benchmark_drawdown = results.get("benchmark_max_drawdown", float('nan'))
    performance_variance = results.get("performance_variance", {})

    print(f"Sharpe Ratio: Strategy: {sharpe_ratio * 100:.6f}%")
    print(f"Max Drawdown: Strategy: {max_drawdown * 100:.6f}%")
    print(f"Sharpe Ratio: Benchmark: {benchmark_sharpe * 100:.6f}%")
    print(f"Max Drawdown: Benchmark: {benchmark_drawdown * 100:.6f}%")

    print("Performance Variance (Std Across Chunks):")
    for k, v in performance_variance.items():
        print(f"{k}: ±{v * 100:.6f}%")

    sys.stdout.flush()
