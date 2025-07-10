import os, json, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from features import FTR_FUNC
from backtest import run_backtest
from compute_features import *
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


EARLY_STOP_PATIENCE = 2
INITIAL_CAPITAL = 100.0
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START_DATE = '2012-01-01'
END_DATE = '2025-06-01'
MODEL_PATH = "trained_model.pth"
LOAD_MODEL = True

def load_config():
    keys = ["SPLIT_DATE", "VAL_SPLIT", "PREDICT_DAYS", "LOOKBACK", "EPOCHS", "MAX_HEADS","BATCH_SIZE", "FEATURES", "MAX_LEVERAGE", "LAYER_COUNT", "DROPOUT", "DECAY",
        "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED", "RETURN_PENALTY_ENABLED","LOSS_MIN_MEAN", "LOSS_RETURN_PENALTY", "WARMUP_FRAC"]
    env_present = any(key in os.environ for key in keys)
    if env_present:
        config = {}
        for key in keys:
            val = os.environ.get(key)
            if val is None:
                raise ValueError(f"Missing env var for key: {key}")
            if key == "FEATURES":
                config[key] = val.split(",")
            elif key == "SPLIT_DATE":
                config[key] = pd.Timestamp(val)
            elif key in ["VAL_SPLIT", "MAX_LEVERAGE", "DROPOUT", "DECAY", "LOSS_MIN_MEAN", "LOSS_RETURN_PENALTY", "WARMUP_FRAC"]:
                config[key] = float(val)
            elif key in ["PREDICT_DAYS", "LOOKBACK", "EPOCHS", "MAX_HEADS", "BATCH_SIZE", "LAYER_COUNT"]:
                config[key] = int(val)
            elif key in ["FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED", "RETURN_PENALTY_ENABLED"]:
                config[key] = bool(int(val))
            else:
                config[key] = val
        return config

    else:
        with open("best_hyperparameters.json", "r") as f:
            config = json.load(f)
        config["SPLIT_DATE"] = pd.Timestamp(config["SPLIT_DATE"])
        config["FEATURES"] = config["FEATURES"].split(",") if isinstance(config["FEATURES"], str) else config["FEATURES"]
        config["VAL_SPLIT"] = float(config["VAL_SPLIT"])
        config["PREDICT_DAYS"] = int(config["PREDICT_DAYS"])
        config["LOOKBACK"] = int(config["LOOKBACK"])
        config["EPOCHS"] = int(config["EPOCHS"])
        config["MAX_HEADS"] = int(config["MAX_HEADS"])
        config["BATCH_SIZE"] = int(config["BATCH_SIZE"])
        config["MAX_LEVERAGE"] = float(config["MAX_LEVERAGE"])
        config["LAYER_COUNT"] = int(config["LAYER_COUNT"])
        config["DROPOUT"] = float(config["DROPOUT"])
        config["DECAY"] = float(config["DECAY"])
        config["FEATURE_ATTENTION_ENABLED"] = bool(int(config["FEATURE_ATTENTION_ENABLED"]))
        config["L2_PENALTY_ENABLED"] = bool(int(config["L2_PENALTY_ENABLED"]))
        config["RETURN_PENALTY_ENABLED"] = bool(int(config["RETURN_PENALTY_ENABLED"]))
        config["LOSS_MIN_MEAN"] = float(config["LOSS_MIN_MEAN"])
        config["LOSS_RETURN_PENALTY"] = float(config["LOSS_RETURN_PENALTY"])
        config["WARMUP_FRAC"] = float(config["WARMUP_FRAC"])
        return config
config = load_config()

SPLIT_DATE = config["SPLIT_DATE"]
VAL_SPLIT = config["VAL_SPLIT"]
PREDICT_DAYS = config["PREDICT_DAYS"]
LOOKBACK = config["LOOKBACK"]
EPOCHS = config["EPOCHS"]
MAX_HEADS = config["MAX_HEADS"]
BATCH_SIZE = config["BATCH_SIZE"]
FEATURES = config["FEATURES"]
MAX_LEVERAGE = config["MAX_LEVERAGE"]
LAYER_COUNT = config["LAYER_COUNT"]
DROPOUT = config["DROPOUT"]
DECAY = config["DECAY"]
FEATURE_ATTENTION_ENABLED = config["FEATURE_ATTENTION_ENABLED"]
L2_PENALTY_ENABLED = config["L2_PENALTY_ENABLED"]
RETURN_PENALTY_ENABLED = config["RETURN_PENALTY_ENABLED"]
LOSS_MIN_MEAN = config["LOSS_MIN_MEAN"]
LOSS_RETURN_PENALTY = config["LOSS_RETURN_PENALTY"]
WARMUP_FRAC = config["WARMUP_FRAC"]

class MarketDataset(Dataset):
    def __init__(self, features, returns):
        self.features = features
        self.returns = returns
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.returns[index]

class TransformerTrader(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers=LAYER_COUNT, dropout=DROPOUT, seq_len=LOOKBACK):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
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
            nn.Linear(64, len(TICKERS))
        )
    def forward(self, x):
        x = x * self.feature_weights * FEATURE_ATTENTION_ENABLED
        x = x + self.pos_embedding
        encoded = self.transformer_encoder(x)
        last_hidden = encoded[:, -1, :]
        weights = self.mlp_head(last_hidden)
        return weights

def calculate_heads(input_dim):
    if input_dim % MAX_HEADS != 0:
        for heads in range(MAX_HEADS, 0, -1):
            if input_dim % heads == 0:
                num_heads = heads
                break
    else:
        num_heads = MAX_HEADS
    return num_heads

def create_model(input_dimension):
    heads = calculate_heads(input_dimension)
    print(f"[Model] Creating TransformerTrader with input_dim={input_dimension}, heads={heads}, device={DEVICE}")
    return TransformerTrader(input_dimension, num_heads=heads).to(DEVICE, non_blocking=True)

def split_train_validation(sequences, targets, validation_ratio=VAL_SPLIT):
    total_samples = len(sequences)
    val_size = int(total_samples * validation_ratio)
    train_size = total_samples - val_size
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    val_sequences = sequences[train_size:]
    val_targets = targets[train_size:]
    return train_sequences, train_targets, val_sequences, val_targets

def create_sequences(features, returns, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(features)
    sequences = []
    targets = []
    indices = []
    for i in range(start_idx, end_idx - LOOKBACK - PREDICT_DAYS):
        feature_window = features.iloc[i:i + LOOKBACK].values.astype(np.float32)
        normalized_window = normalize_features(feature_window)
        if np.isnan(normalized_window).any():
            print(f"[Sequence][Warning] NaN detected in normalized features at index {i}")
        future_returns = returns.iloc[i + LOOKBACK:i + LOOKBACK + PREDICT_DAYS].mean().values.astype(np.float32)
        if future_returns.shape[0] != len(TICKERS):
            print(f"[Sequence][Warning] Future returns length mismatch at index {i}: {future_returns.shape}")
        sequences.append(normalized_window)
        targets.append(future_returns)
        indices.append(features.index[i + LOOKBACK])
    return sequences, targets, indices

def prepare_main_datasets(features, returns):
    sequences, targets, seq_dates = create_sequences(features, returns)
    if len(set(seq_dates)) != len(seq_dates):
        print("[Data][Warning] Duplicate dates found in sequence dates.")
    if any(pd.isna(seq_dates)):
        print("[Data][Warning] NaN detected in sequence dates.")
    train_sequences, train_targets = [], []
    test_sequences, test_targets = [], []
    for seq, tgt, date in zip(sequences, targets, seq_dates):
        if date < SPLIT_DATE:
            train_sequences.append(seq)
            train_targets.append(tgt)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)
    train_seq, train_tgt, val_seq, val_tgt = split_train_validation(train_sequences, train_targets)
    train_dataset = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_dataset = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))
    test_dataset = MarketDataset(torch.tensor(np.array(test_sequences)), torch.tensor(np.array(test_targets)))
    print(f"[Data] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, l2_lambda=1e-4):
        super().__init__()
        self.l2_lambda = l2_lambda
    def forward(self, portfolio_weights, target_returns, model=None):
        returns = (portfolio_weights * target_returns).sum(dim=1)
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + 1e-6
        sharpe_ratio = mean_return / std_return

        low_return_penalty = torch.clamp(LOSS_MIN_MEAN - mean_return, min=0.0)
        loss = -sharpe_ratio + LOSS_RETURN_PENALTY * low_return_penalty * RETURN_PENALTY_ENABLED
        l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += self.l2_lambda * l2_penalty * L2_PENALTY_ENABLED
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

def train_model_with_validation(model, train_loader, val_loader, epochs=EPOCHS):
    weight_decay = DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    learning_warmup_steps = int(total_steps * WARMUP_FRAC)
    print(f"[Scheduler] Total steps: {total_steps}, LEARNING_WARMUP steps: {learning_warmup_steps}")
    learning_scheduler = TransformerLRScheduler(optimizer, d_model=model.mlp_head[0].in_features, warmup_steps=learning_warmup_steps)
    loss_function = DifferentiableSharpeLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    lrs = []
    for epoch in range(epochs):
        print(f"[Training] Epoch {epoch+1}/{epochs}")
        model.train()
        train_losses = []

        for batch_features, batch_returns in train_loader:
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            batch_returns = batch_returns.to(DEVICE, non_blocking=True)
            raw_weights = model(batch_features)
            abs_sum = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
            scaling_factor = torch.clamp(MAX_LEVERAGE / abs_sum, max=1.0)
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
                scaling_factor = torch.clamp(MAX_LEVERAGE / weight_sum, max=1.0)
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
            if patience_counter >= EARLY_STOP_PATIENCE:
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

def train_main_model():
    features, returns = compute_features(TICKERS, START_DATE, END_DATE, FEATURES)
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = create_model(train_dataset[0][0].shape[1])
    trained_model = train_model_with_validation(model, train_loader, val_loader)
    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"[Model] Trained model saved to {MODEL_PATH}")
    return trained_model

def load_trained_model(input_dimension, path=MODEL_PATH):
    model = create_model(input_dimension)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded trained model from {path}")
    return model

def calculate_performance_metrics(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()
    if equity_curve.isna().any():
        print("[Performance] Warning: NaNs detected in equity curve after dropna (should not happen)")
    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(returns) / 252
    cagr = total_return ** (1/years) - 1
    sharpe_ratio = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
    peak_values = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak_values) / peak_values
    max_drawdown = drawdowns.min()
    return {'cagr': cagr,'sharpe_ratio': sharpe_ratio,'max_drawdown': max_drawdown}

if __name__ == "__main__":
    features, returns = compute_features(TICKERS, START_DATE, END_DATE, FEATURES)
    _, _, test_dataset = prepare_main_datasets(features, returns)
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        trained_model = load_trained_model(test_dataset[0][0].shape[1])
    else:
        trained_model = train_main_model()
    run_backtest(
        DEVICE, INITIAL_CAPITAL, SPLIT_DATE, LOOKBACK, MAX_LEVERAGE,
        compute_features, normalize_features, calculate_performance_metrics,
        TICKERS, START_DATE, END_DATE, FEATURES, trained_model, plot=True
    )
