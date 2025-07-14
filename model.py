import os, torch, sys, csv
import numpy as np
import torch.nn as nn
from compute_features import *
from loadconfig import load_config
from data_prep import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "trained_model.pth"
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,nhead=num_heads,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(nn.Linear(input_dim, 64),nn.PReLU(),nn.Dropout(dropout),nn.Linear(64, len(tickers)))
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
    return TransformerTrader(input_dim=input_dimension,num_heads=heads,num_layers=config["LAYER_COUNT"],dropout=config["DROPOUT"],seq_len=config["LOOKBACK"],tickers=config["TICKERS"],feature_attention_enabled=config["FEATURE_ATTENTION_ENABLED"]).to(DEVICE, non_blocking=True)

def split_train_validation(sequences, targets, validation_ratio):
    total_samples = len(sequences)
    val_size = int(total_samples * validation_ratio)
    train_size = total_samples - val_size
    return (sequences[:train_size], targets[:train_size],    sequences[train_size:], targets[train_size:])

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, l2_lambda, loss_min_mean, loss_return_penalty, l2_penalty_enabled):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.loss_min_mean = loss_min_mean
        self.loss_return_penalty = loss_return_penalty
        self.l2_penalty_enabled = l2_penalty_enabled
    def forward(self, portfolio_weights, target_returns, model=None):
        returns = (portfolio_weights * target_returns).sum(dim=1)
        mean_return = torch.mean(returns)
        if returns.numel() > 1 and not torch.isnan(returns).all():
            std_return = torch.std(returns, unbiased=False)
            if std_return < 1e-4:
                logging.warning("[Loss] Std dev of returns too low (<1e-6), skipping batch.")
                return None  
        else:
            logging.warning("[Loss] Returns invalid or empty, skipping batch.")
            return None 

        sharpe_ratio = mean_return / (std_return + 1e-6)
        low_return = torch.clamp(self.loss_min_mean - mean_return, min=0.0)
        loss = -sharpe_ratio + self.loss_return_penalty * low_return 
        if self.l2_penalty_enabled and model is not None:
            l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += self.l2_lambda * l2_penalty
        return loss

class TransformerLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=50, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.d_model ** -0.5
        lr = scale * min(step ** (-0.5), step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.optimizer.param_groups]

def load_trained_model(input_dimension, config, path=MODEL_PATH):
    model = create_model(input_dimension, config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded trained model from {path}")
    return model

def save_top_features_csv(model, feature_names, filepath="top_features.csv", top_k=50):
    weights = model.feature_weights.detach().cpu().numpy()
    feature_weight_pairs = list(zip(feature_names, weights))
    feature_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Weight"])
        for feat, weight in feature_weight_pairs[:top_k]:
            writer.writerow([feat, weight])
    print(f"Saved top {top_k} features to {filepath}")


if __name__ == "__main__":
    import sys
    from backtest import run_backtest
    from train import train_main_model
    from compute_features import load_price_data, compute_features, normalize_features
    from data_prep import prepare_main_datasets
    from loadconfig import load_config

    config = load_config()

    # Parse tickers, features, and macro keys from config
    tickers = config["TICKERS"].split(",") if isinstance(config["TICKERS"], str) else config["TICKERS"]
    feature_list = config["FEATURES"].split(",") if isinstance(config["FEATURES"], str) else config["FEATURES"]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [k.strip() for k in macro_keys.split(",") if k.strip()]
    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, feature_list, cached_data, macro_keys)
    _, _, test_dataset = prepare_main_datasets(features, returns, config)

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        trained_model = load_trained_model(test_dataset[0][0].shape[1], config)
    else:
        trained_model = train_main_model(config, features, returns)
    save_top_features_csv(trained_model, features.columns.tolist(), filepath="top_features.csv", top_k=50)
    results = run_backtest(
        device=DEVICE,
        initial_capital=config["INITIAL_CAPITAL"],
        split_date=config["SPLIT_DATE"],
        lookback=config["LOOKBACK"],
        max_leverage=config["MAX_LEVERAGE"],
        compute_features=compute_features,
        normalize_features=normalize_features,
        tickers=tickers,
        start_date=config["START_DATE"],
        end_date=config["END_DATE"],
        features=feature_list,
        macro_keys=macro_keys,
        test_chunk_months=config["TEST_CHUNK_MONTHS"],
        model=trained_model,
        plot=True,
        config=config,
        retrain_window=config["RETRAIN_WINDOW"],
    )

    sharpe_ratio = results["portfolio"].get("sharpe_ratio", float('nan'))
    max_drawdown = results["portfolio"].get("max_drawdown", float('nan'))
    benchmark_sharpe = results["benchmark"].get("sharpe_ratio", float('nan'))
    benchmark_drawdown = results["benchmark"].get("max_drawdown", float('nan'))
    cagr = results["portfolio"].get("cagr", float('nan'))
    benchmark_cagr = results["benchmark"].get("cagr", float('nan'))

    print(f"\nSharpe Ratio: Strategy: {sharpe_ratio * 100:.6f}%")
    print(f"Sharpe Ratio: Benchmark: {benchmark_sharpe * 100:.6f}%")
    print(f"Max Drawdown: Strategy: {max_drawdown * 100:.6f}%")
    print(f"Max Drawdown: Benchmark: {benchmark_drawdown * 100:.6f}%")
    print(f"CAGR: Strategy: {cagr * 100:.6f}%")
    print(f"CAGR: Benchmark: {benchmark_cagr * 100:.6f}%\n")

    print("Average Benchmark Outperformance Across Chunks:")
    for k, v in results["performance_outperformance"].items():
        print(f"{k}: {v * 100:.6f}%")
    sys.stdout.flush()
