import os, torch, sys, csv, logging
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
    def __init__(self, dimen, num_heads, num_layers, dropout, seq_len, tickers, feature_attention_enabled):
        super().__init__()
        self.seq_len = seq_len
        self.feature_attention_enabled = feature_attention_enabled
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dimen))
        self.feature_attention = nn.Sequential(nn.Linear(dimen, dimen), nn.Tanh(), nn.Linear(dimen, dimen), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(d_model=dimen,nhead=num_heads,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(nn.Linear(dimen, 64),nn.PReLU(),nn.Dropout(dropout),nn.Linear(64, len(tickers)))
        print(f"[DEBUG] Model MLP head output dim: {len(tickers)}")
    def forward(self, x):
        if self.feature_attention_enabled:
            x = x * self.feature_attention(x)
        x = x + self.pos_embedding
        encoded = self.transformer_encoder(x)
        last_hidden = encoded[:, -1, :]
        return self.mlp_head(last_hidden)

def calculate_heads(dimen, max_heads):
    if dimen % max_heads != 0:
        for heads in range(max_heads, 0, -1):
            if dimen % heads == 0:
                return heads
    else:
        return max_heads

def create_model(dimenension, config):
    heads = calculate_heads(dimenension, config["MAX_HEADS"])
    print(f"[Model] Creating TransformerTrader with dimen={dimenension}, heads={heads}, device={DEVICE}")
    return TransformerTrader(dimen=dimenension,num_heads=heads,num_layers=config["LAYER_COUNT"],dropout=config["DROPOUT"],seq_len=config["LOOKBACK"],tickers=config["TICKERS"],feature_attention_enabled=config["FEATURE_ATTENTION_ENABLED"]).to(DEVICE, non_blocking=True)

def split_train_validation(sequences, targets, validation_ratio):
    total_samples = len(sequences)
    val_size = int(total_samples * validation_ratio)
    train_size = total_samples - val_size
    return (sequences[:train_size], targets[:train_size],    sequences[train_size:], targets[train_size:])

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, return_penalty, l1_penalty, drawdown_penalty):
        super().__init__()
        self.return_penalty = return_penalty
        self.drawdown_penalty = drawdown_penalty
        self.l1_penalty = l1_penalty
    def forward(self, portfolio_weights, target_returns, model=None):
        returns = (portfolio_weights * target_returns).sum(dim=1)
        mean_return = torch.mean(returns)
        if returns.numel() > 1 and not torch.isnan(returns).all():
            std_return = torch.std(returns, unbiased=False)
            if std_return < 1e-4:
                logging.warning("[Loss] SD - returns too low (<1e-4), skipping batch.")
                return None  
        else:
            logging.warning("[Loss] Returns invalid, skipping batch.")
            return None 
        sharpe_ratio = mean_return / (std_return + 1e-6)
        cum_returns = torch.cumsum(returns, dim=0)
        drawdown_approx = torch.nn.functional.relu(torch.cummax(cum_returns, dim=0).values - cum_returns)
        max_drawdown = torch.mean(drawdown_approx)
        loss = -sharpe_ratio - (self.return_penalty * mean_return) + (self.drawdown_penalty * max_drawdown)
        loss += self.l1_penalty * sum(p.abs().sum() for p in model.parameters())
        #beta = torch.cov(portfolio_returns, benchmark_returns)[0,1] / torch.var(benchmark_returns)
        #loss += self.beta_penalty * torch.abs(beta - target_beta)
        #print(f"[Loss] Mean Return: {mean_return.item():.6f}")
        #print(f"[Loss] Std Return: {std_return.item():.6f}")
        print(f"[Loss] Sharpe Ratio: {sharpe_ratio.item():.6f}")
        print(f"[Loss] Low Return Penalty Term: {self.return_penalty * mean_return.item():.6f}")
        print(f"[Loss] Max Drawdown Penalty Term: {self.drawdown_penalty * max_drawdown.item():.6f}")
        print(f"[Loss] L1 Penalty Term: {self.l1_penalty * l1.item() if self.l1_penalty else 0.0:.6f}")
        print(f"[Loss] Final Loss: {loss.item():.6f}\n")
        #print(f"[Returns] {returns.detach().cpu().numpy()}")
        return loss

class TransformerLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.d_model ** -0.5
        lr = scale * min(step ** (-0.5), step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.optimizer.param_groups]

def load_trained_model(dimenension, config, path=MODEL_PATH):
    model = create_model(dimenension, config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"[Model] Loaded trained model from {path}")
    return model

if __name__ == "__main__":
    import sys
    from backtest import run_backtest
    from train import train_main_model
    from compute_features import load_price_data, compute_features, normalize_features
    from data_prep import prepare_main_datasets
    from loadconfig import load_config
    from tune_data import MACRO_LIST
    config = load_config()
    print(f"[DEBUG] Configured TICKERS: {config['TICKERS']} (count: {len(config['TICKERS'])})")
    tickers = config["TICKERS"].split(",") if isinstance(config["TICKERS"], str) else config["TICKERS"]
    feature_list = config["FEATURES"].split(",") if isinstance(config["FEATURES"], str) else config["FEATURES"]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [k.strip() for k in macro_keys.split(",") if k.strip()]
    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], MACRO_LIST)
    features, returns = compute_features(tickers, feature_list, cached_data, macro_keys)
    print(f"[DEBUG] Features shape: {features.shape}, Columns: {features.columns[:5].tolist()}...")
    print(f"[DEBUG] Returns shape: {returns.shape}, Columns: {returns.columns[:5].tolist()}...")

    _, _, test_dataset = prepare_main_datasets(features, returns, config)

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        trained_model = load_trained_model(test_dataset[0][0].shape[1], config)
    else:
        trained_model = train_main_model(config, features, returns)
        torch.save(trained_model.state_dict(), MODEL_PATH)
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
    cagr = results["portfolio"].get("cagr", float('nan'))

    benchmark_sharpe = results["benchmark"].get("sharpe_ratio", float('nan'))
    benchmark_drawdown = results["benchmark"].get("max_drawdown", float('nan'))
    benchmark_cagr = results["benchmark"].get("cagr", float('nan'))
    
    weights_df = pd.read_csv("weights.csv", index_col="Date", parse_dates=True)
    exp_delta = weights_df["total_exposure"].max() - weights_df["total_exposure"].min()

    print(f"\nSharpe Ratio: Strategy: {sharpe_ratio * 100:.6f}%")
    print(f"Sharpe Ratio: Benchmark: {benchmark_sharpe * 100:.6f}%")
    print(f"Max Drawdown: Strategy: {max_drawdown * 100:.6f}%")
    print(f"Max Drawdown: Benchmark: {benchmark_drawdown * 100:.6f}%")
    print(f"CAGR: Strategy: {cagr * 100:.6f}%")
    print(f"CAGR: Benchmark: {benchmark_cagr * 100:.6f}%\n")
    print(f"Total Exposure Delta: {exp_delta:.4f}")
    print("Average Benchmark Outperformance Across Chunks:")
    for k, v in results["performance_outperformance"].items():
        print(f"{k}: {v * 100:.6f}%")
    sys.stdout.flush()
