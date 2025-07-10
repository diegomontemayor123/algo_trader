import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from features_factory import FEATURE_FUNCTIONS
from sklearn.decomposition import DictionaryLearning

# === Configs ===
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START = '2019-01-01'
END = '2025-06-01'
INITIAL_CAPITAL = 100.0

LOOKBACK = int(os.environ.get("LOOKBACK", 52)) 
EPOCHS = int(os.environ.get("EPOCHS", 6))
N_HEADS = int(os.environ.get("N_HEADS", 6))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))
PREDICT_DAYS = int(os.environ.get("PREDICT_DAYS", 1))
FEATURES = os.environ.get("FEATURES", "ret,sma,vol,boll,cmo").split(",")
MAX_SHORT = float(os.environ.get("MAX_SHORT", -1))
MAX_LONG = float(os.environ.get("MAX_LONG", 1))
MAX_LEVERAGE = float(os.environ.get("MAX_LEVERAGE", 1.0))
SPLIT_DATE = pd.Timestamp(os.environ.get("SPLIT_DATE", "2024-01-01"))
ROLLING_SHARPE_WINDOW = int(os.environ.get("ROLLING_SHARPE_WINDOW", 20))
EARLY_STOPPING_PATIENCE = 5  # epochs
CACHE_FILE = "cached_prices.csv"
FEATURES_CACHE = "features.pkl"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Dataset ===
class MarketDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# === Model ===
class TransformerTrader(nn.Module):
    def __init__(self, input_dim, n_heads=5, n_layers=2, dropout=0.1):
        super().__init__()
        if input_dim % n_heads != 0:
            for nh in range(n_heads, 0, -1):
                if input_dim % nh == 0:
                    n_heads = nh
                    break
        self.n_heads = n_heads

        print(f"[Model] Using {n_heads} heads for input_dim {input_dim}")
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.linear = nn.Linear(input_dim, len(TICKERS))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return self.tanh(x)

def build_model(input_dim):
    print(f"[Build] Building model with input dim {input_dim}")
    return TransformerTrader(input_dim, n_heads=N_HEADS).to(DEVICE, non_blocking=True)

# === Data preparation and normalization ===
def get_price_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        df = yf.download(TICKERS, start=START, end=END)['Adj Close']
        df.to_csv(CACHE_FILE)
    return df

def compute_features_OLD(df):
    feature_data = {}
    for t in TICKERS:
        price = df[t].ffill().dropna()
        feats = []
        for f in FEATURES:
            f_func = FEATURE_FUNCTIONS[f]
            feats.append(f_func(price))
        all_feats = pd.concat(feats, axis=1)
        all_feats.columns = [f"{f}_{t}" for f in FEATURES]
        feature_data[t] = all_feats
    return pd.concat(feature_data.values(), axis=1).dropna()

def compute_features(df, lookback=LOOKBACK, n_atoms=20):
    """
    Compute features using sparse dictionary learning on rolling windows of returns.
    Each window becomes a single sparse feature vector.
    """
    # Compute returns and drop NaNs
    returns = df.pct_change().dropna()
    return_vals = returns.values
    # Create rolling windows: shape (num_samples, lookback * num_assets)
    windows = []
    for i in range(len(return_vals) - lookback):
        window = return_vals[i:i+lookback].flatten()
        windows.append(window)
    windows = np.array(windows)
    # Fit sparse dictionary learning
    dl = DictionaryLearning(n_components=n_atoms, alpha=1, max_iter=500, random_state=42)
    sparse_codes = dl.fit_transform(windows)  # shape: (num_samples, n_atoms)
    # Create index to align with returns
    new_index = returns.index[lookback:]
    feats = pd.DataFrame(sparse_codes, index=new_index, columns=[f"atom_{i}" for i in range(n_atoms)])
    # Align target returns to match feature dates
    rets = returns.loc[new_index]
    return feats, rets


def get_features():
    if os.path.exists(FEATURES_CACHE):
        with open(FEATURES_CACHE, 'rb') as f:
            feats, rets = pickle.load(f)
    else:
        df = get_price_data()
        feats = compute_features(df)
        rets = df.pct_change().shift(-1).reindex(feats.index)
        with open(FEATURES_CACHE, 'wb') as f:
            pickle.dump((feats, rets), f)
    return feats, rets

def normalize_window(window):
    mean = window.mean(axis=0)
    std = window.std(axis=0) + 1e-6
    return (window - mean) / std

def prepare_datasets(feats, rets):
    X_train, y_train = [], []
    X_val, y_val = [], []
    for i in range(len(feats) - LOOKBACK - PREDICT_DAYS):
        date = feats.index[i + LOOKBACK]
        window = feats.iloc[i:i + LOOKBACK].values.astype(np.float32)
        norm_window = normalize_window(window)
        avg_fwd_return = rets.iloc[i + LOOKBACK:i + LOOKBACK + PREDICT_DAYS].mean().values.astype(np.float32)

        if date < SPLIT_DATE:
            X_train.append(norm_window)
            y_train.append(avg_fwd_return)
        else:
            X_val.append(norm_window)
            y_val.append(avg_fwd_return)

    train_ds = MarketDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = MarketDataset(torch.tensor(X_val), torch.tensor(y_val))
    print(f"[Data] Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    return train_ds, val_ds

# === Rolling Sharpe loss with smoothing ===
# === Rolling Sharpe loss with smoothing ===
class RollingSharpeLoss:
    def __init__(self, model, l2_lambda=1e-4, window=ROLLING_SHARPE_WINDOW):
        self.model = model
        self.l2_lambda = l2_lambda
        self.window = int(window)  # in number of recent samples
        self.port_returns_buffer = []

    def reset(self):
        self.port_returns_buffer = []

    def __call__(self, weights, target_returns):
        # weights, target_returns shape: (batch_size, num_assets)
        port_returns = (weights * target_returns).sum(dim=1).detach().cpu().numpy()
        self.port_returns_buffer.extend(port_returns.tolist())
        if len(self.port_returns_buffer) > self.window:
            self.port_returns_buffer = self.port_returns_buffer[-self.window:]
        rolling_array = np.array(self.port_returns_buffer)
        mean_ret = rolling_array.mean()
        std_ret = rolling_array.std() + 1e-6
        rolling_sharpe = mean_ret / std_ret
        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss = -rolling_sharpe + self.l2_lambda * l2_norm
        return torch.tensor(loss, requires_grad=True, device=weights.device)


# === Training with validation and early stopping ===
def train_model():
   # print("[Train] Starting training...")
    feats, rets = get_features()
    train_ds, val_ds = prepare_datasets(feats, rets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = build_model(train_ds[0][0].shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = RollingSharpeLoss(model)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
       # print(f"[Train] Epoch {epoch+1}/{EPOCHS}")
        model.train()
        loss_fn.reset()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            outputs = model(batch_X)
            clipped = torch.clamp(outputs, MAX_SHORT, MAX_LONG)
            abs_sum = torch.sum(torch.abs(clipped), dim=1, keepdim=True) + 1e-6
            weights = clipped / abs_sum * MAX_LEVERAGE
            loss = loss_fn(weights, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation (using rolling-sharpe)
        model.eval()
        val_port_returns = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                outputs = model(batch_X)
                clipped = torch.clamp(outputs, MAX_SHORT, MAX_LONG)
                abs_sum = torch.sum(torch.abs(clipped), dim=1, keepdim=True) + 1e-6
                weights = clipped / abs_sum * MAX_LEVERAGE
                port_returns = (weights * batch_y).sum(dim=1)
                val_port_returns.extend(port_returns.cpu().numpy())

        # Compute rolling Sharpe on full validation series
        val_port_returns = np.array(val_port_returns)
        if len(val_port_returns) >= ROLLING_SHARPE_WINDOW:
            rolling_sharpes = [
                val_port_returns[i:i+ROLLING_SHARPE_WINDOW].mean() / 
                (val_port_returns[i:i+ROLLING_SHARPE_WINDOW].std() + 1e-6)
                for i in range(len(val_port_returns) - ROLLING_SHARPE_WINDOW)
            ]
            avg_val_loss = -np.mean(rolling_sharpes)
        else:
            avg_val_loss = float('inf')
        print(f"[Train] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "transformer_trader_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("[Train] Early stopping triggered.")
                break

# === Backtest ===
def backtest():
    print("[Backtest] Starting backtest...")
    feats, rets = get_features()
    model = build_model(feats.shape[1])
    model.load_state_dict(torch.load("transformer_trader_best.pt", map_location=DEVICE))
    model.eval()
    port_val, eq_val = [INITIAL_CAPITAL], [INITIAL_CAPITAL]
    start_idx = feats.index.get_indexer([SPLIT_DATE], method='bfill')[0]
    dates = rets.index[start_idx:]
    for i in range(start_idx - LOOKBACK, len(feats) - LOOKBACK):
        window = feats.iloc[i:i + LOOKBACK].values.astype(np.float32)
        norm = (window - window.mean()) / (window.std() + 1e-6)
        xt = torch.tensor(norm).unsqueeze(0).to(DEVICE, non_blocking=True)

        with torch.no_grad():
            raw_weights = model(xt).cpu().numpy().flatten()
        clipped = np.clip(raw_weights, MAX_SHORT, MAX_LONG)
        abs_sum = np.sum(np.abs(clipped))
        weights = clipped / abs_sum * MAX_LEVERAGE if abs_sum > 0 else clipped
        returns = rets.iloc[i + LOOKBACK].values
        port_ret = np.dot(weights, returns)
        eq_ret = np.mean(returns)
        port_val.append(port_val[-1] * (1 + port_ret))
        eq_val.append(eq_val[-1] * (1 + eq_ret))
    strat_metrics = compute_metrics(port_val)
    bench_metrics = compute_metrics(eq_val)

    print("\n[Backtest] --- Transformer Performance Summary ---")
    for k in strat_metrics:
        print(f"  {k.capitalize().replace('_', ' ')}:")
        print(f"    Transformer: {strat_metrics[k]:.2%}")
        print(f"    Benchmark:   {bench_metrics[k]:.2%}")
    plt.figure(figsize=(12, 6))
    plt.plot(dates, port_val[1:], label="Transformer Portfolio")
    plt.plot(dates, eq_val[1:], label="Equal-Weighted Benchmark")
    plt.title("Transformer Portfolio Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transformer_equity_curve.png")
    #print("[Backtest] Saved transformer_equity_curve.png")

    weights_records = []
    for i in range(len(feats) - LOOKBACK):
        window = feats.iloc[i:i + LOOKBACK].values.astype(np.float32)
        norm = (window - window.mean()) / (window.std() + 1e-6)
        xt = torch.tensor(norm).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            raw_weights = model(xt).cpu().numpy().flatten()
        clipped = np.clip(raw_weights, MAX_SHORT, MAX_LONG)
        abs_sum = np.sum(np.abs(clipped))
        weights = clipped / abs_sum * MAX_LEVERAGE if abs_sum > 0 else clipped
        weights_records.append(pd.Series(weights, index=TICKERS, name=rets.index[i + LOOKBACK]))
    pd.DataFrame(weights_records).round(4).to_csv("transformer_weights.csv")
    #print("[Backtest] Saved daily weights to transformer_weights.csv")

def compute_metrics(portfolio):
    rets = pd.Series(portfolio).pct_change().dropna()
    mean = rets.mean()
    std = rets.std()
    sharpe = mean / std * np.sqrt(252)
    total_return = portfolio[-1] / portfolio[0] - 1
    max_drawdown = ((np.maximum.accumulate(portfolio) - portfolio) / np.maximum.accumulate(portfolio)).max()
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

if __name__ == "__main__":
    train_model()
    backtest()
