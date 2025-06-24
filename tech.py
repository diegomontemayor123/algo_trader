import os, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from features_factory import FTR_FUNC
from walkforward import run_walkforward_test_with_validation
from backtest import run_backtest
from compute_features import *
from torch.optim.lr_scheduler import _LRScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOP_PATIENCE = 3
INITIAL_CAPITAL = 100.0
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START_DATE = '2012-01-01'
END_DATE = '2025-06-01'
SPLIT_DATE = pd.Timestamp(os.environ.get("SPLIT_DATE", "2023-01-01"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.2)) 
PREDICT_DAYS = int(os.environ.get("PREDICT_DAYS", 3))
LOOKBACK = int(os.environ.get("LOOKBACK", 80))
EPOCHS = int(os.environ.get("EPOCHS", 20))
MAX_HEADS = int(os.environ.get("MAX_HEADS", 20))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 60))
FEATURES = os.environ.get("FEATURES", "ret,vol,log_ret,rolling_ret,volume").split(",")
MAX_LEVERAGE = float(os.environ.get("MAX_LEVERAGE", 1.0))
LAYER_COUNT = int(os.environ.get("LAYER_COUNT", 6))
DROPOUT = float(os.environ.get("DROPOUT", 0.2))
LEARNING_WARMUP = int(os.environ.get("LEARNING_WARMUP", 350))
DECAY = float(os.environ.get("DECAY", 0.0175))

FEATURE_ATTENTION_ENABLED = bool(int(os.environ.get("FEATURE_ATTENTION_ENABLED", 0)))
L2_PENALTY_ENABLED = bool(int(os.environ.get("L2_PENALTY_ENABLED", 0)))
RETURN_PENALTY_ENABLED = bool(int(os.environ.get("RETURN_PENALTY_ENABLED", 0)))
LOSS_MIN_MEAN = float(os.environ.get("LOSS_MIN_MEAN", 0.005)) #.02 returns are 'high'
LOSS_RETURN_PENALTY = float(os.environ.get("LOSS_RETURN_PENALTY", 1))

WALKFORWARD_ENABLED = bool(int(os.environ.get("WALKFWD_ENABLED", 0)))
WALKFORWARD_STEP_SIZE = int(os.environ.get("WALKFWD_STEP", 60))
WALKFORWARD_TRAIN_WINDOW = int(os.environ.get("WALKFWD_WNDW", 365))
#Variables to add -

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
            nn.Linear(64, len(TICKERS)))
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
    def __init__(self, optimizer, d_model, warmup_steps=LEARNING_WARMUP, last_epoch=-1):
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay) #lr not used if scheduler
    learning_scheduler = TransformerLRScheduler(optimizer, d_model=model.mlp_head[0].in_features)
    loss_function = DifferentiableSharpeLoss()
    best_val_loss = float('inf')
    patience_counter = 0
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
    return model

def train_main_model():
    features, returns = compute_features(TICKERS,START_DATE,END_DATE,FEATURES)
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = create_model(train_dataset[0][0].shape[1])
    trained_model = train_model_with_validation(model, train_loader, val_loader)
    return trained_model

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
    if WALKFORWARD_ENABLED:
        run_walkforward_test_with_validation(compute_features,create_sequences,split_train_validation,MarketDataset,create_model,train_model_with_validation,normalize_features,calculate_performance_metrics,SPLIT_DATE,WALKFORWARD_STEP_SIZE,WALKFORWARD_TRAIN_WINDOW,LOOKBACK,PREDICT_DAYS,BATCH_SIZE,INITIAL_CAPITAL,MAX_LEVERAGE,DEVICE,TICKERS,START_DATE,END_DATE,FEATURES)
    else:
        trained_model = train_main_model()
        run_backtest(DEVICE,INITIAL_CAPITAL,SPLIT_DATE,LOOKBACK,MAX_LEVERAGE,compute_features,normalize_features,calculate_performance_metrics,TICKERS,START_DATE,END_DATE,FEATURES,trained_model,plot=True)

