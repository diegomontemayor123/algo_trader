import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load hyperparams from env or defaults
DROPOUT = float(os.environ.get("DROPOUT", 0.15))
MAX_LEVERAGE = float(os.environ.get("MAX_LEVERAGE", 1.0))
LOOKBACK = int(os.environ.get("LOOKBACK", 94))
PREDICT_DAYS = int(os.environ.get("PREDICT_DAYS", 8))
LAYER_COUNT = int(os.environ.get("LAYER_COUNT", 6))
DECAY = float(os.environ.get("DECAY", 0.04))
FEATURE_ATTENTION_ENABLED = bool(int(os.environ.get("FEATURE_ATTENTION_ENABLED", 1)))
RETURN_PENALTY_ENABLED = bool(int(os.environ.get("RETURN_PENALTY_ENABLED", 1)))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 55))
EPOCHS = int(os.environ.get("EPOCHS", 20))
INITIAL_CAPITAL = 100.0
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

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

def calculate_heads(input_dim, max_heads=20):
    if input_dim % max_heads != 0:
        for heads in range(max_heads, 0, -1):
            if input_dim % heads == 0:
                return heads
    return max_heads

def create_model(input_dim, dropout=None):
    heads = calculate_heads(input_dim)
    effective_dropout = dropout if dropout is not None else DROPOUT

    print(f"[Model] Creating TransformerTrader with input_dim={input_dim}, heads={heads}, "
          f"layers={LAYER_COUNT}, dropout={effective_dropout}")

    model = TransformerTrader(input_dim, heads, LAYER_COUNT, effective_dropout, LOOKBACK)
    return model.to(DEVICE)


def create_sequences(features, returns):
    X, y, dates = [], [], []
    for i in range(len(features) - LOOKBACK - PREDICT_DAYS + 1):
        X.append(features.iloc[i:i+LOOKBACK].values)
        y.append(returns.iloc[i+LOOKBACK + PREDICT_DAYS - 1].values)
        dates.append(features.index[i + LOOKBACK + PREDICT_DAYS - 1])
    return X, y, dates

def split_train_validation(sequences, targets, validation_ratio=0.2):
    n = len(sequences)
    split_idx = int(n * (1 - validation_ratio))
    return sequences[:split_idx], targets[:split_idx], sequences[split_idx:], targets[split_idx:]
import torch.optim as optim
import torch.nn as nn
import numpy as np

def train_model_with_validation(model, train_loader, val_loader, learning_rate=1e-3, epochs=20, weight_decay=0.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.float().to(DEVICE), y_batch.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.float().to(DEVICE), y_val.float().to(DEVICE)
                pred_val = model(x_val)
                val_loss = criterion(pred_val, y_val)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model


def calculate_performance_metrics(portfolio_values):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    cagr = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_drawdown = ((pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)).max()) / pd.Series(portfolio_values).cummax().max()
    return {
        "total_return": total_return,
        "CAGR": cagr,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

def calculate_additional_metrics(portfolio_values):
    return calculate_performance_metrics(portfolio_values)
