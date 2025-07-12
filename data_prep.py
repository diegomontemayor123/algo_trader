import torch
import numpy as np
import pandas as pd
from compute_features import normalize_features

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, features, returns):
        self.features = features
        self.returns = returns
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.returns[index]

def create_sequences(features, returns, lookback, predict_days, tickers):
    sequences = []
    targets = []
    indices = []
    for i in range(len(features) - lookback - predict_days):
        feature_window = features.iloc[i:i + lookback].values.astype(np.float32)
        normalized_window = normalize_features(feature_window)
        if np.isnan(normalized_window).any():
            print(f"[Sequence][Warning] NaN detected in normalized features at index {i}")
        future_returns = returns.iloc[i + lookback:i + lookback + predict_days].mean().values.astype(np.float32)
        if len(future_returns) != len(tickers):
            print(f"[Sequence][Warning] Future returns length mismatch at index {i}: {future_returns.shape}")
        sequences.append(normalized_window)
        targets.append(future_returns)
        indices.append(features.index[i + lookback])
    return sequences, targets, indices

def prepare_main_datasets(features, returns, config):
    sequences, targets, seq_dates = create_sequences(features, returns, config["LOOKBACK"], config["PREDICT_DAYS"], config["TICKERS"])
    if len(set(seq_dates)) != len(seq_dates):
        print("[Data][Warning] Duplicate dates found in sequence dates.")
    if any(pd.isna(seq_dates)):
        print("[Data][Warning] NaN detected in sequence dates.")
    train_sequences, train_targets, test_sequences, test_targets = [], [], [], []
    split_date = pd.to_datetime(config["SPLIT_DATE"])
    for seq, tgt, date in zip(sequences, targets, pd.to_datetime(seq_dates)):
        if date < split_date:
            train_sequences.append(seq)
            train_targets.append(tgt)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)
    val_split = config.get("VAL_SPLIT", 0.2)
    val_size = int(len(train_sequences) * val_split)
    train_size = len(train_sequences) - val_size
    train_seq = train_sequences[:train_size]
    train_tgt = train_targets[:train_size]
    val_seq = train_sequences[train_size:]
    val_tgt = train_targets[train_size:]
    train_dataset = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_dataset = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))
    test_dataset = MarketDataset(torch.tensor(np.array(test_sequences)), torch.tensor(np.array(test_targets)))
    print(f"[Data] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset
