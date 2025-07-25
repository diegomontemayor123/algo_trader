import torch
import numpy as np
import pandas as pd
from feat import norm_feat

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, feat, ret):
        self.feat = feat
        self.ret = ret

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        return self.feat[index], self.ret[index]

def create_sequences(feat, ret, lback, pred_days, TICK):
    #print(f"[Info] Creating sequences with lookback={lback}, pred_days={pred_days}")
    #print(f"[Info] Feature DataFrame shape: {feat.shape}")
    #print(f"[Info] Return DataFrame shape: {ret.shape}")
    sequences = []
    targets = []
    indices = []
    for i in range(len(feat) - lback - pred_days):
        feat_win = feat.iloc[i:i + lback].values.astype(np.float32)
        norm_win = norm_feat(feat_win)

        if np.isnan(norm_win).any():
            print(f"[Warning] NaN detected in norm feat at index {i}")

        future_ret = ret.iloc[i + lback:i + lback + pred_days].mean().values.astype(np.float32)
        if len(future_ret) != len(TICK):
            print(f"[Warning] Future ret length mismatch at index {i}: {future_ret.shape}")

        sequences.append(norm_win)
        targets.append(future_ret)
        indices.append(feat.index[i + lback])

    #print(f"[Info] Total sequences created: {len(sequences)}")
    return sequences, targets, indices

def prep_data(feat, ret, config):
    #print(f"[Config] Start: {config['START']}, End: {config['END']}, Split: {config['SPLIT']}")
    #print(f"Feature date range: {feat.index[0]} to {feat.index[-1]}")
    #print(f"Return date range: {ret.index[0]} to {ret.index[-1]}")
    sequences, targets, seq_dates = create_sequences(feat, ret, config["LBACK"], config["PRED_DAYS"], config["TICK"])

    if len(set(seq_dates)) != len(seq_dates):
        print("[Warning] Duplicate dates found in sequence dates.")
    if any(pd.isna(seq_dates)):
        print("[Warning] NaN detected in sequence dates.")

    train_sequences, train_targets, train_dates = [], [], []
    test_sequences, test_targets, test_dates = [], [], []

    split = pd.to_datetime(config["SPLIT"])
    #print(f"[Info] Splitting data at {split.date()}")

    for seq, tgt, date in zip(sequences, targets, pd.to_datetime(seq_dates)):
        if date < split:
            train_sequences.append(seq)
            train_targets.append(tgt)
            train_dates.append(date)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)
            test_dates.append(date)

    #print(f"[Split] Total train sequences: {len(train_sequences)}")
    #print(f"[Split] Total test sequences: {len(test_sequences)}")
    if train_sequences:
        print(f"[Split] Train date range: {train_dates[0].date()} to {train_dates[-1].date()}")
    if test_sequences:
        print(f"[Split] Test date range: {test_dates[0].date()} to {test_dates[-1].date()}")

    val_split = config["VAL_SPLIT"]
    val_size = int(len(train_sequences) * val_split)
    train_size = len(train_sequences) - val_size

    #print(f"[Split] Training set size: {train_size}, Validation set size: {val_size}")

    train_seq = train_sequences[:train_size]
    train_tgt = train_targets[:train_size]
    train_dates = train_dates[:train_size]

    val_seq = train_sequences[train_size:]
    val_tgt = train_targets[train_size:]

    train_data = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_data = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))
    test_data = MarketDataset(torch.tensor(np.array(test_sequences)), torch.tensor(np.array(test_targets)))

    print(f"[Data] Final sample counts - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    if train_dates:
        print(f"[Data] Final training date range: {train_dates[0].date()} to {train_dates[-1].date()}")

    return train_data, val_data, test_data
