import torch
import numpy as np
import pandas as pd
from feat import norm_feat
np.seterr(all='raise')

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, feat, ret):
        self.feat = feat
        self.ret = ret

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        return self.feat[index], self.ret[index]

def create_sequences(feat, ret, lback, pred_days, TICK):

    sequences = []
    targets = []
    indices = []
    for i in range(len(feat) - lback - pred_days):
        feat_win = feat.iloc[i:i + lback].values.astype(np.float32)
        norm_win = norm_feat(feat_win)

        if np.isnan(norm_win).any():
            print(f"[Prep] NaN detected in norm feat at index {i}")

        future_ret = ret.iloc[i + lback:i + lback + pred_days].mean().values.astype(np.float32)
        if len(future_ret) != len(TICK):
            print(f"[Prep] Future ret length mismatch at index {i}: {future_ret.shape}")

        sequences.append(norm_win)
        targets.append(future_ret)
        indices.append(feat.index[i + lback])
    print(f"Feature/Return DataFrame shape: {feat.shape} / {ret.shape}, num_seq: {len(sequences)} ")
    return sequences, targets, indices

def prep_data(feat, ret, config):
    print(f"Feat/Ret date range: {feat.index[0].date()} to {feat.index[-1].date()} / {ret.index[0].date()} to {ret.index[-1].date()}")
    sequences, targets, seq_dates = create_sequences(feat, ret, config["LBACK"], config["PRED_DAYS"], config["TICK"])
    if len(set(seq_dates)) != len(seq_dates): print("[Prep] Duplicate dates found in sequence dates.")
    if any(pd.isna(seq_dates)): print("[Prep] NaN detected in sequence dates.")
    train_sequences, train_targets, train_dates = [], [], []
    test_sequences, test_targets, test_dates = [], [], []
    split = pd.to_datetime(config["SPLIT"])
    for seq, tgt, date in zip(sequences, targets, pd.to_datetime(seq_dates)):
        if date < split:
            train_sequences.append(seq)
            train_targets.append(tgt)
            train_dates.append(date)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)
            test_dates.append(date)
    print(f"Train/Test date range: {train_dates[0].date()}-{train_dates[-1].date()} / {test_dates[0].date()}-{test_dates[-1].date()}, train/test_seq: {len(train_sequences)}/{len(test_sequences)}")
    val_split = config["VAL_SPLIT"]
    val_size = int(len(train_sequences) * val_split)
    train_size = len(train_sequences) - val_size
    train_seq = train_sequences[:train_size]
    train_tgt = train_targets[:train_size]
    train_dates = train_dates[:train_size]
    val_seq = train_sequences[train_size:]
    val_tgt = train_targets[train_size:]
    train_data = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_data = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))
    test_data = MarketDataset(torch.tensor(np.array(test_sequences)), torch.tensor(np.array(test_targets)))
    print(f"[Prep] Final sample counts - Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    if train_dates:  print(f"[Prep] Adj.training dates: {train_dates[0].date()}-{train_dates[-1].date()}")

    return train_data, val_data, test_data
