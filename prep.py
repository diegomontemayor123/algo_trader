import torch
import numpy as np
import pandas as pd
np.seterr(all='raise')

class RollingScaler:
    def __init__(self, eps=1e-10):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < self.eps] = 1.0

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            return (X - self.mean_) / self.std_
        elif X.ndim == 3:
            shp = X.shape
            flat = X.reshape(-1, shp[-1])
            flat = (flat - self.mean_) / self.std_
            return flat.reshape(shp)
        else:
            raise ValueError("Unsupported shape for transform")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, feat, ret):
        self.feat = feat
        self.ret = ret

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        return self.feat[index], self.ret[index]

def create_sequences(feat, ret, lback, pred_days, TICK):
    sequences, targets, indices = [], [], []
    for i in range(len(feat) - lback - pred_days + 1):
        feat_win = feat.iloc[i:i + lback].values.astype(np.float32)
        future_ret = ret.iloc[i + lback:i + lback + pred_days].mean().values.astype(np.float32)
        if len(future_ret) != len(TICK):
            print(f"[Prep] Future ret length mismatch at index {i}: {future_ret.shape}")
        sequences.append(feat_win)
        targets.append(future_ret)
        indices.append(feat.index[i + lback])
    print(f"[Prep] Feature/Return DataFrame shape: {feat.shape} / {ret.shape}, num_seq: {len(sequences)} ")
    return sequences, targets, indices

def zscore_shrink(local_seq, global_seq, anchor_seq, alpha=None, beta=None):
    local_z = (local_seq - local_seq.mean(axis=0, keepdims=True)) / (local_seq.std(axis=0, keepdims=True) + 1e-8)
    anchor_seq = np.broadcast_to(anchor_seq, local_seq.shape)
    return alpha * local_z + (1 - alpha - beta) * global_seq + beta * anchor_seq

def prep_data(feat, ret, config, anchor_date=None):
    print(f"[Prep] Feat/Ret date range: {feat.index[0].date()} - {feat.index[-1].date()} / {ret.index[0].date()} - {ret.index[-1].date()}")
    sequences, targets, seq_dates = create_sequences(feat, ret, config["LBACK"], config["PRED_DAYS"], config["TICK"])

    split = pd.to_datetime(config["SPLIT"])
    train_sequences, train_targets, train_dates = [], [], []
    test_sequences, test_targets, test_dates = [], [], []

    seq_dates_dt = pd.to_datetime(seq_dates)
    for seq, tgt, date in zip(sequences, targets, seq_dates_dt):
        if date < split:
            train_sequences.append(seq)
            train_targets.append(tgt)
            train_dates.append(date)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)
            test_dates.append(date)

    print(f"[Prep] Train/Test date range: {train_dates[0].date()} - {train_dates[-1].date()} / {test_dates[0].date()} - {test_dates[-1].date()}, train/test_seq: {len(train_sequences)}/{len(test_sequences)}")

    val_size = int(len(train_sequences) * config["VAL_SPLIT"])
    train_size = len(train_sequences) - val_size
    train_seq, train_tgt, train_dates_adj = train_sequences[:train_size], train_targets[:train_size], train_dates[:train_size]
    val_seq, val_tgt = train_sequences[train_size:], train_targets[train_size:]

    def compute_anchor_with_decay(anchor_seqs, decay=config["Z_DECAY"]):
        if len(anchor_seqs) == 0:  return None 
        weights = np.array([decay**(len(anchor_seqs)-i-1) for i in range(len(anchor_seqs))])
        weights = weights / weights.sum() 
        anchor_seq = np.zeros_like(anchor_seqs[0])
        for w, seq in zip(weights, anchor_seqs):anchor_seq += w * seq
        return anchor_seq

    if anchor_date is not None:
        anchor_date = pd.to_datetime(anchor_date)
        anchor_seqs = [s for s, date in zip(sequences, seq_dates_dt) if (date >= anchor_date) and (date < split)]
        if len(anchor_seqs) == 0:
            print(f"[Prep] No sequences found for anchor date {anchor_date}. Using global mean as anchor.")
            anchor_seq = np.zeros_like(np.vstack(train_seq)[0])
        else: anchor_seq = compute_anchor_with_decay(anchor_seqs, decay=config["Z_DECAY"])
        seq_for_fit = anchor_seqs
    else:
        anchor_seq = np.zeros_like(np.vstack(train_seq)[0])
        seq_for_fit = train_seq

    scaler = RollingScaler()
    scaler.fit(np.vstack(seq_for_fit))

    alpha = config.get("Z_ALPHA")
    beta =  config.get("Z_BETA")

    train_seq_scaled = np.array([
        zscore_shrink(local_seq, scaler.transform(local_seq), anchor_seq, alpha=alpha, beta=beta)
        for local_seq in train_seq])
    val_seq_scaled = np.array([
        zscore_shrink(local_seq, scaler.transform(local_seq), anchor_seq, alpha=alpha, beta=beta)
        for local_seq in val_seq]) if len(val_seq) else np.empty((0,))
    test_seq_scaled = np.array([
        zscore_shrink(local_seq, scaler.transform(local_seq), anchor_seq, alpha=alpha, beta=beta)
        for local_seq in test_sequences]) if len(test_sequences) else np.empty((0,))

    train_data = MarketDataset(torch.tensor(train_seq_scaled), torch.tensor(np.array(train_tgt)))
    val_data = MarketDataset(torch.tensor(val_seq_scaled), torch.tensor(np.array(val_tgt)))
    test_data = MarketDataset(torch.tensor(test_seq_scaled), torch.tensor(np.array(test_targets)))

    print(f"[Prep] Final sample counts - Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    if train_dates_adj:
        print(f"[Prep] Adj.training dates: {train_dates_adj[0].date()} - {train_dates_adj[-1].date()}")

    return train_data, val_data, test_data, scaler, anchor_seq
