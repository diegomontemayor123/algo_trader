import pandas as pd
import numpy as np

PERIODS = [15, 40]

def add_ret(data):
    data['ret'] = data['close'].pct_change()

def add_price(data):
    data['price'] = data['close']

def add_log_return(data):
    days = 5

    ratio = data['close'] / data['close'].shift(1)

    # mask for problematic values (NaN or <= 0)
    problematic_mask = ratio.isna() | (ratio <= 0)

    if problematic_mask.any():
        problematic_indices = ratio.index[problematic_mask]
        for idx in problematic_indices:
            val = ratio.loc[idx]
            prev_idx_pos = data.index.get_loc(idx) - 1
            prev_idx = data.index[prev_idx_pos] if prev_idx_pos >= 0 else None
            prev_close = data['close'].loc[prev_idx] if prev_idx is not None else None
            curr_close = data['close'].loc[idx]

            print(f"[Warning] Problematic ratio value found at Index {idx}: value={val} (<=0 or NaN)")
            print(f"    Previous close (index {prev_idx}): {prev_close}")
            print(f"    Current close (index {idx}): {curr_close}")

    # Then compute log return normally (NaNs will remain)
    data['log_ret'] = np.log(ratio)
    data['log_ret_norm5'] = (data['log_ret'] - data['log_ret'].rolling(days).mean()) / (data['log_ret'].rolling(days).std() + 1e-6)

    
    # Replace invalid values with NaN for safe log
    ratio_safe = ratio.where(ratio > 0, np.nan)
    data['log_ret'] = np.log(ratio_safe)
    data['log_ret_norm5'] = (
        data['log_ret'] - data['log_ret'].rolling(days).mean()
    ) / (data['log_ret'].rolling(days).std() + 1e-6)


    if problematic.any():
        print("[Warning] Problematic ratio values found:")
        for idx in data.index[problematic]:
            val = ratio.loc[idx]
            try:
                val_log = np.log(val) if val > 0 else "invalid (<=0)"
            except Exception as e:
                val_log = f"error: {e}"
            print(f"  Index {idx}: value={val}, log(value)={val_log}")

    # Replace invalid values with NaN before log to avoid runtime errors
    ratio_safe = ratio.where(ratio > 0, np.nan)

    data['log_ret'] = np.log(ratio_safe)
    data['log_ret_norm5'] = (
        data['log_ret'] - data['log_ret'].rolling(days).mean()
    ) / (data['log_ret'].rolling(days).std() + 1e-6)

def add_rolling_returns(data):
    for p in PERIODS:
        data[f'rolling_ret{p}'] = data['close'].pct_change(p)


def add_volume(volume_df):
    col = volume_df.columns[0]
    series = volume_df[col]
    volume_features = pd.DataFrame(index=series.index)
    for p in PERIODS:
        volume_features[f'volume_zscore{p}'] = (series - series.rolling(p).mean()) / (series.rolling(p).std() + 1e-6)

    return volume_features

def add_vol(data):
    if 'ret' not in data:
        data['ret'] = data['close'].pct_change()
    for p in PERIODS:
        data[f'vol{p}'] = data['ret'].rolling(p).std()

def add_sma(data):
    for p in PERIODS:
        data[f'sma{p}'] = data['close'].rolling(p).mean()

def add_rsi(data):
    for p in PERIODS:
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        data[f'rsi{p}'] = 100 - 100 / (1 + gain / (loss + 1e-6))

def add_macd(data):
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26

def add_momentum(data):
    for p in PERIODS:
        data[f'momentum{p}'] = data['close'] / data['close'].shift(p) - 1

def add_ema(data):
    for p in PERIODS:
        data[f'ema{p}'] = data['close'].ewm(span=p).mean()

def add_bollinger(data):
    for p in PERIODS:
        sma = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'boll_upper{p}'] = sma + 2 * std
        data[f'boll_lower{p}'] = sma - 2 * std

def add_williams_r(data):
    for p in PERIODS:
        high = data['close'].rolling(p).max()
        low = data['close'].rolling(p).min()
        data[f'williams_r{p}'] = (high - data['close']) / (high - low + 1e-6)

def add_cmo(data):
    for p in PERIODS:
        delta = data['close'].diff()
        up = delta.clip(lower=0).rolling(p).sum()
        down = -delta.clip(upper=0).rolling(p).sum()
        data[f'cmo{p}'] = 100 * (up - down) / (up + down + 1e-6)

FTR_FUNC = {
    "ret": add_ret,
    "price": add_price,
    "log_ret": add_log_return,
    "rolling_ret": add_rolling_returns,
    "volume": add_volume,
    "vol": add_vol,
    "sma": add_sma,
    "rsi": add_rsi,
    "macd": add_macd,
    "momentum": add_momentum,
    "ema": add_ema,
    "boll": add_bollinger,
    "williams": add_williams_r,
    "cmo": add_cmo,
}
