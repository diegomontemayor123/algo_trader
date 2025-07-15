import pandas as pd
import numpy as np

PERIODS = [15, 40]

def add_ret(data):
    data['ret'] = data['close'].pct_change()

def add_price(data):
    data['price'] = data['close']

import numpy as np

def add_log_return(data):
    days = 5
    ratio = data['close'] / data['close'].shift(1)

    # Create a copy to clean
    ratio_clean = ratio.copy()

    # For the entire data, replace invalid ratios with NaN (<=0 or NaN)
    ratio_clean[(ratio_clean <= 0) | (ratio_clean.isna())] = np.nan

    # For the first 4 rows ONLY, replace NaNs with a small positive number to avoid warnings
    first4 = ratio_clean.index[:4]
    ratio_clean.loc[first4] = ratio_clean.loc[first4].fillna(1e-6)
    ratio_clean.loc[first4] = ratio_clean.loc[first4].apply(lambda x: max(x, 1e-6))

    # Calculate log returns on cleaned ratio
    data['log_ret'] = np.log(ratio_clean)

    # Compute rolling mean and std
    roll_mean = data['log_ret'].rolling(days).mean()
    roll_std = data['log_ret'].rolling(days).std()

    # Calculate normalized log returns
    data['log_ret_norm5'] = (data['log_ret'] - roll_mean) / (roll_std + 1e-6)

    # Override first 4 rows of normalized returns with 0 or any safe value
    data.loc[first4, 'log_ret_norm5'] = 0

    # You keep all rows, no drops


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
