import pandas as pd
import numpy as np
from load import load_config
config= load_config()

raw_pers = config["FEAT_PER"]

if isinstance(raw_pers, str):       per = list(map(int, raw_pers.split(",")))
elif isinstance(raw_pers, list):    per = list(map(int, raw_pers))
else:                               raise TypeError(f"Invalid type for FEAT_PER: {type(raw_pers)}")

def add_ret(data): data['ret'] = data['close'].pct_change()

def add_price(data): data['price'] = data['close']

def add_log_ret(data):
    days = 5
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['log_ret_norm5'] = (data['log_ret'] - data['log_ret'].rolling(days).mean()) / (data['log_ret'].rolling(days).std() + 1e-10)

def add_roll_ret(data):
    for p in per: data[f'roll_ret{p}'] = data['close'].pct_change(p)

def add_volume(volume_df):
    col = volume_df.columns[0]
    series = volume_df[col]
    volume_feat = pd.DataFrame(index=series.index)
    for p in per: volume_feat[f'volume_zscore{p}'] = (series - series.rolling(p).mean()) / (series.rolling(p).std() + 1e-10)
    return volume_feat

def add_vol(data):
    if 'ret' not in data: data['ret'] = data['close'].pct_change()
    for p in per: data[f'vol{p}'] = data['ret'].rolling(p).std()

def add_sma(data):
    for p in per: data[f'sma{p}'] = data['close'].rolling(p).mean()

def add_rsi(data):
    for p in per:
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        data[f'rsi{p}'] = 100 - 100 / (1 + gain / (loss + 1e-10))

def add_macd(data):
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26

def add_momentum(data):
    for p in per: data[f'momentum{p}'] = data['close'] / data['close'].shift(p) - 1

def add_ema(data):
    for p in per: data[f'ema{p}'] = data['close'].ewm(span=p).mean()

def add_boll(data):
    for p in per:
        sma = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'boll_upper{p}'] = sma + 2 * std
        data[f'boll_lower{p}'] = sma - 2 * std

def add_williams_r(data):
    for p in per:
        high = data['close'].rolling(p).max()
        low = data['close'].rolling(p).min()
        data[f'williams_r{p}'] = (high - data['close']) / (high - low + 1e-10)

def add_cmo(data):
    for p in per:
        delta = data['close'].diff()
        up = delta.clip(lower=0).rolling(p).sum()
        down = -delta.clip(upper=0).rolling(p).sum()
        data[f'cmo{p}'] = 100 * (up - down) / (up + down + 1e-10)


CROSS_FEAT = {} 
def set_all_cross_feat(cross_feat_dict):
    global CROSS_FEAT
    CROSS_FEAT = cross_feat_dict

def add_ret_cross_z(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        print(f"[Cross-Z] Skipping: ALL_FEAT not set.")
        return
    try:
        ret_df = pd.DataFrame({k: df[f"ret"] for k, df in CROSS_FEAT.items() if 'ret' in df})
        ret_df = ret_df.dropna()
        zscores = (ret_df.sub(ret_df.mean(axis=1), axis=0).div(ret_df.std(axis=1) + 1e-10, axis=0))
        if ticker in zscores:data["ret_cross_z"] = zscores[ticker]
    except Exception as e:print(f"[Cross-Z] Failed for {ticker}: {e}")



FTR_FUNC = {"ret": add_ret,"price": add_price,
            "log_ret": add_log_ret,"roll_ret": add_roll_ret,
             "volume": add_volume,"vol": add_vol,
             "sma": add_sma,"rsi": add_rsi,
             "macd": add_macd,"momentum": add_momentum,
             "ema": add_ema,"boll": add_boll,
             "williams": add_williams_r,"cmo": add_cmo,
             "ret_cross_z": add_ret_cross_z, 
            }
