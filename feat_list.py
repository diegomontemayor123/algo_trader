import pandas as pd
import numpy as np
import ta
from load import load_config

config = load_config()

raw_pers = config["FEAT_PER"]

if isinstance(raw_pers, str):
    per = list(map(int, raw_pers.split(",")))
elif isinstance(raw_pers, list):
    per = list(map(int, raw_pers))
else:
    raise TypeError(f"Invalid type for FEAT_PER: {type(raw_pers)}")

# ========================== BASIC FEATURES ==========================
def add_ret(data): data['ret'] = data['close'].pct_change()
def add_price(data): data['price'] = data['close']

def add_log_ret(data):
    days = 5
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['log_ret_norm5'] = (data['log_ret'] - data['log_ret'].rolling(days).mean()) / (data['log_ret'].rolling(days).std() + 1e-10)

def add_roll_ret(data):
    for p in per:
        data[f'roll_ret_{p}'] = data['close'].pct_change(p)

# ========================== TREND & MOMENTUM ==========================
def add_sma(data):
    for p in per:
        data[f'sma_{p}'] = data['close'].rolling(p).mean()

def add_ema(data):
    for p in per:
        data[f'ema_{p}'] = data['close'].ewm(span=p).mean()

def add_momentum(data):
    for p in per:
        data[f'momentum_{p}'] = data['close'] / data['close'].shift(p) - 1

def add_macd(data):
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26

def add_acceleration(data):
    for p in per:
        momentum_col = f'momentum_{p}'
        if momentum_col not in data.columns:
            data[momentum_col] = data['close'] / data['close'].shift(p) - 1
        data[f'acceleration_{p}'] = data[momentum_col].diff()

def add_price_vs_high(data):
    for p in per:
        data[f'price_vs_high_{p}'] = data['close'] / (data['close'].rolling(p).max() + 1e-10)

def add_up_down_ratio(data):
    for p in per:
        up = (data['close'].diff() > 0).rolling(p).sum()
        down = (data['close'].diff() < 0).rolling(p).sum()
        data[f'up_down_ratio_{p}'] = up / (down + 1e-10)

# ========================== vol ==========================
def add_vol(data):
    if 'ret' not in data:
        data['ret'] = data['close'].pct_change()
    for p in per:
        data[f'vol_{p}'] = data['ret'].rolling(p).std()

def add_atr(data):
    for p in per:
        high = data['high']
        low = data['low']
        close = data['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        data[f'atr_{p}'] = tr.rolling(p).mean()

def add_range(data):
    data['range'] = (data['high'] - data['low']) / (data['close'] + 1e-10)

def add_vol_change(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'vol_change_{p}'] = vol.pct_change()


def add_vol_ptile(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'vol_ptile_{p}'] = vol.rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

# ========================== OSCILLATORS / MEAN REVERSION  ==========================
def add_rsi(data):
    for p in per:
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        data[f'rsi_{p}'] = 100 - 100 / (1 + gain / (loss + 1e-10))

def add_cmo(data):
    for p in per:
        delta = data['close'].diff()
        up = delta.clip(lower=0).rolling(p).sum()
        down = -delta.clip(upper=0).rolling(p).sum()
        data[f'cmo_{p}'] = 100 * (up - down) / (up + down + 1e-10)

def add_williams(data):
    for p in per:
        high = data['close'].rolling(p).max()
        low = data['close'].rolling(p).min()
        data[f'williams_{p}'] = (high - data['close']) / (high - low + 1e-10)

def add_zscore(data):
    for p in per:
        mean = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'zscore_{p}'] = (data['close'] - mean) / (std + 1e-10)

def add_stoch(data):
    for p in per:
        low_min = data['low'].rolling(p).min()
        high_max = data['high'].rolling(p).max()
        data[f'stoch_{p}'] = 100 * (data['close'] - low_min) / (high_max - low_min + 1e-10)

# ========================== REGIME ==========================

def add_price_ptiles(data):
    for p in per:
        data[f'ptile_rank_{p}'] = data['close'].rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def add_adx(data):
    for p in per:
        try:
            adx = ta.trend.adx(data['high'], data['low'], data['close'], window=p)
            data[f'adx_{p}'] = adx
        except:
            pass
            
def add_entropy(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        ret = data['ret'].fillna(0)
        def shannon_entropy(x):
            counts = np.histogram(x, bins=10)[0]
            probs = counts / counts.sum()
            return -np.sum([p * np.log2(p + 1e-10) for p in probs if p > 0])
        data[f'entropy_{p}'] = ret.rolling(p).apply(shannon_entropy, raw=True)

def add_mean_abs_return(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        data[f'mean_abs_ret_{p}'] = data['ret'].abs().rolling(p).mean()

# ========================== BANDS / CHANNELS ==========================
def add_boll(data):
    for p in per:
        sma = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'boll_Upper_{p}'] = sma + 2 * std
        data[f'boll_Lower_{p}'] = sma - 2 * std

def add_donchian(data):
    for p in per:
        high = data['high'].rolling(p).max()
        low = data['low'].rolling(p).min()
        data[f'donchian_{p}'] = (high - low) / data['close']

# ========================== VOLUME ==========================
def add_volume(volume_df):
    col = volume_df.columns[0]
    series = volume_df[col]
    volume_feat = pd.DataFrame(index=series.index)
    for p in per:
        volume_feat[f'volume_zscore_{p}'] = (series - series.rolling(p).mean()) / (series.rolling(p).std() + 1e-10)
    return volume_feat

# ========================== LAG / TIME FEATURES ==========================
def add_lags(data):
    for p in per:
        data[f'lag_close_{p}'] = data['close'].shift(p)

def add_trend_combo(data):
    for fast, med, slow in [(5, 20, 100), (10, 30, 200)]:
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_med = data['close'].ewm(span=med).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        data[f'trend_combo_{fast}_{med}_{slow}'] = (ema_fast > ema_med) & (ema_med > ema_slow)

# ========================== CROSS-SECTIONAL ==========================
CROSS_FEAT = {}
def set_all_cross_feat(cross_feat_dict):
    global CROSS_FEAT
    CROSS_FEAT = {k: df for k, df in cross_feat_dict.items() if 'close' in df.columns}

def add_ret_cross_z(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        ret_df = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df})
        ret_df = ret_df.dropna()
        zscores = (ret_df.sub(ret_df.mean(axis=1), axis=0).div(ret_df.std(axis=1) + 1e-10, axis=0))
        if ticker in zscores:
            data["ret_cross_z"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Z] Failed for {ticker}: {e}")

def add_cross_momentum(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT: return
    try:
        for p in per:
            mom_df = pd.DataFrame({k: df['close'] / df['close'].shift(p) - 1 for k, df in CROSS_FEAT.items()})
            zscores = (mom_df.sub(mom_df.mean(axis=1), axis=0).div(mom_df.std(axis=1) + 1e-10, axis=0))
            if ticker in zscores: data[f"cross_momentum_z_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Momentum] Failed for {ticker}: {e}")

def add_cross_vol_z(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT: return
    try:
        for p in per:
            vol_df = pd.DataFrame({k: df['ret'].rolling(p).std() for k, df in CROSS_FEAT.items() if 'ret' in df})
            zscores = (vol_df.sub(vol_df.mean(axis=1), axis=0).div(vol_df.std(axis=1) + 1e-10, axis=0))
            if ticker in zscores: data[f"cross_vol_z_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Vol-Z] Failed for {ticker}: {e}")

def add_cross_ret_rank(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT: return
    try:
        for p in per:
            roll_ret = {k: df['close'].pct_change(p) for k, df in CROSS_FEAT.items()}
            ret_df = pd.DataFrame(roll_ret)
            ranks = ret_df.rank(axis=1, pct=True)
            if ticker in ranks: data[f"cross_ret_rank_{p}"] = ranks[ticker]
    except Exception as e:
        print(f"[Cross-Rank] Failed for {ticker}: {e}")

def add_cross_rel_strength(data, ticker, benchmark='SPY'):
    global CROSS_FEAT
    if not CROSS_FEAT or benchmark not in CROSS_FEAT: return
    try:
        for p in per:
            rel_strength = data['close'] / CROSS_FEAT[benchmark]['close']
            data[f"cross_rel_strength_{p}"] = rel_strength / rel_strength.rolling(p).mean()
    except Exception as e:
        print(f"[Cross-RelStrength] Failed for {ticker}: {e}")

def add_cross_beta(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT: return
    try:
        basket = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df}).drop(columns=[ticker], errors='ignore')
        if 'ret' not in data:
            data['ret'] = data['close'].pct_change()
        for p in per:
            rolling_cov = data['ret'].rolling(p).cov(basket.mean(axis=1))
            rolling_var = basket.mean(axis=1).rolling(p).var()
            data[f"cross_beta_{p}"] = rolling_cov / (rolling_var + 1e-10)
    except Exception as e:
        print(f"[Cross-Beta] Failed for {ticker}: {e}")

def add_cross_corr(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT: return
    try:
        for p in per:
            peers = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df and k != ticker})
            data[f'cross_corr_{p}'] = data['ret'].rolling(p).corr(peers.mean(axis=1))
    except Exception as e:
        print(f"[Cross-Corr] Failed for {ticker}: {e}")

FTR_FUNC = {
    # Basic
    "ret": add_ret, "price": add_price, "log_ret": add_log_ret, "roll_ret": add_roll_ret,
    # Trend / Momentum
    "sma": add_sma, "ema": add_ema, "momentum": add_momentum, 
    "macd": add_macd,
    "acceleration":add_acceleration,
    "price_vs_high":add_price_vs_high,
    "up_down_ratio":add_up_down_ratio,
    # vol
    "vol": add_vol, "atr": add_atr,
    "range":add_range,"vol_change":add_vol_change,
    "vol_ptile":add_vol_ptile,
    # Oscillators/Mean ReversionRegime
     "zscore": add_zscore,"rsi": add_rsi, "cmo": add_cmo, "williams": add_williams,
     "stoch":add_stoch,
    # Structure / Regime
     "price_ptile": add_price_ptiles,
     "adx":add_adx, "entropy":add_entropy, 
     "mean_abs_return":add_mean_abs_return,
    # Bands
    "boll": add_boll, "donchian": add_donchian,
    # Volume
    "volume": add_volume,
    # Lag/Time
    "lags": add_lags,
    "trend_combo":add_trend_combo,
    # Cross-sectional
    "ret_cross_z": add_ret_cross_z, "cross_momentum_z": add_cross_momentum,
    "cross_vol_z": add_cross_vol_z, "cross_ret_rank": add_cross_ret_rank,
    "cross_rel_strength": add_cross_rel_strength, "cross_beta": add_cross_beta,
    "cross_corr":add_cross_corr,
}
