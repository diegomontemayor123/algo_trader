import pandas as pd
import numpy as np
import ta
from load import load_config
np.seterr(all='raise')

config = load_config()
per = [int(config["SHORT_PER"]), int(config["MED_PER"]), int(config["LONG_PER"])]

# ========================== BASIC FEATURES ==========================
def add_ret(data): data['ret'] = data['close'].pct_change()
def add_price(data): data['price'] = data['close']
def add_logret(data):
    ratio = (data['close'] / data['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1)
    ratio_clean = ratio.where(ratio > 0, 1)
    logret = np.log(ratio_clean)
    data['logret'] = logret.reindex(data.index)
    return data
def add_rollret(data):
    for p in per:
        data[f'rollret_{p}'] = data['close'].pct_change(p)

# ========================== TREND & MOMENTUM ==========================
def add_sma(data):
    for p in per:
        data[f'sma_{p}'] = data['close'].rolling(p).mean()
def add_ema(data):
    for p in per: data[f'ema_{p}'] = data['close'].ewm(span=p).mean()
def add_momentum(data):
    for p in per: data[f'momentum_{p}'] = data['close'] / data['close'].shift(p) - 1
def add_macd(data):
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26
def add_acceleration(data):
    for p in per:
        momentum_col = f'momentum_{p}'
        if momentum_col not in data.columns: data[momentum_col] = data['close'] / data['close'].shift(p) - 1
        data[f'acceleration_{p}'] = data[momentum_col].diff()
def add_pricevshigh(data):
    for p in per: data[f'pricevshigh_{p}'] = data['close'] / (data['close'].rolling(p).max() + 1e-10)
def add_updownratio(data):
    for p in per:
        up = (data['close'].diff() > 0).rolling(p).sum()
        down = (data['close'].diff() < 0).rolling(p).sum()
        data[f'updownratio_{p}'] = up / (down + 1e-10)

# ========================== vol ==========================
def add_vol(data):
    if 'ret' not in data: data['ret'] = data['close'].pct_change()
    for p in per: data[f'vol_{p}'] = data['ret'].rolling(p).std()
def add_atr(data):
    for p in per:
        high = data['high']
        low = data['low']
        close = data['close']
        tr = pd.concat([high - low,(high - close.shift()).abs(),(low - close.shift()).abs()], axis=1).max(axis=1)
        data[f'atr_{p}'] = tr.rolling(p).mean()

def add_range(data): data['range'] = (data['high'] - data['low']) / (data['close'] + 1e-10)

def add_volchange(data):
    if 'ret' not in data.columns: data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'volchange_{p}'] = vol.pct_change()
def add_volptile(data):
    if 'ret' not in data.columns: data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'volptile_{p}'] = vol.rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).fillna(0.5)

# ========================== OSCILLATORS / MEAN REVERSION  ==========================
def add_rsi(data):
    delta = data['close'].diff().replace([np.inf, -np.inf], np.nan).fillna(0)
    for p in per:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        rs = gain / (loss + 1e-10)
        data[f'rsi_{p}'] = 100 - (100 / (1 + rs))

def add_cmo(data):
    delta = data['close'].diff().replace([np.inf, -np.inf], np.nan).fillna(0)
    for p in per:
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
def add_priceptile(data):
    for p in per: data[f'ptile_rank_{p}'] = data['close'].rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def add_adx(data):
    for p in per:
        try:
            adx = ta.trend.adx(data['high'], data['low'], data['close'], window=p)
            data[f'adx_{p}'] = adx
        except: pass
def add_entropy(data):
    if 'ret' not in data.columns: data['ret'] = data['close'].pct_change()
    for p in per:
        ret = data['ret'].fillna(0)
        def shannon_entropy(x):
            counts = np.histogram(x, bins=10)[0]
            probs = counts / counts.sum()
            return -np.sum([p * np.log2(p + 1e-10) for p in probs if p > 0])
        data[f'entropy_{p}'] = ret.rolling(p).apply(shannon_entropy, raw=True)

def add_meanabsret(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per: data[f'meanabsret_{p}'] = data['ret'].abs().rolling(p).mean()

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
    for p in per: volume_feat[f'volume_zscore_{p}'] = (series - series.rolling(p).mean()) / (series.rolling(p).std() + 1e-10)
    return volume_feat

# ========================== LAG / TIME FEATURES ==========================
def add_lag(data):
    for p in per:
        data[f'lag_{p}'] = data['close'].shift(p)
def add_trendcombo(data):
    for fast, med, slow in [(5, 20, 100), (10, 30, 200)]:
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_med = data['close'].ewm(span=med).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        data[f'trendcombo_{fast}_{med}_{slow}'] = (ema_fast > ema_med) & (ema_med > ema_slow)

# ========================== CROSS-SECTIONAL ==========================
CROSS_FEAT = {}
def setallcrossfeat(cross_feat_dict):
    global CROSS_FEAT
    CROSS_FEAT = {k: df for k, df in cross_feat_dict.items() if 'close' in df.columns}

def add_retcrossz(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        ret_df = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df})
        # align to the target asset index (do not drop rows)
        ret_df = ret_df.reindex(data.index)
        # row-wise mean/std skipping NaNs
        row_mean = ret_df.mean(axis=1, skipna=True)
        row_std = ret_df.std(axis=1, skipna=True) + 1e-10
        zscores = (ret_df.sub(row_mean, axis=0)).div(row_std, axis=0)
        if ticker in zscores.columns:
            data["retcrossz"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Z] Failed for {ticker}: {e}")

def add_crossmomentum(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        for p in per:
            mom_df = pd.DataFrame({k: df['close'] / df['close'].shift(p) - 1 for k, df in CROSS_FEAT.items()})
            mom_df = mom_df.reindex(data.index)
            row_mean = mom_df.mean(axis=1, skipna=True)
            row_std = mom_df.std(axis=1, skipna=True) + 1e-10
            zscores = (mom_df.sub(row_mean, axis=0)).div(row_std, axis=0)
            if ticker in zscores.columns:
                data[f"cross_momentum_z_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Momentum] Failed for {ticker}: {e}")

def add_crossvolz(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        for p in per:
            vol_df = pd.DataFrame({k: df['ret'].rolling(p).std() for k, df in CROSS_FEAT.items() if 'ret' in df})
            vol_df = vol_df.reindex(data.index)
            row_mean = vol_df.mean(axis=1, skipna=True)
            row_std = vol_df.std(axis=1, skipna=True) + 1e-10
            zscores = (vol_df.sub(row_mean, axis=0)).div(row_std, axis=0)
            if ticker in zscores.columns:
                data[f"cross_volz_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Vol-Z] Failed for {ticker}: {e}")

def add_crossretrank(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        for p in per:
            rollret = {k: df['close'].pct_change(p) for k, df in CROSS_FEAT.items()}
            ret_df = pd.DataFrame(rollret)
            ret_df = ret_df.reindex(data.index)
            ranks = ret_df.rank(axis=1, pct=True)
            if ticker in ranks.columns:
                data[f"cross_retrank_{p}"] = ranks[ticker]
    except Exception as e:
        print(f"[Cross-Rank] Failed for {ticker}: {e}")

def add_crossbeta(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        basket = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df}).drop(columns=[ticker], errors='ignore')
        if 'ret' not in data:
            data['ret'] = data['close'].pct_change()
        for p in per:
            # rolling cov with basket mean series; reindex to ensure alignment
            basket_mean = basket.mean(axis=1)
            rolling_cov = data['ret'].rolling(p).cov(basket_mean).reindex(data.index)
            rolling_var = basket_mean.rolling(p).var().reindex(data.index)
            data[f"cross_beta_{p}"] = rolling_cov / (rolling_var + 1e-10)
    except Exception as e:
        print(f"[Cross-Beta] Failed for {ticker}: {e}")

def add_crosscorr(data, ticker):
    global CROSS_FEAT
    if not CROSS_FEAT:
        return
    try:
        for p in per:
            peers = pd.DataFrame({k: df['ret'] for k, df in CROSS_FEAT.items() if 'ret' in df and k != ticker})
            peers = peers.reindex(data.index)
            peers_mean = peers.mean(axis=1, skipna=True)
            data[f'cross_corr_{p}'] = data['ret'].rolling(p).corr(peers_mean)
    except Exception as e:
        print(f"[Cross-Corr] Failed for {ticker}: {e}")



def add_crossrelstrength(data, ticker, benchmark='SPY'):
    global CROSS_FEAT
    if not CROSS_FEAT or benchmark not in CROSS_FEAT: return
    try:
        for p in per:
            relstrength = data['close'] / CROSS_FEAT[benchmark]['close']
            data[f"cross_relstrength_{p}"] = relstrength / relstrength.rolling(p).mean()
    except Exception as e: print(f"[Cross-RelStrength] Failed for {ticker}: {e}")


FTR_FUNC = {
    # Basic
    "ret": add_ret, "price": add_price, "logret": add_logret, "rollret": add_rollret,
    # Trend / Momentum
    "sma": add_sma, "ema": add_ema, "momentum": add_momentum, 
    "macd": add_macd,
    "acceleration":add_acceleration,
    "pricevshigh":add_pricevshigh,
    "updownratio":add_updownratio,
    # vol
    "vol": add_vol, "atr": add_atr,
    "range":add_range,"volchange":add_volchange,
    "volptile":add_volptile,
    # Oscillators/Mean ReversionRegime
     "zscore": add_zscore,"rsi": add_rsi, "cmo": add_cmo, "williams": add_williams,
     "stoch":add_stoch,
    # Structure / Regime
     "priceptile": add_priceptile,
     "adx":add_adx, "entropy":add_entropy, 
     "meanabsret":add_meanabsret,
    # Bands
    "boll": add_boll, "donchian": add_donchian,
    # Volume
    "volume": add_volume,
    # Lag/Time
    "lag": add_lag,
    "trendcombo":add_trendcombo,
    # Cross-sectional
    "retcrossz": add_retcrossz, "cross_momentum_z": add_crossmomentum,
    "cross_volz": add_crossvolz, "cross_retrank": add_crossretrank,
    "cross_relstrength": add_crossrelstrength, "cross_beta": add_crossbeta,
    "cross_corr":add_crosscorr,
}
