import pandas as pd
import numpy as np
import ta
from load import load_config
np.seterr(all='raise')

config = load_config()

raw_pers = config["FEAT_PER"]

if isinstance(raw_pers, str):
    per = list(map(int, raw_pers.split(",")))
elif isinstance(raw_pers, list):
    per = list(map(int, raw_pers))
else:
    raise TypeError(f"Invalid type for FEAT_PER: {type(raw_pers)}")

# ========================== BASIC FEATURES ==========================
def ret(data): data['ret'] = data['close'].pct_change()
def price(data): data['price'] = data['close']

import numpy as np

def logret(data):
    ratio = (data['close'] / data['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1)
    ratioclean = ratio.where(ratio > 0, 1)
    logret = np.log(ratioclean)
    data['logret'] = logret.reindex(data.index)
    return data

def rollret(data):
    for p in per:
        data[f'rollret_{p}'] = data['close'].pct_change(p)

# ========================== TREND & MOMENTUM ==========================
def sma(data):
    for p in per:
        data[f'sma_{p}'] = data['close'].rolling(p).mean()

def ema(data):
    for p in per:
        data[f'ema_{p}'] = data['close'].ewm(span=p).mean()

def momentum(data):
    for p in per:
        data[f'momentum_{p}'] = data['close'] / data['close'].shift(p) - 1

def macd(data):
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26

def acceleration(data):
    for p in per:
        momentumcol = f'momentum_{p}'
        if momentumcol not in data.columns:
            data[momentumcol] = data['close'] / data['close'].shift(p) - 1
        data[f'acceleration_{p}'] = data[momentumcol].diff()

def pricevshigh(data):
    for p in per:
        data[f'pricevshigh_{p}'] = data['close'] / (data['close'].rolling(p).max() + 1e-10)

#def updownratio(data):
   # for p in per:
       # up = (data['close'].diff() > 0).rolling(p).sum()
        #down = (data['close'].diff() < 0).rolling(p).sum()
        #data[f'updownratio_{p}'] = up / (down + 1e-10)

# ========================== vol ==========================
def vol(data):
    if 'ret' not in data:
        data['ret'] = data['close'].pct_change()
    for p in per:
        data[f'vol_{p}'] = data['ret'].rolling(p).std()

def atr(data):
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

def range(data):
    data['range'] = (data['high'] - data['low']) / (data['close'] + 1e-10)

def volchange(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'volchange_{p}'] = vol.pct_change()


def volptile(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        vol = data['ret'].rolling(p).std()
        data[f'volptile_{p}'] = vol.rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

# ========================== OSCILLATORS / MEAN REVERSION  ==========================

def rsi(data):
    delta = data['close'].diff().replace([np.inf, -np.inf], np.nan).fillna(0)
    for p in per:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        rs = gain / (loss + 1e-10)
        data[f'rsi_{p}'] = 100 - (100 / (1 + rs))

def cmo(data):
    delta = data['close'].diff().replace([np.inf, -np.inf], np.nan).fillna(0)
    for p in per:
        up = delta.clip(lower=0).rolling(p).sum()
        down = -delta.clip(upper=0).rolling(p).sum()
        data[f'cmo_{p}'] = 100 * (up - down) / (up + down + 1e-10)


def williams(data):
    for p in per:
        high = data['close'].rolling(p).max()
        low = data['close'].rolling(p).min()
        data[f'williams_{p}'] = (high - data['close']) / (high - low + 1e-10)

def zscore(data):
    for p in per:
        mean = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'zscore_{p}'] = (data['close'] - mean) / (std + 1e-10)

def stoch(data):
    for p in per:
        lowmin = data['low'].rolling(p).min()
        highmax = data['high'].rolling(p).max()
        data[f'stoch_{p}'] = 100 * (data['close'] - lowmin) / (highmax - lowmin + 1e-10)

# ========================== REGIME ==========================

def priceptile(data):
    for p in per:
        data[f'ptilerank_{p}'] = data['close'].rolling(p).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def adx(data):
    for p in per:
        try:
            adx = ta.trend.adx(data['high'], data['low'], data['close'], window=p)
            data[f'adx_{p}'] = adx
        except:
            pass
            
def entropy(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        ret = data['ret'].fillna(0)
        def shannonentropy(x):
            counts = np.histogram(x, bins=10)[0]
            probs = counts / counts.sum()
            return -np.sum([p * np.log2(p + 1e-10) for p in probs if p > 0])
        data[f'entropy_{p}'] = ret.rolling(p).apply(shannonentropy, raw=True)

def meanabsret(data):
    if 'ret' not in data.columns:
        data['ret'] = data['close'].pct_change()
    for p in per:
        data[f'meanabsret_{p}'] = data['ret'].abs().rolling(p).mean()

# ========================== BANDS / CHANNELS ==========================
def boll(data):
    for p in per:
        sma = data['close'].rolling(p).mean()
        std = data['close'].rolling(p).std()
        data[f'bollupper_{p}'] = sma + 2 * std
        data[f'bolllower_{p}'] = sma - 2 * std

def donchian(data):
    for p in per:
        high = data['high'].rolling(p).max()
        low = data['low'].rolling(p).min()
        data[f'donchian_{p}'] = (high - low) / data['close']

# ========================== VOLUME ==========================
def volume(volumedf):
    col = volumedf.columns[0]
    series = volumedf[col]
    volumefeat = pd.DataFrame(index=series.index)
    for p in per:
        volumefeat[f'volumezscore_{p}'] = (series - series.rolling(p).mean()) / (series.rolling(p).std() + 1e-10)
    return volumefeat

# ========================== LAG / TIME FEATURES ==========================
def lag(data):
    for p in per:
        data[f'lag_{p}'] = data['close'].shift(p)

def trendcombo(data):
    for fast, med, slow in [(5, 20, 100), (10, 30, 200)]:
        emafast = data['close'].ewm(span=fast).mean()
        emamed = data['close'].ewm(span=med).mean()
        emaslow = data['close'].ewm(span=slow).mean()
        data[f'trendcombo{fast}{med}{slow}'] = (emafast > emamed) & (emamed > emaslow)

# ========================== CROSS-SECTIONAL ==========================
CROSSFEAT = {}
def setallcrossfeat(crossfeatdict):
    global CROSSFEAT
    CROSSFEAT = {k: df for k, df in crossfeatdict.items() if 'close' in df.columns}

def retcrossz(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT:
        return
    try:
        retdf = pd.DataFrame({k: df['ret'] for k, df in CROSSFEAT.items() if 'ret' in df})
        retdf = retdf.dropna()
        zscores = (retdf.sub(retdf.mean(axis=1), axis=0).div(retdf.std(axis=1) + 1e-10, axis=0))
        if ticker in zscores:
            data["retcrossz"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Z] Failed for {ticker}: {e}")

def crossmomentum(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT: return
    try:
        for p in per:
            momdf = pd.DataFrame({k: df['close'] / df['close'].shift(p) - 1 for k, df in CROSSFEAT.items()})
            zscores = (momdf.sub(momdf.mean(axis=1), axis=0).div(momdf.std(axis=1) + 1e-10, axis=0))
            if ticker in zscores: data[f"crossmomentumz_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Momentum] Failed for {ticker}: {e}")

def crossvolz(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT: return
    try:
        for p in per:
            voldf = pd.DataFrame({k: df['ret'].rolling(p).std() for k, df in CROSSFEAT.items() if 'ret' in df})
            zscores = (voldf.sub(voldf.mean(axis=1), axis=0).div(voldf.std(axis=1) + 1e-10, axis=0))
            if ticker in zscores: data[f"crossvolz_{p}"] = zscores[ticker]
    except Exception as e:
        print(f"[Cross-Vol-Z] Failed for {ticker}: {e}")

def crossretrank(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT: return
    try:
        for p in per:
            rollret = {k: df['close'].pct_change(p) for k, df in CROSSFEAT.items()}
            retdf = pd.DataFrame(rollret)
            ranks = retdf.rank(axis=1, pct=True)
            if ticker in ranks: data[f"crossretrank_{p}"] = ranks[ticker]
    except Exception as e:
        print(f"[Cross-Rank] Failed for {ticker}: {e}")

def crossrelstrength(data, ticker, benchmark='SPY'):
    global CROSSFEAT
    if not CROSSFEAT or benchmark not in CROSSFEAT: return
    try:
        for p in per:
            relstrength = data['close'] / CROSSFEAT[benchmark]['close']
            data[f"crossrelstrength_{p}"] = relstrength / relstrength.rolling(p).mean()
    except Exception as e:
        print(f"[Cross-RelStrength] Failed for {ticker}: {e}")

def crossbeta(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT: return
    try:
        basket = pd.DataFrame({k: df['ret'] for k, df in CROSSFEAT.items() if 'ret' in df}).drop(columns=[ticker], errors='ignore')
        if 'ret' not in data:
            data['ret'] = data['close'].pct_change()
        for p in per:
            rollingcov = data['ret'].rolling(p).cov(basket.mean(axis=1))
            rollingvar = basket.mean(axis=1).rolling(p).var()
            data[f"crossbeta_{p}"] = rollingcov / (rollingvar + 1e-10)
    except Exception as e:
        print(f"[Cross-Beta] Failed for {ticker}: {e}")

def crosscorr(data, ticker):
    global CROSSFEAT
    if not CROSSFEAT: return
    try:
        for p in per:
            peers = pd.DataFrame({k: df['ret'] for k, df in CROSSFEAT.items() if 'ret' in df and k != ticker})
            data[f'crosscorr_{p}'] = data['ret'].rolling(p).corr(peers.mean(axis=1))
    except Exception as e:
        print(f"[Cross-Corr] Failed for {ticker}: {e}")

FTR_FUNC = {
    # Basic
    "ret": ret, "price": price, "logret": logret, "rollret": rollret,
    # Trend / Momentum
    "sma": sma, "ema": ema, "momentum": momentum, 
    "macd": macd,
    "acceleration":acceleration,
    "pricevshigh":pricevshigh,
    #"updownratio":updownratio,
    # vol
    "vol": vol, "atr": atr,
    "range":range,"volchange":volchange,
    "volptile":volptile,
    # Oscillators/Mean ReversionRegime
     "zscore": zscore,"rsi": rsi, "cmo": cmo, "williams": williams,
     "stoch":stoch,
    # Structure / Regime
     "priceptile": priceptile,
     "adx":adx, "entropy":entropy, 
     "meanabsret":meanabsret,
    # Bands
    "boll": boll, "donchian": donchian,
    # Volume
    "volume": volume,
    # Lag/Time
    "lag": lag,
    "trendcombo":trendcombo,
    # Cross-sectional
    "retcrossz": retcrossz, "crossmomentumz": crossmomentum,
    "crossvolz": crossvolz, "crossretrank": crossretrank,
    "crossrelstrength": crossrelstrength, "crossbeta": crossbeta,
    "crosscorr":crosscorr,
}
