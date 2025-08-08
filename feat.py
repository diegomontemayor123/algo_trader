import os
import pandas as pd
import yfinance as yf
from feat_list import FTR_FUNC, add_volume, setallcrossfeat
from load import load_config
from prune import select_features 

config = load_config()
PRICE_CACHE = "csv/prices.csv"; TICK = ['JPM', 'MSFT', 'NVDA', 'AVGO', 'LLY', 'COST', 'MA', 'XOM', 'UNH', 'AMZN', 'CAT', 'ADBE', 'TSLA']

def fetch_macro(name, ticker, start, end):
    try:
        yf_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if 'Adj Close' in yf_data:series = yf_data['Adj Close']
        elif 'Close' in yf_data:series = yf_data['Close']
        else:raise ValueError(f"No valid price column found for {ticker}")
        series.name = name
        return series
    except Exception as e: print(f"[Feat] Failed to fetch {name} ({ticker}): {e}");return pd.Series(dtype='float64')
    
def load_prices(START, END, macro_keys):
    if os.path.exists(PRICE_CACHE): data = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    else:
        print("[Feat] Downloading price and macro data...")
        raw_data = yf.download(" ".join(TICK), start=START, end=END, auto_adjust=False)
        price = raw_data['Adj Close']
        volume = raw_data['Volume']
        high = raw_data['High']
        low = raw_data['Low']
        volume.columns = [f"{ticker}_volume" for ticker in volume.columns]
        high.columns = [f"{ticker}_high" for ticker in high.columns]
        low.columns = [f"{ticker}_low" for ticker in low.columns]
        data = pd.concat([price, volume, high, low], axis=1)
        for macro_name in macro_keys:
            macro_series = fetch_macro(macro_name, macro_name, START, END)
            data[macro_name] = macro_series
        data = data.sort_index()
        data.to_csv(PRICE_CACHE)
        print(f"[Feat] Cached to {PRICE_CACHE}")
    data = data.loc[START:END]
    return data

def process_macro_feat(cached_data, index, macro_keys, min_non_na_ratio=0.1):
    macro_feat = {}
    for col in macro_keys:
        if col not in cached_data.columns:continue
        series = cached_data[col].reindex(index)
        non_na_ratio = series.notna().mean()
        if  non_na_ratio < min_non_na_ratio:
            print(f"[Feat] Excluding {col} due to low data coverage ({non_na_ratio:.2%} non-NA)")
            continue
        series = series.bfill().ffill()
        try:pct_series = series.pct_change(fill_method=None).fillna(0)
        except Exception as e:
            print(f"[Feat] Warning: pct_change failed for {col} with error {e}, filling zeros")
            pct_series = pd.Series(0, index=series.index)
        macro_feat[col] = pct_series
    return pd.DataFrame(macro_feat, index=index)

def comp_feat(TICK, FEAT, cached_data, macro_keys, thresh=config["THRESH"], split_date=None, method=None,train_end_date=None):
    price_cols = [col for col in cached_data.columns if not col.endswith(("_volume", "_high", "_low")) and col in TICK]
    volume_cols = [f"{ticker}_volume" for ticker in TICK if f"{ticker}_volume" in cached_data.columns]
    high_cols = {ticker: f"{ticker}_high" for ticker in TICK if f"{ticker}_high" in cached_data.columns}
    low_cols = {ticker: f"{ticker}_low" for ticker in TICK if f"{ticker}_low" in cached_data.columns}
    prices = cached_data[price_cols]
    volume = cached_data[volume_cols] if volume_cols else None
    all_feat = {}
    for ticker in TICK:
        if ticker not in prices:  continue
        df = pd.DataFrame(index=prices.index)
        df['close'] = prices[ticker].ffill().dropna()
        if ticker in high_cols: 
            df['high'] = cached_data[high_cols[ticker]].reindex(df.index).ffill()
        if ticker in low_cols: 
            df['low'] = cached_data[low_cols[ticker]].reindex(df.index).ffill()
        for feat_name in FEAT:
            if feat_name.startswith('volume'):  continue
            feat_func = FTR_FUNC.get(feat_name)
            if feat_func:
                if "cross" in feat_name:  feat_func(df, ticker)
                else:  feat_func(df)
        if volume is not None and any(ftr.startswith('volume') for ftr in FEAT):
            vol_col = f"{ticker}_volume"
            if vol_col in volume.columns:
                vol_series = volume[vol_col].reindex(df.index).ffill().dropna()
                if len(vol_series) == len(df):
                    vol_feat = add_volume(pd.DataFrame({vol_col: vol_series}))
                    df = pd.concat([df, vol_feat], axis=1)
        df = df.drop(columns=['close', 'high', 'low'], errors='ignore')
        df.columns = [f"{col}_{ticker}" for col in df.columns]
        all_feat[ticker] = df
    
    setallcrossfeat(all_feat)
    feat = pd.concat(all_feat.values(), axis=1).dropna()
    ret = prices.pct_change().shift(-1).reindex(feat.index)
    if ret.iloc[-1].isna().all():
        ret = ret.iloc[:-1]
        feat = feat.loc[ret.index]
    
    macro_df = process_macro_feat(cached_data, feat.index, macro_keys)
    feat = pd.concat([feat, macro_df], axis=1)
    feat['day_of_week'] = feat.index.dayofweek
    feat['month'] = feat.index.month - 1
    #feat = norm_feat(feat)
    feat = select_features(feat, ret, split_date, thresh=thresh, method=method, train_end_date=train_end_date)
    feat.to_csv("csv/feat_all.csv")
    return feat, ret

def norm_feat(feat_win):
    mean = feat_win.mean(axis=0)
    std = feat_win.std(axis=0) + 1e-10
    return (feat_win - mean) / std