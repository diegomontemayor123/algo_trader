import os
import pandas as pd
import yfinance as yf
from features import FTR_FUNC, add_volume
from tune_data import TICKER_LIST

PRICE_CACHE_FILE = "prices.csv"
TICKERS = TICKER_LIST

def fetch_macro_series(name, ticker, start, end):
    try:
        yf_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if 'Adj Close' in yf_data:
            series = yf_data['Adj Close']
        elif 'Close' in yf_data:
            series = yf_data['Close']
        else:
            raise ValueError(f"No valid price column found for {ticker}")
        series.name = name
        return series
    except Exception as e:
        print(f"[Macro] Failed to fetch {name} ({ticker}): {e}")
        return pd.Series(dtype='float64')
    
def load_price_data(START_DATE, END_DATE, macro_keys):
    if os.path.exists(PRICE_CACHE_FILE):
        print(f"[Data] Using cached data from {PRICE_CACHE_FILE}")
        data = pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)
    else:
        print("[Data] Downloading price and macro data...")
        raw_data = yf.download(" ".join(TICKERS), start=START_DATE, end=END_DATE, auto_adjust=False)
        price = raw_data['Adj Close']
        volume = raw_data['Volume']
        volume.columns = [f"{ticker}_volume" for ticker in volume.columns]
        data = pd.concat([price, volume], axis=1)
        for macro_name in macro_keys:
            macro_series = fetch_macro_series(macro_name, macro_name, START_DATE, END_DATE)
            data[macro_name] = macro_series

        data = data.sort_index()
        data.to_csv(PRICE_CACHE_FILE)
        print(f"[Data] Cached to {PRICE_CACHE_FILE}")
    return data

def process_macro_features(cached_data, index, macro_keys, min_non_na_ratio=0.1):
    macro_features = {}
    for col in macro_keys:
        if col not in cached_data.columns:
            continue
        series = cached_data[col].reindex(index)
        non_na_ratio = series.notna().mean()
        if  non_na_ratio < min_non_na_ratio:
            print(f"[Macro] Excluding {col} due to low data coverage ({non_na_ratio:.2%} non-NA)")
            continue
        series = series.bfill().ffill()
        try:
            pct_series = series.pct_change(fill_method=None).fillna(0)
        except Exception as e:
            print(f"[Macro] Warning: pct_change failed for {col} with error {e}, filling zeros")
            pct_series = pd.Series(0, index=series.index)
        macro_features[col] = pct_series
    return pd.DataFrame(macro_features, index=index)

def compute_features(TICKERS, FEATURES, cached_data, macro_keys):
    price_cols = [col for col in cached_data.columns if not col.endswith("_volume") and col in TICKERS]
    volume_cols = [f"{ticker}_volume" for ticker in TICKERS if f"{ticker}_volume" in cached_data.columns]
    prices = cached_data[price_cols]
    volume = cached_data[volume_cols] if volume_cols else None
    all_features = {}
    for ticker in TICKERS:
        if ticker not in prices:
            continue
        df = pd.DataFrame(index=prices.index)
        df['close'] = prices[ticker].ffill().dropna()
        for feature_name in FEATURES:
            if feature_name.startswith('volume'):
                continue
            feature_function = FTR_FUNC.get(feature_name)
            if feature_function:
                feature_function(df)
        if volume is not None and any(ftr.startswith('volume') for ftr in FEATURES):
            vol_col = f"{ticker}_volume"
            if vol_col in volume.columns:
                vol_series = volume[vol_col].reindex(df.index).ffill().dropna()
                if len(vol_series) == len(df):
                    vol_feats = add_volume(pd.DataFrame({vol_col: vol_series}))
                    df = pd.concat([df, vol_feats], axis=1)
        df = df.drop(columns=['close'], errors='ignore')
        df.columns = [f"{col}_{ticker}" for col in df.columns]
        all_features[ticker] = df
    features = pd.concat(all_features.values(), axis=1).dropna()
    returns = prices.pct_change().shift(-1)
    returns = returns.reindex(features.index)
    returns = returns.fillna(0)  # Or drop rows with NaNs if preferred

    macro_df = process_macro_features(cached_data, features.index, macro_keys)
    features = pd.concat([features, macro_df], axis=1)
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month - 1 
    features = normalize_features(features)
    features.to_csv("features_all.csv")
    return features, returns


def normalize_features(feature_window):
    mean = feature_window.mean(axis=0)
    std = feature_window.std(axis=0) + 1e-6
    return (feature_window - mean) / std
