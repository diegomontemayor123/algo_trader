
import pandas as pd
from features_factory import add_volume
import os
import pandas as pd
import yfinance as yf
from features_factory import FTR_FUNC

PRICE_CACHE_FILE = "cached_prices.csv"

def load_price_data(TICKERS,START_DATE,END_DATE):
    if os.path.exists(PRICE_CACHE_FILE):
        price_data = pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)
    else:
        raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
        price = raw_data['Adj Close']
        volume = raw_data['Volume']
        volume.columns = [f"{ticker}_volume" for ticker in volume.columns]
        price_data = pd.concat([price, volume], axis=1)
        price_data.to_csv(PRICE_CACHE_FILE)
    return price_data

def compute_features(TICKERS,START_DATE,END_DATE,FEATURES):
    data = load_price_data(TICKERS,START_DATE,END_DATE)
    price_cols = [col for col in data.columns if not col.endswith("_volume")]
    volume_cols = [col for col in data.columns if col.endswith("_volume")]
    prices = data[price_cols]
    volume = data[volume_cols] if volume_cols else None
    all_features = {}
    for ticker in TICKERS:
        price_data = prices[ticker].ffill().dropna()
        df = pd.DataFrame(index=price_data.index)
        df['close'] = price_data
        for feature_name in FEATURES:
            if feature_name.startswith('volume'):
                continue
            feature_function = FTR_FUNC.get(feature_name)
            if feature_function is not None:
                feature_function(df)
        if volume is not None and any(ftr.startswith('volume') for ftr in FEATURES):
            volume_col = f"{ticker}_volume"
            if volume_col in volume.columns:
                vol_series = volume[volume_col].reindex(df.index).ffill().dropna()
                if len(vol_series) == len(df):
                    volume_feats = add_volume(pd.DataFrame({volume_col: vol_series}))
                    df = pd.concat([df, volume_feats], axis=1)
                else:
                    print(f"[Warning] Volume series length mismatch for {ticker}")
        df = df.drop(columns=['close'], errors='ignore')
        df.columns = [f"{col}_{ticker}" for col in df.columns]
        all_features[ticker] = df
    features = pd.concat(all_features.values(), axis=1).dropna()
    returns = prices.pct_change().shift(-1).reindex(features.index)
    print(f"Features shape: {features.shape}, Returns shape: {returns.shape}")
    print(f"Features sample:\n{features.head()}")
    print(f"[Features] Returns sample:\n{returns.head()}")
    if not features.index.equals(returns.index):
        print("[Features][Warning] Feature and Return indices do not match!")
    return features, returns

def normalize_features(feature_window):
    mean = feature_window.mean(axis=0)
    std = feature_window.std(axis=0) + 1e-6
    return (feature_window - mean) / std