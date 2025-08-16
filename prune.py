import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config
import hashlib
import numpy as np

config = load_config()

def hash_df(df):
    """Return a quick hash of a DataFrame (including index)"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    print(f"[Debug] Starting feature selection for split_date={split_date}")
    if method is None: 
        print("[Debug] Method is None, returning all features")
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    print(f"[Debug] Split date: {split_date_ts.date()}, Start date: {start.date()}, Window: {window}")
    print(f"[Debug] Masked ret shape: {ret.loc[mask].shape}")

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    print("[Debug] Portfolio return head:\n", portfolio_ret.head())
    print("[Debug] Portfolio return hash:", hash_df(portfolio_ret.to_frame()))

    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 

    y = (portfolio_ret.shift(-window)
         .rolling(window)
         .mean() - config["PRUNEDOWN"] * (portfolio_ret.shift(-window)
                                          .rolling(window)
                                          .apply(max_drawdown, raw=False) + 1e-10)
        ).dropna()

    print("[Debug] Target y head:\n", y.head())
    print("[Debug] Target y stats:", y.describe())
    print("[Debug] Target y hash:", hash_df(y.to_frame()))

    X = feat.loc[y.index]
    print("[Debug] Feature matrix X shape:", X.shape)
    print("[Debug] Feature matrix head:\n", X.head())
    print("[Debug] Feature matrix dtypes:\n", X.dtypes.value_counts())
    print("[Debug] Feature matrix hash:", hash_df(X))

    # Feature filtering
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Debug] Dropping {len(dropped_features)} features:")
        for f in dropped_features: 
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    else:
        print("[Debug] No features dropped")

    if X.empty or len(X) < 10:
        print("[Debug] Not enough data for feature selection, returning all features")
        return feat

    # RandomForest
    if method[0] == "rf":
        print(f"[Debug] Fitting RandomForest with n_estimators={config['NESTIM']}, random_state={config['SEED']}")
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    else: 
        print("[Debug] Unknown method, returning all features")
        return feat

    print("[Debug] Feature importance head:\n", scores.sort_values(ascending=False).head(10))
    print("[Debug] Feature importance stats:\n", scores.describe())

    # Threshold selection
    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Debug] Selected top {int(thresh)} features by RF from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Debug] Selected {len(selected_features)} features with RF > {thresh} from {start.date()}–{split_date_ts.date()}")
    else:
        print("[Debug] Invalid threshold, returning all features")
        return feat

    print(f"[Debug] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    print(f"[Debug] Selected features count: {len(selected_features)}")
    print(f"[Debug] Selected features head: {list(selected_features[:20])}")

    return feat[selected_features]
