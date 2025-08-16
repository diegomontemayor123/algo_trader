import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from load import load_config
import hashlib

config = load_config()

def hash_array(arr):
    """Return a reproducible hash for a numpy array or pandas Series"""
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64)).hexdigest()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None:
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    # Step 1: masked returns
    masked_ret = ret.loc[mask]
    print(f"[Debug] Step 1: masked_ret shape: {masked_ret.shape}")
    print(f"[Debug] Step 1: masked_ret hash: {hash_array(masked_ret.values)}")

    # Step 2: portfolio return
    portfolio_ret = masked_ret.mean(axis=1, skipna=True)
    print(f"[Debug] Step 2: portfolio_ret head:\n{portfolio_ret.head()}")
    print(f"[Debug] Step 2: portfolio_ret hash: {hash_array(portfolio_ret.values)}")

    # Step 3: shifted rolling mean - for target y
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    shifted = portfolio_ret.shift(-window)
    print(f"[Debug] Step 3a: shifted head:\n{shifted.head()}")
    print(f"[Debug] Step 3a: shifted hash: {hash_array(shifted.values)}")

    rolled_mean = shifted.rolling(window).mean()
    print(f"[Debug] Step 3b: rolled_mean head:\n{rolled_mean.head()}")
    print(f"[Debug] Step 3b: rolled_mean hash: {hash_array(rolled_mean.values)}")

    rolled_drawdown = shifted.rolling(window).apply(max_drawdown, raw=False)
    print(f"[Debug] Step 3c: rolled_drawdown head:\n{rolled_drawdown.head()}")
    print(f"[Debug] Step 3c: rolled_drawdown hash: {hash_array(rolled_drawdown.values)}")

    y = (rolled_mean - config["PRUNEDOWN"] * (rolled_drawdown + 1e-10)).dropna()
    print(f"[Debug] Step 4: target y head:\n{y.head()}")
    print(f"[Debug] Step 4: target y hash: {hash_array(y.values)}")

    # Step 5: subset features
    X = feat.loc[y.index]
    print(f"[Debug] Step 5: X shape: {X.shape}")
    print(f"[Debug] Step 5: X hash: {hash_array(X.values)}")

    # Step 6: drop problematic features
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} Features have issues:")
        for f in dropped_features:
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    if X.empty or len(X) < 10:
        print("[Prune] Not enough data for feature selection.")
        return feat

    # Step 7: Random Forest feature importances
    if method[0] == "rf":
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"],
            random_state=config["SEED"],
            n_jobs=-1
        )
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        print(f"[Debug] Step 7: top 10 feature importances:\n{scores.sort_values(ascending=False).head(10)}")
        print(f"[Debug] Step 7: feature importances hash: {hash_array(scores.values)}")
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with threshold > {thresh}")
    else:
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")

    return feat[selected_features]
