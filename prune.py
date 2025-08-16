import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    print(f"\n[DEBUG] Starting feature selection for split_date={split_date}")

    if method is None: 
        print("[DEBUG] No method provided, returning all features.")
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]

    mask = (ret.index >= start) & (ret.index < split_date_ts)
    print(f"[DEBUG] Mask applied: {mask.sum()} rows selected")

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)

    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y_raw = ret.loc[mask].mean(axis=1)
    y_shifted = y_raw.shift(-window)
    y_rolling_mean = y_shifted.rolling(window).mean()
    y_drawdown = y_shifted.rolling(window).apply(max_drawdown, raw=False) + 1e-10
    y = (y_rolling_mean - config["PRUNEDOWN"] * y_drawdown).dropna()
    print(f"[DEBUG] y computed (head 10):\n{y.head(10)}")

    X = feat.loc[y.index]
    print(f"[DEBUG] X shape after aligning with y: {X.shape}")

    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features

    if dropped_features:
        print(f"[DEBUG] {len(dropped_features)} features have issues:")
        for f in dropped_features: 
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")

    if X.empty or len(X) < 10: 
        print("[DEBUG] Not enough data for feature selection, returning all features.")
        return feat

    if method[0] == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        print(f"[DEBUG] Feature importances (top 10):\n{scores.sort_values(ascending=False).head(10)}")
    else: 
        print("[DEBUG] Unknown method, returning all features.")
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[DEBUG] Selected top {int(thresh)} features")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[DEBUG] Selected {len(selected_features)} features above threshold {thresh}")
    else: 
        print("[DEBUG] Invalid threshold, returning all features.")
        return feat

    print(f"[DEBUG] Top feature: {combined_scores.loc[selected_features].head(1)}")
    return feat[selected_features]
