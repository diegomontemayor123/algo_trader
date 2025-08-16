import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"], debug_log="prune_debug.log"):
    with open(debug_log, "a") as log:
        log.write(f"\n[DEBUG] Starting feature selection for split_date={split_date}\n")

        if method is None: 
            log.write("[DEBUG] No method provided, returning all features.\n")
            return feat

        split_date_ts = pd.to_datetime(split_date)
        start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
        window = config["YWIN"]

        mask = (ret.index >= start) & (ret.index < split_date_ts)
        log.write(f"[DEBUG] Mask applied: {mask.sum()} rows selected\n")

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
        log.write(f"[DEBUG] y computed: {y.head(10).to_string()}\n")

        X = feat.loc[y.index]
        log.write(f"[DEBUG] X shape after aligning with y: {X.shape}\n")

        constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
        sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
        dropped_features = constant_features + sparse_features

        if dropped_features:
            log.write(f"[DEBUG] {len(dropped_features)} features have issues:\n")
            for f in dropped_features: 
                reason = "constant" if f in constant_features else "too many NaNs"
                log.write(f" - {f} ({reason})\n")

        if X.empty or len(X) < 10: 
            log.write("[DEBUG] Not enough data for feature selection, returning all features.\n")
            return feat

        if method[0] == "rf":
            model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
            model.fit(X, y)
            scores = pd.Series(model.feature_importances_, index=X.columns)
            log.write(f"[DEBUG] Feature importances (top 10): {scores.sort_values(ascending=False).head(10).to_string()}\n")
        else: 
            log.write("[DEBUG] Unknown method, returning all features.\n")
            return feat

        combined_scores = scores.sort_values(ascending=False)
        if thresh > 1:
            selected_features = combined_scores.nlargest(int(thresh)).index
            log.write(f"[DEBUG] Selected top {int(thresh)} features\n")
        elif 0 < thresh <= 1:
            selected_features = combined_scores[combined_scores > thresh].index
            log.write(f"[DEBUG] Selected {len(selected_features)} features above threshold {thresh}\n")
        else: 
            log.write("[DEBUG] Invalid threshold, returning all features.\n")
            return feat

        log.write(f"[DEBUG] Top feature: {combined_scores.loc[selected_features].head(1).to_string()}\n")
        return feat[selected_features]
