import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None: 
        print("[Prune] No method provided, returning all features")
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    print(f"[Prune] Split date: {split_date_ts.date()}, Start date: {start.date()}, Window: {window}")
    print(f"[Prune] Using {mask.sum()} data points for feature selection")

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    print(f"[Prune] Portfolio return head:\n{portfolio_ret.head()}")

    # Max drawdown function
    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 

    # Compute target
    y_raw = ret.loc[mask].mean(axis=1)
    y_shifted = y_raw.shift(-window)
    y_rolling_mean = y_shifted.rolling(window).mean()
    y_rolling_draw = y_shifted.rolling(window).apply(max_drawdown, raw=False)
    y = (y_rolling_mean - config["PRUNEDOWN"] * (y_rolling_draw + 1e-10)).dropna()

    print(f"[Prune] Target y head:\n{y.head()}")
    print(f"[Prune] Target y stats: count={len(y)}, mean={y.mean()}, std={y.std()}, min={y.min()}, max={y.max()}")

    X = feat.loc[y.index].copy()
    print(f"[Prune] Feature matrix X shape: {X.shape}")
    print(f"[Prune] NaN summary:\n{X.isna().sum()[X.isna().sum() > 0]}")

    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} Features have issues:")
        for f in dropped_features: 
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
        X = X.drop(columns=dropped_features)
    else:
        print("[Prune] No features dropped")

    if X.empty or len(X) < 10: 
        print("[Prune] Not enough data for feature selection, returning all features")
        return feat

    if method[0] == "rf":
        print(f"[Prune] Fitting RandomForest with n_estimators={config['NESTIM']}, random_state={config['SEED']}")
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"], 
            random_state=config["SEED"], 
            n_jobs=-1
        )
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        print(f"[Prune] Feature importance head:\n{scores.sort_values(ascending=False).head(10)}")
    else: 
        print(f"[Prune] Method {method[0]} not recognized, returning all features")
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with importance > {thresh} from {start.date()}–{split_date_ts.date()}")
    else: 
        print("[Prune] Threshold invalid, returning all features")
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    print(f"[Prune] Selected features:\n{selected_features.tolist()}")

    return feat[selected_features]
