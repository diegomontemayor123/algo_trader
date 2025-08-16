import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None: return feat
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    print(f"[Log] Portfolio return range: {portfolio_ret.index.min()} - {portfolio_ret.index.max()}")

    # Rolling calculation logs
    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()
    
    shifted = ret.loc[mask].mean(axis=1).shift(-window)
    print(f"[Log] Shifted returns head:\n{shifted.head(window+5)}")
    
    rolling_mean = shifted.rolling(window, min_periods=window).mean()
    rolling_dd = shifted.rolling(window, min_periods=window).apply(max_drawdown, raw=False)
    print(f"[Log] Rolling mean head:\n{rolling_mean.head(window+5)}")
    print(f"[Log] Rolling drawdown head:\n{rolling_dd.head(window+5)}")
    
    y = (rolling_mean - config["PRUNEDOWN"] * (rolling_dd + 1e-10)).dropna()
    print(f"[Log] y after rolling calc head:\n{y.head()} | len={len(y)}")

    # Feature slicing logs
    X = feat.loc[y.index]
    print(f"[Log] X shape after slicing: {X.shape} | index range: {X.index.min()} - {X.index.max()}")

    # Dropped features logs
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} features dropped:")
        for f in dropped_features:
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    
    if X.empty or len(X) < 10: 
        print("[Prune] Not enough data for feature selection."); 
        return feat

    # Random Forest feature importance
    if method[0] == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh}")
    else: 
        return feat
    
  # Assuming combined_scores is sorted descending by default
    bottom_20 = combined_scores.loc[selected_features].nsmallest(20)

    print(f"[Prune] Bottom 20 features and scores:")
    for f, s in bottom_20.items(): 
        print(f" - {f}: {s:.6f}")
