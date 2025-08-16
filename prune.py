import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None: return feat
    
    # Set random seed for exact reproducibility
    np.random.seed(config["SEED"])
    
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    
    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 
    
    y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - config["PRUNEDOWN"]*(ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()
    
    X = feat.loc[y.index]
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} Features have issues (look into dropping):")
        for f in dropped_features: 
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    
    if X.empty or len(X) < 10: 
        print("[Prune] Not enough data for feature selection.")
        return feat

    if method[0] == "rf":
        # CRITICAL CHANGES for exact reproducibility:
        # 1. Sort data to ensure consistent ordering
        X_sorted = X.sort_index().sort_index(axis=1)
        y_sorted = y.sort_index()
        
        # 2. Use single thread instead of n_jobs=-1
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"], 
            random_state=config["SEED"],
            n_jobs=1  # Changed from -1 to 1 for exact reproducibility
        )
        
        model.fit(X_sorted, y_sorted)
        scores = pd.Series(model.feature_importances_, index=X_sorted.columns)
    else: 
        return feat
    
    combined_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else: 
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    
    return feat[selected_features]