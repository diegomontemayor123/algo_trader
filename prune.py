import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"], importance_features=None):
    if method is None: return feat
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    #y = (portfolio_ret.shift(-window).rolling(window).mean() / (portfolio_ret.shift(-window).rolling(window).std() + 1e-10)).dropna()
    
    def max_drawdown(returns): cum = (1 + returns).cumprod();drawdown = (cum - cum.cummax()) / cum.cummax();return drawdown.min() 
    y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - config["PRUNEDOWN"]*(ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()
    
    X = feat.loc[y.index]
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist(); sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} Features have issues (look into dropping):")
        for f in dropped_features: reason = "constant" if f in constant_features else "too many NaNs";print(f" - {f} ({reason})")
    if X.empty or len(X) < 10: print("[Prune] Not enough data for feature selection."); return feat

    if method[0] == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"],n_jobs=-1 )
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)

        #import lightgbm as lgb
        #model = lgb.LGBMRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"])
        #model.fit(X, y)
        #from sklearn.inspection import permutation_importance
        #result = permutation_importance(model, X, y, n_repeats=10, random_state=config["SEED"])
        #scores = pd.Series(result.importances_mean, index=X.columns)

    else: return feat
    
    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else: return feat
    #for f, s in combined_scores.loc[selected_features].items(): print(f" - {f}: {s:.6f}")

    if importance_features is not None and len(importance_features) > 0:
        current_features = list(selected_features)
        bottom_features = combined_scores.loc[current_features].nsmallest(config["TOPIMP"]).index
        new_candidates = [f for f in importance_features if f not in current_features and f in feat.columns]
        if new_candidates:
            replace_count = min(len(bottom_features), len(new_candidates))
            features_to_remove = bottom_features[:replace_count]
            features_to_add = new_candidates[:replace_count]
            updated_features = [f for f in current_features if f not in features_to_remove] + features_to_add
            selected_features = pd.Index(updated_features)
            print(f"[Prune] Replaced {replace_count} bottom RF features with transformer importance")
        
    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    return feat[selected_features]
