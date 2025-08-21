import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from load import load_config

config = load_config()
PERSIST_FILE = "csv/persistent.txt"

def select_features(
    feat,
    ret,
    split_date,
    thresh=config["THRESH"],
    method=["rf"],
    importance_features=None,

    rf_weight=config["RF_WEIGHT"],                   # weight for RF vs permutation
    transformer_weight=config["TRANS_WEIGHT"],          # weight for transformer features
    persistence=False,               # carry features forward across runs
):
    if method is None:return feat
        
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    #y = (portfolio_ret.shift(-window).rolling(window).mean() / (portfolio_ret.shift(-window).rolling(window).std() + 1e-10)).dropna()
    
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()  # <- single float
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
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)     
        rf_scores = pd.Series(model.feature_importances_, index=X.columns)     
        if rf_weight<1:
            perm = permutation_importance(model, X, y, n_repeats=5, random_state=config["SEED"], n_jobs=-1)
            perm_scores = pd.Series(perm.importances_mean, index=X.columns)
            scores = (rf_weight * rf_scores.rank(pct=True)) + ((1 - rf_weight) * perm_scores.rank(pct=True))
        else: scores = rf_scores  
    else: return feat
    combined_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    else: return feat

    if importance_features is not None and len(importance_features) > 0:
        if transformer_weight > 0:
            blended = pd.Series(0.0, index=X.columns)
            blended.loc[combined_scores.index] += scores.rank(pct=True)
            blended.loc[importance_features] += transformer_weight
            selected_features = blended.nlargest(len(selected_features)).index
            print(f"[Prune] Blended RF and transformer importance")

    persistent_features = set()
    if os.path.exists(PERSIST_FILE):
        with open(PERSIST_FILE) as f: persistent_features = set(f.read().splitlines())
    if persistence:
        if persistent_features:
            missing = persistent_features - set(selected_features)
            if missing:
                lowest = combined_scores.loc[selected_features].nsmallest(len(missing)).index
                new_selection = [f for f in selected_features if f not in lowest] + list(missing)
                selected_features = pd.Index(new_selection)
                print(f"[Prune - Persistence] Replaced {len(missing)} features.)")
    else:
        if persistent_features:
            survivors = set(selected_features).intersection(persistent_features) 
            with open(PERSIST_FILE, "w") as f: f.write("\n".join(sorted(survivors)))

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    #for f, s in combined_scores.loc[selected_features].items(): print(f" - {f}: {s:.6f}")
    return feat[selected_features]