import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from load import load_config

config = load_config()

def select_features(
    feat,
    ret,
    split_date,
    thresh=config["THRESH"],
    method=["rf"],
    importance_features=None,
    rf_weight=config["RF_WEIGHT"],              # weight for RF vs permutation
    transformer_weight=config["TRANS_WEIGHT"],  # weight for transformer features
):
    if method is None:
        return feat
        
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y = (
        ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean()
        - config["PRUNEDOWN"]
        * (
            ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False)
            + 1e-10
        )
    ).dropna()
        
    X = feat.loc[y.index]

    # Drop constants / sparse
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} features dropped (constant/sparse)")
    if X.empty or len(X) < 10:
        print("[Prune] Not enough data for feature selection.")
        return feat

    if method[0] == "rf":
        rf_scores_all = []
        for s in range(config["SEED"] -0 , config["SEED"]):
            model = RandomForestRegressor(
                n_estimators=config["NESTIM"], 
                random_state=s, 
                n_jobs=-1
            )
            model.fit(X, y)
            rf_scores_all.append(pd.Series(model.feature_importances_, index=X.columns))

        rf_scores = pd.concat(rf_scores_all, axis=1).mean(axis=1)  # average across seeds

        if rf_weight < 1:
            perm = permutation_importance(
                model, X, y, n_repeats=5, random_state=config["SEED"], n_jobs=-1
            )
            perm_scores = pd.Series(perm.importances_mean, index=X.columns)
            scores = (rf_weight * rf_scores.rank(pct=True)) + ((1 - rf_weight) * perm_scores.rank(pct=True))
        else:
            scores = rf_scores
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)

    # --- Thresholding ---
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features from {start.date()}–{split_date_ts.date()}")
    else:
        return feat

    # --- Blend with transformer importances ---
    if importance_features is not None and len(importance_features) > 0:
        if transformer_weight > 0:
            blended = pd.Series(0.0, index=X.columns)
            blended.loc[combined_scores.index] += scores.rank(pct=True)
            blended.loc[importance_features] += transformer_weight
            selected_features = blended.nlargest(len(selected_features)).index
            print(f"[Prune] Blended RF and transformer importance")

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    return feat[selected_features]
