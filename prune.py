import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None:
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    print(f"[Log] Portfolio mean returns ({mask.sum()} points) from {start.date()} to {split_date_ts.date()}:")
    print(portfolio_ret.head(5).to_string(), "...\n")

    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 

    rolling_mean = portfolio_ret.shift(-window).rolling(window).mean()
    rolling_mdd = portfolio_ret.shift(-window).rolling(window).apply(max_drawdown, raw=False)
    y = (rolling_mean - config["PRUNEDOWN"] * (rolling_mdd + 1e-10)).dropna()

    print(f"[Log] Rolling mean sample:\n{rolling_mean.head(5).to_string()} ...")
    print(f"[Log] Rolling max drawdown sample:\n{rolling_mdd.head(5).to_string()} ...")
    print(f"[Log] Computed target y sample:\n{y.head(5).to_string()} ...\n")

    X = feat.loc[y.index]
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} features with issues:")
        for f in dropped_features:
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")

    if X.empty or len(X) < 10:
        print("[Prune] Not enough data for feature selection.")
        return feat

    if method[0] == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        print(f"[Log] Raw feature importances:\n{scores.sort_values(ascending=False).head(10).to_string()} ...\n")
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)

    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with score > {thresh} from {start.date()}–{split_date_ts.date()}")
    else:
        return feat

    print(f"[Prune] Top feature score:\n{combined_scores.loc[selected_features].head(5).to_string()} ...")

    # Optional: save detailed logs for debugging across environments
    log_df = pd.DataFrame({
        "feature": X.columns,
        "score": scores.values,
        "selected": [f in selected_features for f in X.columns],
        "constant": [f in constant_features for f in X.columns],
        "sparse": [f in sparse_features for f in X.columns]
    })
    log_file = "feature_selection_debug.csv"
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', index=False, header=False)
    else:
        log_df.to_csv(log_file, index=False)

    return feat[selected_features]
