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

    print(f"[Debug] Split date: {split_date_ts}")
    print(f"[Debug] Start date: {start}")
    print(f"[Debug] Window size: {window}")

    # Time mask
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    print(f"[Debug] Mask covers {mask.sum()} rows in return data.")

    # Target variable y
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y = (ret.loc[mask].mean(axis=1)
         .shift(-window)
         .rolling(window)
         .mean()
         - config["PRUNEDOWN"] * (
             ret.loc[mask].mean(axis=1)
             .shift(-window)
             .rolling(window)
             .apply(max_drawdown, raw=False) + 1e-10
         )
    ).dropna()

    print(f"[Debug] y.shape: {y.shape}, y sample:\n{y.head()}")

    # Features aligned to y
    X = feat.loc[y.index]
    print(f"[Debug] X shape: {X.shape}, Features: {list(X.columns[:5])}...")

    # Dropping constant or sparse features
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

    if method[0] == "rf":
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"],
            random_state=config["SEED"],
            n_jobs=4
        )
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)
    print(f"[Debug] Top 5 scores:\n{combined_scores.head()}")

    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else:
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    print(f"[Debug] Final selected feature count: {len(selected_features)}")

    return feat[selected_features]
