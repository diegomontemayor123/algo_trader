import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=config["FILTERMETHOD"]):
    if method is None: return feat
    method = method[0]
    mask = ret.notna().all(axis=1)
    start = (pd.to_datetime(split_date) - pd.DateOffset(months=config["FILTERWIN"]))
    mask = mask & (mask.index < pd.to_datetime(split_date)) & (mask.index >= start)

    X = feat.loc[mask]
    y = ret.loc[mask].mean(axis=1) / (ret.loc[mask].std(axis=1) + 1e-10) if method == "rf" else ret.loc[mask].mean(axis=1)

    if method == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=config["SEED"])
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    elif method == "mutual":
        scores = pd.Series(mutual_info_regression(X, y, random_state=config["SEED"]), index=X.columns)
    elif method == "correl":
        scores = X.apply(lambda col: col.corr(y)).abs()
    else:
        print(f"[FILTER] Unknown method '{method}' specified, skipping filtering")
        return feat
    score_name=method
    if 1<thresh:
        selected_features = scores.nlargest(thresh).index
        print(f"[FILTER] Selected top {thresh} features by {score_name} between {start.date()} - {split_date.date()}")
    elif  0 < thresh < 1:
        selected_features = scores[scores > thresh].index
        print(f"[FILTER] Selected {len(selected_features)} features with {score_name} > {thresh} between {start.date()} - {split_date.date()}")
    else:
        selected_features = scores.index
        print(f"[FILTER] No valid threshold specified, using all {len(selected_features)} features")

    return feat[selected_features]
