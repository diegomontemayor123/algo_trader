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
    if method == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=config["SEED"])
        X = feat.loc[mask]
        y = ret.loc[mask].mean(axis=1) / (ret.loc[mask].std(axis=1) + 1e-10)
        model.fit(X, y)
        importances = model.feature_importances_
        selected_features = X.columns[importances > thresh]
        print(f"[FILTER] Selected {len(selected_features)} features with RF importance >{thresh} between {start.date()} - {split_date.date()}")
    elif method == "mutual":
        X = feat.loc[mask]
        y = ret.loc[mask].mean(axis=1)
        mi = mutual_info_regression(X, y, random_state=config["SEED"])
        mi_series = pd.Series(mi, index=X.columns)
        selected_features = mi_series[mi_series > thresh].index
        print(f"[FILTER] Selected {len(selected_features)} features with mutual info > {thresh} between {start.date()} - {split_date.date()}")
    elif method == "correl":
        X = feat.loc[mask]
        y = ret.loc[mask].mean(axis=1)
        corr = X.apply(lambda x: x.corr(y)).abs()
        selected_features = corr[corr > thresh].index
        print(f"[FILTER] Selected {len(selected_features)} features with correlation > {thresh} between {start.date()} - {split_date.date()}")
    else:
        print(f"[FILTER] Unknown method '{method}' specified, skipping filtering")
        return feat

    return feat[selected_features]
