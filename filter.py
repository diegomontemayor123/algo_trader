import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=config["FILTER"]):
    if method is None: 
        return feat
    method = method[0]

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["FILTERWIN"])

    mask = ret.notna().all(axis=1)
    mask = mask & (mask.index < split_date_ts) & (mask.index >= start)


    def max_drawdown(returns): cum = (1 + returns).cumprod();drawdown = (cum - cum.cummax()) / cum.cummax();return drawdown.min()
    window = 60  
    y = ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - (-ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10).dropna()
    #y = ret.loc[mask].mean(axis=1) / (ret.loc[mask].std(axis=1) + 1e-10)
    X = feat.loc[y.index]

    if method == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"])
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    elif method == "mutual":
        scores = pd.Series(mutual_info_regression(X, y, random_state=config["SEED"]), index=X.columns)
    elif method == "correl":
        scores = X.apply(lambda col: col.corr(y)).abs()
    else:
        print(f"[Filter] Unknown method '{method}' specified, skipping filtering")
        return feat

    score_name = method

    if thresh > 1:
        selected_features = scores.nlargest(int(thresh)).index
        print(f"[Filter] Selected top {int(thresh)} features by {score_name} between {start.date()} - {split_date_ts.date()}")
        print(f"[Filter] x/y date range: {X.index.min().date()} to {X.index.max().date()} / {y.index.min().date()} to {y.index.max().date()}")
    elif 0 < thresh <= 1:
        selected_features = scores[scores > thresh].index
        print(f"[Filter] Selected {len(selected_features)} features with {score_name} > {thresh} between {start.date()} - {split_date_ts.date()}")
        print(f"[Filter] x/y date range: {X.index.min().date()} to {X.index.max().date()} / {y.index.min().date()} to {y.index.max().date()}")
        


    return feat[selected_features]
