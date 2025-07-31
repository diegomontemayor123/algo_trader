import os
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
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)

    y = (portfolio_ret.shift(-window).rolling(window).mean() / (portfolio_ret.shift(-window).rolling(window).std() + 1e-10)).dropna()

    #def max_drawdown(returns): cum = (1 + returns).cumprod();drawdown = (cum - cum.cummax()) / cum.cummax();return drawdown.min() 
    #FWD RETURN adj. DOWN y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()


    X = feat.loc[y.index]
    if X.empty or len(X) < 10: print("[Filter] Not enough data for feature selection."); return feat

    if method == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"])
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    elif method == "mutual": scores = pd.Series(mutual_info_regression(X, y, random_state=config["SEED"]), index=X.columns)
    elif method == "correl": scores = X.apply(lambda col: col.corr(y)).abs()
    else: return feat
    
    combined_scores = scores.sort_values(ascending=False)

    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Filter] Selected top {int(thresh)} features by {method} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Filter] Selected {len(selected_features)} features with {method} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else: return feat

    top_features_log = pd.DataFrame({"feature": selected_features,"avg_score": combined_scores.loc[selected_features].values,"split_date": split_date,"start_date": start.date()})

    if os.path.exists("rf_features.csv"): top_features_log.to_csv("rf_features.csv", mode='a', index=False, header=False)
    else: top_features_log.to_csv("rf_features.csv", index=False)
    print(f"[Filter] Top 10 feature scores:\n{combined_scores.loc[selected_features].head(10).to_string()}")
    return feat[selected_features]
