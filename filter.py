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
    all_scores = []

    # We'll store sets of low-importance features per asset
    low_rf_feature_sets = []

    for asset in ret.columns:
        asset_ret = ret[asset]
        mask = asset_ret.notna()
        mask &= (asset_ret.index < split_date_ts) & (asset_ret.index >= start)

        y = (asset_ret.loc[mask].shift(-window).rolling(window).mean() /
            (asset_ret.loc[mask].shift(-window).rolling(window).std() + 1e-10)).dropna()
        
        #def max_drawdown(returns): cum = (1 + returns).cumprod();drawdown = (cum - cum.cummax()) / cum.cummax();return drawdown.min()
        #FWD RETURN adj. DOWN y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()

        asset_feat_cols = [col for col in feat.columns if col.endswith(f"_{asset}")]
        if not asset_feat_cols:continue  
        X = feat.loc[y.index, asset_feat_cols]
        if X.empty or len(X) < 10:continue  
        if method == "rf":
            model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"])
            model.fit(X, y)
            scores = pd.Series(model.feature_importances_, index=X.columns)
            low_rf_feature_sets.append(set(scores[scores <= 1e-1].index))
        elif method == "mutual":scores = pd.Series(mutual_info_regression(X, y, random_state=config["SEED"]), index=X.columns)
        elif method == "correl":scores = X.apply(lambda col: col.corr(y)).abs()
        else:return feat
        all_scores.append(scores)

    if method == "rf" and low_rf_feature_sets: 
        common_low_feats = set.intersection(*low_rf_feature_sets) if len(low_rf_feature_sets) > 1 else low_rf_feature_sets[0]

        if common_low_feats:
            low_rf_records = []
            for feat_name in common_low_feats:
                # extract asset from feature name (assumed format: something_asset)
                asset_part = feat_name.split('_')[-1]  
                low_rf_records.append({"feature": feat_name,"asset": asset_part,
                    "split_date": split_date,"start_date": start.date()})
            df_low_rf = pd.DataFrame(low_rf_records)
            if os.path.exists("rf_prune.csv"):df_low_rf.to_csv("rf_prune.csv", mode='a', index=False, header=False)
            else:df_low_rf.to_csv("rf_prune.csv", index=False)
            print(f"[Filter] Logged {len(common_low_feats)} low-importance RF features common across all assets to rf_prune.csv")

    if not all_scores:
        print("[Filter] No valid scores computed.")
        return feat

    combined_scores = pd.concat(all_scores, axis=1).mean(axis=1).sort_values(ascending=False)

    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Filter] Selected top {int(thresh)} features by {method} from {start.date()}-{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Filter] Selected {len(selected_features)} features with {method} > {thresh} from {start.date()}-{split_date_ts.date()}")
    else:
        return feat

    print(f"[Filter] Top 10 feature scores:\n{combined_scores.loc[selected_features].head(10).to_string()}")
    return feat[selected_features]
