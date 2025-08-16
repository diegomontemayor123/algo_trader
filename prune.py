import os
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def hash_series(s, name="series"):
    """Compute a quick hash of a pandas Series or DataFrame for debugging."""
    h = hashlib.md5(pd.util.hash_pandas_object(s, index=True).values).hexdigest()
    print(f"[Hash] {name}: {h}")
    return h

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None: 
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    
    # Step 1: Masked returns
    masked_ret = ret.loc[mask]
    print(f"[Debug] Step 1: masked_ret shape: {masked_ret.shape}")
    print(masked_ret.head())
    print(masked_ret.describe())
    hash_series(masked_ret, "masked_ret")

    # Step 2: Portfolio returns
    portfolio_ret = masked_ret.mean(axis=1, skipna=True)
    print(f"[Debug] Step 2: portfolio_ret head:\n{portfolio_ret.head()}")
    print(portfolio_ret.describe())
    hash_series(portfolio_ret, "portfolio_ret")

    # Step 3: Shifted returns
    y_shifted = portfolio_ret.shift(-window)
    print(f"[Debug] Step 3: y_shifted head:\n{y_shifted.head()}")
    print(y_shifted.describe())
    hash_series(y_shifted, "y_shifted")

    # Step 4: Rolling mean
    y_rolled_mean = y_shifted.rolling(window).mean()
    print(f"[Debug] Step 4: y_rolled_mean head:\n{y_rolled_mean.head()}")
    print(y_rolled_mean.describe())
    hash_series(y_rolled_mean, "y_rolled_mean")

    # Step 5: Rolling max drawdown
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y_rolled_dd = y_shifted.rolling(window).apply(max_drawdown, raw=False)
    print(f"[Debug] Step 5: y_rolled_dd head:\n{y_rolled_dd.head()}")
    print(y_rolled_dd.describe())
    hash_series(y_rolled_dd, "y_rolled_dd")

    # Step 6: Final target
    y = (y_rolled_mean - config["PRUNEDOWN"] * (y_rolled_dd + 1e-10)).dropna()
    print(f"[Debug] Step 6: final target y head:\n{y.head()}")
    print(y.describe())
    hash_series(y, "y_final")

    # Step 7: Align features
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

    # Step 8: Random Forest feature importance
    if method[0] == "rf":
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"],
            random_state=config["SEED"],
            n_jobs=-1
        )
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else:
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    return feat[selected_features]
