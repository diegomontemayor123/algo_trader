import os
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

# Utility functions for logging
def hash_df(df):
    """Compute MD5 hash of DataFrame/Series for comparison."""
    if isinstance(df, pd.DataFrame):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    elif isinstance(df, pd.Series):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    else:
        return None

def log_step(name, df):
    """Log details for a DataFrame/Series."""
    print(f"[Debug] {name} shape: {df.shape if hasattr(df,'shape') else 'N/A'}")
    print(f"[Debug] {name} head:\n{df.head() if hasattr(df,'head') else df}")
    print(f"[Debug] {name} hash: {hash_df(df)}")
    if isinstance(df, (pd.DataFrame, pd.Series)):
        print(f"[Debug] {name} stats: min {df.min().min() if isinstance(df, pd.DataFrame) else df.min()}, "
              f"max {df.max().max() if isinstance(df, pd.DataFrame) else df.max()}, "
              f"mean {df.mean().mean() if isinstance(df, pd.DataFrame) else df.mean()}, "
              f"count {df.count().sum() if isinstance(df, pd.DataFrame) else df.count()}")

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None:
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    masked_ret = ret.loc[mask]
    log_step("Step 1: masked_ret", masked_ret)

    portfolio_ret = masked_ret.mean(axis=1, skipna=True)
    log_step("Step 2: portfolio_ret", portfolio_ret)

    shifted = portfolio_ret.shift(-window)
    log_step("Step 3a: shifted", shifted)

    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 

    rolled_mean = shifted.rolling(window).mean()
    log_step("Step 3b: rolled_mean", rolled_mean)

    rolled_dd = shifted.rolling(window).apply(max_drawdown, raw=False)
    log_step("Step 3c: rolled_dd", rolled_dd)

    y = (rolled_mean - config["PRUNEDOWN"] * (rolled_dd + 1e-10)).dropna()
    log_step("Step 4: y (target)", y)

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
    log_step("Step 5: selected_features_scores", combined_scores.loc[selected_features])

    return feat[selected_features]
