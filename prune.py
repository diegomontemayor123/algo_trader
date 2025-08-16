import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def log_series_info(s, name):
    if isinstance(s, pd.DataFrame):
        print(f"[Info] {name}: shape={s.shape}, dtypes=\n{s.dtypes}")
        print(f"[Info] {name}: NaNs per column:\n{s.isna().sum()}")
        print(f"[Info] {name}: infs per column:\n{(~np.isfinite(s)).sum()}")
        if not s.empty:
            print(f"[Sample] {name} head:\n{s.head()}")
            print(f"[Stats] {name} describe:\n{s.describe()}")
    else:  # Series
        print(f"[Info] {name}: shape={s.shape}, dtype={s.dtype}, NaNs={s.isna().sum()}, infs={(~np.isfinite(s)).sum()}")
        if len(s) > 0:
            print(f"[Sample] {name} head:\n{s.head()}")
            print(f"[Stats] {name} min={s.min()}, max={s.max()}, mean={s.mean()}, std={s.std()}")


def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None:
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    
    # Step 1: Masked returns
    masked_ret = ret.loc[mask]
    log_series_info(masked_ret, "masked_ret")

    # Step 2: Portfolio returns
    portfolio_ret = masked_ret.mean(axis=1, skipna=True)
    log_series_info(portfolio_ret, "portfolio_ret")

    # Step 3: Shifted returns
    y_shifted = portfolio_ret.shift(-window)
    log_series_info(y_shifted, "y_shifted")

    # Step 4: Rolling mean
    y_rolled_mean = y_shifted.rolling(window).mean()
    log_series_info(y_rolled_mean, "y_rolled_mean")

    # Step 5: Rolling max drawdown
    def max_drawdown(returns):
        if returns.isna().any() or len(returns) == 0:
            return np.nan
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y_rolled_dd = y_shifted.rolling(window).apply(max_drawdown, raw=False)
    log_series_info(y_rolled_dd, "y_rolled_dd")

    # Step 6: Final target
    y = (y_rolled_mean - config["PRUNEDOWN"] * (y_rolled_dd + 1e-10)).dropna()
    log_series_info(y, "y_final")

    # Step 7: Align features
    X = feat.loc[y.index]
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

    # Step 8: Random Forest feature importance
    if method[0] == "rf":
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
    else:
        return feat

    combined_scores = scores.sort_values(ascending=False)
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features by {method[0]}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with {method[0]} > {thresh}")
    else:
        return feat

    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    return feat[selected_features]
