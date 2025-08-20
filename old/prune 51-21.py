import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

# Full predefined list of 175 features
PREDEFINED_FEATURES = [
    "boll_Lower_21_LLY","boll_Lower_14_LLY","price_XOM","ema_69_LLY","donchian_21_CAT",
    "sma_21_LLY","donchian_21_AVGO","donchian_21_JPM","macd_JPM","month",
    "momentum_69_ADBE","zscore_69_ADBE","rollret_69_ADBE","lag_14_LLY","boll_Upper_21_XOM",
    "boll_Lower_69_AVGO","ema_21_XOM","rsi_69_ADBE","volume_zscore_69_LLY","ema_21_LLY",
    "sma_14_XOM","volptile_69_JPM","ema_14_XOM","sma_14_UNH","ema_69_MA",
    "cmo_69_ADBE","volume_zscore_14_LLY","meanabsret_69_NVDA","vol_14_MSFT","sma_21_XOM",
    "rollret_69_CAT","volptile_69_LLY","donchian_21_COST","price_CAT","volptile_14_XOM",
    "donchian_14_AVGO","vol_21_UNH","boll_Lower_69_MA","donchian_69_COST","sma_69_MSFT",
    "boll_Upper_69_LLY","sma_69_UNH","volptile_69_UNH","macd_ADBE","boll_Upper_21_AVGO",
    "rsi_69_CAT","boll_Upper_69_CAT","boll_Upper_21_COST","boll_Lower_14_XOM","donchian_21_XOM",
    "vol_14_XOM","sma_21_AVGO","momentum_69_AVGO","atr_21_UNH","volptile_21_AMZN",
    "boll_Upper_21_AMZN","meanabsret_21_XOM","donchian_69_MSFT","lag_21_AVGO","vol_21_XOM",
    "atr_21_COST","boll_Upper_69_AMZN","boll_Upper_21_CAT","cmo_14_MA","vol_21_MSFT",
    "boll_Lower_21_COST","ema_69_CAT","boll_Upper_69_MSFT","volptile_69_MA","ema_69_JPM",
    "sma_69_MA","macd_COST","donchian_14_MSFT","rsi_14_MA","boll_Upper_69_UNH",
    "lag_21_ADBE","vol_69_ADBE","vol_69_MA","stoch_69_ADBE","donchian_69_MA",
    "boll_Lower_69_COST","volume_zscore_14_MSFT","momentum_69_CAT","boll_Lower_21_XOM","vol_14_AVGO",
    "boll_Upper_14_COST","ema_69_ADBE","atr_69_LLY","boll_Upper_21_NVDA","sma_14_COST",
    "rsi_69_AVGO","volptile_21_MSFT","sma_21_AMZN","sma_21_ADBE","macd_CAT",
    "cmo_69_AVGO","boll_Upper_14_JPM","donchian_21_UNH","boll_Lower_14_COST","boll_Upper_14_XOM",
    "volume_zscore_21_LLY","vol_21_COST","lag_14_COST","lag_21_MA","zscore_69_NVDA",
    "sma_21_CAT","donchian_21_MA","ema_69_MSFT","zscore_69_AVGO","stoch_69_UNH",
    "zscore_14_COST","atr_69_COST","atr_21_AVGO","ema_14_COST","williams_69_ADBE",
    "sma_69_LLY","lag_14_AMZN","sma_69_CAT","boll_Lower_21_CAT","donchian_69_XOM",
    "vol_14_LLY","atr_69_UNH","macd_AVGO","zscore_69_CAT","boll_Lower_14_MSFT",
    "cmo_69_CAT","vol_21_NVDA","sma_21_MA","boll_Lower_21_AVGO","boll_Upper_69_AVGO",
    "lag_21_AMZN","atr_21_JPM","ema_14_AMZN","range_LLY","ema_21_MSFT",
    "atr_21_MA","zscore_21_NVDA","macd_NVDA","ema_21_NVDA","boll_Upper_21_MA",
    "atr_14_MA","atr_21_LLY","cmo_21_MSFT","ema_21_COST","volptile_14_ADBE",
    "ema_21_MA","volptile_69_AVGO","volume_zscore_69_XOM","momentum_21_MSFT","sma_21_NVDA",
    "price_COST","pricevshigh_21_XOM","ema_69_AMZN","volume_zscore_14_AVGO","pricevshigh_69_ADBE",
    "ema_69_COST","boll_Lower_69_LLY","boll_Lower_21_MSFT","ret_NVDA","macd_AMZN",
    "boll_Upper_69_MA","ptile_rank_14_MA","rollret_21_MSFT","meanabsret_14_ADBE","donchian_14_CAT",
    "meanabsret_14_XOM","ema_14_MSFT","price_ADBE","ema_21_ADBE","sma_14_CAT",
    "boll_Lower_21_UNH","donchian_14_LLY","boll_Upper_14_MA","meanabsret_69_CAT","pricevshigh_21_CAT"
]

def select_features(feat, ret=None, split_date=None, thresh=config["THRESH"], method=["rf"], use_predefined=False):
    """
    Original feature selection function. If use_predefined=True, it will
    return only the predefined list of 175 features.
    """
    if use_predefined:
        available_features = [f for f in PREDEFINED_FEATURES if f in feat.columns]
        missing_features = set(PREDEFINED_FEATURES) - set(available_features)
        if missing_features:
            print(f"[Info] {len(missing_features)} predefined features not in DataFrame and were skipped.")
        print(f"[Info] Selecting {len(available_features)} predefined features.")
        return feat[available_features]

    if method is None or ret is None or split_date is None: 
        return feat

    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()
    
    y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() -
         config["PRUNEDOWN"] * (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()
    
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

    print("[Prune] Selected features and their scores:")
    for f, s in combined_scores.loc[selected_features].items():
        print(f" - {f}: {s:.6f}")
    
    print(f"[Prune] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    return feat[selected_features]
