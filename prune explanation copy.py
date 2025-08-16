import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None: return feat
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    # Log basic parameters
    print(f"[Y_LOG] Split date: {split_date}, Start: {start.date()}, Window: {window}")
    print(f"[Y_LOG] Config PRUNEDOWN: {config['PRUNEDOWN']}")
    
    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    print(f"[Y_LOG] Portfolio returns shape: {portfolio_ret.shape}")
    print(f"[Y_LOG] Portfolio returns date range: {portfolio_ret.index.min()} to {portfolio_ret.index.max()}")
    print(f"[Y_LOG] Portfolio returns first 5 values:\n{portfolio_ret.head()}")
    print(f"[Y_LOG] Portfolio returns stats: mean={portfolio_ret.mean():.8f}, std={portfolio_ret.std():.8f}")
    
    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 
    
    # Break down y calculation step by step for logging
    shifted_returns = ret.loc[mask].mean(axis=1).shift(-window)
    print(f"[Y_LOG] Shifted returns (-{window}) shape: {shifted_returns.shape}")
    print(f"[Y_LOG] Shifted returns first 5 values:\n{shifted_returns.head()}")
    
    rolling_mean = shifted_returns.rolling(window).mean()
    print(f"[Y_LOG] Rolling mean shape: {rolling_mean.shape}")
    print(f"[Y_LOG] Rolling mean first 5 values:\n{rolling_mean.head()}")
    print(f"[Y_LOG] Rolling mean last 5 values:\n{rolling_mean.tail()}")
    
    rolling_drawdown = shifted_returns.rolling(window).apply(max_drawdown, raw=False)
    print(f"[Y_LOG] Rolling drawdown shape: {rolling_drawdown.shape}")
    print(f"[Y_LOG] Rolling drawdown first 5 values:\n{rolling_drawdown.head()}")
    print(f"[Y_LOG] Rolling drawdown last 5 values:\n{rolling_drawdown.tail()}")
    print(f"[Y_LOG] Rolling drawdown stats: mean={rolling_drawdown.mean():.8f}, std={rolling_drawdown.std():.8f}")
    
    # Calculate penalty term
    penalty_term = config["PRUNEDOWN"] * (rolling_drawdown + 1e-10)
    print(f"[Y_LOG] Penalty term first 5 values:\n{penalty_term.head()}")
    print(f"[Y_LOG] Penalty term stats: mean={penalty_term.mean():.8f}, std={penalty_term.std():.8f}")
    
    # Final y calculation
    y_before_dropna = rolling_mean - penalty_term
    print(f"[Y_LOG] Y before dropna shape: {y_before_dropna.shape}")
    print(f"[Y_LOG] Y before dropna first 5 values:\n{y_before_dropna.head()}")
    print(f"[Y_LOG] Y before dropna last 5 values:\n{y_before_dropna.tail()}")
    print(f"[Y_LOG] Y before dropna NaN count: {y_before_dropna.isna().sum()}")
    
    y = y_before_dropna.dropna()
    print(f"[Y_LOG] Final Y shape: {y.shape}")
    print(f"[Y_LOG] Final Y date range: {y.index.min()} to {y.index.max()}")
    print(f"[Y_LOG] Final Y first 10 values:\n{y.head(10)}")
    print(f"[Y_LOG] Final Y last 10 values:\n{y.tail(10)}")
    print(f"[Y_LOG] Final Y stats: mean={y.mean():.8f}, std={y.std():.8f}, min={y.min():.8f}, max={y.max():.8f}")
    
    # Log Y hash for exact comparison
    y_hash = pd.util.hash_pandas_object(y).sum()
    print(f"[Y_LOG] Y hash (for exact comparison): {y_hash}")
    
    # Additional precision logging to find the source of differences
    print(f"[Y_LOG] Pandas version: {pd.__version__}")
    print(f"[Y_LOG] Y dtype: {y.dtype}")
    print(f"[Y_LOG] Y precision check - first 3 values with full precision:")
    for i in range(min(3, len(y))):
        print(f"[Y_LOG]   y[{i}] = {y.iloc[i]:.20f}")
    
    # Check if the issue is in rolling calculations
    rolling_mean_hash = pd.util.hash_pandas_object(rolling_mean.dropna()).sum()
    rolling_drawdown_hash = pd.util.hash_pandas_object(rolling_drawdown.dropna()).sum()
    print(f"[Y_LOG] Rolling mean hash: {rolling_mean_hash}")
    print(f"[Y_LOG] Rolling drawdown hash: {rolling_drawdown_hash}")
    
    # Save Y to CSV for manual inspection
    y_df = pd.DataFrame({'date': y.index, 'y_value': y.values})
    y_filename = f"y_values_{split_date.replace('-', '_')}.csv"
    y_df.to_csv(y_filename, index=False)
    print(f"[Y_LOG] Y values saved to: {y_filename}")
    
    X = feat.loc[y.index]
    print(f"[Y_LOG] X shape after aligning with Y: {X.shape}")
    print(f"[Y_LOG] X and Y index alignment check: {X.index.equals(y.index)}")
    
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} Features have issues (look into dropping):")
        for f in dropped_features: 
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    if X.empty or len(X) < 10: 
        print("[Prune] Not enough data for feature selection."); 
        return feat

    if method[0] == "rf":
        # Final logging before RF training
        print(f"[Y_LOG] About to train RF with X shape: {X.shape}, Y shape: {y.shape}")
        print(f"[Y_LOG] X NaN count: {X.isna().sum().sum()}")
        print(f"[Y_LOG] Final Y for RF (first 5): {y.iloc[:5].tolist()}")
        print(f"[Y_LOG] Final Y for RF (last 5): {y.iloc[-5:].tolist()}")
        
        model = RandomForestRegressor(n_estimators=config["NESTIM"], random_state=config["SEED"], n_jobs=-1)
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        
        print(f"[Y_LOG] RF training completed successfully")
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