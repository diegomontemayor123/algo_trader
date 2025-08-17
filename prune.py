import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def debug_hash(data, name):
    """Create consistent hash for debugging"""
    if isinstance(data, pd.Series):
        # Convert to string with consistent precision and sort by index
        data_str = data.sort_index().round(12).to_string()
    elif isinstance(data, pd.DataFrame):
        data_str = data.sort_index().round(12).to_string()
    else:
        data_str = str(data)
    
    hash_val = hashlib.md5(data_str.encode()).hexdigest()[:8]
    print(f"[DEBUG] {name}: hash={hash_val}, shape={getattr(data, 'shape', 'N/A')}, type={type(data).__name__}")
    
    if hasattr(data, 'index'):
        print(f"        Index range: {data.index.min()} to {data.index.max()}")
    if hasattr(data, 'describe'):
        stats = data.describe()
        print(f"        Stats: mean={stats['mean']:.8f}, std={stats['std']:.8f}")
    
    return hash_val

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    print(f"\n{'='*60}")
    print(f"DEBUGGING FEATURE SELECTION")
    print(f"Split date: {split_date}")
    print(f"Config: PRUNEWIN={config['PRUNEWIN']}, YWIN={config['YWIN']}, PRUNEDOWN={config['PRUNEDOWN']}")
    print(f"{'='*60}")
    
    if method is None: 
        return feat
    
    # Debug environment info
    print(f"[ENV] Python: {os.sys.version}")
    print(f"[ENV] Pandas: {pd.__version__}")
    print(f"[ENV] Numpy: {np.__version__}")
    print(f"[ENV] Random seed: {config['SEED']}")
    
    # Set seeds
    np.random.seed(config["SEED"])
    
    # Step 1: Date calculations
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    
    print(f"\n[STEP 1] Date Setup:")
    print(f"  split_date_ts: {split_date_ts}")
    print(f"  start: {start}")
    print(f"  window: {window}")
    
    # Step 2: Input data hashing
    print(f"\n[STEP 2] Input Data:")
    debug_hash(ret, "ret (input)")
    debug_hash(feat, "feat (input)")
    
    # Step 3: Mask creation
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    print(f"\n[STEP 3] Mask:")
    print(f"  Mask sum: {mask.sum()} out of {len(mask)} rows")
    print(f"  Date range: {ret.index[mask].min()} to {ret.index[mask].max()}")
    
    # Step 4: Portfolio returns calculation
    ret_masked = ret.loc[mask]
    debug_hash(ret_masked, "ret_masked")
    
    # Test different mean calculation methods
    portfolio_ret_v1 = ret_masked.mean(axis=1, skipna=True)
    portfolio_ret_v2 = ret_masked.mean(axis=1, skipna=False)
    portfolio_ret_v3 = ret_masked.fillna(0).mean(axis=1)
    
    print(f"\n[STEP 4] Portfolio Returns (different methods):")
    debug_hash(portfolio_ret_v1, "portfolio_ret (skipna=True)")
    debug_hash(portfolio_ret_v2, "portfolio_ret (skipna=False)")  
    debug_hash(portfolio_ret_v3, "portfolio_ret (fillna(0))")
    
    # Use the most likely method (skipna=True)
    portfolio_ret = portfolio_ret_v1
    
    # Step 5: Shift operation
    shifted_ret = portfolio_ret.shift(-window)
    print(f"\n[STEP 5] Shift Operation:")
    debug_hash(shifted_ret, f"shifted_ret (shift -{window})")
    
    # Step 6: Rolling mean
    rolling_mean = shifted_ret.rolling(window).mean()
    print(f"\n[STEP 6] Rolling Mean:")
    debug_hash(rolling_mean, f"rolling_mean (window={window})")
    
    # Step 7: Max drawdown calculation
    def max_drawdown_v1(returns):
        """Original version"""
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()
    
    def max_drawdown_v2(returns):
        """Version with explicit NaN handling"""
        returns_clean = returns.fillna(0)
        cum = (1 + returns_clean).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()
    
    print(f"\n[STEP 7] Rolling Drawdown Calculation:")
    
    # Test both versions
    rolling_dd_v1 = shifted_ret.rolling(window).apply(max_drawdown_v1, raw=False)
    rolling_dd_v2 = shifted_ret.rolling(window).apply(max_drawdown_v2, raw=False)
    
    debug_hash(rolling_dd_v1, "rolling_drawdown_v1 (original)")
    debug_hash(rolling_dd_v2, "rolling_drawdown_v2 (fillna)")
    
    # Use original version
    rolling_drawdown = rolling_dd_v1
    
    # Step 8: Final y calculation
    print(f"\n[STEP 8] Final Y Calculation:")
    
    # Test the exact original calculation
    y_original = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - 
                  config["PRUNEDOWN"]*(ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown_v1, raw=False) + 1e-10)).dropna()
    
    # Test step-by-step calculation
    y_stepwise = (rolling_mean - config["PRUNEDOWN"]*(rolling_drawdown + 1e-10)).dropna()
    
    debug_hash(y_original, "y_original (chained)")
    debug_hash(y_stepwise, "y_stepwise (explicit)")
    
    # Compare the two
    if len(y_original) == len(y_stepwise):
        diff = (y_original - y_stepwise).abs().max()
        print(f"  Max difference between methods: {diff}")
    
    # Use original method to match Kaggle
    y = y_original
    
    # Step 9: Feature alignment
    print(f"\n[STEP 9] Feature Alignment:")
    X = feat.loc[y.index]
    debug_hash(X, "X (aligned features)")
    debug_hash(y, "y (final target)")
    
    # Step 10: Sort consistency check
    X_sorted = X.sort_index()
    y_sorted = y.sort_index()
    
    print(f"\n[STEP 10] Sorting Check:")
    debug_hash(X_sorted, "X_sorted")
    debug_hash(y_sorted, "y_sorted")
    
    # Check if sorting changes anything
    if not X.index.equals(X_sorted.index):
        print("  WARNING: X index was not sorted!")
    if not y.index.equals(y_sorted.index):
        print("  WARNING: y index was not sorted!")
    
    # Feature preprocessing
    print(f"\n[STEP 11] Feature Preprocessing:")
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    
    print(f"  Constant features: {len(constant_features)}")
    print(f"  Sparse features: {len(sparse_features)}")
    print(f"  Total dropped: {len(dropped_features)}")
    
    if X.empty or len(X) < 10: 
        print("[ERROR] Not enough data for feature selection.")
        return feat

    # Random Forest
    if method[0] == "rf":
        print(f"\n[STEP 12] Random Forest:")
        print(f"  n_estimators: {config['NESTIM']}")
        print(f"  random_state: {config['SEED']}")
        
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"], 
            random_state=config["SEED"],
            n_jobs=1
        )
        
        # Final data check before fitting
        print(f"  Final X shape: {X.shape}")
        print(f"  Final y shape: {y.shape}")
        print(f"  X dtypes: {X.dtypes.value_counts().to_dict()}")
        print(f"  y dtype: {y.dtype}")
        
        model.fit(X, y)
        scores = pd.Series(model.feature_importances_, index=X.columns)
        debug_hash(scores, "feature_scores")
        
    else: 
        return feat

    # Feature selection
    combined_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"\n[FINAL] Selected top {int(thresh)} features")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"\n[FINAL] Selected {len(selected_features)} features with score > {thresh}")
    else: 
        return feat

    print(f"Selected features: {len(selected_features)}")
    print(f"Top 5 features: {list(selected_features[:5])}")
    print(f"Top 5 scores: {combined_scores.head().round(6).tolist()}")
    
    return feat[selected_features]

# Usage:
# selected_feat = debug_select_features(feat, ret, split_date)