import os
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    if method is None:
        return feat
    
    # ==========================================
    # REPRODUCIBILITY SETUP - ADD AT THE START
    # ==========================================
    
    # Set all random seeds for full reproducibility
    np.random.seed(config["SEED"])
    random.seed(config["SEED"])
    os.environ['PYTHONHASHSEED'] = str(config["SEED"])
    
    # Force single-threaded operations for consistency
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1' 
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Print environment info for debugging
    import sklearn
    print(f"[Debug] Sklearn: {sklearn.__version__}, NumPy: {np.__version__}, Pandas: {pd.__version__}")
    print(f"[Debug] Random seed: {config['SEED']}, Single-threaded mode enabled")
    
    # ==========================================
    # ORIGINAL LOGIC WITH DETERMINISTIC FIXES
    # ==========================================
    
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)
    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    
    print(f"[Log] Portfolio mean returns ({mask.sum()} points) from {start.date()} to {split_date_ts.date()}:")
    print(portfolio_ret.head(5).to_string(), "...\n")
    
    def max_drawdown(returns):
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 
    
    rolling_mean = portfolio_ret.shift(-window).rolling(window).mean()
    rolling_mdd = portfolio_ret.shift(-window).rolling(window).apply(max_drawdown, raw=False)
    y = (rolling_mean - config["PRUNEDOWN"] * (rolling_mdd + 1e-10)).dropna()
    
    print(f"[Log] Rolling mean sample:\n{rolling_mean.head(5).to_string()} ...")
    print(f"[Log] Rolling max drawdown sample:\n{rolling_mdd.head(5).to_string()} ...")
    print(f"[Log] Computed target y sample:\n{y.head(5).to_string()} ...\n")
    
    X = feat.loc[y.index]
    
    # Feature filtering
    constant_features = X.nunique(dropna=True)[X.nunique(dropna=True) <= 1].index.tolist()
    sparse_features = X.columns[X.isna().sum() > (len(X) - int(0.9 * len(X)))].tolist()
    dropped_features = constant_features + sparse_features
    
    if dropped_features:
        print(f"[Prune] {len(dropped_features)} features with issues:")
        for f in dropped_features:
            reason = "constant" if f in constant_features else "too many NaNs"
            print(f" - {f} ({reason})")
    
    if X.empty or len(X) < 10:
        print("[Prune] Not enough data for feature selection.")
        return feat
    
    # ==========================================
    # DETERMINISTIC RANDOM FOREST
    # ==========================================
    
    if method[0] == "rf":
        # Reset random seed right before model creation
        np.random.seed(config["SEED"])
        random.seed(config["SEED"])
        
        # Sort data for consistent ordering across environments
        print(f"[Debug] Sorting X and y for consistency...")
        X_sorted = X.sort_index().sort_index(axis=1)  # Sort both rows and columns
        y_sorted = y.sort_index()
        
        # Verify sorting worked
        print(f"[Debug] X shape after sorting: {X_sorted.shape}")
        print(f"[Debug] First 3 features: {list(X_sorted.columns[:3])}")
        print(f"[Debug] First 3 y values: {y_sorted.head(3).values}")
        
        # Create fully deterministic RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"], 
            random_state=config["SEED"],
            n_jobs=1,  # CRITICAL: Single thread for exact reproducibility
            bootstrap=True,  # Explicit bootstrap setting
            max_features='sqrt',  # Ensure consistent feature sampling strategy
            min_samples_split=2,  # Explicit default values
            min_samples_leaf=1,
            max_depth=None,
            criterion='squared_error'  # Explicit criterion
        )
        
        print(f"[Debug] Model params: n_estimators={config['NESTIM']}, random_state={config['SEED']}, n_jobs=1")
        
        # Fit model with sorted data
        model.fit(X_sorted, y_sorted)
        
        # Get feature importances in same order as sorted features
        scores = pd.Series(model.feature_importances_, index=X_sorted.columns)
        
        print(f"[Log] Raw feature importances:\n{scores.sort_values(ascending=False).head(10).to_string()} ...\n")
        
        # Verify top scores match expected values
        top_feature = scores.idxmax()
        top_score = scores.max()
        print(f"[Debug] Top feature: {top_feature} with score: {top_score:.6f}")
        
    else:
        return feat
    
    # Feature selection logic (unchanged)
    combined_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"[Prune] Selected top {int(thresh)} features from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"[Prune] Selected {len(selected_features)} features with score > {thresh} from {start.date()}–{split_date_ts.date()}")
    else:
        return feat
    
    print(f"[Prune] Top feature score:\n{combined_scores.loc[selected_features].head(5).to_string()} ...")
    
    # ==========================================
    # ENHANCED DEBUGGING LOG
    # ==========================================
    
    # Create detailed log with additional debug info
    log_df = pd.DataFrame({
        "feature": X_sorted.columns,  # Use sorted column order
        "score": scores.values,
        "selected": [f in selected_features for f in X_sorted.columns],
        "constant": [f in constant_features for f in X_sorted.columns],
        "sparse": [f in sparse_features for f in X_sorted.columns],
        # Add debug columns
        "rank": scores.rank(ascending=False, method='min').astype(int),
        "environment": "local" if os.path.exists("/working") else "kaggle",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "seed_used": config["SEED"]
    })
    
    # Sort log by score for easier comparison
    log_df = log_df.sort_values('score', ascending=False)
    
    log_file = "feature_selection_debug.csv"
    
    # Add timestamp to distinguish runs
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_file = f"feature_selection_debug_{timestamp}.csv"
    
    # Save both files
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', index=False, header=False)
    else:
        log_df.to_csv(log_file, index=False)
    
    # Always save timestamped version
    log_df.to_csv(detailed_log_file, index=False)
    print(f"[Debug] Saved detailed log to {detailed_log_file}")
    
    # Print final verification
    print(f"[Debug] Final verification - Top 3 selected features:")
    for i, (feat, score) in enumerate(combined_scores.head(3).items()):
        print(f"  {i+1}. {feat}: {score:.6f}")
    
    return feat[selected_features]