import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    # IMMEDIATE INPUT VALIDATION - catch the error at the source
    print(f"[CRITICAL DEBUG] select_features called with:")
    print(f"  - feat type: {type(feat)}")
    print(f"  - feat value preview: {str(feat)[:200] if isinstance(feat, str) else 'DataFrame with shape ' + str(getattr(feat, 'shape', 'unknown'))}")
    print(f"  - ret type: {type(ret)}")
    print(f"  - split_date: {split_date}")
    print(f"  - thresh: {thresh}")
    print(f"  - method: {method}")
    
    if not isinstance(feat, pd.DataFrame):
        print(f"[ERROR] FEAT IS NOT A DATAFRAME!")
        print(f"[ERROR] feat type: {type(feat)}")
        if isinstance(feat, str):
            print(f"[ERROR] feat is a string with length {len(feat)}")
            print(f"[ERROR] feat content (first 500 chars): {feat[:500]}")
        else:
            print(f"[ERROR] feat content: {feat}")
        
        # Import traceback to see the call stack
        import traceback
        print(f"[ERROR] Call stack:")
        traceback.print_stack()
        
        raise TypeError(f"Expected pandas DataFrame for 'feat', got {type(feat)}. This suggests an error in the calling function.")
    
    if not isinstance(ret, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for 'ret', got {type(ret)}")
    
    if method is None: 
        return feat
    
    # MAXIMUM REPRODUCIBILITY - Force identical environment
    import random
    import sklearn
    from sklearn.utils import check_random_state
    
    # Print versions for debugging
    print(f"[Debug] Sklearn: {sklearn.__version__}, NumPy: {np.__version__}, Random seed: {config['SEED']}")
    
    # Set ALL possible random seeds and control threading
    np.random.seed(config["SEED"])
    random.seed(config["SEED"])
    os.environ['PYTHONHASHSEED'] = str(config["SEED"])
    
    # Force single-threaded operations across all libraries
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Additional threading controls
    try:
        import threadpoolctl
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            pass
    except ImportError:
        pass
    
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    portfolio_ret = ret.loc[mask].mean(axis=1, skipna=True)
    
    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min() 
    
    y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - config["PRUNEDOWN"]*(ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()
    
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
        # MAXIMUM REPRODUCIBILITY for RandomForest
        
        # Reset ALL random states immediately before model creation
        np.random.seed(config["SEED"])
        random.seed(config["SEED"])
        
        # Create and manually set the sklearn random state
        rng = check_random_state(config["SEED"])
        np.random.set_state(rng.get_state())
        
        # Sort data to ensure identical ordering
        X_sorted = X.sort_index().sort_index(axis=1)
        y_sorted = y.sort_index()
        
        print(f"[Debug] Data shapes - X: {X_sorted.shape}, y: {y_sorted.shape}")
        print(f"[Debug] First 3 features: {list(X_sorted.columns[:3])}")
        print(f"[Debug] X data hash: {hash(str(X_sorted.values.tobytes()))}")
        print(f"[Debug] y data hash: {hash(str(y_sorted.values.tobytes()))}")
        
        # Create RandomForest with maximum determinism
        model = RandomForestRegressor(
            n_estimators=config["NESTIM"], 
            random_state=config["SEED"],
            n_jobs=1,  # Absolutely critical for reproducibility
            bootstrap=True,
            max_features='sqrt',  # Explicit instead of 'auto'
            criterion='squared_error',
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            warm_start=False,
            oob_score=False,
            verbose=0
        )
        
        # Fit with maximum precision control
        with np.errstate(all='raise'):  # Catch any numerical issues
            try:
                model.fit(X_sorted, y_sorted)
            except FloatingPointError as e:
                print(f"[Warning] Floating point error: {e}")
                # Reset and try again
                np.random.seed(config["SEED"])
                model.fit(X_sorted, y_sorted)
        
        # Get feature importances
        scores = pd.Series(model.feature_importances_, index=X_sorted.columns)
        
        # Debug output for comparison
        top_5_scores = scores.sort_values(ascending=False).head(5)
        print(f"[Debug] Top 5 feature scores:")
        for feat, score in top_5_scores.items():
            print(f"  {feat}: {score:.10f}")
        
        print(f"[Debug] Feature importance sum: {scores.sum():.10f}")
        print(f"[Debug] Feature importance std: {scores.std():.10f}")
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