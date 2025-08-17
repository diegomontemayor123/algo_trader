import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from load import load_config

config = load_config()

def select_features(feat, ret, split_date, thresh=config["THRESH"], method=["rf"]):
    """Comprehensive RandomForest settings inspection"""
    
    if method is None or method[0] != "rf": 
        return feat
    
    print(f"\n{'='*80}")
    print(f"COMPLETE RANDOMFOREST SETTINGS INSPECTION")
    print(f"{'='*80}")
    
    # Environment inspection
    print(f"[ENVIRONMENT]")
    print(f"  Python version: {os.sys.version}")
    print(f"  Sklearn version: {sklearn.__version__}")
    print(f"  Numpy version: {np.__version__}")
    print(f"  Pandas version: {pd.__version__}")
    print(f"  Platform: {os.sys.platform}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'NOT SET')}")
    
    # Random state inspection
    print(f"\n[RANDOM STATE BEFORE]")
    print(f"  Numpy random state sample: {np.random.get_state()[1][:5]}")
    print(f"  Config seed: {config['SEED']}")
    
    # Set seed
    np.random.seed(config["SEED"])
    
    print(f"\n[RANDOM STATE AFTER SEED]")
    print(f"  Numpy random state sample: {np.random.get_state()[1][:5]}")
    
    # Data preparation (same as your original)
    split_date_ts = pd.to_datetime(split_date)
    start = split_date_ts - pd.DateOffset(months=config["PRUNEWIN"])
    window = config["YWIN"]
    mask = (ret.index >= start) & (ret.index < split_date_ts)

    def max_drawdown(returns): 
        cum = (1 + returns).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        return drawdown.min()

    y = (ret.loc[mask].mean(axis=1).shift(-window).rolling(window).mean() - 
         config["PRUNEDOWN"]*(ret.loc[mask].mean(axis=1).shift(-window).rolling(window).apply(max_drawdown, raw=False) + 1e-10)).dropna()
    
    X = feat.loc[y.index]
    
    # Data inspection
    print(f"\n[DATA INSPECTION]")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X dtypes: {X.dtypes.value_counts().to_dict()}")
    print(f"  y dtype: {y.dtype}")
    print(f"  X memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  X has NaN: {X.isnull().any().any()}")
    print(f"  y has NaN: {y.isnull().any()}")
    print(f"  X min/max: {X.min().min():.6f} / {X.max().max():.6f}")
    print(f"  y min/max: {y.min():.6f} / {y.max():.6f}")
    
    # Create RandomForest with YOUR current settings
    print(f"\n[RANDOMFOREST CREATION]")
    print(f"  n_estimators from config: {config['NESTIM']}")
    print(f"  random_state from config: {config['SEED']}")
    
    model = RandomForestRegressor(
        n_estimators=config["NESTIM"], 
        random_state=config["SEED"],
        n_jobs=1  # Your current setting
    )
    
    # INSPECT ALL MODEL PARAMETERS
    print(f"\n[RANDOMFOREST PARAMETERS - BEFORE FIT]")
    params = model.get_params()
    for key in sorted(params.keys()):
        print(f"  {key}: {params[key]}")
    
    # Check what sklearn defaults are
    print(f"\n[SKLEARN DEFAULTS COMPARISON]")
    default_rf = RandomForestRegressor()
    default_params = default_rf.get_params()
    
    for key in sorted(default_params.keys()):
        if params[key] != default_params[key]:
            print(f"  {key}: YOUR={params[key]} vs DEFAULT={default_params[key]} *** DIFFERENT")
        else:
            print(f"  {key}: {params[key]} (same as default)")
    
    # Fit the model
    print(f"\n[RANDOMFOREST FITTING]")
    print(f"  Fitting with X shape {X.shape}, y shape {y.shape}")
    
    # Random state just before fitting
    pre_fit_state = np.random.get_state()[1][:5]
    print(f"  Random state just before fit: {pre_fit_state}")
    
    model.fit(X, y)
    
    # Random state after fitting
    post_fit_state = np.random.get_state()[1][:5]
    print(f"  Random state just after fit: {post_fit_state}")
    
    # INSPECT FITTED MODEL
    print(f"\n[RANDOMFOREST AFTER FITTING]")
    print(f"  n_estimators (actual): {model.n_estimators}")
    print(f"  n_features_in_: {model.n_features_in_}")
    print(f"  feature_names_in_: {getattr(model, 'feature_names_in_', 'NOT AVAILABLE')}")
    print(f"  estimators count: {len(model.estimators_)}")
    
    # Check first few estimators' random states
    print(f"\n[INDIVIDUAL TREE INSPECTION]")
    for i in range(min(3, len(model.estimators_))):
        tree = model.estimators_[i]
        print(f"  Tree {i}: random_state={tree.random_state}, max_depth={tree.tree_.max_depth}, n_leaves={tree.tree_.n_leaves}")
    
    # Get feature importances
    scores = pd.Series(model.feature_importances_, index=X.columns)
    
    print(f"\n[FEATURE IMPORTANCE ANALYSIS]")
    print(f"  Total features: {len(scores)}")
    print(f"  Feature importance sum: {scores.sum():.10f}")
    print(f"  Feature importance mean: {scores.mean():.10f}")
    print(f"  Feature importance std: {scores.std():.10f}")
    print(f"  Non-zero importances: {(scores > 0).sum()}")
    print(f"  Zero importances: {(scores == 0).sum()}")
    
    # Show top 10 features with high precision
    print(f"\n[TOP 10 FEATURES - HIGH PRECISION]")
    top_scores = scores.sort_values(ascending=False).head(10)
    for i, (feat_name, score) in enumerate(top_scores.items()):
        print(f"  {i+1:2d}. {feat_name}: {score:.12f}")
    
    # Check if there are ties around the 175th position
    print(f"\n[FEATURE SELECTION ANALYSIS]")
    sorted_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        thresh_int = int(thresh)
        print(f"  Selecting top {thresh_int} features")
        
        # Show features around the cutoff
        start_idx = max(0, thresh_int - 5)
        end_idx = min(len(sorted_scores), thresh_int + 5)
        
        print(f"  Features around position {thresh_int}:")
        for i in range(start_idx, end_idx):
            marker = " >>> CUTOFF <<<" if i == thresh_int else ""
            score = sorted_scores.iloc[i]
            feat_name = sorted_scores.index[i]
            print(f"    {i+1:3d}. {feat_name}: {score:.12f}{marker}")
        
        # Check for ties
        cutoff_score = sorted_scores.iloc[thresh_int-1]
        ties = (sorted_scores == cutoff_score).sum()
        if ties > 1:
            print(f"  WARNING: {ties} features have the same score as the {thresh_int}th feature!")
            print(f"  Cutoff score: {cutoff_score:.12f}")
    
    # Memory and performance info
    print(f"\n[PERFORMANCE INFO]")
    print(f"  Model memory usage: ~{model.estimators_[0].tree_.capacity * len(model.estimators_) * 8 / 1024**2:.2f} MB (estimated)")
    
    # Final random state
    final_state = np.random.get_state()[1][:5]
    print(f"  Final random state: {final_state}")
    
    # FEATURE SELECTION (like original function)
    combined_scores = scores.sort_values(ascending=False)
    
    if thresh > 1:
        selected_features = combined_scores.nlargest(int(thresh)).index
        print(f"\n[FINAL SELECTION] Selected top {int(thresh)} features by {method[0]} from {start.date()}–{split_date_ts.date()}")
    elif 0 < thresh <= 1:
        selected_features = combined_scores[combined_scores > thresh].index
        print(f"\n[FINAL SELECTION] Selected {len(selected_features)} features with {method[0]} > {thresh} from {start.date()}–{split_date_ts.date()}")
    else: 
        return feat

    print(f"[FINAL SELECTION] Top feature score: {combined_scores.loc[selected_features].head(1).to_string()}")
    
    # RETURN SELECTED FEATURES LIKE ORIGINAL
    return feat[selected_features]

# Usage - add this to your debug script:
# scores, model = inspect_rf_settings(feat, ret, split_date)