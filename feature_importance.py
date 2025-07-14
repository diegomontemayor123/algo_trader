import os, sys, torch, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compute_features import compute_features, load_price_data, normalize_features
from loadconfig import load_config
from data_prep import prepare_main_datasets
from model import create_model, save_top_features_csv, TransformerTrader
from train import train_main_model
from backtest import run_backtest
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_nan_or_constant(features: pd.DataFrame):
    """Check for constant or NaN features."""
    logging.info("[Check] Validating feature matrix integrity...")
    nan_columns = features.columns[features.isna().any()]
    constant_columns = [col for col in features.columns if features[col].nunique() == 1]
    if len(nan_columns):
        logging.warning(f"[Data] NaN detected in features: {nan_columns.tolist()}")
    if len(constant_columns):
        logging.warning(f"[Data] Constant features: {constant_columns}")
    return nan_columns, constant_columns


def check_feature_weight_diversity(model: TransformerTrader):
    """Check for uniform feature weights — indicates weak feature discrimination."""
    weights = model.feature_weights.detach().cpu().numpy()
    std = np.std(weights)
    logging.info(f"[Model] Feature weight std: {std:.6f}")
    if std < 0.01:
        logging.warning("[Model] All feature weights nearly equal — possible signal absence or over-regularization.")
    return std


def check_model_under_overfit(train_perf: dict, test_perf: dict):
    """Compare Sharpe ratio and drawdown to determine generalization."""
    sharpe_gap = train_perf["sharpe_ratio"] - test_perf["sharpe_ratio"]
    drawdown_gap = test_perf["max_drawdown"] - train_perf["max_drawdown"]

    if sharpe_gap > 0.5:
        logging.warning("[Eval] Model likely overfitting — large Sharpe drop from train to test.")
    elif sharpe_gap < -0.5:
        logging.warning("[Eval] Model possibly underfitting — better test than train (unexpected).")
    else:
        logging.info("[Eval] Model generalization within expected bounds.")

    return sharpe_gap, drawdown_gap


def run_evaluation():
    config = load_config()

    tickers = config["TICKERS"]
    feature_list = config["FEATURES"]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [k.strip() for k in macro_keys.split(",") if k.strip()]

    raw_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, feature_list, raw_data, macro_keys)

    # Step 1: Check NaNs & constants
    nan_cols, const_cols = check_nan_or_constant(features)

    # Step 2: Prepare data splits
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns, config)
    input_dim = train_dataset[0][0].shape[1]

    # Step 3: Train model
    logging.info("[Train] Initiating training cycle...")
    model = train_main_model(config, features, returns)

    # Step 4: Evaluate feature weight variance
    weight_std = check_feature_weight_diversity(model)

    # Step 5: Save top features
    save_top_features_csv(model, features.columns.tolist())

    # Step 6: Run backtest
    logging.info("[Backtest] Executing validation on test set...")
    results = run_backtest(
        device=DEVICE,
        initial_capital=config["INITIAL_CAPITAL"],
        split_date=config["SPLIT_DATE"],
        lookback=config["LOOKBACK"],
        max_leverage=config["MAX_LEVERAGE"],
        compute_features=compute_features,
        normalize_features=normalize_features,
        tickers=tickers,
        start_date=config["START_DATE"],
        end_date=config["END_DATE"],
        features=feature_list,
        macro_keys=macro_keys,
        test_chunk_months=config["TEST_CHUNK_MONTHS"],
        model=model,
        plot=False,
        config=config,
        retrain_window=config["RETRAIN_WINDOW"],
    )

    test_perf = results["portfolio"]

    # Optional dummy train_perf assumption (for diagnostic continuity)
    train_perf = {
        "sharpe_ratio": float("nan"),
        "max_drawdown": float("nan"),
        "cagr": float("nan"),
    }

    # Step 7: Diagnostics (limited without train_perf)
    check_model_under_overfit(train_perf, test_perf)

    # Step 8: Print KPIs
    logging.info(f"\n[Results] Test Sharpe: {test_perf.get('sharpe_ratio', float('nan')):.4f} | Max DD: {test_perf.get('max_drawdown', float('nan')):.4f} | CAGR: {test_perf.get('cagr', float('nan')):.4f}")
    
    logging.info("\n[Conclusion]")
    if len(nan_cols) or len(const_cols):
        logging.info("- Feature engineering requires cleanup.")
    if weight_std < 0.01:
        logging.info("- Feature weights are uniform — model may not be learning any signal.")
    if test_perf['sharpe_ratio'] < 0.5:
        logging.info("- Strategy underperforms — consider boosting model capacity, tuning lookback, or improving alpha features.")
    else:
        logging.info("- Model performing within acceptable thresholds.")

    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_evaluation()
