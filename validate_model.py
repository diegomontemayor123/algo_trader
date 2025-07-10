import os
import logging
import numpy as np
import pandas as pd
import torch
import optuna
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from compute_features import compute_features, normalize_features
from walkforward import run_walkforward_test_with_validation
from backtest import run_backtest
from tech import (
    create_model, create_sequences, split_train_validation,
    MarketDataset, train_model_with_validation, calculate_performance_metrics,
    TICKERS, INITIAL_CAPITAL, DEVICE, FEATURES,
    LOOKBACK, PREDICT_DAYS, BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DATA_WARNINGS = []

# ---------------------
# Parameter Sensitivity via Optuna
# ---------------------

def objective(trial):
    # Define tunable hyperparameters
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    max_leverage = trial.suggest_float('max_leverage', 0.5, 2.0)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    lookback = trial.suggest_int('lookback', 50, 120)
    predict_days = trial.suggest_int('predict_days', 3, 15)
    layer_count = trial.suggest_int('layer_count', 2, 8)
    decay = trial.suggest_float('decay', 0.0, 0.1)
    feature_attention_enabled = trial.suggest_categorical('feature_attention_enabled', [True, False])
    return_penalty_enabled = trial.suggest_categorical('return_penalty_enabled', [True, False])

    # Update global / environment vars dynamically for tech.py consumption
    os.environ["DROPOUT"] = str(dropout)
    os.environ["MAX_LEVERAGE"] = str(max_leverage)
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["LOOKBACK"] = str(lookback)
    os.environ["PREDICT_DAYS"] = str(predict_days)
    os.environ["LAYER_COUNT"] = str(layer_count)
    os.environ["DECAY"] = str(decay)
    os.environ["FEATURE_ATTENTION_ENABLED"] = '1' if feature_attention_enabled else '0'
    os.environ["RETURN_PENALTY_ENABLED"] = '1' if return_penalty_enabled else '0'

    # Reload updated constants from env for safety
    global LOOKBACK, PREDICT_DAYS
    LOOKBACK = int(os.environ["LOOKBACK"])
    PREDICT_DAYS = int(os.environ["PREDICT_DAYS"])

    # Compute features and returns (cache if large dataset)
    features, returns = compute_features(TICKERS, pd.Timestamp('2012-01-01'), pd.Timestamp('2025-06-01'), FEATURES)
    features.index = pd.to_datetime(features.index)
    returns.index = pd.to_datetime(returns.index)

    # Prepare datasets
    sequences, targets, _ = create_sequences(features, returns)
    train_seq, train_tgt, val_seq, val_tgt = split_train_validation(sequences, targets)

    train_dataset = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_dataset = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model and train
    model = create_model(train_dataset[0][0].shape[1])
    trained_model = train_model_with_validation(model, train_loader, val_loader, learning_rate=learning_rate, epochs=10)

    # Backtest on validation data or small holdout set
    portfolio_values = run_backtest(
        DEVICE, INITIAL_CAPITAL, pd.Timestamp('2023-01-01'), LOOKBACK, max_leverage,
        compute_features, normalize_features, calculate_performance_metrics,
        TICKERS, pd.Timestamp('2023-01-01'), pd.Timestamp('2023-06-30'), FEATURES,
        trained_model, plot=False
    )

    metrics = calculate_performance_metrics(portfolio_values)
    sharpe = metrics.get('sharpe_ratio', -np.inf)

    logging.info(f"Trial completed: Dropout={dropout:.3f}, Leverage={max_leverage:.2f}, Sharpe={sharpe:.4f}")

    # Optuna tries to maximize the objective, so return sharpe ratio as is
    return sharpe

def run_optuna_tuning(n_trials=25):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logging.info(f"Optuna best params: {best_params}")

    # Save param results to CSV for audit
    df = study.trials_dataframe()
    df.to_csv("optuna_trials.csv", index=False)

    return best_params

# ---------------------
# Equal Weighted Benchmark
# ---------------------

def run_equal_weight_benchmark(returns, initial_capital=100):
    weights = np.ones(len(TICKERS)) / len(TICKERS)
    portfolio_values = [initial_capital]
    for i in range(len(returns)):
        period_returns = returns.iloc[i].values
        portfolio_return = np.dot(weights, period_returns)
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    return portfolio_values

# ---------------------
# Full Pipeline Main
# ---------------------

def generate_summary_report(metrics, feature_importance, sensitivity_results, walkforward_df, holdout_metrics, benchmark_metrics, filename="validation_summary.txt"):
    with open(filename, "w") as f:
        def write_line(line=""):
            f.write(line + "\n")
            logging.info(line)

        write_line("=== Validation Pipeline Summary Report ===\n")

        # Data warnings
        if DATA_WARNINGS:
            write_line("Data Warnings:")
            for w in DATA_WARNINGS:
                write_line(f"  {w}")
            write_line()
        else:
            write_line("No data warnings detected.\n")

        # Performance metrics (final backtest)
        write_line("Final Backtest Metrics:")
        for k, v in metrics.items():
            write_line(f"  {k}: {v:.4f}")
        write_line()

        # Feature importance
        if feature_importance:
            write_line("Feature Importance:")
            for feat, weight in feature_importance.items():
                write_line(f"  {feat}: {weight:.4f}")
        else:
            write_line("Feature importance not available.")
        write_line()

        # Parameter sensitivity (best results)
        if sensitivity_results:
            write_line("Parameter Sensitivity Best Results:")
            for res in sensitivity_results:
                params_str = ", ".join(f"{k}={v}" for k, v in res['params'].items())
                write_line(f"  Params: {params_str} -> Metric: {res['metric']:.4f}")
        else:
            write_line("No parameter sensitivity test results.")
        write_line()

        # Walkforward aggregate
        if walkforward_df is not None and not walkforward_df.empty:
            write_line("Walkforward Aggregate Metrics:")
            write_line(f"  Avg CAGR: {walkforward_df['cagr'].mean():.4f}")
            write_line(f"  Avg Sharpe Ratio: {walkforward_df['sharpe_ratio'].mean():.4f}")
            write_line(f"  Avg Max Drawdown: {walkforward_df['max_drawdown'].mean():.4f}")
            write_line(f"  Number of Walkforward Periods: {len(walkforward_df)}")
            write_line()

        # Holdout results
        if holdout_metrics:
            write_line("Holdout (2024–2025) Metrics:")
            for k, v in holdout_metrics.items():
                write_line(f"  {k}: {v:.4f}")
            write_line()

        # Benchmark comparison
        if benchmark_metrics:
            write_line("Equal Weighted Benchmark Metrics:")
            for k, v in benchmark_metrics.items():
                write_line(f"  {k}: {v:.4f}")
            write_line()

        # Comparison summary
        if walkforward_df is not None and not walkforward_df.empty and benchmark_metrics and metrics:
            write_line("Performance Comparison Summary:")
            write_line(f"  Backtest CAGR vs Benchmark: {metrics['CAGR']:.4f} vs {benchmark_metrics['CAGR']:.4f}")
            write_line(f"  Walkforward Avg CAGR vs Benchmark: {walkforward_df['cagr'].mean():.4f} vs {benchmark_metrics['CAGR']:.4f}")

        write_line("\n=== End of Report ===")

    logging.info(f"Validation summary report saved as {filename}")

def main():
    logging.info("Starting Validation Pipeline with Optuna tuning...")

    # 1. Optuna Hyperparameter tuning
    best_params = run_optuna_tuning(n_trials=30)

    # Apply best params globally for full retrain
    for k, v in best_params.items():
        os.environ[k.upper()] = str(v)

    global LOOKBACK, PREDICT_DAYS, DROPOUT, MAX_LEVERAGE, DECAY, LAYER_COUNT, FEATURE_ATTENTION_ENABLED, RETURN_PENALTY_ENABLED
    LOOKBACK = int(os.environ.get("LOOKBACK", LOOKBACK))
    PREDICT_DAYS = int(os.environ.get("PREDICT_DAYS", PREDICT_DAYS))
    DROPOUT = float(os.environ.get("DROPOUT", DROPOUT))
    MAX_LEVERAGE = float(os.environ.get("MAX_LEVERAGE", MAX_LEVERAGE))
    DECAY = float(os.environ.get("DECAY", DECAY))
    LAYER_COUNT = int(os.environ.get("LAYER_COUNT", LAYER_COUNT))
    FEATURE_ATTENTION_ENABLED = bool(int(os.environ.get("FEATURE_ATTENTION_ENABLED", FEATURE_ATTENTION_ENABLED)))
    RETURN_PENALTY_ENABLED = bool(int(os.environ.get("RETURN_PENALTY_ENABLED", RETURN_PENALTY_ENABLED)))

    # 2. Prepare full dataset
    features, returns = compute_features(TICKERS, pd.Timestamp('2012-01-01'), pd.Timestamp('2025-06-01'), FEATURES)
    features.index = pd.to_datetime(features.index)
    returns.index = pd.to_datetime(returns.index)

    # 3. Prepare main datasets
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Train model with best hyperparams
    model = create_model(train_dataset[0][0].shape[1])
    trained_model = train_model_with_validation(model, train_loader, val_loader, learning_rate=float(os.environ.get("LEARNING_RATE", 1e-3)), epochs=20)

    # 5. Extract feature importance
    feature_importance = None
    if hasattr(trained_model, 'feature_weights'):
        weights = trained_model.feature_weights.detach().cpu().numpy()
        feature_importance = dict(zip(FEATURES, weights))

    # 6. Run final backtest over entire test period
    portfolio_values = run_backtest(
        DEVICE, INITIAL_CAPITAL, pd.Timestamp('2023-01-01'), LOOKBACK, MAX_LEVERAGE,
        compute_features, normalize_features, calculate_performance_metrics,
        TICKERS, pd.Timestamp('2023-01-01'), pd.Timestamp('2025-06-01'), FEATURES,
        trained_model, plot=True
    )

    metrics = calculate_additional_metrics(portfolio_values)

    # 7. Run walkforward test and save results
    walkforward_results = run_walkforward_test_with_validation(
        compute_features, create_sequences, split_train_validation,
        MarketDataset, create_model, train_model_with_validation,
        normalize_features, calculate_performance_metrics,
        pd.Timestamp('2023-01-01'), 60, 365,
        LOOKBACK, PREDICT_DAYS, BATCH_SIZE, INITIAL_CAPITAL,
        MAX_LEVERAGE, DEVICE, TICKERS, pd.Timestamp('2012-01-01'), pd.Timestamp('2025-06-01'),
        FEATURES,
        save_results_path="walkforward_results.csv"
    )

    walkforward_df = None
    if walkforward_results:
        walkforward_df = pd.DataFrame(walkforward_results)
        walkforward_df.to_csv("walkforward_results.csv", index=False)
        logging.info("[Main] Walkforward results saved to walkforward_results.csv")

    # 8. Run holdout test on 2024-2025 data
    holdout_start = pd.Timestamp('2024-01-01')
    holdout_end = pd.Timestamp('2025-06-01')

    logging.info("Running holdout backtest 2024–2025...")
    holdout_portfolio = run_backtest(
        DEVICE, INITIAL_CAPITAL, holdout_start, LOOKBACK, MAX_LEVERAGE,
        compute_features, normalize_features, calculate_performance_metrics,
        TICKERS, holdout_start, holdout_end, FEATURES,
        trained_model, plot=True
    )
    holdout_metrics = calculate_additional_metrics(holdout_portfolio)

    # 9. Equal weighted benchmark for test period
    benchmark_portfolio = run_equal_weight_benchmark(returns.loc[holdout_start:holdout_end], INITIAL_CAPITAL)
    benchmark_metrics = calculate_additional_metrics(benchmark_portfolio)

    logging.info("Equal Weighted Benchmark metrics:")
    for k, v in benchmark_metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    # 10. Generate comprehensive summary report
    generate_summary_report(metrics, feature_importance, [], walkforward_df, holdout_metrics, benchmark_metrics)

    logging.info("Validation pipeline completed successfully.")


if __name__ == "__main__":
    main()
