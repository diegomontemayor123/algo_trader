import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
import csv

from compute_features import compute_features, normalize_features
from walkforward import run_walkforward_test_with_validation
from backtest import run_backtest
from tech import (
    create_model, create_sequences, split_train_validation,
    MarketDataset, train_model_with_validation, calculate_performance_metrics,
    TICKERS, LOOKBACK, PREDICT_DAYS, DEVICE, MAX_LEVERAGE, INITIAL_CAPITAL,
    FEATURES, SPLIT_DATE, BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DATA_WARNINGS = []  # Collect data issues here for reporting


def calculate_additional_metrics(equity_curve):
    equity = pd.Series(equity_curve).dropna()
    returns = equity.pct_change().dropna()
    years = len(returns) / 252

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)

    downside_returns = returns[returns < 0]
    sortino = returns.mean() / (downside_returns.std() + 1e-6) * np.sqrt(252) if len(downside_returns) > 0 else np.nan

    peak = equity.cummax()
    drawdowns = (equity - peak) / peak
    max_drawdown = drawdowns.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar
    }
    return metrics


def plot_equity_curve(equity_curve, title='Equity Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved plot: {filename}")


def parameter_sensitivity_test(param_grid, features, returns):
    results = []
    logging.info("Starting parameter sensitivity testing...")
    for params in param_grid:
        logging.info(f"Testing parameters: {params}")
        model = create_model(features.shape[1])

        # Customize model or training with params as needed
        train_dataset, val_dataset, _ = prepare_main_datasets(features, returns)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        trained_model = train_model_with_validation(model, train_loader, val_loader, epochs=10)

        # Placeholder metric, replace with real metric extraction
        dummy_metric = np.random.rand()
        logging.info(f"Completed test for params {params} with dummy metric {dummy_metric:.4f}")
        results.append({'params': params, 'metric': dummy_metric})
    return results


def extract_feature_importance(model):
    if hasattr(model, 'feature_weights'):
        weights = model.feature_weights.detach().cpu().numpy()
        feature_importance = dict(zip(FEATURES, weights))
        logging.info("Feature importance extracted:")
        for feat, weight in feature_importance.items():
            logging.info(f"  {feat}: {weight:.4f}")
        return feature_importance
    else:
        logging.warning("Model does not have feature_weights attribute.")
        return None


def prepare_main_datasets(features, returns):
    sequences, targets, seq_dates = create_sequences(features, returns)
    if len(set(seq_dates)) != len(seq_dates):
        DATA_WARNINGS.append("[Data] Warning: Duplicate dates found in sequence dates.")
    if any(pd.isna(seq_dates)):
        DATA_WARNINGS.append("[Data] Warning: NaN detected in sequence dates.")
    train_sequences, train_targets = [], []
    test_sequences, test_targets = [], []

    for seq, tgt, date in zip(sequences, targets, seq_dates):
        if date < SPLIT_DATE:
            train_sequences.append(seq)
            train_targets.append(tgt)
        else:
            test_sequences.append(seq)
            test_targets.append(tgt)

    train_seq, train_tgt, val_seq, val_tgt = split_train_validation(train_sequences, train_targets)

    train_dataset = MarketDataset(torch.tensor(np.array(train_seq)), torch.tensor(np.array(train_tgt)))
    val_dataset = MarketDataset(torch.tensor(np.array(val_seq)), torch.tensor(np.array(val_tgt)))
    test_dataset = MarketDataset(torch.tensor(np.array(test_sequences)), torch.tensor(np.array(test_targets)))

    logging.info(f"Datasets prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def save_sensitivity_results_to_csv(results, filename="parameter_sensitivity_results.csv"):
    keys = results[0]['params'].keys() if results else []
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = list(keys) + ['metric']
        writer.writerow(header)
        for res in results:
            row = list(res['params'].values()) + [res['metric']]
            writer.writerow(row)
    logging.info(f"Parameter sensitivity results saved to {filename}")


def generate_summary_report(metrics, feature_importance, sensitivity_results, filename="validation_summary.txt"):
    with open(filename, "w") as f:
        def write_line(line=""):
            f.write(line + "\n")
            logging.info(line)

        write_line("=== Validation Pipeline Summary Report ===\n")

        # Data Warnings
        if DATA_WARNINGS:
            write_line("Data Warnings:")
            for warn in DATA_WARNINGS:
                write_line(f"  {warn}")
            write_line()
        else:
            write_line("No data warnings detected.\n")

        # Performance Metrics
        write_line("Performance Metrics (Final Backtest):")
        for k, v in metrics.items():
            write_line(f"  {k}: {v:.4f}")
        write_line()

        # Feature Importance
        if feature_importance:
            write_line("Feature Importance:")
            for feat, weight in feature_importance.items():
                write_line(f"  {feat}: {weight:.4f}")
        else:
            write_line("Feature importance not available.")
        write_line()

        # Parameter Sensitivity Results
        if sensitivity_results:
            write_line("Parameter Sensitivity Test Results:")
            for res in sensitivity_results:
                params_str = ", ".join(f"{k}={v}" for k, v in res['params'].items())
                write_line(f"  Params: {params_str} -> Metric: {res['metric']:.4f}")
        else:
            write_line("No parameter sensitivity test results.")
        write_line()

        write_line("=== End of Report ===")

    logging.info(f"Validation summary report saved as {filename}")


def main():
    logging.info("Starting validation pipeline...")

    start_date = pd.Timestamp('2012-01-01')
    end_date = pd.Timestamp('2025-06-01')
    split_date = SPLIT_DATE

    features, returns = compute_features(TICKERS, start_date, end_date, FEATURES)
    features.index = pd.to_datetime(features.index)
    returns.index = pd.to_datetime(returns.index)

    step_sizes = [30, 60, 90]
    train_windows = [180, 365, 730]

    for step in step_sizes:
        for wndw in train_windows:
            logging.info(f"Running walk-forward test: Step={step} days, Train Window={wndw} days")
            run_walkforward_test_with_validation(
                compute_features, create_sequences, split_train_validation,
                MarketDataset, create_model, train_model_with_validation,
                normalize_features, calculate_performance_metrics,
                split_date, step, wndw,
                LOOKBACK, PREDICT_DAYS, BATCH_SIZE, INITIAL_CAPITAL,
                MAX_LEVERAGE, DEVICE, TICKERS, start_date, end_date,
                FEATURES
            )
            logging.info("Walk-forward iteration completed.")

    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(train_dataset[0][0].shape[1])
    trained_model = train_model_with_validation(model, train_loader, val_loader)

    feature_importance = extract_feature_importance(trained_model)

    portfolio_values = run_backtest(
        DEVICE, INITIAL_CAPITAL, split_date, LOOKBACK, MAX_LEVERAGE,
        compute_features, normalize_features, calculate_performance_metrics,
        TICKERS, start_date, end_date, FEATURES, trained_model, plot=False
    )

    plot_equity_curve(portfolio_values, title="Final Backtest Equity Curve")

    metrics = calculate_additional_metrics(portfolio_values)

    param_grid = [
        {'dropout': 0.10, 'max_leverage': 1.0},
        {'dropout': 0.15, 'max_leverage': 1.0},
        {'dropout': 0.20, 'max_leverage': 1.5},
    ]
    sensitivity_results = parameter_sensitivity_test(param_grid, features, returns)
    save_sensitivity_results_to_csv(sensitivity_results)

    generate_summary_report(metrics, feature_importance, sensitivity_results)

    logging.info("Validation pipeline completed successfully.")


if __name__ == "__main__":
    main()
