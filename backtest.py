import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def run_backtest (DEVICE, INITIAL_CAPITAL, SPLIT_DATE, LOOKBACK, MAX_LEVERAGE, compute_features, normalize_features,calculate_performance_metrics, TICKERS,START_DATE,END_DATE,FEATURES,model=None, plot=False,weights_csv_path="daily_portfolio_weights.csv", ):
    print("[Backtest] Starting backtest...")
    features, returns = compute_features(TICKERS,START_DATE,END_DATE,FEATURES)
    if model is None:
        raise ValueError("Model must be passed to run_backtest() to avoid checkpoint mismatch.")
    model.eval()
    portfolio_values = [INITIAL_CAPITAL]
    benchmark_values = [INITIAL_CAPITAL]

    # Store daily weights
    daily_weights = []
    start_index = features.index.get_indexer([pd.to_datetime(SPLIT_DATE)], method='bfill')[0]
    test_dates = returns.index[start_index:]
    asset_names = returns.columns
    for i in range(start_index - LOOKBACK, len(features) - LOOKBACK):
        feature_window = features.iloc[i:i + LOOKBACK].values.astype(np.float32)
        normalized_features = normalize_features(feature_window)
        input_tensor = torch.tensor(normalized_features).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            raw_weights = model(input_tensor).cpu().numpy().flatten()
        weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
        scaling_factor = min(MAX_LEVERAGE / weight_sum, 1.0)  
        final_weights = raw_weights * scaling_factor
        period_returns = returns.iloc[i + LOOKBACK].values
        portfolio_return = np.dot(final_weights, period_returns)
        benchmark_return = np.mean(period_returns)
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
        weight_date = returns.index[i + LOOKBACK]
        daily_weights.append(pd.Series(final_weights, index=asset_names, name=weight_date))
    weights_df = pd.DataFrame(daily_weights)
    weights_df["total_exposure"] = weights_df.sum(axis=1)  # or .abs().sum(axis=1) for gross leverage
    weights_df.index.name = "Date"
    weights_df.to_csv(weights_csv_path)
    portfolio_metrics = calculate_performance_metrics(portfolio_values)
    benchmark_metrics = calculate_performance_metrics(benchmark_values)

    print("\n[Backtest] === Performance Summary ===")
    for metric_name in portfolio_metrics:
        print(f"  {metric_name.replace('_', ' ').title()}:")
        print(f"    Strategy: {portfolio_metrics[metric_name]:.2%}")
        print(f"    Benchmark: {benchmark_metrics[metric_name]:.2%}")
    print("\n[Backtest] === Portfolio Weights Summary ===")
    weights_summary = weights_df.describe().loc[["min", "mean", "max"]]
    for column in weights_summary.columns:
        print(f"\n  {column}:")
        print(f"    Min:  {weights_summary.loc['min', column]:.4f}")
        print(f"    Mean: {weights_summary.loc['mean', column]:.4f}")
        print(f"    Max:  {weights_summary.loc['max', column]:.4f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, portfolio_values[1:], label="Transformer Strategy", linewidth=2)
        plt.plot(test_dates, benchmark_values[1:], label="Equal-Weight Benchmark", linewidth=2)
        plt.title("Portfolio Performance Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("portfolio_performance.png", dpi=300)
        plt.close()
