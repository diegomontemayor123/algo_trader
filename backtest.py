import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging

def run_backtest(
    device,
    initial_capital,
    split_date,
    lookback,
    max_leverage,
    compute_features,
    normalize_features,
    calculate_performance_metrics,
    tickers,
    start_date,
    end_date,
    features,
    model=None,
    plot=False,
    weights_csv_path="daily_portfolio_weights.csv"
):
    logging.info("[Backtest] Starting backtest...")
    try:
        features_df, returns_df = compute_features(tickers, start_date, end_date, features)
        features_df.index = pd.to_datetime(features_df.index)
        returns_df.index = pd.to_datetime(returns_df.index)

        if features_df.empty or returns_df.empty:
            logging.warning("[Backtest] Feature or return data is empty. Returning flat curve.")
            return pd.Series([initial_capital])

        if model is None:
            raise ValueError("Model must be provided to run_backtest()")

        model.eval()
        portfolio_values = [initial_capital]
        benchmark_values = [initial_capital]
        daily_weights = []

        try:
            start_index = features_df.index.get_indexer([pd.to_datetime(split_date)], method='bfill')[0]
        except Exception as e:
            logging.error(f"[Backtest] Error finding split_date index: {e}")
            return pd.Series([initial_capital])

        test_dates = returns_df.index[start_index:]
        asset_names = returns_df.columns

        for i in range(start_index - lookback, len(features_df) - lookback):
            feature_window = features_df.iloc[i:i + lookback].values.astype(np.float32)
            normalized_features = normalize_features(feature_window)
            input_tensor = torch.tensor(normalized_features).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                raw_weights = model(input_tensor).cpu().numpy().flatten()

            weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
            scaling_factor = min(max_leverage / weight_sum, 1.0)
            final_weights = raw_weights * scaling_factor

            period_returns = returns_df.iloc[i + lookback].values
            portfolio_return = np.dot(final_weights, period_returns)
            benchmark_return = np.mean(period_returns)

            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))

            weight_date = returns_df.index[i + lookback]
            daily_weights.append(pd.Series(final_weights, index=asset_names, name=weight_date))

        if not daily_weights:
            logging.warning("[Backtest] No daily weights generated. Possibly insufficient data.")
            return pd.Series([initial_capital])

        weights_df = pd.DataFrame(daily_weights)
        weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
        weights_df.index.name = "Date"
        weights_df.to_csv(weights_csv_path)
        logging.info(f"[Backtest] Saved daily portfolio weights to {weights_csv_path}")

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

        return pd.Series(portfolio_values[1:], index=test_dates)

    except Exception as e:
        logging.error("[Backtest] Critical failure during backtest: %s", str(e))
        return pd.Series([initial_capital])
