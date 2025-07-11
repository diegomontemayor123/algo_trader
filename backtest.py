import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta

def calculate_performance_metrics(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        print("[Performance] Not enough data points to calculate metrics.")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    returns = equity_curve.pct_change().dropna()
    if returns.empty or equity_curve.iloc[0] <= 0:
        print("[Performance] Invalid returns or initial capital.")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(returns) / 252
    if years <= 0:
        print("[Performance] Invalid time span (years <= 0).")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    try:
        cagr = total_return ** (1 / years) - 1
    except Exception as e:
        print(f"[Performance] Error calculating CAGR: {e}")
        cagr = 0.0
    std_returns = returns.std()

    if std_returns == 0 or np.isnan(std_returns):
        print("[Performance] Std dev of returns is zero or NaN — Sharpe set to 0")
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = returns.mean() / std_returns * np.sqrt(252)
    peak_values = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak_values) / peak_values
    max_drawdown = drawdowns.min()

    return {
        'cagr': float(cagr),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown)
    }


def run_backtest(
    device,
    initial_capital,
    split_date,
    lookback,
    max_leverage,
    compute_features,
    normalize_features,
    tickers,
    start_date,
    end_date,
    features,
    model=None,
    plot=False,
    test_chunk_months=6,
    weights_csv_path="daily_portfolio_weights.csv"
):
    if model is None:
        raise ValueError("Model must be provided to run_backtest()")

    model.eval()

    # --- Compute full features and returns once ---
    features_df, returns_df = compute_features(tickers, start_date, end_date, features)
    features_df.index = pd.to_datetime(features_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)

    # Determine starting index in features_df for backtest start
    split_date_dt = pd.to_datetime(split_date)
    end_date_dt = pd.to_datetime(end_date)
    try:
        start_index = features_df.index.get_indexer([split_date_dt], method='bfill')[0]
    except Exception as e:
        logging.error(f"[Backtest] Error finding start index for backtest: {e}")
        return None

    asset_names = returns_df.columns
    portfolio_values = [initial_capital]
    benchmark_values = [initial_capital]
    daily_weights = []

    logging.info(f"[Backtest] Running full backtest: {split_date_dt.date()} to {end_date_dt.date()} with initial capital ${initial_capital:.2f}")

    # --- Run full backtest ---
    for i in range(start_index - lookback, len(features_df) - lookback):
        current_date = returns_df.index[i + lookback]
        if current_date > end_date_dt:
            break
        if current_date < split_date_dt:
            continue

        feature_window = features_df.iloc[i:i + lookback].values.astype(np.float32)
        normalized_features = normalize_features(feature_window)
        input_tensor = torch.tensor(normalized_features).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_weights = model(input_tensor).cpu().numpy().flatten()

        weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
        scaling_factor = min(max_leverage / weight_sum, 1.0)
        final_weights = raw_weights * scaling_factor

        period_returns = returns_df.loc[current_date].values
        portfolio_return = np.dot(final_weights, period_returns)
        benchmark_return = np.mean(period_returns)

        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))

        daily_weights.append(pd.Series(final_weights, index=asset_names, name=current_date))

    if not daily_weights:
        logging.warning("[Backtest] No daily weights generated; returning empty results")
        flat_series = pd.Series([initial_capital], index=pd.date_range(split_date_dt, end_date_dt))
        flat_metrics = {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        return {
            'portfolio': flat_metrics,
            'benchmark': flat_metrics,
            'combined_equity_curve': flat_series,
            'combined_benchmark_equity_curve': flat_series,
            'performance_variance': {}
        }

    # Convert to DataFrames / Series
    weights_df = pd.DataFrame(daily_weights)
    weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
    weights_df.index.name = "Date"

    portfolio_series = pd.Series(portfolio_values[1:], index=weights_df.index)
    benchmark_series = pd.Series(benchmark_values[1:], index=weights_df.index)

    # --- Save full weights CSV ---
    weights_df.to_csv(weights_csv_path)

    # --- Generate chunks ---
    chunks = []
    current_start = split_date_dt
    while current_start < end_date_dt:
        current_end = current_start + relativedelta(months=test_chunk_months) - pd.Timedelta(days=1)
        if current_end > end_date_dt:
            current_end = end_date_dt
        chunks.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)

    # --- Post-process each chunk for metrics, CSVs, PNGs, logs ---
    all_portfolio_metrics = []
    all_benchmark_metrics = []

    for idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_portfolio = portfolio_series.loc[chunk_start:chunk_end]
        chunk_benchmark = benchmark_series.loc[chunk_start:chunk_end]
        chunk_weights = weights_df.loc[chunk_start:chunk_end]

        # Save chunk weights CSV
        chunk_weights.to_csv(weights_csv_path.replace(".csv", f"_chunk{idx + 1}.csv"))

        # Calculate performance metrics for chunk
        portfolio_metrics = calculate_performance_metrics(chunk_portfolio)
        benchmark_metrics = calculate_performance_metrics(chunk_benchmark)

        all_portfolio_metrics.append(portfolio_metrics)
        all_benchmark_metrics.append(benchmark_metrics)

        # Log performance summary
        print(f"\n[Backtest] === Performance Summary (Chunk {idx + 1}) ===")
        for metric_name in portfolio_metrics:
            print(f"  {metric_name.replace('_', ' ').title()}:")
            print(f"    Strategy: {portfolio_metrics[metric_name]:.2%}")
            print(f"    Benchmark: {benchmark_metrics[metric_name]:.2%}")

        # Plot chunk equity curves
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(chunk_portfolio.index, chunk_portfolio.values, label="Strategy Equity Curve")
            plt.plot(chunk_benchmark.index, chunk_benchmark.values, label="Benchmark Equity Curve")
            plt.title(f"Equity Curve - Test Chunk {idx + 1} ({chunk_start.date()} to {chunk_end.date()})")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"equity_curve_chunk_{idx + 1}.png", dpi=300)
            plt.close()

    # --- Calculate and log combined metrics over full period ---
    combined_portfolio_metrics = calculate_performance_metrics(portfolio_series)
    combined_benchmark_metrics = calculate_performance_metrics(benchmark_series)

    print(f"\n[Backtest] === Combined Performance Over Full Period ===")
    for key in combined_portfolio_metrics:
        print(f"{key.replace('_', ' ').title()}:")
        print(f"    Strategy: {combined_portfolio_metrics[key]:.2%}")
        print(f"    Benchmark: {combined_benchmark_metrics[key]:.2%}")

    # Calculate performance variance (std across chunks)
    metrics_df = pd.DataFrame(all_portfolio_metrics)
    metrics_std = metrics_df.std()

    print(f"\nPerformance Variance (Std Across Chunks):")
    if isinstance(metrics_std, pd.Series):
        for key, val in metrics_std.items():
            print(f"  {key.replace('_', ' ').title()}: ±{val:.2%}")
    else:
        print(f"  Performance Variance: ±{metrics_std:.2%}")

    # Plot combined equity curve
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_series.index, portfolio_series.values, label="Combined Strategy Equity Curve", linewidth=2)
        plt.plot(benchmark_series.index, benchmark_series.values, label="Combined Benchmark Equity Curve", linewidth=2)
        plt.title("Combined Equity Curve Over Full Test Period")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("combined_equity_curve.png", dpi=300)
        plt.close()

    return {
        'portfolio': combined_portfolio_metrics,
        'benchmark': combined_benchmark_metrics,
        'combined_equity_curve': portfolio_series,
        'combined_benchmark_equity_curve': benchmark_series,
        'performance_variance': metrics_std.to_dict()
    }
