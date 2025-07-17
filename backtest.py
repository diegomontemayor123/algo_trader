import numpy as np
import torch, logging, multiprocessing, os
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from data_prep import prepare_main_datasets
from torch.utils.data import DataLoader
from model import create_model
from train import train_model_with_validation
from copy import deepcopy
from compute_features import load_price_data
from calc_perform import calculate_performance_metrics

def run_backtest(device, initial_capital, split_date, lookback, max_leverage,
                 compute_features, normalize_features, tickers, start_date, end_date,
                 features, macro_keys, test_chunk_months, retrain_window, model=None, plot=False,
                 weights_csv_path="weights.csv", config=None):
    if retrain_window > 0 and config is None:
        raise ValueError("Config needed when retrain_window>0")
    logging.info(f"[Backtest] Starting w retrain_window={retrain_window}")
    split_date_dt = pd.to_datetime(split_date)
    end_date_dt = pd.to_datetime(end_date)
    cached_data = load_price_data(start_date, end_date, macro_keys)
    features_df, returns_df = compute_features(tickers, features, cached_data, macro_keys)
    features_df.index = pd.to_datetime(features_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)
    chunks = []
    current_start = split_date_dt
    while current_start < end_date_dt:
        current_end = current_start + relativedelta(months=test_chunk_months) - pd.Timedelta(days=1)
        if current_end > end_date_dt:
            current_end = end_date_dt
        chunks.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
    if len(chunks) >= 2:
        final_start, final_end = chunks[-1]
        duration_months = (final_end.year - final_start.year) * 12 + (final_end.month - final_start.month)
        if duration_months < test_chunk_months:
            logging.info(f"[Chunk Merge] Merging short final chunk ({final_start.date()} to {final_end.date()}) into previous.")
            prev_start, _ = chunks[-2]
            chunks[-2] = (prev_start, final_end)
            chunks.pop()
    portfolio_values = [initial_capital]
    benchmark_values = [initial_capital]
    daily_weights = []
    all_portfolio_metrics = []
    all_benchmark_metrics = []
    avg_outperformance = {}

    if retrain_window < 1:
        logging.info("[Backtest] Running w/o retraining.")
        model.eval()
        try:
            start_index = features_df.index.get_indexer([split_date_dt], method='bfill')[0]
        except Exception as e:
            logging.error(f"[Backtest] Error finding start index for backtest: {e}")
            return None
        asset_names = returns_df.columns
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
            logging.debug(f"[Backtest] Date {current_date.date()} - Raw weights sample: {raw_weights[:5]}")
            weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
            scaling_factor = min(max_leverage / weight_sum, 1.0)
            final_weights = raw_weights * scaling_factor
            logging.debug(f"[Backtest] Date {current_date.date()} - Final weights sample: {final_weights[:5]} (sum abs: {np.sum(np.abs(final_weights)):.4f})")
            period_returns = returns_df.loc[current_date].values
            portfolio_return = np.dot(final_weights, period_returns)
            benchmark_return = np.mean(period_returns)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
            daily_weights.append(pd.Series(final_weights, index=asset_names, name=current_date))
        logging.info(f"[Backtest] Number of daily weight entries: {len(daily_weights)}")
        if len(daily_weights) == 0:
            logging.warning("[Backtest] No daily weights collected!")
        else:
            logging.debug(f"[Backtest] Sample daily weight for date {daily_weights[0].name}: {daily_weights[0].head()}")
        weights_df = pd.DataFrame(daily_weights)
        weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
        weights_df.index.name = "Date"
        logging.info(f"[Backtest] Saving combined weights DataFrame to CSV at: {os.path.abspath(weights_csv_path)}")
        try:
            weights_df.to_csv(weights_csv_path)
            logging.info(f"[Backtest] Successfully saved combined weights to {weights_csv_path}")
        except Exception as e:
            logging.error(f"[Backtest] Error saving daily weights to {weights_csv_path}: {e}")
        portfolio_series = pd.Series(portfolio_values[1:], index=weights_df.index)
        benchmark_series = pd.Series(benchmark_values[1:], index=weights_df.index)
        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_portfolio = portfolio_series.loc[chunk_start:chunk_end]
            chunk_benchmark = benchmark_series.loc[chunk_start:chunk_end]
            if len(chunk_portfolio) < 2:
                continue
            portfolio_metrics = calculate_performance_metrics(chunk_portfolio)
            benchmark_metrics = calculate_performance_metrics(chunk_benchmark)
            all_portfolio_metrics.append(portfolio_metrics)
            all_benchmark_metrics.append(benchmark_metrics)
            logging.info(f"[Backtest] Chunk {idx+1}: Portfolio Metrics: {portfolio_metrics}")
            logging.info(f"[Backtest] Chunk {idx+1}: Benchmark Metrics: {benchmark_metrics}")
        if all_portfolio_metrics and all_benchmark_metrics:
            metrics_keys = all_portfolio_metrics[0].keys()
            for key in metrics_keys:
                port_vals = [m[key] for m in all_portfolio_metrics]
                bench_vals = [m[key] for m in all_benchmark_metrics]
                diffs = [p - b for p, b in zip(port_vals, bench_vals)]
                avg_outperformance[key] = np.mean(diffs)
    else:
        logging.info(f"[Backtest] Running w test_chunk_months {test_chunk_months} and retrain_window {retrain_window}")
        previous_model = None
        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            logging.info(f"[Backtest] Starting Chunk {idx+1} | Period: {chunk_start.date()} to {chunk_end.date()} ===")
            train_start_date = chunk_start - relativedelta(months=retrain_window)
            train_end_date = chunk_start - pd.Timedelta(days=1)
            train_start_date = max(train_start_date, pd.to_datetime(start_date))
            train_end_date = max(train_end_date, train_start_date)
            training_days = (train_end_date - train_start_date).days
            if training_days < ((chunk_start - (chunk_start - relativedelta(months=retrain_window))).days) - 30:
                logging.warning(f"[Backtest] Skipping chunk {idx+1} due to insufficient training data: {training_days} days")
                continue
            logging.info(f"[Backtest] Chunk {idx+1}: Training from {train_start_date.date()} to {train_end_date.date()}")
            chunk_config = config.copy()
            chunk_config["START_DATE"] = str(train_start_date.date())
            chunk_config["END_DATE"] = str(train_end_date.date())
            chunk_config["SPLIT_DATE"] = str(chunk_start.date())
            features_train, returns_train = compute_features(tickers, chunk_config["START_DATE"], chunk_config["END_DATE"], features)
            train_dataset, val_dataset, _ = prepare_main_datasets(features_train, returns_train, chunk_config)
            num_workers = min(2, multiprocessing.cpu_count())
            train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=num_workers)
            model_dim = train_dataset[0][0].shape[1]
            model = deepcopy(previous_model) if previous_model else create_model(model_dim, config)
            model = train_model_with_validation(model, train_loader, val_loader, config)
            if model is None:
                logging.warning(f"[Backtest] Skipping chunk {idx+1} due to failed training (NaNs or early exit).")
                continue
            model.eval()
            previous_model = deepcopy(model)
            try:
                start_idx = features_df.index.get_indexer([chunk_start], method='bfill')[0]
                end_idx = features_df.index.get_indexer([chunk_end], method='ffill')[0]
            except Exception as e:
                logging.error(f"[Backtest] Error getting index for chunk {idx+1}: {e}")
                continue
            asset_names = returns_df.columns
            for i in range(start_idx - lookback, end_idx - lookback + 1):
                current_date = returns_df.index[i + lookback]
                if current_date < chunk_start or current_date > chunk_end:
                    continue
                feature_window = features_df.iloc[i:i + lookback].values.astype(np.float32)
                normalized_features = normalize_features(feature_window)
                input_tensor = torch.tensor(normalized_features).unsqueeze(0).to(device)
                with torch.no_grad():
                    raw_weights = model(input_tensor).cpu().numpy().flatten()
                logging.debug(f"[Backtest] Chunk {idx+1} Date {current_date.date()} - Raw weights sample: {raw_weights[:5]}")
                weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
                scaling_factor = min(max_leverage / weight_sum, 1.0)
                final_weights = raw_weights * scaling_factor
                logging.debug(f"[Backtest] Chunk {idx+1} Date {current_date.date()} - Final weights sample: {final_weights[:5]} (sum abs: {np.sum(np.abs(final_weights)):.4f})")
                period_returns = returns_df.loc[current_date].values
                portfolio_return = np.dot(final_weights, period_returns)
                benchmark_return = np.mean(period_returns)
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
                daily_weights.append(pd.Series(final_weights, index=asset_names, name=current_date))
            logging.info(f"[Backtest] Chunk {idx+1}: Number of daily weight entries so far: {len(daily_weights)}")
            logging.info(f"[Backtest] Chunk {idx+1}: Completed inference.")
        logging.info(f"[Backtest] Total daily weight entries collected: {len(daily_weights)}")
        if len(daily_weights) == 0:
            logging.warning("[Backtest] No daily weights collected!")
        else:
            logging.debug(f"[Backtest] Sample daily weight for date {daily_weights[0].name}: {daily_weights[0].head()}")

        weights_df = pd.DataFrame(daily_weights)
        weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
        weights_df.index.name = "Date"

        try:
            weights_df.to_csv(weights_csv_path)
            logging.info(f"[Backtest] Successfully saved combined weights to {weights_csv_path}")
        except Exception as e:
            logging.error(f"[Backtest] Error saving daily weights to {weights_csv_path}: {e}")

        portfolio_series = pd.Series(portfolio_values[1:], index=weights_df.index)
        benchmark_series = pd.Series(benchmark_values[1:], index=weights_df.index)

        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_portfolio = portfolio_series.loc[chunk_start:chunk_end]
            chunk_benchmark = benchmark_series.loc[chunk_start:chunk_end]
            if len(chunk_portfolio) < 2:
                continue
            portfolio_metrics = calculate_performance_metrics(chunk_portfolio)
            benchmark_metrics = calculate_performance_metrics(chunk_benchmark)
            all_portfolio_metrics.append(portfolio_metrics)
            all_benchmark_metrics.append(benchmark_metrics)
            logging.info(f"[Backtest] Chunk {idx+1}: Portfolio Metrics: {portfolio_metrics}")
            logging.info(f"[Backtest] Chunk {idx+1}: Benchmark Metrics: {benchmark_metrics}")

        if all_portfolio_metrics and all_benchmark_metrics:
            metrics_keys = all_portfolio_metrics[0].keys()
            for key in metrics_keys:
                port_vals = [m[key] for m in all_portfolio_metrics]
                bench_vals = [m[key] for m in all_benchmark_metrics]
                diffs = [p - b for p, b in zip(port_vals, bench_vals)]
                avg_outperformance[key] = np.mean(diffs)
    combined_portfolio_metrics = calculate_performance_metrics(portfolio_series)
    combined_benchmark_metrics = calculate_performance_metrics(benchmark_series)
    report_path = "img/backtest_report.txt"
    with open(report_path, "w") as f:
        f.write("=Combined Perform Over Full Period=\n")
        for key in combined_portfolio_metrics:
            f.write(f"{key.title()}: Strategy {combined_portfolio_metrics[key]:.2%}, Benchmark {combined_benchmark_metrics[key]:.2%}\n")
        f.write("\n=Per-Chunk Metrics=\n")
        for i, (pm, bm) in enumerate(zip(all_portfolio_metrics, all_benchmark_metrics)):
            chunk_start, chunk_end = chunks[i]
            f.write(f"-Chunk {i+1} ({chunk_start.date()} to {chunk_end.date()})-\n")
            for key in pm:
                f.write(f"{key.title()}: Strategy {pm[key]:.2%}, Benchmark {bm[key]:.2%}\n")
        f.write("\n=Std Dev of Metrics Across Chunks=\n")
        metrics_df = pd.DataFrame(all_portfolio_metrics)
        metrics_std = metrics_df.std()
        for key, val in metrics_std.items():
            f.write(f"{key.title()}: ±{val:.2%}\n")
        f.write("\n=Average Outperformance Across Chunks (Strategy - Benchmark)=\n")
        for key, val in avg_outperformance.items():
            f.write(f"{key.title()}: {val:.2%}\n")
    logging.info(f"[Backtest] Saved perform report to {report_path}")
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
        plt.savefig("img/combined_equity_curve.png", dpi=300)
        plt.close()
        logging.info("[Backtest] Saved combined equity curve plot.")
        plt.figure(figsize=(14, 8))
        weights_df.drop(columns=["total_exposure"]).plot(lw=1)
        plt.title("Daily Portfolio Weights")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.savefig("/img/daily_weights_plot.png", dpi=300)
        plt.close()
        logging.info("[Backtest] Saved daily_weights_plot.png")
    return {
        'portfolio': combined_portfolio_metrics,
        'benchmark': combined_benchmark_metrics,
        'combined_equity_curve': portfolio_series,
        'combined_benchmark_equity_curve': benchmark_series,
        'performance_outperformance': avg_outperformance,
        'cagr': combined_portfolio_metrics.get('cagr', float('nan'))
    }
