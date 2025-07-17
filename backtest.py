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

def run_backtest(device, initial_capital, split_date, lookback, max_leverage, compute_features, normalize_features, tickers, start_date, end_date, features, macro_keys, test_chunk_months, retrain_window, model=None, plot=False, weights_csv_path="weights.csv", config=None):
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
    cur_start = split_date_dt
    while cur_start < end_date_dt:
        cur_end = cur_start + relativedelta(months=test_chunk_months) - pd.Timedelta(days=1)
        cur_end = min(cur_end, end_date_dt)
        chunks.append((cur_start, cur_end))
        cur_start = cur_end + pd.Timedelta(days=1)
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
    asset_names = returns_df.columns
    def infer_weight_vector(feature_window_np):
        norm_feats = normalize_features(feature_window_np.astype(np.float32))
        with torch.no_grad():
            raw = model(torch.tensor(norm_feats).unsqueeze(0).to(device)).cpu().numpy().flatten()
        return raw 
    if retrain_window < 1:
        logging.info("[Backtest] Running w/o retraining.")
        model.eval()
        start_idx = features_df.index.get_indexer([split_date_dt], method='bfill')[0]
        for i in range(start_idx - lookback, len(features_df) - lookback):
            cur_date = returns_df.index[i + lookback]
            if not split_date_dt <= cur_date <= end_date_dt:
                continue
            w = infer_weight_vector(features_df.iloc[i:i + lookback].values)
            r = returns_df.loc[cur_date].values
            port_ret = np.dot(w, r)
            bench_ret = np.mean(r)
            portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
            benchmark_values.append(benchmark_values[-1] * (1 + bench_ret))
            daily_weights.append(pd.Series(w, index=asset_names, name=cur_date))
    else:
        logging.info(f"[Backtest] Running w test_chunk_months={test_chunk_months} & retrain_window={retrain_window}")
        prev_model = None
        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            logging.info(f"[Backtest] Chunk {idx+1}: {chunk_start.date()} ➜ {chunk_end.date()}")
            train_start = max(chunk_start - relativedelta(months=retrain_window), pd.to_datetime(start_date))
            train_end = chunk_start - pd.Timedelta(days=1)
            if (train_end - train_start).days < 30:
                logging.warning(f"[Backtest] Skipping chunk {idx+1}: too little training data.")
                continue
            chunk_cfg = {**config, "START_DATE": str(train_start.date()), "END_DATE": str(train_end.date()), "SPLIT_DATE": str(chunk_start.date())}
            feats_tr, rets_tr = compute_features(tickers, chunk_cfg["START_DATE"], chunk_cfg["END_DATE"], features)
            tr_ds, val_ds, _ = prepare_main_datasets(feats_tr, rets_tr, chunk_cfg)
            num_workers = min(2, multiprocessing.cpu_count())
            tr_loader = DataLoader(tr_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=num_workers)
            va_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=num_workers)
            model_dim = tr_ds[0][0].shape[1]
            model = deepcopy(prev_model) if prev_model else create_model(model_dim, config)
            model = train_model_with_validation(model, tr_loader, va_loader, config)
            if model is None:
                logging.warning(f"[Backtest] Chunk {idx+1}: training failed, skipping.")
                continue
            model.eval()
            prev_model = deepcopy(model)
            start_i = features_df.index.get_indexer([chunk_start], method='bfill')[0]
            end_i = features_df.index.get_indexer([chunk_end], method='ffill')[0]
            for i in range(start_i - lookback, end_i - lookback + 1):
                cur_date = returns_df.index[i + lookback]
                if not chunk_start <= cur_date <= chunk_end:
                    continue
                w = infer_weight_vector(features_df.iloc[i:i + lookback].values)
                r = returns_df.loc[cur_date].values
                port_ret = np.dot(w, r)
                bench_ret = np.mean(r)
                portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
                benchmark_values.append(benchmark_values[-1] * (1 + bench_ret))
                daily_weights.append(pd.Series(w, index=asset_names, name=cur_date))
    if len(daily_weights) == 0:
        logging.warning("[Backtest] No daily weights collected!")
        return None
    weights_df = pd.DataFrame(daily_weights)
    weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
    weights_df.index.name = "Date"
    try:
        weights_df.to_csv(weights_csv_path)
        logging.info(f"[Backtest] Saved weights CSV ➜ {os.path.abspath(weights_csv_path)}")
    except Exception as e:
        logging.error(f"[Backtest] Error saving weights CSV: {e}")
    portfolio_series = pd.Series(portfolio_values[1:], index=weights_df.index)
    benchmark_series = pd.Series(benchmark_values[1:], index=weights_df.index)
    for c_idx, (c_start, c_end) in enumerate(chunks):
        port_chunk = portfolio_series.loc[c_start:c_end]
        bench_chunk = benchmark_series.loc[c_start:c_end]
        if len(port_chunk) < 2:
            continue
        pm = calculate_performance_metrics(port_chunk)
        bm = calculate_performance_metrics(bench_chunk)
        all_portfolio_metrics.append(pm)
        all_benchmark_metrics.append(bm)
        logging.info(f"[Backtest] Chunk {c_idx+1}: Portfolio {pm}")
        logging.info(f"[Backtest] Chunk {c_idx+1}: Benchmark {bm}")
    if all_portfolio_metrics:
        for key in all_portfolio_metrics[0]:
            diffs = [pm[key] - bm[key] for pm, bm in zip(all_portfolio_metrics, all_benchmark_metrics)]
            avg_outperformance[key] = np.mean(diffs)
    comb_port_metrics = calculate_performance_metrics(portfolio_series)
    comb_bench_metrics = calculate_performance_metrics(benchmark_series)
    with open("img/backtest_report.txt", "w") as f:
        f.write("=Combined Performance=\n")
        for k in comb_port_metrics:
            f.write(f"{k.title()}: Strategy {comb_port_metrics[k]:.2%}, Benchmark {comb_bench_metrics[k]:.2%}\n")
        f.write("\n=Per‑Chunk Metrics=\n")
        for i, (pm, bm) in enumerate(zip(all_portfolio_metrics, all_benchmark_metrics)):
            cs, ce = chunks[i]
            f.write(f"-Chunk {i+1} ({cs.date()}–{ce.date()})-\n")
            for k in pm:
                f.write(f"{k.title()}: Strategy {pm[k]:.2%}, Benchmark {bm[k]:.2%}\n")
        f.write("\n=Std Dev of Metrics Across Chunks=\n")
        metrics_std = pd.DataFrame(all_portfolio_metrics).std()
        for k, v in metrics_std.items():
            f.write(f"{k.title()}: ±{v:.2%}\n")
        f.write("\n=Avg Outperformance Across Chunks=\n")
        for k, v in avg_outperformance.items():
            f.write(f"{k.title()}: {v:.2%}\n")
    logging.info("[Backtest] Wrote backtest_report.txt")
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_series.index, portfolio_series.values, label="Strategy", linewidth=2)
        plt.plot(benchmark_series.index, benchmark_series.values, label="Benchmark", linewidth=2)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("img/combined_equity_curve.png", dpi=300)
        plt.close()
        logging.info("[Backtest] Saved combined_equity_curve.png")
        plt.figure(figsize=(14, 8))
        weights_df.drop(columns=["total_exposure"]).plot(lw=1)
        plt.title("Daily Portfolio Weights")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.savefig("img/weights_plot.png", dpi=300)
        plt.close()
        logging.info("[Backtest] Saved weights_plot.png")
    return {
        "portfolio": comb_port_metrics,
        "benchmark": comb_bench_metrics,
        "combined_equity_curve": portfolio_series,
        "combined_benchmark_equity_curve": benchmark_series,
        "performance_outperformance": avg_outperformance,
        "cagr": float(comb_port_metrics.get("cagr") or 0.0),
    }
