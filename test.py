import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from perf import calc_perf_metrics
from train import run_training_chunks
np.seterr(all='raise')

def save_to_csv(var,csv_path):
    x = pd.DataFrame(var)
    pct_change = x.pct_change().abs().sum(axis=1) * 100
    x["total"] = pct_change
    x.index.name = "Date"
    try:x.to_csv(csv_path);return x
    except Exception as e:print(f"Error saving {var} to {csv_path}: {e}")

def run_btest(device, initial_capital, split, lback,comp_feat, norm_feat, TICK, start, end,feat, macro_keys,
                test_chunk, plot=False,weight_csv_path="csv/weight.csv", config=None):
    split_dt = pd.to_datetime(split);   end_dt = pd.to_datetime(end)
    chunks = [];    curr_start = split_dt
    while curr_start < end_dt:
        curr_end = min(curr_start + relativedelta(months=test_chunk) - pd.Timedelta(days=1),end_dt)
        chunks.append((curr_start, curr_end))
        curr_start = curr_end + pd.Timedelta(days=1)
    if len(chunks) >= 2:
        fin_start, final_end = chunks[-1]
        dur_months = (final_end.year - fin_start.year) * 12 + (final_end.month - fin_start.month)
        if dur_months < test_chunk -1 :
            print(f"[Test] Merging short final chunk ({fin_start.date()} to {final_end.date()}) into previous.")
            prev_start, _ = chunks[-2]
            chunks[-2] = (prev_start, final_end)
            chunks.pop()
    pfo_values = [initial_capital];bench_values = [initial_capital];daily_weight = [];all_pfo_metrics = [];all_bench_metrics = [];avg_outperf = {}
    pfo_values, bench_values, daily_weight, all_pfo_metrics, all_bench_metrics, avg_outperf = run_training_chunks(chunks,  lback, norm_feat, TICK,feat, comp_feat, macro_keys, config, start, device, initial_capital)
    save_to_csv(daily_weight, weight_csv_path);weight_df = pd.read_csv(weight_csv_path, index_col="Date", parse_dates=True)
    pfo_series = pd.Series(pfo_values[1:], index=weight_df.index);  bench_series = pd.Series(bench_values[1:], index=weight_df.index)
    comb_pfo_metrics = calc_perf_metrics(pfo_series) ;  comb_bench_metrics = calc_perf_metrics(bench_series)
    report_path = "img/test.txt"
    with open(report_path, "w") as f:
        f.write("=Combined Perf Over Full Per=\n")
        for key in comb_pfo_metrics:f.write(f"{key.title()}: Strat {comb_pfo_metrics[key]:.2%}, Bench {comb_bench_metrics[key]:.2%}\n")
        f.write("\n=Per-Chunk Metrics=\n")
        for i, (pm, bm) in enumerate(zip(all_pfo_metrics, all_bench_metrics)):
            chunk_start, chunk_end = chunks[i]
            f.write(f"\n-Chunk {i+1} ({chunk_start.date()} to {chunk_end.date()})-\n")
            for key in pm:f.write(f"{key.title()}: Strat {pm[key]:.2%}, Bench {bm[key]:.2%}\n")
        f.write("\n=SD - Metrics Across Chunks=\n")
        metrics_df = pd.DataFrame(all_pfo_metrics) ; metrics_std = metrics_df.std()
        for key, val in metrics_std.items():f.write(f"{key.title()}: ±{val:.2%}\n")
        f.write("\n=Average Outperf Across Chunks (Strat - Bench)=\n")
        for key, val in avg_outperf.items():f.write(f"{key.title()}: {val:.2%}\n")
    if plot:
        plt.figure(figsize=(12, 6));plt.plot(pfo_series.index, pfo_series.values, label="Combined Strat Equity Curve", linewidth=2)
        plt.plot(bench_series.index, bench_series.values, label="Combined Bench Equity Curve", linewidth=2)
        plt.title("Combined Equity Curve Over Full Test Per");plt.xlabel("Date");plt.ylabel("Pfo Value ($)");plt.legend()
        plt.grid(True, alpha=0.3);plt.tight_layout();plt.savefig("img/eq_curve.png", dpi=300);plt.close()
        plt.figure(figsize=(14, 8));weight_df.drop(columns=["total"]).plot(lw=1)
        plt.title("Daily Pfo weight");plt.xlabel("Date");plt.ylabel("Weight");plt.grid(alpha=0.3)
        plt.legend(ncol=2, fontsize="small");plt.tight_layout();plt.savefig("img/weight_plot.png", dpi=300);plt.close()

    return {'pfo': comb_pfo_metrics,'bench': comb_bench_metrics,'eq_curve': pfo_series,'comb_bench_eq_curve': bench_series,'perf_outperf': avg_outperf,'cagr': comb_pfo_metrics.get('cagr', float('nan'))}