import torch, multiprocessing, copy
import pandas as pd
import numpy as np
from feat import load_prices
from model import create_model
from train import train_model
from torch.utils.data import DataLoader
from perf import calc_perf_metrics  
from prep import prep_data

def reset_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_retraining_chunks(chunks, feat_df, ret_df, lback, norm_feat, TICK, comp_feat, macro_keys, config, start, device, initial_capital, model0=None):
    pfo_values = [initial_capital]
    bench_values = [initial_capital]
    daily_weight = []

    all_pfo_metrics = []
    all_bench_metrics = []
    
    for idx, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"Starting Chunk {idx+1} | Period: {chunk_start.date()} to {chunk_end.date()} ===")
        if idx == 0: current_model = model0
        else:
            orig_train = (chunks[0][0] - pd.to_datetime(start))
            train_end = chunk_start - pd.Timedelta(days=1)
            train_start = max(train_end - orig_train, pd.to_datetime(start))
            training_days = (train_end - train_start).days
            if training_days < orig_train.days - 30:
                print(f"Skipping chunk {idx+1} due to short training window: {training_days} days"); continue
            print(f"Chunk {idx+1}: Training from {train_start.date()} to {train_end.date()} ({training_days} days)")
            chunk_config = copy.deepcopy(config)
            chunk_config["START"] = str(train_start.date())
            chunk_config["END"] = str(train_end.date())
            chunk_config["SPLIT"] = str((train_end + pd.Timedelta(days=1)).date())
            cached_chunk_data = load_prices(chunk_config["START"], config["END"], macro_keys)
            feat_list = config["FEAT"].split(",") if isinstance(config["FEAT"], str) else config["FEAT"]
            feat_train, ret_train = comp_feat(TICK, feat_list, cached_chunk_data, macro_keys, split_date=chunk_config["SPLIT"], method=config["FILTER"])
            print(f"Feature train shape: {feat_train.shape}, Return train shape: {ret_train.shape}")
            train_data, val_data, _ = prep_data(feat_train, ret_train, chunk_config)
            train_loader = DataLoader(train_data, batch_size=config["BATCH"], shuffle=True, num_workers=min(2, multiprocessing.cpu_count()))
            val_loader = DataLoader(val_data, batch_size=config["BATCH"], shuffle=False, num_workers=min(2, multiprocessing.cpu_count()))
            current_model = create_model(train_data[0][0].shape[1], config)
            asset_sd = torch.tensor(ret_train.std(axis=0).values.astype(np.float32), device=device)
            reset_seeds(config["SEED"])
            current_model = train_model(current_model, train_loader, val_loader, config, asset_sd=asset_sd)
            if current_model is None:
                print(f"Skipping chunk {idx+1} due to failure.");continue
        current_model.eval()       
        start_idx = feat_df.index.get_indexer([chunk_start], method='bfill')[0]
        end_idx = feat_df.index.get_indexer([chunk_end], method='ffill')[0]
        assets = ret_df.columns
        chunk_pfo_values = [pfo_values[-1]]  
        chunk_bench_values = [bench_values[-1]]
        for i in range(start_idx - lback, end_idx - lback + 1):
            current_date = ret_df.index[i + lback]
            if current_date < chunk_start or current_date > chunk_end: continue
            feat_win = feat_df.iloc[i:i + lback].values.astype(np.float32)
            normalized_feat = norm_feat(feat_win)
            input_tensor = torch.tensor(normalized_feat).unsqueeze(0).to(device)
            with torch.no_grad(): weight = current_model(input_tensor).cpu().numpy().flatten()
            per_ret = ret_df.loc[current_date].values
            pfo_ret = np.dot(weight, per_ret)
            bench_ret = np.mean(per_ret)
            pfo_values.append(pfo_values[-1] * (1 + pfo_ret))
            bench_values.append(bench_values[-1] * (1 + bench_ret))
            chunk_pfo_values.append(chunk_pfo_values[-1] * (1 + pfo_ret))
            chunk_bench_values.append(chunk_bench_values[-1] * (1 + bench_ret))
            daily_weight.append(pd.Series(weight, index=assets, name=current_date))
        chunk_pfo_series = pd.Series(chunk_pfo_values[1:], index=ret_df.loc[chunk_start:chunk_end].index)
        chunk_bench_series = pd.Series(chunk_bench_values[1:], index=ret_df.loc[chunk_start:chunk_end].index)
        pfo_metrics = calc_perf_metrics(chunk_pfo_series)
        bench_metrics = calc_perf_metrics(chunk_bench_series)
        all_pfo_metrics.append(pfo_metrics)
        all_bench_metrics.append(bench_metrics)
        print(f"[Test] Chunk {idx+1}: Number of daily weights so far: {len(daily_weight)}")
        print(f"[Test] Chunk {idx+1}: Performance Metrics: {pfo_metrics}\n")
    avg_outperf = {}
    if all_pfo_metrics and all_bench_metrics:
        metrics_keys = all_pfo_metrics[0].keys()
        for key in metrics_keys:
            port_vals = [m[key] for m in all_pfo_metrics]
            bench_vals = [m[key] for m in all_bench_metrics]
            diffs = [p - b for p, b in zip(port_vals, bench_vals)]
            avg_outperf[key] = np.mean(diffs)
    return pfo_values, bench_values, daily_weight, all_pfo_metrics, all_bench_metrics, avg_outperf
