import torch, copy, random
import pandas as pd
import numpy as np
from feat import load_prices
from model import train
from load import load_config
from perf import calc_perf_metrics  
from minitune import minitune_for_chunk
np.seterr(all='raise')

SEED = load_config()["SEED"]
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def run_training_chunks(chunks, lback, norm_feat, TICK, comp_feat, macro_keys, config, start, device, initial_capital):
    pfo_values = [initial_capital];bench_values = [initial_capital]; daily_weight = []; all_pfo_metrics = []; all_bench_metrics = []
    
    for idx, (chunk_start, chunk_end) in enumerate(chunks):
        train_end = chunk_start - pd.Timedelta(days=1)
        train_start = max(train_end - (chunks[0][0] - pd.to_datetime(start)), pd.to_datetime(start))
        training_days = (train_end - train_start).days
        if training_days < (chunks[0][0] - pd.to_datetime(start)).days - 30: print(f"Skipping chunk {idx+1} due to short training window: {training_days} days"); continue
        chunk_config = copy.deepcopy(config)
        #chunk_config = minitune_for_chunk(chunk_start)
        chunk_config["START"] = str(train_start.date()); chunk_config["END"] = str(chunk_end.date()); chunk_config["SPLIT"] = str(chunk_start.date())

        cached_chunk_data = load_prices(chunk_config["START"], chunk_config["END"], macro_keys)
        feat_train, ret_train = comp_feat(TICK, config["FEAT"], cached_chunk_data, macro_keys, split_date=chunk_config["SPLIT"], method=["rf"])
        #current_model, scaler = train(chunk_config, feat_train, ret_train)
        current_model = train(chunk_config, feat_train, ret_train)
        current_model.eval()       
        start_idx = feat_train.index.get_indexer([chunk_start], method='bfill')[0]
        end_idx = feat_train.index.get_indexer([chunk_end], method='ffill')[0]
        assets = ret_train.columns
        
        print(f"[Train] Starting Chunk {idx+1} | Period: {chunk_start.date()} to {chunk_end.date()} ===")
        print(f"[Train] Chunk {idx+1}: Training from {train_start.date()} to {train_end.date()} ({training_days} days)")
        print(f"[Train] Feature train shape: {feat_train.shape}, Return train shape: {ret_train.shape}")
        chunk_pfo_values = [pfo_values[-1]]  ; chunk_bench_values = [bench_values[-1]]

        for i in range(start_idx - lback, end_idx - lback + 1):
            current_date = ret_train.index[i + lback]
            if current_date < chunk_start or current_date > chunk_end: continue
            normalized_feat = norm_feat(feat_train.iloc[i:i + lback].values.astype(np.float32))
            #normalized_feat = scaler.transform(feat_train.iloc[i:i + lback].values.astype(np.float32)) 
            input_tensor = torch.tensor(normalized_feat).unsqueeze(0).to(device)
            with torch.no_grad(): weight = current_model(input_tensor).cpu().numpy().flatten()
            pfo_ret = np.dot(weight, ret_train.loc[current_date].values)
            bench_ret = np.mean(ret_train.loc[current_date].values)
            pfo_values.append(pfo_values[-1] * (1 + pfo_ret))
            bench_values.append(bench_values[-1] * (1 + bench_ret))
            chunk_pfo_values.append(chunk_pfo_values[-1] * (1 + pfo_ret))
            chunk_bench_values.append(chunk_bench_values[-1] * (1 + bench_ret))
            daily_weight.append(pd.Series(weight, index=assets, name=current_date))

        chunk_pfo_series = pd.Series(chunk_pfo_values[1:], index=ret_train.loc[chunk_start:chunk_end].index)
        chunk_bench_series = pd.Series(chunk_bench_values[1:], index=ret_train.loc[chunk_start:chunk_end].index)
        pfo_metrics = calc_perf_metrics(chunk_pfo_series)
        bench_metrics = calc_perf_metrics(chunk_bench_series)
        all_pfo_metrics.append(pfo_metrics)
        all_bench_metrics.append(bench_metrics)
        print(f"[Train] Chunk {idx+1}: Performance Metrics: {pfo_metrics} Bench: {bench_metrics}\n")
        #if pfo_metrics["max_down"] < - 0.4 and (idx + 1) < 4 : print("KILLRUN - pfo sharpe below threshold.")

    avg_outperf = {}
    if all_pfo_metrics and all_bench_metrics:
        metrics_keys = all_pfo_metrics[0].keys()
        for key in metrics_keys:
            port_vals = [m[key] for m in all_pfo_metrics]
            bench_vals = [m[key] for m in all_bench_metrics]
            diffs = [p - b for p, b in zip(port_vals, bench_vals)]
            avg_outperf[key] = np.mean(diffs)
            
    return pfo_values, bench_values, daily_weight, all_pfo_metrics, all_bench_metrics, avg_outperf
