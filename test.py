import os, subprocess, csv, re, multiprocessing
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

feature_sets = ["ret,vol,log_ret,rolling_ret,volume",
                #"ret,sma,williams,cmo,momentum",
    #"ret,sma,vol,boll,cmo",
    #"ret,momentum,macd",
    #"ret,cmo,momentum",
    #"ret,sma,vol,cmo",
]

grid = {
    "SPLIT_DATE": ["2023-01-01"],   
    "VAL_SPLIT": [0.2], 
    "PREDICT_DAYS": [1], #1,3 
    "LOOKBACK": [80], #70-90
    "EPOCHS": [20], 
    "MAX_HEADS": [20], 
    "BATCH_SIZE": [60], #60-80
    "FEATURES": feature_sets,       
    "MAX_LEVERAGE": [1.0], 
    "LAYER_COUNT": [3],
    "DROPOUT": [0.3], #0.3
    "LEARNING_WARMUP": [460,500,540],#>=300
    "DECAY":[0.0175,0.02,0.0225], #0.01775
    #"LOSS_MIN_MEAN": [0.005],
    #"LOSS_RETURN_PENALTY": [0.4],
    
    "WALKFWD_ENABLED": [0],
    "WALKFWD_STEP": [60],
    "WALKFWD_WNDW": [365],
    } 

param_combos = list(product(*grid.values()))
param_keys = list(grid.keys())

def run_experiment(index, total, params):
    env = os.environ.copy()
    config = dict(zip(param_keys, params))
    for k, v in config.items():
        env[k] = str(v)
    print(f"Running {index + 1} of {total}")
    try:
        result = subprocess.run(
            ["python", "tech.py"],
            capture_output=True,
            text=True,
            env=env,
            timeout=1600
        )
        if result.returncode != 0:
            print("❌ Script failed to run:")
            print(result.stdout)
            print(result.stderr)
        output = result.stdout + result.stderr
        def extract_metric(label, output):
            pattern = rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%"
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100  
            return None
        def extract_weights(output):
            pattern = r"\s*(\w+):\n\s*Min:\s*([-+]?\d*\.\d+)\n\s*Mean:\s*([-+]?\d*\.\d+)\n\s*Max:\s*([-+]?\d*\.\d+)"
            matches = re.findall(pattern, output)
            weights = {}
            if matches:
                ticker, wmin, _, wmax = matches[-1]
                weights[f"{ticker}_w_delta"] = float(wmax) - float(wmin)
            return weights
        sharpe = extract_metric("Sharpe Ratio", output)
        cagr = extract_metric("Cagr", output)
        drawdown = extract_metric("Max Drawdown", output)
        weights = extract_weights(output)
        return {**config, "sharpe": sharpe, "cagr": cagr, "drawdown": drawdown, **weights}

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for {config}")
        return {**config, "sharpe": None, "cagr": None, "drawdown": None}

def ratio(sharpe, drawdown):
    if sharpe is None or drawdown is None or drawdown == 0:
        return -float('inf')
    return sharpe / abs(drawdown)
def main():
    max_workers = min(12, multiprocessing.cpu_count())
    total = len(param_combos)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment, i, total, p) for i, p in enumerate(param_combos)]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"Config: { {k: row[k] for k in param_keys} } | Sharpe: {row['sharpe']}, CAGR: {row['cagr']}, Drawdown: {row['drawdown']}")
    if results:
        varying_keys = [k for k in param_keys if len(set(r[k] for r in results)) > 1]
        metric_keys = ["sharpe", "cagr", "drawdown"]
        extra_keys = [k for k in results[0].keys() if k.endswith("_w_delta")]
        keys_to_write = varying_keys + metric_keys + extra_keys
        with open("results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys_to_write)
            writer.writeheader()
            for row in results:
                filtered_row = {k: row.get(k) for k in keys_to_write}
                writer.writerow(filtered_row)
        print(f"\n✅ Saved {len(results)} results to results.csv with varying hyperparameters only.")

if __name__ == "__main__":
    main()