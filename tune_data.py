import os
import subprocess
import re
import json
import optuna
from optuna.samplers import TPESampler
from collections import Counter

TICKER_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "XOM", "PFE", "KO", "WMT", "BA", "PG", "IBM", "CAT",
    "CVX", "MCD", "GE", "VZ", "T", "UNH", "NKE", "ORCL"
]

MACRO_LIST = [
    "FEDFUNDS", "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^N225",
    "CL=F", "GC=F", "SI=F", "NG=F", "ZW=F",
    "EURUSD=X", "JPY=X", "GBPUSD=X", "USDJPY=X",
    "^TNX", "^FVX", "^IRX", "^TYX", "TLT", "IEF", "GLD", "USO", "UUP"
]

FEATURE_LIST = [
    "ret", "price", "log_ret", "rolling_ret", "volume", "vol",
    "sma", "rsi", "macd", "momentum", "ema", "boll", "williams", "cmo"
]

def load_fixed_params(filepath="hyparams.json"):
    with open(filepath, "r") as f:
        params = json.load(f)
    return params

def binary_select(trial, items, prefix):
    return [item for item in items if trial.suggest_categorical(f"{prefix}_{item}", [False, True])]

def run_experiment(trial):
    selected_macros = binary_select(trial, MACRO_LIST, "macro")
    selected_features = binary_select(trial, FEATURE_LIST, "feature")
    selected_tickers = binary_select(trial, TICKER_LIST, "ticker")
    if len(selected_features) == 0 or len(selected_macros) == 0 or len(selected_tickers) == 0:
        return -float("inf")  
    fixed_params = load_fixed_params()
    config = fixed_params.copy()
    config.update({
        "TICKERS": ",".join(selected_tickers),
        "MACRO": ",".join(selected_macros),
        "FEATURES": ",".join(selected_features),
    })
    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    try:
        result = subprocess.run(
            ["python", "model.py"],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800
        )
        output = result.stdout + result.stderr
        with open("tune_output.log", "a") as f:
            f.write("\n\n=== Trial output start ===\n")
            f.write(output)
            f.write("\n=== Trial output end ===\n")
        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None
        def extract_avg_benchmark_outperformance(output):
            matches = re.findall(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            for val in reversed(matches):
                try:
                    return float(val) / 100.0
                except:
                    continue
            return 0.0
        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        avg_benchmark_outperformance = extract_avg_benchmark_outperformance(output)
        if sharpe is None or drawdown is None:
            return -float("inf")
        score = (1 * sharpe) + (0 * avg_benchmark_outperformance) - (0.7 * abs(drawdown))
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("selected_macro", selected_macros)
        trial.set_user_attr("selected_tickers", selected_tickers)
        return score
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")

def main():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=50, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best Trial Metrics ===")
    for m in ["sharpe", "drawdown", "avg_benchmark_outperformance"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")
    print(f"Selected features: {best.user_attrs.get('selected_features')}")
    print(f"Selected macros: {best.user_attrs.get('selected_macro')}")
    print(f"Selected tickers: {best.user_attrs.get('selected_tickers')}")
    macro_counter = Counter()
    feature_counter = Counter()
    ticker_counter = Counter()
    for trial in study.trials:
        if trial.value is None or trial.value == -float("inf"):
            continue
        for macro in MACRO_LIST:
            if trial.params.get(f"macro_{macro}"):
                macro_counter[macro] += 1
        for feature in FEATURE_LIST:
            if trial.params.get(f"feature_{feature}"):
                feature_counter[feature] += 1
        for ticker in TICKER_LIST:
            if trial.params.get(f"ticker_{ticker}"):
                ticker_counter[ticker] += 1
    print("\nMacro inclusion frequency:")
    for macro, count in macro_counter.most_common():
        print(f"{macro}: {count}")
    print("\nFeature inclusion frequency:")
    for feature, count in feature_counter.most_common():
        print(f"{feature}: {count}")
    print("\nTicker inclusion frequency:")
    for ticker, count in ticker_counter.most_common():
        print(f"{ticker}: {count}")
    print("\nParam importances:")
    importance = optuna.importance.get_param_importances(study)
    for k, v in importance.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
