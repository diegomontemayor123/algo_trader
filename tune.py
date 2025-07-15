import os
import subprocess
import re
import optuna
import json
from optuna.samplers import TPESampler
from itertools import combinations, product

# === Settings ===
macro_options = [
    "FEDFUNDS", "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^N225",
    "CL=F", "GC=F", "SI=F", "NG=F", "ZW=F",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "^TNX", "^FVX", "^IRX", "^TYX", "TLT", "IEF", "GLD", "USO", "UUP"
]
feature_options = [
    "ret", "price", "log_ret", "rolling_ret", "volume", "vol",
    "sma", "rsi", "macd", "momentum", "ema", "boll", "williams", "cmo"
]


def get_powerset(options, max_len=2):
    return [",".join(combo) for r in range(1, max_len+1) for combo in combinations(options, r)]

macro_combos = get_powerset(macro_options, max_len=10)
feature_combos = get_powerset(feature_options, max_len=5)
macro_feature_pairs = list(product(macro_combos, feature_combos))

def run_experiment(trial, macro, features):
    config = {
        "START_DATE": "2012-01-01",
        "END_DATE": "2025-07-01",
        "SPLIT_DATE": "2023-07-01",
        "TICKERS": "AAPL,MSFT,GOOGL,NVDA,JPM,WMT,CVX,MCD,T,NKE",
        "MACRO": macro,
        "FEATURES": features,
        "INITIAL_CAPITAL": trial.suggest_float("INITIAL_CAPITAL", 100, 100),
        "MAX_LEVERAGE": trial.suggest_float("MAX_LEVERAGE", 1.3, 1.3),
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 53, 53),
        "LOOKBACK": trial.suggest_int("LOOKBACK", 75, 75),
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 4, 4),
        "WARMUP_FRAC": 0.17,
        "DROPOUT": trial.suggest_float("DROPOUT", 0.0, 0.07),
        "DECAY": trial.suggest_float("DECAY", 0.04, 0.04),
        "FEATURE_ATTENTION_ENABLED": 1,
        "L1_PENALTY": trial.suggest_float("L1_PENALTY", 0.0006,0.0006),
        "L2_PENALTY": 0,
        "INIT_LR": trial.suggest_float("INIT_LR", 0.5,0.5),
        "LOSS_MIN_MEAN": trial.suggest_float("LOSS_MIN_MEAN", 0.01, 0.01),
        "LOSS_RETURN_PENALTY": trial.suggest_float("LOSS_RETURN_PENALTY", 0.02, 0.02),
        "TEST_CHUNK_MONTHS": 24,
        "RETRAIN_WINDOW": 0,
        "EPOCHS": 20,
        "MAX_HEADS": 20,
        "LAYER_COUNT": 6,
        "EARLY_STOP_PATIENCE": 5,
        "VAL_SPLIT": 0.16,
    }

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

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        def extract_avg_benchmark_outperformance(output):
            match = re.findall(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            if match:
                return float(match[-1]) / 100.0
            multi = re.search(r"Average Benchmark Outperformance Across Chunks:\s*\ncagr:\s*([-+]?\d*\.\d+|\d+)%", output)
            return float(multi.group(1)) / 100.0 if multi else 0.0

        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        avg_benchmark_outperformance = extract_avg_benchmark_outperformance(output)

        if sharpe is None or drawdown is None:
            return -float("inf")

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)

        score = (1 * sharpe) + (1 * avg_benchmark_outperformance) - (0.3 * abs(drawdown))
        return score

    except subprocess.TimeoutExpired:
        return -float("inf")

def main():
    best_overall = {
        "score": -float("inf"),
        "macro": None,
        "features": None,
        "params": None,
        "attrs": {},
    }

    for idx, (macro, features) in enumerate(macro_feature_pairs):
        print(f"\n=== [{idx+1}/{len(macro_feature_pairs)}] Testing MACRO: {macro} | FEATURES: {features} ===")
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def wrapped(trial):
            return run_experiment(trial, macro, features)

        study.optimize(wrapped, n_trials=5, n_jobs=1)

        best_trial = study.best_trial
        score = best_trial.value

        if score > best_overall["score"]:
            best_overall.update({
                "score": score,
                "macro": macro,
                "features": features,
                "params": best_trial.params.copy(),
                "attrs": best_trial.user_attrs.copy()
            })

    print("\n\n=== ✅ Best Macro + Feature Combo ===")
    print(f"MACRO: {best_overall['macro']}")
    print(f"FEATURES: {best_overall['features']}")
    print("Score:", round(best_overall["score"], 4))
    print("Hyperparams:")
    for k, v in best_overall["params"].items():
        print(f"  {k}: {v}")
    for m in ["sharpe", "drawdown", "avg_benchmark_outperformance"]:
        print(f"{m}: {best_overall['attrs'].get(m, float('nan')):.4f}")

    with open("hyparams.json", "w") as f:
        json.dump({
            "macro": best_overall["macro"],
            "features": best_overall["features"],
            "params": best_overall["params"],
            "metrics": best_overall["attrs"],
            "score": best_overall["score"]
        }, f, indent=4)

if __name__ == "__main__":
    main()
