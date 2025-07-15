import os
import subprocess
import re
import json
import optuna
from optuna.samplers import TPESampler
from collections import Counter

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

def binary_select(trial, items, prefix):
    return [item for item in items if trial.suggest_categorical(f"{prefix}_{item}", [False, True])]

def run_experiment(trial):
    selected_macros = binary_select(trial, MACRO_LIST, "macro")
    selected_features = binary_select(trial, FEATURE_LIST, "feature")

    if len(selected_features) == 0 or len(selected_macros) == 0:
        return -float("inf")  # Skip useless trials

    macro_str = ",".join(selected_macros)
    features_str = ",".join(selected_features)

    config = {
        "START_DATE": "2012-01-01",
        "END_DATE": "2025-07-01",
        "SPLIT_DATE": "2023-07-01",
        "TICKERS": "AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,WMT,CVX,MCD,T,NKE",
        "MACRO": macro_str,
        "FEATURES": features_str,
        "INITIAL_CAPITAL": 100.0,
        "MAX_LEVERAGE": 1.3,
        "BATCH_SIZE": 53,
        "LOOKBACK": 75,
        "PREDICT_DAYS": 4,
        "WARMUP_FRAC": 0.17,
        "DROPOUT": 0.0791,
        "DECAY": 0.04,
        "FEATURE_ATTENTION_ENABLED": 1,
        "L1_PENALTY": 0.000667,
        "L2_PENALTY": 0.0,
        "INIT_LR": 0.50,
        "LOSS_MIN_MEAN": 0.016676,
        "LOSS_RETURN_PENALTY": 0.0247,
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

        score = (1 * sharpe) + (1 * avg_benchmark_outperformance) - (0.3 * abs(drawdown))

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("selected_macro", selected_macros)

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

    # --- Frequency tracking ---
    macro_counter = Counter()
    feature_counter = Counter()

    for trial in study.trials:
        if trial.value is None or trial.value == -float("inf"):
            continue
        for macro in MACRO_LIST:
            if trial.params.get(f"macro_{macro}"):
                macro_counter[macro] += 1
        for feature in FEATURE_LIST:
            if trial.params.get(f"feature_{feature}"):
                feature_counter[feature] += 1

    print("\nMacro inclusion frequency (top drivers):")
    for macro, count in macro_counter.most_common():
        print(f"{macro}: {count}")

    print("\nFeature inclusion frequency (top drivers):")
    for feature, count in feature_counter.most_common():
        print(f"{feature}: {count}")

    # --- Optional: Parameter importance ---
    print("\nParam importances:")
    importance = optuna.importance.get_param_importances(study)
    for k, v in importance.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
