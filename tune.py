import os
import subprocess
import re
import optuna
import json
from optuna.samplers import TPESampler

def run_experiment(trial):
    config = {
        "START_DATE": trial.suggest_categorical("START_DATE", ["2012-01-01"]),
        "END_DATE": trial.suggest_categorical("END_DATE", ["2025-07-01"]),
        "SPLIT_DATE": trial.suggest_categorical("SPLIT_DATE", ["2021-07-01"]),
        "TICKERS": trial.suggest_categorical("TICKERS", ["AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,WMT,CVX,MCD,T,NKE"]),
        "MACRO": trial.suggest_categorical("MACRO",["USO,^N225, NG=F, GBPUSD=X,^IXIC,GLD,^RUT,CL=F,TLT,^IRX,^FTSE,^GSPC,^DJI,^FVX,IEF"]),
        "FEATURES": trial.suggest_categorical("FEATURES", ["ret,log_ret,ema,vol,rsi"]),
        "INITIAL_CAPITAL": trial.suggest_float("INITIAL_CAPITAL", 100.0, 100.0),
        "MAX_LEVERAGE": trial.suggest_float("MAX_LEVERAGE", 1.3, 1.3),
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 53, 53),
        "LOOKBACK": trial.suggest_int("LOOKBACK", 75, 75),
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 4, 4),
        "WARMUP_FRAC": trial.suggest_float("WARMUP_FRAC", 0.17, 0.17),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.0791, 0.0791),
        "DECAY": trial.suggest_float("DECAY", 0.04, 0.04),
        "FEATURE_ATTENTION_ENABLED": trial.suggest_int("FEATURE_ATTENTION_ENABLED", 1, 1),
        "FEATURE_PERIODS": trial.suggest_categorical("FEATURE_PERIODS",["5,10,20,40"]),
        "L1_PENALTY": trial.suggest_float("L1_PENALTY", 0.000667, 0.000667), #can be negative now
        "INIT_LR": trial.suggest_float("INIT_LR", 0.5, 0.5),
        "LOSS_MIN_MEAN": trial.suggest_float("LOSS_MIN_MEAN", 0.016676, 0.016676),
        "LOSS_RETURN_PENALTY": trial.suggest_float("LOSS_RETURN_PENALTY", 0.0247, 0.0247),
        "TEST_CHUNK_MONTHS": trial.suggest_int("TEST_CHUNK_MONTHS", 12, 12),
        "RETRAIN_WINDOW": trial.suggest_int("RETRAIN_WINDOW", 0, 0),
        "EPOCHS": trial.suggest_int("EPOCHS", 20, 20),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 20, 20),
        "LAYER_COUNT": trial.suggest_int("LAYER_COUNT", 6, 6),
        "EARLY_STOP_PATIENCE": trial.suggest_int("EARLY_STOP_PATIENCE", 5, 5),
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.16, 0.16),
    }

    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"],capture_output=True,text=True,env=env,timeout=1800)
        output = result.stdout + result.stderr
        with open("tune_output.log", "a") as f:
            f.write("\n\n=== Trial output start ===\n")
            f.write(output)
            f.write("\n=== Trial output end ===\n")
        print(f"[Subprocess output]\n{output}\n")

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        def extract_avg_benchmark_outperformance(output):
            single_line_match = re.findall(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            if single_line_match:
                for val in reversed(single_line_match):
                    try:
                        return float(val) / 100.0
                    except:
                        pass
            multiline_match = re.search(r"Average Benchmark Outperformance Across Chunks:\s*\ncagr:\s*([-+]?\d*\.\d+|\d+)%", output, re.MULTILINE)
            if multiline_match:
                try:
                    return float(multiline_match.group(1)) / 100.0
                except:
                    return 0.0
            return 0.0
        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        avg_benchmark_outperformance = extract_avg_benchmark_outperformance(output)
        if sharpe is None or drawdown is None:
            return -float("inf")
        score = (1 * sharpe) - (0.5 * abs(drawdown)) +(0 * avg_benchmark_outperformance) 
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)
        return score
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")

def main():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=30, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    for m in ["sharpe", "drawdown", "avg_benchmark_outperformance"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__":
    main()

