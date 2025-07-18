import os
import subprocess
import re
import optuna
import json
from optuna.samplers import TPESampler

def run_experiment(trial):
    config = {
        "START_DATE": trial.suggest_categorical("START_DATE", ["2016-01-01"]),
        "END_DATE": trial.suggest_categorical("END_DATE", ["2025-07-01"]),
        "SPLIT_DATE": trial.suggest_categorical("SPLIT_DATE", ["2023-01-01"]),
        "TICKERS": trial.suggest_categorical("TICKERS", ['JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE']),
        "MACRO": trial.suggest_categorical("MACRO",['^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX, IEF, HYG']),
        "FEATURES": trial.suggest_categorical("FEATURES", ['price,vol,macd']),
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 55, 55), #68
        "LOOKBACK": trial.suggest_int("LOOKBACK", 84, 84),#71
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 2,2),#4
        "WARMUP_FRAC": trial.suggest_float("WARMUP_FRAC", 0.15, 0.15), #.12
        "DROPOUT": trial.suggest_float("DROPOUT", 0.001, 0.0315),#.024
        "DECAY": trial.suggest_float("DECAY", 0.00385, 0.00385),#.015
        "FEATURE_ATTENTION_ENABLED": trial.suggest_int("FEATURE_ATTENTION_ENABLED", 1, 1),
        "FEATURE_PERIODS": trial.suggest_categorical("FEATURE_PERIODS",["8,12,24"]),
        "INIT_LR": trial.suggest_float("INIT_LR",0.037,0.037,),     
        "EXPOSURE_PENALTY": trial.suggest_float("EXPOSURE_PENALTY", 0.007,0.007),   
        "RETURN_PENALTY": trial.suggest_float("RETURN_PENALTY", 0.182,0.182),
        "DRAWDOWN_PENALTY": trial.suggest_float("DRAWDOWN_PENALTY", 4.82,4.82),
        "DRAWDOWN_CUTOFF": trial.suggest_float("DRAWDOWN_CUTOFF", 0.279,0.279),
        "TEST_CHUNK_MONTHS": trial.suggest_int("TEST_CHUNK_MONTHS", 12, 12),
        "RETRAIN_WINDOW": trial.suggest_int("RETRAIN_WINDOW", 0, 0),
        "EPOCHS": trial.suggest_int("EPOCHS", 20, 20),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 20, 20),
        "LAYER_COUNT": trial.suggest_int("LAYER_COUNT", 6, 6),
        "EARLY_STOP_PATIENCE": trial.suggest_int("EARLY_STOP_PATIENCE", 6,6),
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.13, 0.13),
    }
    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        output = result.stdout + result.stderr

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        def extract_avg_benchmark_outperformance(output):
            match = re.search(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*\ncagr:\s*([-+]?\d*\.\d+|\d+)%", output, re.MULTILINE)
            if match:
                return float(match.group(1)) / 100.0
            matches = re.findall(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            for val in reversed(matches):
                try:
                    return float(val) / 100.0
                except:
                    pass
            return 0.0

        def extract_exposure_delta(output):
            match = re.search(r"Total Exposure Delta:\s*([-+]?\d*\.\d+|\d+)", output)
            return float(match.group(1)) if match else None

        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        cagr = extract_metric("CAGR", output) 
        avg_benchmark_outperformance = extract_avg_benchmark_outperformance(output)
        exp_delta = extract_exposure_delta(output)

        if sharpe is None or drawdown is None or exp_delta is None:
            return -float("inf")


        score = (1 * sharpe
                - 3 * abs(drawdown)
                + 1 * cagr
                + 0 * avg_benchmark_outperformance)
        if exp_delta<.2:
            score-= 10
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)
        trial.set_user_attr("exp_delta", exp_delta)
        with open("tune_output.log", "a") as f:
            f.write("\n\n=== Trial output start ===\n")
            f.write(f"Trial #{trial.number}\n")
            f.write(f"Sharpe: {sharpe:.4f}\n")
            f.write(f"Drawdown: {drawdown:.4f}\n")
            f.write(f"CAGR: {cagr:.4f}\n")
            f.write(f"Avg Benchmark Outperformance: {avg_benchmark_outperformance:.4f}\n")
            f.write(f"Total Exposure Delta: {exp_delta:.4f}\n")
            f.write(output)
            f.write("\n=== Trial output end ===\n")
        print(
            f"[Trial {trial.number}] Sharpe: {sharpe:.4f}, Drawdown: {drawdown:.4f}, "
            f"CAGR: {cagr:.4f}, Benchmark Outperformance: {avg_benchmark_outperformance:.4f}, "
            f"Exposure Δ: {exp_delta:.4f}"
        )
        return score

    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")

def main():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=10, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    for m in ["sharpe", "drawdown", "CAGR", "avg_benchmark_outperformance", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")


if __name__ == "__main__":
    main()

