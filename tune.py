import os, subprocess, re, optuna, json
from optuna.samplers import TPESampler

def run_experiment(trial):
    config = {
        "EARLY_STOP_PATIENCE": trial.suggest_int("EARLY_STOP_PATIENCE", 2, 5),
        "INITIAL_CAPITAL": trial.suggest_float("INITIAL_CAPITAL", 100.0, 100.0),
        "TICKERS": trial.suggest_categorical("TICKERS", ["AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA"]),
        "START_DATE": trial.suggest_categorical("START_DATE", ["2017-01-01" ]),
        "END_DATE": trial.suggest_categorical("END_DATE", [ "2025-05-29"]),
        "SPLIT_DATE": trial.suggest_categorical("SPLIT_DATE", [ "2021-01-01"]),
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.05, 0.2),
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 1, 10),
        "LOOKBACK": trial.suggest_int("LOOKBACK", 40, 120),
        "EPOCHS": trial.suggest_int("EPOCHS", 20, 20),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 20, 20),
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 40, 90),
        "FEATURES": trial.suggest_categorical("FEATURES", ["ret,vol,log_ret,rolling_ret,volume"]),
        "MAX_LEVERAGE": trial.suggest_float("MAX_LEVERAGE", 1.0, 2.0),
        "LAYER_COUNT": trial.suggest_int("LAYER_COUNT", 6, 6),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.1, 0.5),
        "WARMUP_FRAC": trial.suggest_float("WARMUP_FRAC", 0.02, 0.3),
        "DECAY": trial.suggest_float("DECAY", 0.001, 0.1),
        "FEATURE_ATTENTION_ENABLED": trial.suggest_int("FEATURE_ATTENTION_ENABLED", 0, 1),
        "L2_PENALTY_ENABLED": trial.suggest_int("L2_PENALTY_ENABLED", 0, 1),
        "RETURN_PENALTY_ENABLED": trial.suggest_int("RETURN_PENALTY_ENABLED", 1),
        "LOSS_MIN_MEAN": trial.suggest_float("LOSS_MIN_MEAN", 0.0001, 0.1),
        "LOSS_RETURN_PENALTY": trial.suggest_float("LOSS_RETURN_PENALTY", 0.0, 1.0),
        "TEST_CHUNK_MONTHS": trial.suggest_int("TEST_CHUNK_MONTHS", 6, 6),
        "RETRAIN": trial.suggest_categorical("RETRAIN", [False])
    }
    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        output = result.stdout + result.stderr
        print(f"[Subprocess output]\n{output}\n--- End of output ---")
        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        def extract_performance_variance(out):
            pattern = r"Performance Variance \(Std Across Chunks\):\s*((?:\s*\w+:\s*±?[-+]?\d*\.\d+%?\n?)+)"
            var_block = re.search(pattern, out)
            if not var_block:
                return {}
            var_lines = var_block.group(1).strip().split('\n')
            variance_dict = {}
            for line in var_lines:
                parts = line.strip().split(': ±')
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    val = parts[1].strip().replace('%','')
                    try:
                        variance_dict[key] = float(val) / 100.0
                    except:
                        pass
            return variance_dict
        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        variance_metrics = extract_performance_variance(output)
        if sharpe is None or drawdown is None:
            return -float("inf")
        sharpe_var = variance_metrics.get("sharpe_ratio", 0.0)
        score = (1 * sharpe) - (0.5  * abs(drawdown)) - (0.7 * sharpe_var)
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("sharpe_variance", sharpe_var)
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
    with open("hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    for m in ["sharpe", "drawdown", "sharpe_variance"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__":
    main()
