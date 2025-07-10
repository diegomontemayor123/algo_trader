import os
import subprocess
import re
import optuna
from optuna.samplers import TPESampler

def run_experiment(trial):
    config = {
        "SPLIT_DATE": "2023-01-01",
        "VAL_SPLIT": 0.2,
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 1, 10),
        "LOOKBACK": trial.suggest_int("LOOKBACK", 50, 120),
        "EPOCHS": 20,
        "MAX_HEADS": 20,
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 40, 100),
        "FEATURES": trial.suggest_categorical("FEATURES", [
            "ret,vol,log_ret,rolling_ret,volume",
            "ret,sma,williams,cmo,momentum",
            "ret,sma,vol,boll,cmo",
            "ret,momentum,macd",
            "ret,cmo,momentum",
            "ret,sma,vol,cmo"
        ]),
        "MAX_LEVERAGE": trial.suggest_float("MAX_LEVERAGE", 1.0, 1.0),  # fixed at 1.0
        "LAYER_COUNT": 6,
        "DROPOUT": trial.suggest_float("DROPOUT", 0.1, 0.5),
        "DECAY": trial.suggest_float("DECAY", 0.005, 0.1),
        "FEATURE_ATTENTION_ENABLED": 1,
        "RETURN_PENALTY_ENABLED": 0,
        "WALKFWD_ENABLED": 0,
        "WALKFWD_STEP": 60,
        "WALKFWD_WNDW": 365,
        "ADVANCED_LOSS": 0  # signals to use MSE instead of Sharpe
    }

    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)

    try:
        result = subprocess.run(
            ["python", "tech.py"],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800
        )
        output = result.stdout + result.stderr

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)

        if sharpe is None or drawdown is None:
            print(f"[Warning] Failed to extract metrics for config: {config}")
            return -float("inf")

        score = sharpe - 0.5 * abs(drawdown)

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)

        return score

    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")


def main():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=30, n_jobs=1)

    print("\n=== Best trial ===")
    best = study.best_trial
    for k, v in best.params.items():
        print(f"{k}: {v}")
    print("\nMetrics:")
    for m in ["sharpe", "drawdown"]:
        print(f"{m}: {best.user_attrs[m]:.4f}")

    print("\nUse this config in your environment:")
    fixed_config = {
        "SPLIT_DATE": "2023-01-01",
        "VAL_SPLIT": 0.2,
        "PREDICT_DAYS": 3,
        "EPOCHS": 20,
        "MAX_HEADS": 20,
        "LAYER_COUNT": 6,
        "FEATURE_ATTENTION_ENABLED": 1,
        "RETURN_PENALTY_ENABLED": 0,
        "WALKFWD_ENABLED": 0,
        "WALKFWD_STEP": 60,
        "WALKFWD_WNDW": 365,
        "ADVANCED_LOSS": 0
    }

    full_config = {**fixed_config, **best.params}
    ordered_keys = [
        "SPLIT_DATE", "VAL_SPLIT", "PREDICT_DAYS", "LOOKBACK",
        "EPOCHS", "MAX_HEADS", "BATCH_SIZE", "FEATURES", "MAX_LEVERAGE",
        "LAYER_COUNT", "DROPOUT", "DECAY",
        "FEATURE_ATTENTION_ENABLED", "RETURN_PENALTY_ENABLED",
        "ADVANCED_LOSS",
        "WALKFWD_ENABLED", "WALKFWD_STEP", "WALKFWD_WNDW"
    ]
    for k in ordered_keys:
        print(f"{k}={full_config[k]}")


if __name__ == "__main__":
    main()
