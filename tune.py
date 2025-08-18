import os
import subprocess
import re
import optuna
import json
import csv
from optuna.samplers import TPESampler

TRIALS = 400

def run_single_seed_experiment(config, study):
    """Run the experiment subprocess for a single seed and extract metrics."""
    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    dup_value = is_duplicate_trial(study, {k: str(v) for k, v in config.items()})
    if dup_value is not None:
        return dup_value

    try:
        python_exe = os.path.join(env.get("VIRTUAL_ENV", ""), "Scripts", "python.exe") if "VIRTUAL_ENV" in env else "python"

        proc = subprocess.Popen([python_exe, "model.py"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                env=env,
                                bufsize=1)

        output_lines = []
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
            if "KILLRUN" in line:
                print("[KILLRUN] — aborting trial early.")
                proc.kill()
                proc.wait()
                return -float("inf")

        proc.wait()
        output = "".join(output_lines)

        if proc.returncode != 0 or not output.strip():
            print(f"[error] Subprocess failed or empty output: {output}")
            return -float("inf")

        def extract_metric(label, out):
            pattern = rf"{re.escape(label)}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%"
            match = re.search(pattern, out, re.IGNORECASE)
            return float(match.group(1)) / 100 if match else None

        def extract_avgoutperf(output):
            match = re.search(r"Avg Bench Outperf(?: thru Chunks)?:\s*\ncagr:\s*([-+]?\d+(?:\.\d+)?)%", output, re.MULTILINE)
            if match:
                return float(match.group(1)) / 100.0
            matches = re.findall(r"Avg Bench Outperf(?: thru Chunks)?:\s*([-+]?\d+(?:\.\d+)?)%", output)
            for val in reversed(matches):
                try:
                    return float(val) / 100.0
                except:
                    pass
            return 0.0

        def extract_exp_delta(output):
            match = re.search(r"Total Exp Delta:\s*([-+]?\d+(?:\.\d+)?)", output)
            return float(match.group(1)) if match else None

        sharpe = extract_metric("Sharpe Ratio", output)
        down = extract_metric("Max Down", output)
        cagr = extract_metric("CAGR", output)
        avg_outperf = extract_avgoutperf(output)
        exp_delta = extract_exp_delta(output)

        if sharpe is None or down is None or exp_delta is None:
            print("[error] Missing metric(s) — skipping trial.")
            return -float("inf")

        score = 1 * sharpe - 6 * abs(down) + 1 * cagr
        if avg_outperf > 0:
            score += 10
        if exp_delta > 100:
            score += 90

        return score

    except subprocess.TimeoutExpired:
        print(f"[Tune] Trial failed for config: {config}")
        return -float("inf")
    except Exception as e:
        print(f"[Tune] Trial failed with exception: {e}")
        return -float("inf")


def run_experiment(trial, study=None):
    base_config = {
        "START": trial.suggest_categorical("START", ["2012-01-01"]),
        "END": trial.suggest_categorical("END", ["2023-01-01"]),
        "SPLIT": trial.suggest_categorical("SPLIT", ["2017-01-01"]),
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ["^GSPC,CL=F,SI=F,NG=F,HG=F,ZC=F,^IRX,TLT,IEF,UUP,HYG,EEM,VEA,FXI,^RUT,^FTSE,^TYX,AUDUSD=X,USDJPY=X,EURUSD=X,GBPUSD=X,ZW=F,GC=F"]),
        "FEAT": trial.suggest_categorical("FEAT", ["ret,price,logret,rollret,sma,ema,momentum,macd,pricevshigh,vol,atr,range,volchange,volptile,zscore,rsi,cmo,williams,stoch,priceptile,adx,meanabsret,boll,donchian,volume,lag,retcrossz,crossmomentumz,crossvolz,crossretrank"]),
        "YWIN": trial.suggest_int("YWIN", 21, 22),
        "PRUNEWIN": trial.suggest_int("PRUNEWIN", 22, 24),
        "PRUNEDOWN": trial.suggest_float("PRUNEDOWN", 1.3, 1.35),
        "THRESH": trial.suggest_int("THRESH", 170, 190),
        "NESTIM": trial.suggest_int("NESTIM", 192, 192),
        "BATCH": trial.suggest_int("BATCH", 51, 59),
        "LBACK": trial.suggest_int("LBACK", 81, 88),
        "PRED_DAYS": trial.suggest_int("PRED_DAYS", 6, 6),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.0375, 0.0375, log=True),
        "DECAY": trial.suggest_float("DECAY", 0.0032, 0.0032, log=True),
        "SHORT_PER": trial.suggest_int("SHORT_PER", 12, 15),
        "MED_PER": trial.suggest_int("MED_PER", 19, 23),
        "LONG_PER": trial.suggest_int("LONG_PER", 68, 78),
        "INIT_LR": trial.suggest_float("INIT_LR", 0.0052, 0.006, log=True),
        "EXP_PEN": trial.suggest_float("EXP_PEN", 0.234, 0.234),
        "EXP_EXP": trial.suggest_float("EXP_EXP", 1.8, 1.8),
        "RETURN_PEN": trial.suggest_float("RETURN_PEN", 0.073, 0.073),
        "RETURN_EXP": trial.suggest_float("RETURN_EXP", 0.28, 0.28),
        "SD_PEN": trial.suggest_float("SD_PEN", 0.17, 0.17),
        "SD_EXP": trial.suggest_float("SD_EXP", 0.76, 0.76),
        "Z_ALPHA": trial.suggest_float("Z_ALPHA", 0.66, 0.85),
        "MAX_HEADS": trial.suggest_categorical("MAX_HEADS", [1,2]),
        "LAYERS": trial.suggest_int("LAYERS", 1,5),
        "EARLY_FAIL": trial.suggest_int("EARLY_FAIL", 2, 5),
        "VAL_SPLIT": trial.suggest_categorical("VAL_SPLIT", [0.15]),
        "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK", [12]),
        "ATTENT": trial.suggest_categorical("ATTENT", [1])
    }

    seeds_to_try = [42, 123, 999]  # 3 seeds for averaging
    scores = []

    for seed in seeds_to_try:
        base_config["SEED"] = seed
        score = run_single_seed_experiment(base_config, study)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score


def is_duplicate_trial(study, params):
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == params:
            print(f"[DUPLICATE] Found duplicate trial #{t.number}, skipping...")
            return t.value
    return None


def main():
    from load import load_config
    config = load_config()
    sampler = TPESampler(seed=config["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: run_experiment(trial, study), n_trials=TRIALS, n_jobs=1)

    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")


if __name__ == "__main__":
    main()
