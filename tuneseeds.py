import os, subprocess, re, optuna, json, csv, random
from optuna.samplers import TPESampler

TRIALS = 400
SEEDS_PER_TRIAL = 3  # Number of random seeds per trial

def run_experiment(trial, study=None):
    base_config = {
        "START": trial.suggest_categorical("START", ["2012-01-01"]),
        "SPLIT": trial.suggest_categorical("SPLIT", ["2017-01-01"]),
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ["^GSPC,CL=F,SI=F,NG=F,HG=F,ZC=F,^IRX,TLT,IEF,UUP,HYG,EEM,VEA,FXI,^RUT,^FTSE,^TYX,AUDUSD=X,USDJPY=X,EURUSD=X,GBPUSD=X,ZW=F,GC=F"]),
        "FEAT": trial.suggest_categorical("FEAT", ["ret,price,logret,rollret,sma,ema,momentum,macd,pricevshigh,vol,atr,range,volchange,volptile,zscore,rsi,cmo,williams,stoch,priceptile,adx,meanabsret,boll,donchian,volume,lag,retcrossz,crossmomentumz,crossvolz,crossretrank"]),
        "YWIN": trial.suggest_int("YWIN", 19, 24),
        "PRUNEWIN": trial.suggest_int("PRUNEWIN", 19, 24),
        "PRUNEDOWN": trial.suggest_float("PRUNEDOWN", 1.2, 1.40),
        "THRESH": trial.suggest_int("THRESH", 145, 199),
        "NESTIM": trial.suggest_int("NESTIM", 190, 190),
        "IMPDECAY": trial.suggest_float("IMPDECAY", 0.9, 1),
        "BATCH": trial.suggest_int("BATCH", 52, 65),
        "LBACK": trial.suggest_int("LBACK", 80, 95),
        "PRED_DAYS": trial.suggest_int("PRED_DAYS", 5, 8),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.035, 0.045, log=True),
        "DECAY": trial.suggest_float("DECAY", 0.0030, 0.0035, log=True),
        "SHORT_PER": trial.suggest_int("SHORT_PER", 14, 18),
        "MED_PER": trial.suggest_int("MED_PER", 20, 27),
        "LONG_PER": trial.suggest_int("LONG_PER", 60, 85),
        "INIT_LR": trial.suggest_float("INIT_LR", 0.0063, 0.0063, log=True),
        "EXP_PEN": trial.suggest_float("EXP_PEN", 0.231, 0.231),
        "EXP_EXP": trial.suggest_float("EXP_EXP", 1.8, 1.8),
        "RETURN_PEN": trial.suggest_float("RETURN_PEN", 0.074, 0.074),
        "RETURN_EXP": trial.suggest_float("RETURN_EXP", 0.28, 0.28),
        "SD_PEN": trial.suggest_float("SD_PEN", 0.172, 0.172),
        "SD_EXP": trial.suggest_float("SD_EXP", 0.76, 0.76),
        "Z_LOC": trial.suggest_float("Z_LOC", 0.5, 0.85),
        "MAX_HEADS": trial.suggest_categorical("MAX_HEADS", [1, 2]),
        "LAYERS": trial.suggest_categorical("LAYERS", [1, 2]),
        "EARLY_FAIL": trial.suggest_categorical("EARLY_FAIL", [3, 4, 5, 6]),
        "VAL_SPLIT": trial.suggest_categorical("VAL_SPLIT", [0.15]),
        "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK", [12]),
    }

    metrics = {"sharpe": [], "down": [], "CAGR": [], "avg_outperf": [], "exp_delta": []}

    for seed in random.sample(range(1000), SEEDS_PER_TRIAL):
        config = base_config.copy()
        config["SEED"] = seed
        env = os.environ.copy()
        for k, v in config.items():
            env[k] = str(v)

        dup_value = is_duplicate_trial(study, {k: str(v) for k, v in config.items()})
        if dup_value is not None:
            return dup_value

        try:
            python_exe = os.path.join(env.get("VIRTUAL_ENV", ""), "Scripts", "python.exe") if "VIRTUAL_ENV" in env else "python"
            result = subprocess.run([python_exe, "model.py"], capture_output=True, text=True, env=env, timeout=1800)
            output = result.stdout + result.stderr

            if result.returncode != 0 or not output.strip():
                print(f"[error] Subprocess failed or empty output for seed {seed}.")
                return -float("inf")
            if "KILLRUN" in output:
                print(f"[KILLRUN] Detected for seed {seed}, discarding trial.")
                return -float("inf")

            def extract_metric(label):
                match = re.search(rf"{re.escape(label)}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%", output, re.IGNORECASE)
                return float(match.group(1)) / 100 if match else None

            def extract_avgoutperf():
                match = re.search(r"Avg Bench Outperf(?: thru Chunks)?:\s*\ncagr:\s*([-+]?\d+(?:\.\d+)?)%", output, re.MULTILINE)
                if match: return float(match.group(1)) / 100.0
                matches = re.findall(r"Avg Bench Outperf(?: thru Chunks)?:\s*([-+]?\d+(?:\.\d+)?)%", output)
                for val in reversed(matches):
                    try: return float(val) / 100.0
                    except: pass
                return 0.0

            def extract_exp_delta():
                match = re.search(r"Total Exp Delta:\s*([-+]?\d+(?:\.\d+)?)", output)
                return float(match.group(1)) if match else None

            sharpe = extract_metric("Sharpe Ratio")
            down = extract_metric("Max Down")
            cagr = extract_metric("CAGR")
            avg_outperf = extract_avgoutperf()
            exp_delta = extract_exp_delta()

            if None in [sharpe, down, cagr, exp_delta]:
                print(f"[error] Missing metrics for seed {seed}. Discarding trial.")
                return -float("inf")

            metrics["sharpe"].append(sharpe)
            metrics["down"].append(down)
            metrics["CAGR"].append(cagr)
            metrics["avg_outperf"].append(avg_outperf)
            metrics["exp_delta"].append(exp_delta)

        except subprocess.TimeoutExpired:
            print(f"[timeout] Seed {seed} failed.")
            return -float("inf")
        except Exception as e:
            print(f"[exception] Seed {seed} failed: {e}")
            return -float("inf")

    # Compute averages
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    score = 1 * avg_metrics["sharpe"] - 6 * abs(avg_metrics["down"]) + 1 * avg_metrics["CAGR"]
    if avg_metrics["avg_outperf"] > 0: score += 10
    if avg_metrics["exp_delta"] > 100: score += 90

    print(f"  Avg Sharpe: {avg_metrics['sharpe']:.4f}")
    print(f"  Avg Down: {avg_metrics['down']:.4f}")
    print(f"  Avg CAGR: {avg_metrics['CAGR']:.4f}")
    print(f"  Avg Exp Delta: {avg_metrics['exp_delta']:.4f}")
    print(f"  Avg Outperf: {avg_metrics['avg_outperf']:.4f}")
    print(f"  Score: {score:.4f}")

    trial.set_user_attr("sharpe", avg_metrics["sharpe"])
    trial.set_user_attr("down", avg_metrics["down"])
    trial.set_user_attr("CAGR", avg_metrics["CAGR"])
    trial.set_user_attr("avg_outperf", avg_metrics["avg_outperf"])
    trial.set_user_attr("exp_delta", avg_metrics["exp_delta"])

    # Log to CSV
    fieldnames = ["trial"] + list(trial.params.keys()) + ["sharpe", "down", "CAGR", "avg_bench_perf", "exp_delta", "score"]
    row = {"trial": trial.number, "score": score, **trial.params, **{
        "sharpe": avg_metrics["sharpe"],
        "down": avg_metrics["down"],
        "CAGR": avg_metrics["CAGR"],
        "avg_bench_perf": avg_metrics["avg_outperf"],
        "exp_delta": avg_metrics["exp_delta"]
    }}
    log_path = "csv/tune_log.csv"
    write_header = not os.path.exists(log_path)
    with open(log_path, mode="RF_WEIGHT", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header: writer.writeheader()
        writer.writerow(row)

    return score

def is_duplicate_trial(study, params):
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
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
    for k, v in best_params.items(): print(f"{k}: {v}")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__":
    main()
