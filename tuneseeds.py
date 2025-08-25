import os, subprocess, re, optuna, json, csv
from optuna.samplers import TPESampler

TRIALS = 400
SEEDS = [42, 3, 850]  # seeds we will average across

def run_single(config, seed):
    """Run one experiment for a specific seed and return metrics."""
    env = os.environ.copy()
    config["SEED"] = seed
    for k, v in config.items():
        env[k] = str(v)

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
            return None

    proc.wait()
    output = "".join(output_lines)
    if proc.returncode != 0 or not output.strip():
        return None

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
        return None

    score = 1 * sharpe - 6 * abs(down) + 1 * cagr
    if avg_outperf > 0:
        score += 10
    if exp_delta > 100:
        score += 90

    return {
        "seed": seed,
        "sharpe": sharpe,
        "down": down,
        "CAGR": cagr,
        "avg_outperf": avg_outperf,
        "exp_delta": exp_delta,
        "score": score
    }


def run_experiment(trial, study=None):
    config = {
        "START": trial.suggest_categorical("START", ["2012-01-01"]),
        "SPLIT": trial.suggest_categorical("SPLIT", ["2017-01-01"]),
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ["^GSPC,CL=F,SI=F,NG=F,HG=F,ZC=F,^IRX,TLT,IEF,UUP,HYG,EEM,VEA,FXI,^RUT,^FTSE,^TYX,AUDUSD=X,USDJPY=X,EURUSD=X,GBPUSD=X,ZW=F,GC=F"]),
        "FEAT": trial.suggest_categorical("FEAT", ["ret,price,logret,rollret,sma,ema,momentum,macd,pricevshigh,vol,atr,range,volchange,volptile,zscore,rsi,cmo,williams,stoch,priceptile,meanabsret,boll,donchian,volume,lag,retcrossz,crossmomentumz,crossvolz,crossretrank"]),
        "YWIN": trial.suggest_int("YWIN", 21, 24),
        "PRUNEWIN": trial.suggest_int("PRUNEWIN", 21, 24),
        "PRUNEDOWN": trial.suggest_float("PRUNEDOWN", 1.3, 1.35),
        "THRESH": trial.suggest_int("THRESH", 150, 200),
        "NESTIM": trial.suggest_int("NESTIM", 192, 192),
        "TOPIMP": trial.suggest_int("TOPIMP", 0, 150),
        "IMPDECAY": trial.suggest_float("IMPDECAY", 0.89, 0.95),
        "RF_WEIGHT": trial.suggest_float("RF_WEIGHT", 0, 0.4),
        "TRANS_WEIGHT": trial.suggest_float("TRANS_WEIGHT", 0, 0.2),
        "BATCH": trial.suggest_int("BATCH", 50, 64),
        "LBACK": trial.suggest_int("LBACK", 80, 90),
        "PRED_DAYS": trial.suggest_int("PRED_DAYS", 6, 6),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.03, 0.4, log=True),
        "DECAY": trial.suggest_float("DECAY", 0.003, 0.004, log=True),
        "SHORT_PER": trial.suggest_int("SHORT_PER", 12, 14),
        "MED_PER": trial.suggest_int("MED_PER", 21, 21),
        "LONG_PER": trial.suggest_int("LONG_PER", 69, 80),
        "INIT_LR": trial.suggest_float("INIT_LR", 0.001430727411696868, 0.006430727411696868, log=True),
        "EXP_PEN": trial.suggest_float("EXP_PEN", 0.22, 0.24),
        "EXP_EXP": trial.suggest_float("EXP_EXP", 1.8, 1.8),
        "RETURN_PEN": trial.suggest_float("RETURN_PEN", 0.073, 0.073),
        "RETURN_EXP": trial.suggest_float("RETURN_EXP", 0.28, 0.28),
        "SD_PEN": trial.suggest_float("SD_PEN", 0.17, 0.17),
        "SD_EXP": trial.suggest_float("SD_EXP", 0.76, 0.76),
        "Z_LOC": trial.suggest_float("Z_LOC", 0.7, 0.9),
        "Z_ANCH": trial.suggest_float("Z_ANCH", 0, 0.02),
        "ANCH_DECAY": trial.suggest_float("ANCH_DECAY", 0.99737139944604, 0.99737139944604),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 1, 4),
        "LAYERS": trial.suggest_int("LAYERS", 1, 5),
        "EARLY_FAIL": trial.suggest_int("EARLY_FAIL", 2, 5),
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.15, 0.15),
        "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK", [12]),
        "D": trial.suggest_float("D", 999999, 999999),
    }

    # run across seeds
    results = []
    for seed in SEEDS:
        res = run_single(config.copy(), seed)
        if res:
            results.append(res)

    if not results:
        return -float("inf")

    # average across seeds
    avg_metrics = {k: sum(r[k] for r in results) / len(results) for k in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta", "score"]}

    print(f"\n[Trial {trial.number}] Averaged across seeds: {avg_metrics}")

    # log averaged result
    trial.set_user_attr("sharpe", avg_metrics["sharpe"])
    trial.set_user_attr("down", avg_metrics["down"])
    trial.set_user_attr("CAGR", avg_metrics["CAGR"])
    trial.set_user_attr("avg_outperf", avg_metrics["avg_outperf"])
    trial.set_user_attr("exp_delta", avg_metrics["exp_delta"])

    fieldnames = ["trial"] + list(trial.params.keys()) + ["sharpe", "down", "CAGR", "avg_bench_perf", "exp_delta", "score"]
    row = {"trial": trial.number,
           "sharpe": avg_metrics["sharpe"],
           "down": avg_metrics["down"],
           "CAGR": avg_metrics["CAGR"],
           "avg_bench_perf": avg_metrics["avg_outperf"],
           "exp_delta": avg_metrics["exp_delta"],
           "score": avg_metrics["score"],
           **trial.params}
    log_path = "csv/tune_log.csv"
    write_header = not os.path.exists(log_path)
    with open(log_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return avg_metrics["score"]


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
