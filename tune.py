import os, subprocess, re, optuna, json, csv
from optuna.samplers import TPESampler

TRIALS = 5

def run_experiment(trial):
    
    config = {
        "START": trial.suggest_categorical("START", ["2019-01-01"]), #"2016-01-01"
        "END": trial.suggest_categorical("END", ["2025-07-01"]),"SPLIT": trial.suggest_categorical("SPLIT", ["2023-01-01"]),
        "TICK": trial.suggest_categorical("TICK", ['JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE',
                                                   'JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN',
                                                   'JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, CAT, ADBE',
                                                   'JPM, MSFT, NVDA, AVGO, LLY, XOM, UNH, AMZN, CAT, ADBE',
                                                   'JPM, MSFT, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE',
                                                   'NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE',
                                                ]),
        "MACRO": trial.suggest_categorical("MACRO",['^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX, IEF, HYG']),
        "FEAT": trial.suggest_categorical("FEAT", ['price,vol,macd']),
        "BATCH": trial.suggest_int("BATCH", 55, 55),"LBACK": trial.suggest_int("LBACK", 84, 84),
        "PRED_DAYS": trial.suggest_int("PRED_DAYS", 2,2),"WARMUP": trial.suggest_float("WARMUP", 0.15, 0.15),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.03366, 0.03366),"DECAY": trial.suggest_float("DECAY", 0.00345, 0.00345),
        "ATTENT": trial.suggest_categorical("ATTENT", [1]),"FEAT_PER": trial.suggest_categorical("FEAT_PER",["8,12,24"]),
        "INIT_LR": trial.suggest_float("INIT_LR",0.037,0.037,),     
        "EXP_PEN": trial.suggest_float("EXP_PEN", 0.007,0.007),"RETURN_PEN": trial.suggest_float("RETURN_PEN", 0.182,0.182),
        "DOWN_PEN": trial.suggest_float("DOWN_PEN", 4.82,4.82),"DOWN_CUTOFF": trial.suggest_float("DOWN_CUTOFF", 0.279,0.279),
        "TEST_CHUNK": trial.suggest_int("TEST_CHUNK", 12, 12),
        "RETRAIN_WIN": trial.suggest_int("RETRAIN_WIN", 0, 0),"SEED": trial.suggest_int("SEED", 42, 42),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 20, 20),"LAYERS": trial.suggest_int("LAYERS", 6, 6),
        "EARLY_FAIL": trial.suggest_int("EARLY_FAIL", 6,6),"VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.13, 0.13),
    }
    env = os.environ.copy()
    for k, v in config.items(): env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        output = result.stdout + result.stderr

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strat:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None

        def extract_avgoutperf(output):
            match = re.search(r"Average Bench OutPerf(?: Across Chunks)?:\s*\ncagr:\s*([-+]?\d*\.\d+|\d+)%", output, re.MULTILINE)
            if match: return float(match.group(1)) / 100.0
            matches = re.findall(r"Average Bench OutPerf(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            for val in reversed(matches):
                try: return float(val) / 100.0
                except: pass
            return 0.0

        def extract_exp_delta(output):
            match = re.search(r"Total Exposure Delta:\s*([-+]?\d*\.\d+|\d+)", output)
            return float(match.group(1)) if match else None

        sharpe = extract_metric("Sharpe Ratio", output)
        down = extract_metric("Max Down", output)
        cagr = extract_metric("CAGR", output) 
        avg_outperf = extract_avgoutperf(output)
        exp_delta = extract_exp_delta(output)
        if sharpe is None or down is None or exp_delta is None: return -float("inf")
        score = 1 * sharpe - 3 * abs(down) + 1 * cagr+ 0 * avg_outperf
        if exp_delta>100: score+= 10
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("down", down)
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("avg_outperf", avg_outperf)
        trial.set_user_attr("exp_delta", exp_delta)
        fixed_fields = ["trial", "sharpe", "down", "CAGR", "avg_bench_perf", "exp_delta", "score"]
        fieldnames = fixed_fields[:1] + list(trial.params.keys()) + fixed_fields[1:] 
        row = { "trial": trial.number,"sharpe": sharpe,"down": down,"CAGR": cagr,"avg_bench_perf": avg_benc_perf,
               "exp_delta": exp_delta,"score": score,**trial.params}
        with open("csv/tune_log.csv", mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.isfile("csv/tune_log.csv"): writer.writeheader()
            writer.writerow(row)
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}");return -float("inf")

def main():
    from load import load_config
    config = load_config()
    sampler = TPESampler(seed=config["SEED"]);study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=TRIALS, n_jobs=1)
    best = study.best_trial;best_params = best.params.copy()
    with open("hyparams.json", "w") as f: json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items(): print(f"{k}: {v}")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__":
    main()

