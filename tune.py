import os, subprocess, re, optuna, json, csv
from optuna.samplers import TPESampler

TRIALS = 400


def run_experiment(trial,study=None):
    config = {"START": trial.suggest_categorical("START", ["2013-01-01","2012-10-01","2012-07-01","2012-04-01","2012-01-01"]),#2019 Jan
        "END": trial.suggest_categorical("END", ["2023-01-01"]),#2025 Jul
        "SPLIT": trial.suggest_categorical("SPLIT", ["2017-01-01",]),#2023 Jan
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ["^GSPC,CL=F,SI=F,NG=F,HG=F,ZC=F,^IRX,TLT,IEF,UUP,HYG,EEM,VEA,FXI,^RUT,^FTSE,^TYX,AUDUSD=X,USDJPY=X,EURUSD=X,GBPUSD=X,ZW=F,GC=F",]),#"GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP,EEM"
        "FEAT": trial.suggest_categorical("FEAT", ["ret,price,logret,rollret,sma,ema,momentum,macd,pricevshigh,vol,atr,range,volchange,volptile,zscore,rsi,cmo,williams,stoch,priceptile,adx,meanabsret,boll,donchian,volume,lag,retcrossz,crossmomentumz,crossvolz,crossretrank",]),#"sma,ema,boll,macd,volchange,donchian"
        "YWIN": trial.suggest_int("YWIN",20,40),#29,28
        "PRUNEWIN": trial.suggest_int("PRUNEWIN",20,40),#31,28
        "PRUNEDOWN": trial.suggest_float("PRUNEDOWN",1,2),
        "THRESH": trial.suggest_int("THRESH",100,250),#170
        "NESTIM": trial.suggest_int("NESTIM",150,250),#200
        "BATCH": trial.suggest_int("BATCH",40,60),#53
        "LBACK": trial.suggest_int("LBACK",60,95),#84
        "PRED_DAYS": trial.suggest_int("PRED_DAYS",1,9),#6
        "DROPOUT": trial.suggest_float("DROPOUT",.03,.045),#.035
        "DECAY": trial.suggest_float("DECAY",.003,.004,log=True),#.003
        "SHORT_PER": trial.suggest_int("SHORT_PER",5,18),#12
        "MED_PER": trial.suggest_int("MED_PER",14,35),#17
        "LONG_PER": trial.suggest_int("LONG_PER",30,80),#58
        "INIT_LR": trial.suggest_float("INIT_LR",.001,.006,log=True),#.001
        "EXP_PEN": trial.suggest_float("EXP_PEN",.23,.26),#.24
        "EXP_EXP": trial.suggest_float("EXP_EXP",1.8,1.8),#1.8
        "RETURN_PEN": trial.suggest_float("RETURN_PEN",.07,.08),#.073
        "RETURN_EXP": trial.suggest_float("RETURN_EXP",.28,.28),#.28 
        "SD_PEN": trial.suggest_float("SD_PEN",.15,.2),#.17 
        "SD_EXP": trial.suggest_float("SD_EXP",.7,.8),#.74 
        "SEED": trial.suggest_categorical("SEED",[42]),#42
        "MAX_HEADS": trial.suggest_categorical("MAX_HEADS",[1,2,3,4]),#1
        "LAYERS": trial.suggest_categorical("LAYERS",[1,2,3,4,5]),#2
        "EARLY_FAIL": trial.suggest_categorical("EARLY_FAIL",[2,3,4,5,6,7,8]),#2
        "VAL_SPLIT": trial.suggest_categorical("VAL_SPLIT",[.15]),#.15
        "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK",[6,9,12,15,18,21,24,36,48]),
        "ATTENT": trial.suggest_categorical("ATTENT",[1]),
    }

    env = os.environ.copy()
    for k, v in config.items():env[k] = str(v)
    dup_value = is_duplicate_trial(study, {k: str(v) for k, v in config.items()})
    if dup_value is not None: return dup_value 
    try:
        python_exe = os.path.join(env.get("VIRTUAL_ENV", ""), "Scripts", "python.exe") if "VIRTUAL_ENV" in env else "python"
        result = subprocess.run([python_exe, "model.py"], capture_output=True, text=True, env=env, timeout=1800)

        if result.returncode != 0:print(f"  Subprocess failed: {result.stdout} / {result.stderr}")
        output = result.stdout + result.stderr
        if not output.strip():print("[error] Empty subprocess output.");return -float("inf")
        if "KILLRUN" in output:print("[KILLRUN] — aborting trial.");return -float("inf")
        def extract_metric(label, out):
            pattern = rf"{re.escape(label)}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%"
            match = re.search(pattern, out, re.IGNORECASE)
            return float(match.group(1)) / 100 if match else None
        def extract_avgoutperf(output):
            match = re.search(r"Avg Bench Outperf(?: thru Chunks)?:\s*\ncagr:\s*([-+]?\d+(?:\.\d+)?)%", output, re.MULTILINE)
            if match:return float(match.group(1)) / 100.0
            matches = re.findall(r"Avg Bench Outperf(?: thru Chunks)?:\s*([-+]?\d+(?:\.\d+)?)%", output)
            for val in reversed(matches):
                try:return float(val) / 100.0
                except:pass
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
        if avg_outperf>0: score += 10
        if exp_delta > 100: score += 90

        print(f"  Sharpe: {sharpe}");print(f"  Down: {down}");print(f"  CAGR: {cagr}")
        print(f"  Exp Delta: {exp_delta}");print(f"  Avg Outperf: {avg_outperf}");print(f"  Score: {score}")

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("down", down)
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("avg_outperf", avg_outperf)
        trial.set_user_attr("exp_delta", exp_delta)
        fieldnames = ["trial"] + list(trial.params.keys()) + ["sharpe", "down", "CAGR", "avg_bench_perf", "exp_delta", "score"]
        row = {"trial": trial.number,"sharpe": sharpe,"down": down,"CAGR": cagr,"avg_bench_perf": avg_outperf,"exp_delta": exp_delta,"score": score,**trial.params}
        log_path = "csv/tune_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header: writer.writeheader()
            writer.writerow(row)
        return score
    except subprocess.TimeoutExpired:
        print(f"[Tune] Trial failed for config: {config}")
        return -float("inf")
    except Exception as e:
        print(f"[Tune] Trial failed with exception: {e}")
        return -float("inf")


def is_duplicate_trial(study, params):
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        if t.params == params:
            print(f"[DUPLICATE] Found duplicate trial #{t.number}, skipping...")
            return t.value  
    return None


def main():
    from load import load_config; config = load_config()
    sampler = TPESampler(seed=config["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: run_experiment(trial, study), n_trials=TRIALS, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f: json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items(): print(f"{k}: {v}")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__": main()

