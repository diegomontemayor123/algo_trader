import os, subprocess, re, optuna, json, csv
from optuna.samplers import TPESampler

TRIALS = 30

def run_experiment(trial,study=None):
    config = {"START": trial.suggest_categorical("START", ["2015-01-01"]),#2019 Jan
        "END": trial.suggest_categorical("END", ["2023-01-01"]),#2025 Jul
        "SPLIT": trial.suggest_categorical("SPLIT", ["2019-01-01",]),#2023 Jan
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ['^FTSE,^GSPC,^TYX,EURUSD=X,GBPUSD=X,GC=F,HYG,NG=F,SI=F,TLT,UUP,USDJPY=X,ZC=F,ZW=F,^IRX,EEM,HG=F',]),#"GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP,EEM"
        "FEAT": trial.suggest_categorical("FEAT", ['adx,boll,cmo,cross_corr,cross_rel_strength,cross_ret_rank,cross_vol_z,ema,lags,log_ret,macd,mean_abs_return,price,price_vs_high,range,ret,roll_ret,rsi,sma,stoch,vol_change,vol_ptile,zscore,donchian',]),#"sma,ema,boll,macd,vol_change,donchian"
        "FILTER": trial.suggest_categorical("FILTER", ["rf"]),#"none","mutual","correl","rf"
        "YWIN": trial.suggest_int("YWIN",20,25),#25
        "FILTERWIN": trial.suggest_int("FILTERWIN",26,26),#26
        "THRESH": trial.suggest_int("THRESH",105,120),#113
        "NESTIM": trial.suggest_int("NESTIM",300,340),#406
        "BATCH": trial.suggest_int("BATCH",53,53),#53
        "LBACK": trial.suggest_int("LBACK",84,84),#84
        "PRED_DAYS": trial.suggest_int("PRED_DAYS",6,6),#6
        "DROPOUT": trial.suggest_float("DROPOUT",.03,.04),#.028
        "DECAY": trial.suggest_float("DECAY",.0028,.0034,log=True),#.003
        "FEAT_PER": trial.suggest_categorical("FEAT_PER", ["8,12,24"]),
        "INIT_LR": trial.suggest_float("INIT_LR",.001,.005,log=True),#.006
        "EXP_PEN": trial.suggest_float("EXP_PEN",.21,.25),#.226 long macro/feat
        "EXP_EXP": trial.suggest_float("EXP_EXP",1.76,1.8),#1.8
        "RETURN_PEN": trial.suggest_float("RETURN_PEN",.04,.07),#.07 long macro/feat
        "RETURN_EXP": trial.suggest_float("RETURN_EXP",.25,.28),#.28 
        "SD_PEN": trial.suggest_float("SD_PEN",.16,.19),#.17 
        "SD_EXP": trial.suggest_float("SD_EXP",.74,.77),#.74 
        "SEED": trial.suggest_categorical("SEED",[42]),#42
        "MAX_HEADS": trial.suggest_categorical("MAX_HEADS", [1]),#1
        "LAYERS": trial.suggest_categorical("LAYERS", [1,2]),#1
        "EARLY_FAIL": trial.suggest_categorical("EARLY_FAIL",[2,4]),#2
        "VAL_SPLIT": trial.suggest_categorical("VAL_SPLIT",[.15]),#.15
        "WARMUP": trial.suggest_categorical("WARMUP",[0]),
        "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK",[24]),#12
        "RETRAIN": trial.suggest_categorical("RETRAIN", [1]),
        "ATTENT": trial.suggest_categorical("ATTENT", [1]),
    }

    env = os.environ.copy()
    for k, v in config.items():env[k] = str(v)
    dup_value = is_duplicate_trial(study, {k: str(v) for k, v in config.items()})
    if dup_value is not None: return dup_value 
    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        if result.returncode != 0:print(f"  Subprocess failed: {result.stdout} / {result.stderr}")
        output = result.stdout + result.stderr
        if not output.strip():print("[error] Empty subprocess output.");return -float("inf")
        if "KILLRUN" in output:print("[KILLRUN] Portfolio Sharpe below benchmark Sharpe — aborting trial.");return -float("inf")
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

