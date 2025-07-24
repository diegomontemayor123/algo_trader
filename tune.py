import os, subprocess, re, optuna, json, csv
from optuna.samplers import TPESampler

TRIALS = 4

def run_experiment(trial):
    config = {"START": trial.suggest_categorical("START", ["2017-01-01"]),#2019 Jan
        "END": trial.suggest_categorical("END", ["2023-01-01"]),#2025 Jul
        "SPLIT": trial.suggest_categorical("SPLIT", ["2021-01-01"]),#2023 Jan
        "TICK": trial.suggest_categorical("TICK", ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]),
        "MACRO": trial.suggest_categorical("MACRO", ["GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP,EEM",
                                                     "^GSPC,EEM,HYG,^FTSE,UUP,GBPUSD=X,^IRX,EURUSD=X",]),#"GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP,EEM"
        #'^VIX'
        "FEAT": trial.suggest_categorical("FEAT", ["sma,ema,boll,macd,volatility_change,donchain",
                                                   "ret,williams,rsi,volatility_change"]),#"sma,ema,boll,macd,volatility_change,donchain"
        #"price,ema"
        "BATCH": trial.suggest_int("BATCH",53,53),#53
        "LBACK": trial.suggest_int("LBACK",84,84),#84
        "PRED_DAYS": trial.suggest_int("PRED_DAYS",6,6),#6
        "DROPOUT": trial.suggest_float("DROPOUT",.028,.028),#.028
        "DECAY": trial.suggest_float("DECAY",.003,.003),#.003
        "FEAT_PER": trial.suggest_categorical("FEAT_PER", ["8,12,24"]),
        "INIT_LR": trial.suggest_float("INIT_LR",.006,.006),#.006
        "EXP_PEN": trial.suggest_float("EXP_PEN",.226,.226),#.235 price,ema,vix     / .226 long macro/feat
        "EXP_EXP": trial.suggest_float("EXP_EXP",1.82,1.82),#1.82
        "RETURN_PEN": trial.suggest_float("RETURN_PEN",.07,.07),#.105 price,ema,vix / .07 long macro/feat
        "RETURN_EXP": trial.suggest_float("RETURN_EXP",.28,.28),#.28 
        "SD_PEN": trial.suggest_float("SD_PEN",.17,.17),#.17 
        "SD_EXP": trial.suggest_float("SD_EXP",.74,.74),#.74 
        "SEED": trial.suggest_int("SEED",42,42),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 1, 1),#1
        "LAYERS": trial.suggest_int("LAYERS", 1, 1),#1
        "EARLY_FAIL": trial.suggest_int("EARLY_FAIL", 2, 2),#2
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", .15, .15),#.15
        "WARMUP": trial.suggest_categorical("WARMUP", [0]),
        "TEST_CHUNK": trial.suggest_int("TEST_CHUNK",24,24),
        "RETRAIN": trial.suggest_categorical("RETRAIN_WIN", [0]),
        "ATTENT": trial.suggest_categorical("ATTENT", [1]),
    }

    env = os.environ.copy()
    for k, v in config.items():env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        if result.returncode != 0:print(f"  Subprocess failed: {result.stdout} / {result.stderr}")
        output = result.stdout + result.stderr
        if not output.strip():print("[error] Empty subprocess output.");return -float("inf")
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
        print("[debug] Metrics extracted:")
        print(f"  Sharpe: {sharpe}");print(f"  Down: {down}");print(f"  CAGR: {cagr}")
        print(f"  Exp Delta: {exp_delta}");print(f"  Avg Outperf: {avg_outperf}")

        score = 1* sharpe - 6 * abs(down) + 0.5 * cagr 
        if avg_outperf>0: score += 90
        if exp_delta > 100: score += 10

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
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")
    except Exception as e:
        print(f"[error] Trial failed with exception: {e}")
        return -float("inf")

def main():
    from load import load_config
    config = load_config()
    sampler = TPESampler(seed=config["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=TRIALS, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f: json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items(): print(f"{k}: {v}")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__": main()
