import os, subprocess, re, csv
from optuna.trial import FixedTrial
import itertools

TRIALS = 70

def run_experiment(trial):
    config = {
        "START": trial.params["START"],
        "END": trial.params["END"],
        "SPLIT": trial.params["SPLIT"],
        "TICK": trial.params["TICK"],
        "MACRO": trial.params["MACRO"],
        "FEAT": trial.params["FEAT"],
        "BATCH": trial.params["BATCH"],
        "LBACK": trial.params["LBACK"],
        "PRED_DAYS": trial.params["PRED_DAYS"],
        "DROPOUT": trial.params["DROPOUT"],
        "DECAY": trial.params["DECAY"],
        "FEAT_PER": trial.params["FEAT_PER"],
        "INIT_LR": trial.params["INIT_LR"],
        "EXP_PEN": trial.params["EXP_PEN"],
        "EXP_EXP": trial.params["EXP_EXP"],
        "RETURN_PEN": trial.params["RETURN_PEN"],
        "RETURN_EXP": trial.params["RETURN_EXP"],
        "SD_PEN": trial.params["SD_PEN"],
        "SD_EXP": trial.params["SD_EXP"],
        "TEST_CHUNK": trial.params["TEST_CHUNK"],
        "RETRAIN_WIN": trial.params["RETRAIN_WIN"],
        "SEED": trial.params["SEED"],
        "MAX_HEADS": trial.params["MAX_HEADS"],
        "LAYERS": trial.params["LAYERS"],
        "EARLY_FAIL": trial.params["EARLY_FAIL"],
        "VAL_SPLIT": trial.params["VAL_SPLIT"],
        "WARMUP": trial.params["WARMUP"],
        "ATTENT": trial.params["ATTENT"],
    }

    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)

    try:
        result = subprocess.run(
            ["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800
        )
        if result.returncode != 0:
            print(f"  Subprocess failed: {result.stdout} / {result.stderr}")
        output = result.stdout + result.stderr
        if not output.strip():
            print("[error] Empty subprocess output.")
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

        print("[debug] Metrics extracted:")
        print(f"  Sharpe: {sharpe}")
        print(f"  Down: {down}")
        print(f"  CAGR: {cagr}")
        print(f"  Exp Delta: {exp_delta}")
        print(f"  Avg Outperf: {avg_outperf}")

        score = 1 * sharpe - 4 * abs(down) + 0.5 * cagr
        if avg_outperf > 0:
            score += 90
        if exp_delta > 100:
            score += 10

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("down", down)
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("avg_outperf", avg_outperf)
        trial.set_user_attr("exp_delta", exp_delta)

        fieldnames = ["trial"] + list(trial.params.keys()) + ["sharpe", "down", "CAGR", "avg_bench_perf", "exp_delta", "score"]
        row = {
            "trial": trial_id,  # use trial_id here instead of trial.number
            "sharpe": sharpe,
            "down": down,
            "CAGR": cagr,
            "avg_bench_perf": avg_outperf,
            "exp_delta": exp_delta,
            "score": score,
            **trial.params,
        }

        log_path = "csv/tune_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return score
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")
    except Exception as e:
        print(f"[error] Trial failed with exception: {e}")
        return -float("inf")


def main():
    # Manually specify your parameter grids here
    macros = [
        'HG=F,UUP,HYG,VEA,USDJPY=X,EURUSD=X,GC=F,^RUT,ZC=F,^FTSE,^TYX,EEM',
        'HG=F,UUP,HYG,USDJPY=X,EURUSD=X,GC=F,^FTSE,EEM',
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP,EEM",
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X,UUP",
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC,GBPUSD=X",
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F,^GSPC",
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X,HG=F",
        "GC=F,^IRX,^FTSE,HYG,EURUSD=X",
        "GC=F,^IRX,^FTSE,HYG",
        "GC=F,^IRX,^FTSE",
        "GC=F,^IRX",
        "GC=F",
    ]

    feats = [
"sma,ema,boll,macd,volatility_change,donchain",
    ]

    feats2 = [
        "sma,volatility_percentile",
        "price,ema",
        "sma,ema,boll,macd,volatility_change,donchain",
        "sma,ema,boll,macd,volatility_change,donchain,stochastic,williams,rsi",
        "sma,ema,boll,macd,volatility_change,donchain,stochastic,williams",
        "sma,ema,boll,macd,volatility_change,donchain,stochastic",
    "sma",
    "ema",
    "boll",
    "macd",
    "volatility_change",
    "donchain",
    "sma,ema",
    "sma,boll",
    "sma,macd",
    "sma,volatility_change",
    "sma,donchain",
    "ema,boll",
    "ema,macd",
    "ema,volatility_change",
    "ema,donchain",
    "boll,macd",
    "boll,volatility_change",
    "boll,donchain",
    "macd,volatility_change",
    "macd,donchain",
    "volatility_change,donchain",
    "sma,ema,boll",
    "sma,ema,macd",
    "sma,ema,volatility_change",
    "sma,ema,donchain",
    "sma,boll,macd",
    "sma,boll,volatility_change",
    "sma,boll,donchain",
    "sma,macd,volatility_change",
    "sma,macd,donchain",
    "sma,volatility_change,donchain",
    "ema,boll,macd",
    "ema,boll,volatility_change",
    "ema,boll,donchain",
    "ema,macd,volatility_change",
    "ema,macd,donchain",
    "ema,volatility_change,donchain",
    "boll,macd,volatility_change",
    "boll,macd,donchain",
    "boll,volatility_change,donchain",
    "macd,volatility_change,donchain",
    "sma,ema,boll,macd",
    "sma,ema,boll,volatility_change",
    "sma,ema,boll,donchain",
    "sma,ema,macd,volatility_change",
    "sma,ema,macd,donchain",
    "sma,ema,volatility_change,donchain",
    "sma,boll,macd,volatility_change",
    "sma,boll,macd,donchain",
    "sma,boll,volatility_change,donchain",
    "sma,macd,volatility_change,donchain",
    "ema,boll,macd,volatility_change",
    "ema,boll,macd,donchain",
    "ema,boll,volatility_change,donchain",
    "ema,macd,volatility_change,donchain",
    "boll,macd,volatility_change,donchain",
    "sma,ema,boll,macd,volatility_change",
    "sma,ema,boll,macd,donchain",
    "sma,ema,boll,volatility_change,donchain",
    "sma,ema,macd,volatility_change,donchain",
    "sma,boll,macd,volatility_change,donchain",
    "ema,boll,macd,volatility_change,donchain",
    
]


    fixed_params = {
        "START": "2019-01-01",
        "END": "2025-07-01",
        "SPLIT": "2023-01-01",
        "TICK": "JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE",
        "BATCH": 53,
        "LBACK": 84,
        "PRED_DAYS": 6,
        "DROPOUT": 0.028,
        "DECAY": 0.003,
        "FEAT_PER": "8,12,24",
        "INIT_LR": 0.006,
        "EXP_PEN": 0.235,
        "EXP_EXP": 1.82,
        "RETURN_PEN": 0.105,
        "RETURN_EXP": 0.28,
        "SD_PEN": 0.17,
        "SD_EXP": 0.74,
        "TEST_CHUNK": 12,
        "RETRAIN_WIN": 0,
        "SEED": 42,
        "MAX_HEADS": 1,
        "LAYERS": 1,
        "EARLY_FAIL": 2,
        "VAL_SPLIT": 0.15,
        "WARMUP": 0,
        "ATTENT": 0,
    }

    trial_id = 0
    for macro, feat in itertools.product(macros, feats):
        params = fixed_params.copy()
        params["MACRO"] = macro
        params["FEAT"] = feat
        trial = FixedTrial(params)
        print(f"\n=== Running trial {trial_id} ===")
        score = run_experiment(trial)
        print(f"Trial {trial_id} score: {score:.4f}")
        trial_id += 1


if __name__ == "__main__":
    main()
