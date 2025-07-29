import os, subprocess, re, json, optuna, csv
from optuna.samplers import TPESampler
from collections import Counter
from feat_list import FTR_FUNC

TRIALS = 70
TICKER_LIST = ['JPM', 'MSFT', 'NVDA', 'AVGO', 'LLY', 'COST', 'MA', 'XOM', 'UNH', 'AMZN', 'CAT', 'ADBE', 'TSLA']


FEAT_LONG = list(FTR_FUNC.keys()) 
FEAT_LIST = ['roll_ret', 'sma', 'price_vs_high', 'vol_ptile', 'adx', 'cross_vol_z', 'cross_rel_strength', 'cross_corr', 'ema', 'macd', 'stoch', 'vol_change', 'zscore', 'price','lags','log_ret','ret']
MACRO_LIST =  ['GC=F', 'HYG', 'EURUSD=X', 'UUP', 'ZW=F', 'USDJPY=X', 'NG=F', '^TYX', 'ZC=F','GBPUSD=X']

MACRO = [  'GC=F',       # Gold – safe haven and inflation hedge
                "^IRX",       # 13-Week T-Bill Rate
                '^FTSE',      # UK Index – decent global signal
                'HYG',        # Risk-on/risk-off signal
                'EURUSD=X',   # Euro regime
                'HG=F',       # Copper – strong industrial signal
                "^GSPC",      # S&P 500
                "GBPUSD=X",   # GBP/USD
                'UUP',        # USD Index – macro regime signal
                "EEM",        # EM    
                "ZW=F",       # Wheat Futures
                'USDJPY=X',   # Currency regime
                "NG=F",      # Natural Gas
                'VEA',       # Developed Intl Equities
                '^RUT',      # Russell 2000 – small cap US
                'ZC=F',      # Corn – appears in low trials, but still strong in top
                "^TYX",      #30Y
]

MACRO_LONG = [  "^GSPC",        # S&P 500
                "CL=F",         # Crude Oil (WTI)
                "SI=F",         # Silver
                "NG=F",         # Natural Gas
                "HG=F",         # Copper Futures
                "ZC=F",         # Corn Futures
                "^IRX",         # 13-Week T-Bill Rate
                "TLT",          # iShares 20+ Year Treasury Bond ETF
                "IEF",          # iShares 7-10 Year Treasury Bond ETF
                "UUP",          # Invesco DB US Dollar Index Bullish Fund
                "HYG",          # High Yield Corporate Bond ETF
                "EEM",          # Emerging Markets ETF
                "VEA",          # Developed ex-US Markets ETF
                "FXI",          # China Large-Cap ETF
                "^RUT",         # Russell 2000
                "^FTSE",        # FTSE 100
                "^TYX",         # 30-Year Treasury Yield
                "AUDUSD=X",     # AUD/USD (commodity-linked FX pair)
                "USDJPY=X",     # USD/JPY
                "EURUSD=X",     # EUR/USD
                "GBPUSD=X",     # GBP/USD
                "ZW=F",         # Wheat Futures
                "GC=F",         # Gold
                #"^TNX",        # 10-Year Treasury Yield
                #"^UST2Y",      # 2-Year Treasury Yield (FRED)
                #"PPIACO",      # Producer Price Index (FRED)
                #"CPIAUCSL",    # Consumer Price Index (FRED, monthly)
                #"LQD",         # Investment Grade Corporate Bond ETF
                #"^IXIC",       # Nasdaq Composite
                #"^DJI",        # Dow Jones Industrial Average
                #"BRL=X",       # USD/BRL exchange rate
                #"^VIX",        # CBOE Volatility Index
                #"SHY",         # iShares 1-3 Year Treasury Bond ETF
                #"TIP",         # iShares TIPS Bond ETF (Inflation-Protected)
                #"^FVX",        # 5-Year Treasury Yield
                #"^N225",       # Nikkei 225 (Japan)
]

def load_fixed_params(filepath="hyparams.json"):
    with open(filepath, "r") as f:params = json.load(f)
    return params

def binary_select(trial, items, prefix): return [item for item in items if trial.suggest_categorical(f"{prefix}_{item}", [False, True])]

def run_experiment(trial):
    select_macros = binary_select(trial, MACRO_LIST, "m")
    select_feat = binary_select(trial, FEAT_LIST, "f")
    select_TICK = TICKER_LIST.copy()

    if not select_feat or not select_macros or not select_TICK:
        print("[skip] Empty selection for features/macros")
        return -float("inf")

    fixed_params = load_fixed_params()
    config = fixed_params.copy()
    config.update({"TICK": ",".join(select_TICK),"MACRO": ",".join(select_macros),"FEAT": ",".join(select_feat),})

    env = os.environ.copy()
    for k, v in config.items(): env[k] = str(v)

    try:
        result = subprocess.run(["python", "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        if result.returncode != 0:print(f"  Subprocess failed: {result.stdout} / {result.stderr}")
        output = result.stdout + result.stderr
        if not output.strip():print("[error] Empty subprocess output.");return -float("inf")

        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%", out, re.IGNORECASE)
            return float(match.group(1)) / 100 if match else None

        def extract_avgoutperf(output):
            match = re.search(r"Avg Bench Outperf(?: thru Chunks)?:\s*\ncagr:\s*([-+]?\d+(?:\.\d+)?)%", output, re.MULTILINE)
            if match: return float(match.group(1)) / 100.0
            matches = re.findall(r"Avg Bench Outperf(?: thru Chunks)?:\s*([-+]?\d+(?:\.\d+)?)%", output)
            for val in reversed(matches):
                try: return float(val) / 100.0
                except: pass
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
    


        score = 1 * sharpe - 4.0 * abs(down) + 1 * (cagr or 0)
        if avg_outperf and avg_outperf > 0: score += 10
        if exp_delta and exp_delta > 100: score += 90

        print(f"  Sharpe: {sharpe}");print(f"  Down: {down}");print(f"  CAGR: {cagr}")
        print(f"  Exp Delta: {exp_delta}");print(f"  Avg Outperf: {avg_outperf}");print(f"  Score: {score}")

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("down", down)
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("avg_outperf", avg_outperf)
        trial.set_user_attr("exp_delta", exp_delta)
        trial.set_user_attr("select_feat", select_feat)
        trial.set_user_attr("select_macro", select_macros)
        trial.set_user_attr("select_TICK", select_TICK)

        fieldnames = ["trial"] + list(trial.params.keys()) + ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta", "score"]
        row = {"trial": trial.number,"sharpe": sharpe,"down": down,"CAGR": cagr,"avg_outperf": avg_outperf,"exp_delta": exp_delta,"score": score,**trial.params,}

        log_path = "csv/tune_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header: writer.writeheader()
            writer.writerow(row)
        return score

    except subprocess.TimeoutExpired:
        print(f"[timeout] Trial failed for config: {config}")
        return -float("inf")
    except Exception as e:
        print(f"[exception] Trial failed with exception: {e}")
        return -float("inf")

def main():
    from load import load_config
    config = load_config()
    sampler = TPESampler(n_startup_trials=TRIALS/5,seed=config["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=TRIALS, n_jobs=1)

    best = study.best_trial
    print("\n=== Best Trial Metrics ===")
    for m in ["sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")
    print(f"Selected feat: {best.user_attrs.get('select_feat')}")
    print(f"Selected macros: {best.user_attrs.get('select_macro')}")

    macro_counter = Counter()
    feat_counter = Counter()
    for trial in study.trials:
        if trial.value is None or trial.value == -float("inf"):continue
        for macro in MACRO_LIST:
            if trial.params.get(f"macro_{macro}"): macro_counter[macro] += 1
        for feat in FEAT_LIST:
            if trial.params.get(f"feat_{feat}"): feat_counter[feat] += 1

    print("\nMacro inclusion frequency:")
    for macro, count in macro_counter.most_common():print(f"{macro}: {count}")

    print("\nFeature inclusion frequency:")
    for feat, count in feat_counter.most_common():print(f"{feat}: {count}")

    print("\nParam importances:")
    importance = optuna.importance.get_param_importances(study)
    for k, v in importance.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()