import os, subprocess, re, json, optuna
from optuna.samplers import TPESampler
from collections import Counter

TICKER_LIST = ['NVDA','TSLA']
FEAT_LIST = ["ret", "price", "log_ret", "roll_ret", "volume", "vol", "macd", "momentum", "ema", "cmo"]
MACRO_LIST = [
    "^GSPC",        # S&P 500
    "^N225",        # Nikkei 225 (Japan)
    "CL=F",         # Crude Oil (WTI)
    "SI=F",         # Silver
    "NG=F",         # Natural Gas
    "HG=F",         # Copper Futures
    "ZC=F",         # Corn Futures
    "^FVX",         # 5-Year Treasury Yield
    "^IRX",         # 13-Week T-Bill Rate
    "TLT",          # iShares 20+ Year Treasury Bond ETF
    "IEF",          # iShares 7-10 Year Treasury Bond ETF
    "SHY",          # iShares 1-3 Year Treasury Bond ETF
    "TIP",          # iShares TIPS Bond ETF (Inflation-Protected)
    "UUP",          # Invesco DB US Dollar Index Bullish Fund
    "^VIX",         # CBOE Volatility Index
    "HYG",          # High Yield Corporate Bond ETF
    "BRL=X",        # USD/BRL exchange rate
    "AUDUSD=X"      # AUD/USD (commodity-linked FX pair)
    #"EEM",          # Emerging Markets ETF
    #"VEA",          # Developed ex-US Markets ETF
    #"FXI",          # China Large-Cap ETF
    #"^IXIC",        # Nasdaq Composite
    #"^DJI",         # Dow Jones Industrial Average
    #"^RUT",         # Russell 2000
    #"^FTSE",        # FTSE 100
    #"PPIACO",      # Producer Price Index (FRED)
    #"CPIAUCSL",    # Consumer Price Index (FRED, monthly)
    #"LQD",          # Investment Grade Corporate Bond ETF
    #"^TYX",         # 30-Year Treasury Yield
    #"^UST2Y",      # 2-Year Treasury Yield (FRED)
    #"USDJPY=X",     # USD/JPY
    #"EURUSD=X",     # EUR/USD
    #"GBPUSD=X",     # GBP/USD
    #"^TNX",         # 10-Year Treasury Yield
    #"ZW=F",         # Wheat Futures
    #"GC=F",         # Gold
]

def load_fixed_params(filepath="hyparams.json"):
    with open(filepath, "r") as f:params = json.load(f)
    return params

def binary_select(trial, items, prefix):
    return [item for item in items if trial.suggest_categorical(f"{prefix}_{item}", [False, True])]

def run_experiment(trial):
    select_macros = binary_select(trial, MACRO_LIST, "macro")
    select_feat = binary_select(trial, FEAT_LIST, "feat")
    select_TICK = binary_select(trial, TICKER_LIST, "ticker")
    if len(select_feat) == 0 or len(select_macros) == 0 or len(select_TICK) == 0:return -float("inf")  
    fixed_params = load_fixed_params()
    config = fixed_params.copy()
    config.update({"TICK": ",".join(select_TICK),"MACRO": ",".join(select_macros),"FEAT": ",".join(select_feat),})
    env = os.environ.copy()
    for k, v in config.items(): env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"],capture_output=True,text=True,env=env,timeout=1800)
        output = result.stdout + result.stderr
        with open("tune_log.log", "a") as f:
            f.write("\n\n=== Trial output start ===\n")
            f.write(output)
            f.write("\n=== Trial output end ===\n")
        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strat:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None
        def extract_avgbenchperf(output):
            matches = re.findall(r"Average Bench OutPerf(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            for val in reversed(matches):
                try: return float(val) / 100.0
                except: continue
            return 0.0
        sharpe = extract_metric("Sharpe Ratio", output)
        down = extract_metric("Max Down", output)
        avg_benc_perf = extract_avgbenchperf(output)
        if sharpe is None or down is None:
            return -float("inf")
        score = (1 * sharpe) + (0 * avg_benc_perf) - (0.7 * abs(down))
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("down", down)
        trial.set_user_attr("avg_benc_perf", avg_benc_perf)
        trial.set_user_attr("select_feat", select_feat)
        trial.set_user_attr("select_macro", select_macros)
        trial.set_user_attr("select_TICK", select_TICK)
        return score
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")

def main():
    from load import load_config
    config = load_config()
    sampler = TPESampler(seed=config["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=50, n_jobs=1)
    best = study.best_trial
    print("\n=== Best Trial Metrics ===")
    for m in ["sharpe", "down", "avg_benc_perf"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")
    print(f"Selected feat: {best.user_attrs.get('select_feat')}")
    print(f"Selected macros: {best.user_attrs.get('select_macro')}")
    print(f"Selected TICK: {best.user_attrs.get('select_TICK')}")
    macro_counter = Counter()
    feat_counter = Counter()
    ticker_counter = Counter()
    for trial in study.trials:
        if trial.value is None or trial.value == -float("inf"): continue
        for macro in MACRO_LIST:
            if trial.params.get(f"macro_{macro}"): macro_counter[macro] += 1
        for feat in FEAT_LIST:
            if trial.params.get(f"feat_{feat}"): feat_counter[feat] += 1
        for ticker in TICKER_LIST:
            if trial.params.get(f"ticker_{ticker}"): ticker_counter[ticker] += 1

    print("\nMacro inclusion frequency:")
    for macro, count in macro_counter.most_common(): print(f"{macro}: {count}")
    print("\nFeature inclusion frequency:")
    for feat, count in feat_counter.most_common(): print(f"{feat}: {count}")
    print("\nTicker inclusion frequency:")
    for ticker, count in ticker_counter.most_common(): print(f"{ticker}: {count}")
    print("\nParam importances:")
    importance = optuna.importance.get_param_importances(study)
    for k, v in importance.items(): print(f"{k}: {v:.4f}")
    
if __name__ == "__main__":
    main()
