import os
import subprocess
import re
import optuna
import json
from optuna.samplers import TPESampler

def run_experiment(trial):
    config = {
        "START_DATE": trial.suggest_categorical("START_DATE", ["2012-01-01","2013-01-01","2014-01-01","2015-01-01","2016-01-01","2017-01-01"]),
        "END_DATE": trial.suggest_categorical("END_DATE", ["2025-07-01"]),
        "SPLIT_DATE": trial.suggest_categorical("SPLIT_DATE", ["2021-07-01"]),
        "TICKERS": trial.suggest_categorical("TICKERS", ['ORCL,GOOGL,JPM,PFE,MSFT,T,KO,UNH,GE,CAT,TSLA,AAPL',
                                                         "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,WMT,CVX,MCD,T,NKE",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD, COST, MA, ABBV, VZ, BAC, WMT, LMT, JPM, DE",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD, COST, MA",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD, COST, MA, ABBV, VZ",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD, COST, MA, ABBV, VZ, BAC",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MCD, COST, MA, ABBV, VZ, BAC, WMT, LMT",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA, MSFT, LOW, TMO, CAT COST, MA, ABBV",
                                                         "BRK, TXN, HD, T, BA, GS, NVDA COST, MA, ABBV",
                                                         "BRK, TXN, HD, T, CAT, XOM, LLY, PEP, AVGO, MCD, COST",
                                                         "BRK, HD, T, BA, GS, NVDA, MSFT, LLY, PEP, MA, ABBV",
                                                         "MSFT, LOW, TMO, CAT, XOM, LLY, PEP, AVGO, MA, ABBV",
                                                         "HD, T, BA, GS, MSFT, LOW, TMO, XOM, LLY, PEP, COST, ABBV",
                                                         ]),
        "MACRO": trial.suggest_categorical("MACRO",['^IXIC,CL=F,GC=F,NG=F,ZW=F,USDJPY=X,^TNX,^FVX,IEF,UUP',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX, IEF, HYG, ^DJI, ^RUT, VEA, ^IXIC',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX, IEF, HYG',
                                                    '^N225, HG=F, ZC=F, TLT, ^GSPC, AUDUSD=X, CL=F, SHY, BRL=X, ^VIX, NG=F, ^FVX, UUP, SI=F, TIP, ^IRX, IEF, HYG, ^DJI, ^RUT, VEA']),
        "FEATURES": trial.suggest_categorical("FEATURES", ['price,vol,macd,ema',
                                                           'price,vol,macdcmo',
                                                           'price,vol,ema,cmo',
                                                           'price,vol,macd',
                                                           'price,vol,cmo',
                                                           'price,vol,ema',
                                                            'price,vol,ema',
                                                            'price,vol,macd,log_ret',
                                                           'ret,volume,momentum,cmo,log_ret',
                                                           "ret,vol,log_ret,rolling_ret,volume"]),
        "INITIAL_CAPITAL": trial.suggest_float("INITIAL_CAPITAL", 100.0, 100.0),
        "MAX_LEVERAGE": trial.suggest_float("MAX_LEVERAGE", 1.3, 2.0),
        "BATCH_SIZE": trial.suggest_int("BATCH_SIZE", 60, 70),
        "LOOKBACK": trial.suggest_int("LOOKBACK", 70, 75),
        "PREDICT_DAYS": trial.suggest_int("PREDICT_DAYS", 4, 8),
        "WARMUP_FRAC": trial.suggest_float("WARMUP_FRAC", 0.1, 0.3),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.001, 0.07),
        "DECAY": trial.suggest_float("DECAY", 0.0001, 0.04),
        "FEATURE_ATTENTION_ENABLED": trial.suggest_int("FEATURE_ATTENTION_ENABLED", 1, 1),
        "FEATURE_PERIODS": trial.suggest_categorical("FEATURE_PERIODS",["12,28","10,24","12,24","14,24","14,20","14,28","12,24,30","8,12,24"]),
        "L1_PENALTY": trial.suggest_float("L1_PENALTY", 1e-5, 0.001), #-0.001, 0.001
        "INIT_LR": trial.suggest_float("INIT_LR",0.5,0.5),
        "LOSS_MIN_MEAN": trial.suggest_float("LOSS_MIN_MEAN", 0.01, 0.05),
        "LOSS_RETURN_PENALTY": trial.suggest_float("LOSS_RETURN_PENALTY", 0.001, 0.1),
        "TEST_CHUNK_MONTHS": trial.suggest_int("TEST_CHUNK_MONTHS", 12, 12),
        "RETRAIN_WINDOW": trial.suggest_int("RETRAIN_WINDOW", 0, 0),
        "EPOCHS": trial.suggest_int("EPOCHS", 20, 20),
        "MAX_HEADS": trial.suggest_int("MAX_HEADS", 20, 20),
        "LAYER_COUNT": trial.suggest_int("LAYER_COUNT", 6, 6),
        "EARLY_STOP_PATIENCE": trial.suggest_int("EARLY_STOP_PATIENCE", 5, 5),
        "VAL_SPLIT": trial.suggest_float("VAL_SPLIT", 0.1, 0.1),
    }

    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    try:
        result = subprocess.run(["python", "model.py"],capture_output=True,text=True,env=env,timeout=1800)
        output = result.stdout + result.stderr
        with open("tune_output.log", "a") as f:
            f.write("\n\n=== Trial output start ===\n")
            f.write(output)
            f.write("\n=== Trial output end ===\n")
        print(f"[Subprocess output]\n{output}\n")
        def extract_metric(label, out):
            match = re.search(rf"{label}:\s*Strategy:\s*([-+]?\d*\.\d+|\d+)%", out)
            return float(match.group(1)) / 100 if match else None
        def extract_avg_benchmark_outperformance(output):
            single_line_match = re.findall(r"Average Benchmark Outperformance(?: Across Chunks)?:\s*([-+]?\d*\.\d+|\d+)%", output)
            if single_line_match:
                for val in reversed(single_line_match):
                    try:
                        return float(val) / 100.0
                    except:
                        pass
            multiline_match = re.search(r"Average Benchmark Outperformance Across Chunks:\s*\ncagr:\s*([-+]?\d*\.\d+|\d+)%", output, re.MULTILINE)
            if multiline_match:
                try:
                    return float(multiline_match.group(1)) / 100.0
                except:
                    return 0.0
            return 0.0
        sharpe = extract_metric("Sharpe Ratio", output)
        drawdown = extract_metric("Max Drawdown", output)
        avg_benchmark_outperformance = extract_avg_benchmark_outperformance(output)
        if sharpe is None or drawdown is None:
            return -float("inf")
        score = (1 * sharpe) - (0.5 * abs(drawdown)) +(0 * avg_benchmark_outperformance) 
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("drawdown", drawdown)
        trial.set_user_attr("avg_benchmark_outperformance", avg_benchmark_outperformance)
        return score
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Trial failed for config: {config}")
        return -float("inf")

def main():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(run_experiment, n_trials=50, n_jobs=1)
    best = study.best_trial
    best_params = best.params.copy()
    with open("hyparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("\n=== Best trial parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    for m in ["sharpe", "drawdown", "avg_benchmark_outperformance"]:
        print(f"{m}: {best.user_attrs.get(m, float('nan')):.4f}")

if __name__ == "__main__":
    main()

