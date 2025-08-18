

def run_experiment(trial,study=None):
    config = {
    "START": trial.suggest_categorical("START", ["2012-01-01"]),
    "END": trial.suggest_categorical("END", ["2023-01-01"]),
    "SPLIT": trial.suggest_categorical("SPLIT", ["2017-01-01"]),
    "TICK": trial.suggest_categorical(
        "TICK",
        ["JPM, MSFT, NVDA, AVGO, LLY, COST, MA, XOM, UNH, AMZN, CAT, ADBE"]
    ),
    "MACRO": trial.suggest_categorical(
        "MACRO",
        ["^GSPC,CL=F,SI=F,NG=F,HG=F,ZC=F,^IRX,TLT,IEF,UUP,HYG,EEM,VEA,FXI,^RUT,^FTSE,^TYX,AUDUSD=X,USDJPY=X,EURUSD=X,GBPUSD=X,ZW=F,GC=F"]
    ),
    "FEAT": trial.suggest_categorical(
        "FEAT",
        ["ret,price,logret,rollret,sma,ema,momentum,macd,pricevshigh,vol,atr,range,volchange,volptile,zscore,rsi,cmo,williams,stoch,priceptile,adx,meanabsret,boll,donchian,volume,lag,retcrossz,crossmomentumz,crossvolz,crossretrank"]
    ),
    "YWIN": trial.suggest_int("YWIN", 20, 22),
    "PRUNEWIN": trial.suggest_int("PRUNEWIN", 21, 24),
    "PRUNEDOWN": trial.suggest_float("PRUNEDOWN", 1.3204761367650948, 1.4434385633421247),
    "THRESH": trial.suggest_int("THRESH", 175, 175),
    "NESTIM": trial.suggest_int("NESTIM", 192, 192),
    "BATCH": trial.suggest_int("BATCH", 48, 59),
    "LBACK": trial.suggest_int("LBACK", 81, 93),
    "PRED_DAYS": trial.suggest_int("PRED_DAYS", 5, 6),
    "DROPOUT": trial.suggest_float("DROPOUT", 0.03533460358548267, 0.03869787539579138, log=True),
    "DECAY": trial.suggest_float("DECAY", 0.003190861119627664, 0.0034942679459913813, log=True),
    "SHORT_PER": trial.suggest_int("SHORT_PER", 12, 15),
    "MED_PER": trial.suggest_int("MED_PER", 19, 23),
    "LONG_PER": trial.suggest_int("LONG_PER", 68, 78),
    "INIT_LR": trial.suggest_float("INIT_LR", 0.004, 0.006367100708891438, log=True),
    "EXP_PEN": trial.suggest_float("EXP_PEN", 0.23178869944829034, 0.23728392297076622),
    "EXP_EXP": trial.suggest_float("EXP_EXP", 1.8, 1.8),
    "RETURN_PEN": trial.suggest_float("RETURN_PEN", 0.073, 0.073),
    "RETURN_EXP": trial.suggest_float("RETURN_EXP", 0.28, 0.28),
    "SD_PEN": trial.suggest_float("SD_PEN", 0.17, 0.17),
    "SD_EXP": trial.suggest_float("SD_EXP", 0.76, 0.76),
    "Z_ALPHA": trial.suggest_float("Z_ALPHA", 0.66, 0.85),
    "SEED": trial.suggest_categorical("SEED", [42]),
    "MAX_HEADS": trial.suggest_categorical("MAX_HEADS", [1,2]),
    "LAYERS": trial.suggest_int("LAYERS", 1,6),
    "EARLY_FAIL": trial.suggest_int("EARLY_FAIL", 2, 5),
    "VAL_SPLIT": trial.suggest_categorical("VAL_SPLIT", [0.15]),
    "TEST_CHUNK": trial.suggest_categorical("TEST_CHUNK", [12]),
    "ATTENT": trial.suggest_categorical("ATTENT", [1])
}



