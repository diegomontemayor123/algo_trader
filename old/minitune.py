import os, json, re, copy, optuna, subprocess, math
from optuna.samplers import TPESampler
from load import load_config
import pandas as pd

TRIALS = 7
BASE_CONFIG_PATH = "hyparams.json" 
def load_base_config():
    with open(BASE_CONFIG_PATH, "r") as f: return json.load(f)

def is_duplicate_trial(study, params):
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        if t.params == params:print(f"[DUPLICATE] Trial #{t.number} already ran. Skipping.");return t.value
    return None

def run_chunk_tune(trial, start_date, end_date, split_date, config_base, study=None):

    config = copy.deepcopy(config_base)
    config.update({
"START": str(start_date.date()),
"END": str(end_date.date()),
"SPLIT": str(split_date.date()),
"SEED": trial.suggest_categorical("SEED", [config_base["SEED"]]),

"YWIN": trial.suggest_int("YWIN", max(5, math.floor(config_base["YWIN"] * 0.9)),math.ceil(config_base["YWIN"] * 1.1)),
"PRUNEWIN": trial.suggest_int("PRUNEWIN", max(5, math.floor(config_base["PRUNEWIN"] * 0.9)),math.ceil(config_base["PRUNEWIN"] * 1.1)),
"PRUNEDOWN": trial.suggest_float("PRUNEDOWN", math.floor(config_base["PRUNEDOWN"] * 0.9),math.ceil(config_base["PRUNEDOWN"] * 1.1)),
"THRESH": trial.suggest_int("THRESH", max(10, math.floor(config_base["THRESH"] * 0.9)),math.ceil(config_base["THRESH"] * 1.1)),
"DECAY": trial.suggest_float("DECAY", max(1e-8, config_base["DECAY"] * 0.85),config_base["DECAY"] * 1.15,log=False),
"DROPOUT": trial.suggest_float("DROPOUT", max(0.0, config_base["DROPOUT"] * 0.65),min(1.0, config_base["DROPOUT"] * 1.35)),
"INIT_LR": trial.suggest_float("INIT_LR", config_base["INIT_LR"] * 0.25, config_base["INIT_LR"] * 4,log=False),
"EXP_PEN": trial.suggest_float("EXP_PEN", config_base["EXP_PEN"] * 0.85,config_base["EXP_PEN"] * 1.15),
"RETURN_PEN": trial.suggest_float("RETURN_PEN", config_base["RETURN_PEN"] * 0.8, config_base["RETURN_PEN"] * 1.2),
"LAYERS": trial.suggest_int("LAYERS", max(1, config_base["LAYERS"] - 1),min(3, config_base["LAYERS"] + 1)),
"MAX_HEADS": trial.suggest_int("MAX_HEADS", max(1, config_base["MAX_HEADS"] - 1),min(3, config_base["MAX_HEADS"] + 1)),
"EARLY_FAIL": trial.suggest_int("EARLY_FAIL", max(1, config_base["EARLY_FAIL"] - 2),min(4, config_base["EARLY_FAIL"] + 2)),
"NESTIM": trial.suggest_int("NESTIM", max(10, math.floor(config_base["NESTIM"] * 1)),math.ceil(config_base["NESTIM"] * 1)),
"BATCH": trial.suggest_categorical("BATCH", [max(1, int(config_base["BATCH"] * 1)),config_base["BATCH"], int(config_base["BATCH"] * 1)]),
"LBACK": trial.suggest_int("LBACK", max(5, math.floor(config_base["LBACK"] * 1)),math.ceil(config_base["LBACK"] * 1)),
"PRED_DAYS": trial.suggest_int("PRED_DAYS", max(1, config_base["PRED_DAYS"] - 0),config_base["PRED_DAYS"] + 0)})
    dup_value = is_duplicate_trial(study, config)
    if dup_value is not None:return dup_value
    try:
        env = os.environ.copy()
        for k, v in config.items():env[k] = str(v)
        python_exe = os.path.join(env.get("VIRTUAL_ENV", ""), "Scripts", "python.exe") if "VIRTUAL_ENV" in env else "python"
        result = subprocess.run([python_exe, "model.py"], capture_output=True, text=True, env=env, timeout=1800)
        output = result.stdout + result.stderr
        print(f"[Tune] Raw output:\n{output}")
        if "KILLRUN" in output or not output.strip(): print("[KILLRUN] or empty output — skipping.");return -float("inf")
        def extract_metric(label):
            pattern = rf"{re.escape(label)}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%"
            match = re.search(pattern, output)
            return float(match.group(1)) / 100 if match else None
        sharpe = extract_metric("Sharpe Ratio");down = extract_metric("Max Down");cagr = extract_metric("CAGR")
        if None in (sharpe, down, cagr): print("[Tune] Missing metrics, skipping.");return -float("inf")
        score = 1.0 * sharpe - 6.0 * abs(down) + 1.0 * cagr
        print(f"[Tune] Trial score: {score:.4f} | Sharpe: {sharpe:.3f}, Down: {down:.3f}, CAGR: {cagr:.3f}")
        trial.set_user_attr("score", score)
        return score
    except subprocess.TimeoutExpired:print("[Tune] Timeout — skipping trial.");return -float("inf")
    except Exception as e:print(f"[Tune] Exception: {e}");return -float("inf")

def minitune_for_chunk(chunk_start, config_base=None):
    config_base = config_base or load_base_config()
    split_date = pd.to_datetime(chunk_start) - pd.DateOffset(years=1)
    end_date = pd.to_datetime(chunk_start) - pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=5)
    sampler = TPESampler(seed=config_base["SEED"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: run_chunk_tune(trial, start_date, end_date, split_date, config_base, study),n_trials=TRIALS,n_jobs=1)
    best_config = copy.deepcopy(config_base)
    best_config.update(study.best_params)
    print("\n[MiniTune] Best chunk config:")
    for k, v in best_config.items():print(f"  {k}: {v}")
    with open(BASE_CONFIG_PATH, "w") as f: json.dump(best_config, f, indent=4)
    return best_config

def main():
    config = load_config()
    chunk_start = config["SPLIT"]
    print(f"Starting mini-tuning process for chunk starting at {chunk_start}...")
    best_config = minitune_for_chunk(chunk_start)
    print("Mini-tuning complete. Best configuration saved.")

if __name__ == "__main__":
    main()
