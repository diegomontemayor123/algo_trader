import os
import json
import pandas as pd

keys = ["SPLIT", "VAL_SPLIT", "PRED_DAYS", "LBACK", "SEED","MAX_HEADS", "BATCH", "FEAT","LAYERS",
        "DROPOUT", "DECAY", "ATTENT", "EXP_PEN","RETURN_PEN","RETURN_EXP","INIT_LR", "EXP_EXP","WARMUP", "EARLY_FAIL",
        "SD_PEN","SD_EXP", "TICK","START", "END", "TEST_CHUNK", "RETRAIN","MACRO", "FEAT_PER","FILTERWIN","THRESH","FILTERMETHOD"]

def load_config():
    float_keys = {"VAL_SPLIT", "DROPOUT", "DECAY", "RETURN_PEN","RETURN_EXP", "EXP_EXP", "WARMUP", "SD_PEN","SD_EXP","EXP_PEN","INIT_LR","THRESH"}
    int_keys = {"PRED_DAYS", "LBACK", "SEED", "MAX_HEADS", "BATCH","LAYERS", "EARLY_FAIL", "TEST_CHUNK", "RETRAIN","ATTENT","FILTERWIN"}
    list_keys = {"FEAT", "TICK", "FEAT_PER","FILTERMETHOD"} 
    date_keys = {"SPLIT", "START", "END"}

    def parse_value(key, val):
        if key == "MACRO":
            if isinstance(val, str): return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, list):return val
            else: raise ValueError(f"Expected list or comma-separated string for key '{key}', got {type(val)}")
        if key in float_keys: return float(val)
        elif key in int_keys: return int(val)
        elif key in list_keys:
            if isinstance(val, str): return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, list): return val
            else: raise ValueError(f"Expected list or comma-separated string for key '{key}', got {type(val)}")
        elif key in date_keys: return pd.Timestamp(val)
        return val
    
    miss_env_keys = [k for k in keys if not os.environ.get(k)]
    if not miss_env_keys:
        config = {}
        for key in keys:
            val = os.environ.get(key)
            if val is None or val == "": raise ValueError(f"[ENV Missing] Required key: {key}")
            config[key] = parse_value(key, val)
        return config
    if not os.path.exists("hyparams.json"): raise FileNotFoundError("Configuration file 'hyparams.json' not found and some environment variables missing.")
    with open("hyparams.json", "r") as f: config_raw = json.load(f)
    miss_json_keys = [k for k in keys if k not in config_raw]
    if miss_json_keys: raise ValueError(f"[Config Missing] Keys missing from both environment and JSON: {miss_env_keys + miss_json_keys}")
    config = {}
    for key in keys:
        val = os.environ.get(key, config_raw[key])
        config[key] = parse_value(key, val)
    return config
