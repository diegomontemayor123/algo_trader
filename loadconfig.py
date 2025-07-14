import os
import json
import pandas as pd

def load_config():
    keys = [
        "SPLIT_DATE", "VAL_SPLIT", "PREDICT_DAYS", "LOOKBACK", "EPOCHS",
        "MAX_HEADS", "BATCH_SIZE", "FEATURES", "MAX_LEVERAGE", "LAYER_COUNT",
        "DROPOUT", "DECAY", "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED",
        "LOSS_MIN_MEAN", "LOSS_RETURN_PENALTY",
        "WARMUP_FRAC", "EARLY_STOP_PATIENCE", "INITIAL_CAPITAL", "TICKERS",
        "START_DATE", "END_DATE", "TEST_CHUNK_MONTHS", "RETRAIN_WINDOW", "MACRO"
    ]

    float_keys = {
        "VAL_SPLIT", "MAX_LEVERAGE", "DROPOUT", "DECAY", "LOSS_MIN_MEAN",
        "LOSS_RETURN_PENALTY", "WARMUP_FRAC", "INITIAL_CAPITAL"
    }
    int_keys = {
        "PREDICT_DAYS", "LOOKBACK", "EPOCHS", "MAX_HEADS", "BATCH_SIZE",
        "LAYER_COUNT", "EARLY_STOP_PATIENCE", "TEST_CHUNK_MONTHS", "RETRAIN_WINDOW"
    }
    bool_keys = {
        "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED"
    }
    list_keys = {"FEATURES", "TICKERS"}  # Removed MACRO from here
    date_keys = {"SPLIT_DATE", "START_DATE", "END_DATE"}

    def parse_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ('1', 'true', 'yes'):
                return True
            elif val_lower in ('0', 'false', 'no'):
                return False
            else:
                raise ValueError(f"Cannot parse boolean value from string '{val}'")
        return bool(val)

    def parse_value(key, val):
        if key == "MACRO":
            if isinstance(val, str):
                return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, list):
                return val
            else:
                raise ValueError(f"Expected list or comma-separated string for key '{key}', got {type(val)}")
            
        if key in float_keys:
            return float(val)
        elif key in int_keys:
            return int(val)
        elif key in bool_keys:
            return parse_bool(val)
        elif key in list_keys:
            # Accept list or comma-separated string
            if isinstance(val, str):
                return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, list):
                return val
            else:
                raise ValueError(f"Expected list or comma-separated string for key '{key}', got {type(val)}")
        elif key in date_keys:
            return pd.Timestamp(val)
        return val

    # Consider missing if not set or empty string
    missing_env_keys = [k for k in keys if not os.environ.get(k)]

    if not missing_env_keys:
        print("[Config] Loading configuration from environment variables.")
        config = {}
        for key in keys:
            val = os.environ.get(key)
            if val is None or val == "":
                raise ValueError(f"[ENV Missing] Required key: {key}")
            config[key] = parse_value(key, val)
        return config

    # Fallback: load from JSON
    if not os.path.exists("hyperparameters.json"):
        raise FileNotFoundError("Configuration file 'hyperparameters.json' not found and some environment variables missing.")

    with open("hyperparameters.json", "r") as f:
        config_raw = json.load(f)

    missing_json_keys = [k for k in keys if k not in config_raw]

    if missing_json_keys:
        raise ValueError(f"[Config Missing] Keys missing from both environment and JSON: "
                         f"{missing_env_keys + missing_json_keys}")

    print("[Config] Loading configuration from JSON file 'hyperparameters.json'.")
    config = {}
    for key in keys:
        val = os.environ.get(key, config_raw[key])
        if val == "":
            raise ValueError(f"Key '{key}' is empty in environment or JSON.")
        config[key] = parse_value(key, val)

    return config
