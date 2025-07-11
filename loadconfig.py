import os
import json
import pandas as pd

def load_config():
    keys = [
        "SPLIT_DATE", "VAL_SPLIT", "PREDICT_DAYS", "LOOKBACK", "EPOCHS",
        "MAX_HEADS", "BATCH_SIZE", "FEATURES", "MAX_LEVERAGE", "LAYER_COUNT",
        "DROPOUT", "DECAY", "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED",
        "RETURN_PENALTY_ENABLED", "LOSS_MIN_MEAN", "LOSS_RETURN_PENALTY",
        "WARMUP_FRAC", "EARLY_STOP_PATIENCE", "INITIAL_CAPITAL", "TICKERS",
        "START_DATE", "END_DATE", "TEST_CHUNK_MONTHS", "RETRAIN"
    ]

    float_keys = {
        "VAL_SPLIT", "MAX_LEVERAGE", "DROPOUT", "DECAY", "LOSS_MIN_MEAN",
        "LOSS_RETURN_PENALTY", "WARMUP_FRAC", "INITIAL_CAPITAL"
    }
    int_keys = {
        "PREDICT_DAYS", "LOOKBACK", "EPOCHS", "MAX_HEADS", "BATCH_SIZE",
        "LAYER_COUNT", "EARLY_STOP_PATIENCE", "TEST_CHUNK_MONTHS"
    }
    bool_keys = {
        "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED", "RETURN_PENALTY_ENABLED", "RETRAIN"
    }
    list_keys = {"FEATURES", "TICKERS"}
    date_keys = {"SPLIT_DATE", "START_DATE", "END_DATE"}

    def parse_value(key, val):
        if key in float_keys:
            return float(val)
        elif key in int_keys:
            return int(val)
        elif key in bool_keys:
            return bool(int(val))
        elif key in list_keys:
            return val.split(",") if isinstance(val, str) else val
        elif key in date_keys:
            return pd.Timestamp(val)
        return val

    # Identify missing keys in environment variables
    missing_env_keys = [k for k in keys if k not in os.environ]

    if not missing_env_keys:
        # All keys present in environment, parse from env
        print("[Config] Loading configuration from environment variables.")
        config = {}
        for key in keys:
            val = os.environ.get(key)
            if val is None:
                raise ValueError(f"[ENV Missing] Required key: {key}")
            config[key] = parse_value(key, val)
        return config

    # Fallback: Load from JSON file
    if not os.path.exists("hyperparameters.json"):
        raise FileNotFoundError("Configuration file 'hyperparameters.json' not found and some environment variables missing.")

    with open("hyperparameters.json", "r") as f:
        config_raw = json.load(f)

    # Identify missing keys in JSON as well
    missing_json_keys = [k for k in keys if k not in config_raw]

    if missing_json_keys:
        raise ValueError(f"[Config Missing] Keys missing from both environment and JSON: "
                         f"{missing_env_keys + missing_json_keys}")

    print("[Config] Loading configuration from JSON file 'hyperparameters.json'.")
    config = {}
    for key in keys:
        # Prefer env var if exists, else use JSON value
        val = os.environ.get(key, config_raw[key])
        config[key] = parse_value(key, val)

    return config
