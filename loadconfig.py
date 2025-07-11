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
        "START_DATE", "END_DATE",
    ]
    float_keys = {
        "VAL_SPLIT", "MAX_LEVERAGE", "DROPOUT", "DECAY", "LOSS_MIN_MEAN",
        "LOSS_RETURN_PENALTY", "WARMUP_FRAC", "INITIAL_CAPITAL"
    }
    int_keys = {
        "PREDICT_DAYS", "LOOKBACK", "EPOCHS", "MAX_HEADS", "BATCH_SIZE",
        "LAYER_COUNT", "EARLY_STOP_PATIENCE",
    }
    bool_keys = {
        "FEATURE_ATTENTION_ENABLED", "L2_PENALTY_ENABLED", "RETURN_PENALTY_ENABLED"
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
    if any(k in os.environ for k in keys):
        config = {}
        for key in keys:
            val = os.environ.get(key)
            if val is None:
                raise ValueError(f"[ENV Missing] Required key: {key}")
            config[key] = parse_value(key, val)
        return config
    with open("hyperparameters.json", "r") as f:
        config_raw = json.load(f)
    config = {}
    for key in keys:
        if key not in config_raw:
            raise ValueError(f"[JSON Missing] Required key: {key}")
        config[key] = parse_value(key, config_raw[key])

    return config
