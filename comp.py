import torch
import numpy as np
import pandas as pd

from feat import load_prices, comp_feat, norm_feat
from prep import prep_data
from model import create_model
from train import train_model, train
from load import load_config

def compare_tensor(name, t1, t2):
    if not torch.allclose(t1, t2, rtol=1e-5, atol=1e-7):
        diff = (t1 - t2).abs().mean().item()
        print(f"[DIFF] {name}: mean abs diff = {diff:.6f}")
    else:
        print(f"[OK] {name}: identical")

def test_consistency(chunk_start, chunk_end):
    config = load_config()

    config_chunk = config.copy()
    config_chunk["START"] = str(chunk_start)
    config_chunk["END"] = str(chunk_end)
    config_chunk["SPLIT"] = str(chunk_end)

    # ----- Common Setup ----- #
    TICK = config["TICK"].split(",") if isinstance(config["TICK"], str) else config["TICK"]
    FEAT = config["FEAT"].split(",") if isinstance(config["FEAT"], str) else config["FEAT"]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [k.strip() for k in macro_keys.split(",") if k.strip()]

    SEED = config["SEED"]
    LOOKBACK = config["LBACK"]

    # ----- Load Data ----- #
    data_chunk = load_prices(str(chunk_start), str(chunk_end), macro_keys)
    feat_chunk, ret_chunk = comp_feat(TICK, FEAT, data_chunk, macro_keys)
    feat_norm_chunk = norm_feat(feat_chunk)

    # ----- Prep and Train: Chunked ----- #
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_chunk = train(config_chunk, feat_norm_chunk, ret_chunk)

    # ----- Prep and Train: Standalone (Same Dates) ----- #
    config_standalone = config.copy()
    config_standalone["START"] = str(chunk_start)
    config_standalone["END"] = str(chunk_end)
    config_standalone["SPLIT"] = str(chunk_end)

    data_standalone = load_prices(str(chunk_start), str(chunk_end), macro_keys)
    feat_standalone, ret_standalone = comp_feat(TICK, FEAT, data_standalone, macro_keys)
    feat_norm_standalone = norm_feat(feat_standalone)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_standalone = train(config_standalone, feat_norm_standalone, ret_standalone)

    # ----- Compare Model Weights ----- #
    print("\n=== MODEL WEIGHT COMPARISON ===")
    for (k1, v1), (k2, v2) in zip(model_chunk.state_dict().items(), model_standalone.state_dict().items()):
        if not torch.allclose(v1, v2, rtol=1e-5, atol=1e-7):
            print(f"[DIFF] {k1}: mean abs diff = {(v1 - v2).abs().mean().item():.6f}")
        else:
            print(f"[OK] {k1}")

    # ----- Compare Predictions on Same Input ----- #
    print("\n=== WEIGHT PREDICTION COMPARISON ===")
    last_feat = feat_norm_chunk.iloc[-LOOKBACK:]
    input_tensor = torch.tensor(last_feat.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_chunk = model_chunk(input_tensor)
        pred_stand = model_standalone(input_tensor)
        compare_tensor("Predicted Weights", pred_chunk, pred_stand)

def main():
    # Example period: July 2021 to June 2022
    test_consistency(chunk_start="2021-07-01", chunk_end="2022-06-30")

if __name__ == "__main__":
    main()
