import os
import json
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from compute_features import compute_features, normalize_features
from walkforward import create_sequences, split_train_validation
from model import MarketDataset, create_model, train_model_with_validation, calculate_performance_metrics

# === Configuration Flags ===
LOAD_EXISTING_MODEL = True
SAVED_MODEL_PATH = "walkforward_model.pth"
SAVE_MODEL = not LOAD_EXISTING_MODEL

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def run_walkforward_test_with_validation(config):
    print("[Walk-Forward] Starting walk-forward test with validation...")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INITIAL_CAPITAL = config.get("INITIAL_CAPITAL", 100.0)

    # Feature + return computation
    TICKERS = config["TICKERS"]
    START_DATE = config["START_DATE"]
    END_DATE = config["END_DATE"]
    FEATURES = config["FEATURES"]
    features, returns = compute_features(TICKERS, START_DATE, END_DATE, FEATURES)

    features.index = pd.to_datetime(features.index)
    returns.index = pd.to_datetime(returns.index)
    all_dates = features.index

    # Walkforward settings
    SPLIT_DATE = pd.Timestamp(config["SPLIT_DATE"])
    test_start_index = all_dates.get_indexer([SPLIT_DATE], method="bfill")[0]
    step = config["WALKFWD_STEP"]
    window = config["WALKFWD_WNDW"]

    walkforward_results = []
    model_input_dim = features.shape[1]

    for i in range(test_start_index, len(all_dates) - config["LOOKBACK"] - config["PREDICT_DAYS"], step):
        train_start = all_dates[i - window] if i - window > 0 else all_dates[0]
        train_end = all_dates[i]
        test_start = all_dates[i + config["LOOKBACK"]]
        test_end = all_dates[min(i + config["LOOKBACK"] + step, len(all_dates) - 1)]

        print(f"\n[Walk-Forward] Training: {train_start.date()} to {train_end.date()}")
        print(f"[Walk-Forward] Testing: {test_start.date()} to {test_end.date()}")

        train_mask = (features.index >= train_start) & (features.index < train_end)
        train_features = features.loc[train_mask]
        train_returns = returns.loc[train_mask]

        if len(train_features) < config["LOOKBACK"] + config["PREDICT_DAYS"]:
            print("[Walk-Forward] Insufficient training data, skipping...")
            continue

        sequences, targets, _ = create_sequences(
            train_features, train_returns, 0, len(train_features),
            lookback=config["LOOKBACK"], predict_days=config["PREDICT_DAYS"]
        )
        if len(sequences) == 0:
            print("[Walk-Forward] No valid sequences created, skipping...")
            continue

        train_seq, train_tgt, val_seq, val_tgt = split_train_validation(
            sequences, targets, val_split=config["VAL_SPLIT"]
        )
        train_dataset = MarketDataset(torch.tensor(train_seq), torch.tensor(train_tgt))
        val_dataset = MarketDataset(torch.tensor(val_seq), torch.tensor(val_tgt))

        train_loader = DataLoader(train_dataset, batch_size=min(config["BATCH_SIZE"], len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(config["BATCH_SIZE"], len(val_dataset)), shuffle=False)

        if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
            print(f"[Walk-Forward] Loading model from {SAVED_MODEL_PATH}")
            trained_model = create_model(model_input_dim, **extract_model_kwargs(config))
            trained_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
            trained_model.to(DEVICE)
        else:
            model = create_model(model_input_dim, **extract_model_kwargs(config))
            trained_model = train_model_with_validation(
                model, train_loader, val_loader,
                epochs=config["EPOCHS"],
                warmup_frac=config["WARMUP_FRAC"],
                decay=config["DECAY"],
                loss_min_mean=config["LOSS_MIN_MEAN"],
                loss_return_penalty=config["LOSS_RETURN_PENALTY"],
                device=DEVICE,
            )
            if SAVE_MODEL:
                torch.save(trained_model.state_dict(), SAVED_MODEL_PATH)
                print(f"[Walk-Forward] Saved model to {SAVED_MODEL_PATH}")

        # Evaluation loop
        test_mask = (features.index >= test_start) & (features.index <= test_end)
        test_features = features.loc[test_mask]
        test_returns = returns.loc[test_mask]
        portfolio_values = [INITIAL_CAPITAL]
        trained_model.eval()

        with torch.no_grad():
            for j in range(len(test_features) - config["LOOKBACK"]):
                window = test_features.iloc[j:j + config["LOOKBACK"]].values.astype(np.float32)
                norm_window = normalize_features(window)
                input_tensor = torch.tensor(norm_window).unsqueeze(0).to(DEVICE)
                raw_weights = trained_model(input_tensor).cpu().numpy().flatten()
                weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
                scale = min(config["MAX_LEVERAGE"] / weight_sum, 1.0)
                weights = raw_weights * scale
                period_returns = test_returns.iloc[j + config["LOOKBACK"]].values
                portfolio_return = np.dot(weights, period_returns)
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        if len(portfolio_values) > 1:
            metrics = calculate_performance_metrics(portfolio_values)
            print(f"[Walk-Forward] Period Performance:")
            print(f"  CAGR: {metrics['cagr']:.2%}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            walkforward_results.append({
                'start_date': test_start,
                'end_date': test_end,
                'cagr': metrics['cagr'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
            })

    if walkforward_results:
        avg_cagr = np.mean([r['cagr'] for r in walkforward_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in walkforward_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in walkforward_results])
        print("\n[Walk-Forward] === Aggregate Results ===")
        print(f"  Average CAGR: {avg_cagr:.2%}")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  Average Max Drawdown: {avg_drawdown:.2%}")
        print(f"  Number of Periods: {len(walkforward_results)}")
    else:
        print("[Walk-Forward] No valid test periods completed.")


def extract_model_kwargs(config):
    return {
        "layer_count": config["LAYER_COUNT"],
        "dropout": config["DROPOUT"],
        "max_heads": config["MAX_HEADS"],
        "feature_attention_enabled": config.get("FEATURE_ATTENTION_ENABLED", False),
        "l2_penalty_enabled": config.get("L2_PENALTY_ENABLED", False),
        "return_penalty_enabled": config.get("RETURN_PENALTY_ENABLED", False),
    }


if __name__ == "__main__":
    with open("best_hyperparameters.json", "r") as f:
        raw_params = json.load(f)
    if isinstance(raw_params.get("FEATURES"), str):
        raw_params["FEATURES"] = [f.strip() for f in raw_params["FEATURES"].split(",") if f.strip()]
    run_walkforward_test_with_validation(raw_params)
