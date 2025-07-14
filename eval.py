import os, sys, logging, csv
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from compute_features import load_price_data, compute_features, normalize_features
from loadconfig import load_config
from data_prep import prepare_main_datasets
from train import train_main_model
from model import load_trained_model, save_top_features_csv
from backtest import run_backtest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def compute_generalization_gap(train_losses, val_losses):
    return np.mean(val_losses) - np.mean(train_losses)

def evaluate_loss_curve(model, datasets, loss_fn):
    model.eval()
    with torch.no_grad():
        all_losses = []
        for x_batch, y_batch in datasets:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch, model)
            if loss is not None:
                all_losses.append(loss.item())
    return all_losses

def log_metrics(results):
    strategy = results["portfolio"]
    benchmark = results["benchmark"]
    print("\n[Evaluation] Strategy vs Benchmark Performance")
    print(f"Sharpe Ratio:      Strategy = {strategy['sharpe_ratio']*100:.4f}%, Benchmark = {benchmark['sharpe_ratio']*100:.4f}%")
    print(f"Max Drawdown:      Strategy = {strategy['max_drawdown']*100:.4f}%, Benchmark = {benchmark['max_drawdown']*100:.4f}%")
    print(f"CAGR:              Strategy = {strategy['cagr']*100:.4f}%, Benchmark = {benchmark['cagr']*100:.4f}%")

    print("\n[Outperformance by Period]")
    for period, value in results["performance_outperformance"].items():
        print(f"{period}: {value * 100:.4f}%")

def main():
    config = load_config()
    tickers = config["TICKERS"].split(",") if isinstance(config["TICKERS"], str) else config["TICKERS"]
    feature_list = config["FEATURES"].split(",") if isinstance(config["FEATURES"], str) else config["FEATURES"]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [m.strip() for m in macro_keys.split(",") if m.strip()]

    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, feature_list, cached_data, macro_keys)
    train_dataset, val_dataset, test_dataset = prepare_main_datasets(features, returns, config)

    model = train_main_model(config, features, returns)
    save_top_features_csv(model, features.columns.tolist())

    # Overfitting/Underfitting diagnostics
    loss_fn = lambda preds, targets, m: torch.mean((preds - targets) ** 2)
    print("\n[Diagnostics] Computing Loss Curves")
    train_losses = evaluate_loss_curve(model, train_dataset, loss_fn)
    val_losses = evaluate_loss_curve(model, val_dataset, loss_fn)
    gap = compute_generalization_gap(train_losses, val_losses)
    print(f"Generalization Gap (Val - Train): {gap:.6f}")
    if gap > 0.01:
        print("[Warning] Potential overfitting detected.")
    elif gap < -0.01:
        print("[Warning] Model might be underfitting.")
    else:
        print("[Info] Model generalization is within acceptable bounds.")

    print("\n[Backtesting] Executing Walk-Forward Simulation")
    results = run_backtest(
        device=DEVICE,
        initial_capital=config["INITIAL_CAPITAL"],
        split_date=config["SPLIT_DATE"],
        lookback=config["LOOKBACK"],
        max_leverage=config["MAX_LEVERAGE"],
        compute_features=compute_features,
        normalize_features=normalize_features,
        tickers=tickers,
        start_date=config["START_DATE"],
        end_date=config["END_DATE"],
        features=feature_list,
        macro_keys=macro_keys,
        test_chunk_months=config["TEST_CHUNK_MONTHS"],
        model=model,
        plot=True,
        config=config,
        retrain_window=config["RETRAIN_WINDOW"]
    )

    log_metrics(results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
