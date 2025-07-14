import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from model import create_model, save_top_features_csv
from compute_features import load_price_data, compute_features, normalize_features
from data_prep import prepare_main_datasets
from loadconfig import load_config
from train import train_model_with_validation
from torch.utils.data import DataLoader
import logging


def evaluate_feature_importance(model, feature_names, top_k=30, filepath="top_features_eval.csv"):
    print("[Eval] Saving top features...")
    save_top_features_csv(model, feature_names, filepath=filepath, top_k=top_k)
    weights = model.feature_weights.detach().cpu().numpy()
    for i, (name, weight) in enumerate(sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)[:top_k]):
        print(f"{i+1:02d}. {name:40s} {weight:.4f}")


def run_feature_reduction_analysis(features):
    print("[Eval] Running PCA analysis...")
    pca = PCA()
    pca.fit(features)
    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(explained, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_variance.png")
    plt.close()
    print("[Eval] PCA curve saved to pca_variance.png")


def evaluate_target_distribution(returns):
    print("[Eval] Plotting target return distribution...")
    flat_returns = returns.values.flatten()
    flat_returns = flat_returns[~np.isnan(flat_returns)]
    plt.figure(figsize=(8, 4))
    plt.hist(flat_returns, bins=100, alpha=0.7)
    plt.title("Distribution of Future Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("target_distribution.png")
    plt.close()
    print("[Eval] Return distribution saved to target_distribution.png")


def estimate_sharpe_on_raw_weights(model, dataset):
    print("[Eval] Estimating Sharpe on raw weights...")
    model.eval()
    all_returns = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for x, y in loader:
        with torch.no_grad():
            x = x.to(next(model.parameters()).device)
            y = y.to(x.device)
            w = model(x)
            w = w * (1.0 / (torch.sum(torch.abs(w), dim=1, keepdim=True) + 1e-6))
            returns = (w * y).sum(dim=1)
            all_returns.append(returns.cpu().numpy())
    all_returns = np.concatenate(all_returns)
    sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-6)
    print(f"[Eval] Raw weight Sharpe estimate: {sharpe:.4f}")
    return sharpe


def ablation_study(model, features, returns, config, feature_names, retrain=False):
    print("[Eval] Running ablation study...")
    weights = model.feature_weights.detach().cpu().numpy()
    feature_weight_df = pd.DataFrame({'feature': feature_names, 'weight': weights})
    feature_weight_df['abs_weight'] = feature_weight_df['weight'].abs()
    sorted_features = feature_weight_df.sort_values(by='abs_weight', ascending=False)['feature'].tolist()

    keep_counts = [int(len(sorted_features) * r) for r in [1.0, 0.75, 0.5, 0.25]]
    results = {}

    for keep_n in keep_counts:
        keep_set = set(sorted_features[:keep_n])
        reduced_features = features[[f for f in features.columns if f in keep_set]]
        print(f"[Ablation] Evaluating top {keep_n} features...")
        train_set, val_set, test_set = prepare_main_datasets(reduced_features, returns, config)
        if retrain:
            model = create_model(train_set[0][0].shape[1], config)
            model = train_model_with_validation(model, DataLoader(train_set, batch_size=config["BATCH_SIZE"]), DataLoader(val_set, batch_size=config["BATCH_SIZE"]), config)
        sharpe = estimate_sharpe_on_raw_weights(model, test_set)
        results[f"Top {keep_n}"] = sharpe

    print("\n[Ablation Results]")
    for k, v in results.items():
        print(f"{k:12s}: Sharpe {v:.4f}")


def evaluate_learning(retrain=False):
    config = load_config()
    tickers = config["TICKERS"] if isinstance(config["TICKERS"], list) else [x.strip() for x in config["TICKERS"].split(",")]
    features_list = config["FEATURES"] if isinstance(config["FEATURES"], list) else [x.strip() for x in config["FEATURES"].split(",")]
    macro_keys = config.get("MACRO", [])
    if isinstance(macro_keys, str):
        macro_keys = [x.strip() for x in macro_keys.split(",") if x.strip()]

    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, features_list, cached_data, macro_keys)

    train_set, val_set, test_set = prepare_main_datasets(features, returns, config)
    feature_names = features.columns.tolist()

    model = create_model(train_set[0][0].shape[1], config)
    model = train_model_with_validation(model, DataLoader(train_set, batch_size=config["BATCH_SIZE"]), DataLoader(val_set, batch_size=config["BATCH_SIZE"]), config) if retrain else model

    evaluate_feature_importance(model, feature_names)
    run_feature_reduction_analysis(features)
    evaluate_target_distribution(returns)
    estimate_sharpe_on_raw_weights(model, test_set)
    ablation_study(model, features, returns, config, feature_names, retrain=retrain)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TransformerTrader model learning behavior")
    parser.add_argument("--retrain", action="store_true", help="Toggle to retrain the model during ablation")
    args = parser.parse_args()
    evaluate_learning(retrain=args.retrain)
