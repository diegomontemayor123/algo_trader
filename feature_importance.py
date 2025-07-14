import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from compute_features import load_price_data, compute_features
from data_prep import prepare_main_datasets
from loadconfig import load_config
from model import load_trained_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_feature_weights(model):
    with torch.no_grad():
        weights = model.feature_weights.detach().cpu().numpy()
    return weights

def plot_weights_distribution(weights, save_path="plots/weights_distribution.png"):
    plt.figure(figsize=(10, 5))
    plt.hist(weights, bins=30, edgecolor='black')
    plt.title("Distribution of Feature Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_correlation(features, save_path="plots/feature_correlation.png"):
    corr = features.corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", vmax=1.0, square=True, linewidths=0.1)
    plt.title("Feature Correlation Heatmap")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def compute_sensitivity(model, sample, epsilon=1e-3):
    sample = sample.clone().detach().requires_grad_(False).to(DEVICE)
    baseline_output = model(sample).detach().cpu().numpy()
    sensitivities = []

    for i in range(sample.shape[2]):
        perturbed = sample.clone()
        perturbed[:, :, i] += epsilon
        perturbed_output = model(perturbed).detach().cpu().numpy()
        delta = np.mean(np.abs(perturbed_output - baseline_output))
        sensitivities.append(delta)

    return sensitivities

def generate_html_report(feature_names, weights, sensitivities, save_path="report/feature_analysis.html"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights,
        "Sensitivity": sensitivities
    }).sort_values("Weight", ascending=False)

    html = df.to_html(index=False, float_format="%.6f")
    with open(save_path, "w") as f:
        f.write("<h1>Feature Importance and Sensitivity Analysis</h1>\n")
        f.write('<img src="plots/weights_distribution.png" width="600"><br>\n')
        f.write('<img src="plots/feature_correlation.png" width="600"><br>\n')
        f.write(html)
    print(f"[✔] HTML report saved to {save_path}")

def main():
    config = load_config()
    tickers = config["TICKERS"]
    feature_list = config["FEATURES"]
    macro_keys = config.get("MACRO", [])

    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, feature_list, cached_data, macro_keys)
    train_dataset, _, _ = prepare_main_datasets(features, returns, config)

    model = load_trained_model(features.shape[1], config)

    print("[⚙] Extracting feature weights and sensitivities...")
    weights = get_feature_weights(model)
    first_batch_x, _ = train_dataset[0]
    first_batch_x = first_batch_x.unsqueeze(0)  # Shape: (1, seq_len, input_dim)
    sensitivities = compute_sensitivity(model, first_batch_x)

    print("[📊] Plotting distributions...")
    plot_weights_distribution(weights)
    plot_feature_correlation(features)

    print("[📝] Generating HTML report...")
    generate_html_report(features.columns.tolist(), weights, sensitivities)

if __name__ == "__main__":
    main()
