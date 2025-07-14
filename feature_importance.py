import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from compute_features import load_price_data, compute_features
from data_prep import prepare_main_datasets
from loadconfig import load_config
from train import train_main_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_feature_weights(model):
    """
    Extract model feature weights assuming model exposes 'feature_weights' attribute.
    """
    with torch.no_grad():
        weights = model.feature_weights.detach().cpu().numpy()
    return weights

def plot_weights_distribution(weights, save_path="plots/weights_distribution.png"):
    """
    Visualize the distribution of feature weights to assess spread and uniformity.
    """
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
    """
    Generate and save a heatmap showing pairwise absolute correlations among features.
    """
    corr = features.corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", vmax=1.0, square=True, linewidths=0.1)
    plt.title("Feature Correlation Heatmap")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def compute_sensitivity(model, sample, epsilon=1e-3):
    """
    Compute sensitivity of model output to perturbations in each feature.
    This quantifies how changes in each feature affect model predictions.
    """
    sample = sample.clone().detach().to(DEVICE)
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
    """
    Generate a comprehensive HTML report including plots and a detailed table
    for feature weights and sensitivities, sorted by importance.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights,
        "Sensitivity": sensitivities
    }).sort_values("Weight", ascending=False)

    html = df.to_html(index=False, float_format="%.6f")
    with open(save_path, "w") as f:
        f.write("<h1>Feature Importance and Sensitivity Analysis</h1>\n")
        f.write('<img src="../plots/weights_distribution.png" width="600"><br>\n')
        f.write('<img src="../plots/feature_correlation.png" width="600"><br>\n')
        f.write(html)
    print(f"[✔] HTML report saved to {save_path}")

def main():
    # Load configuration and feature set
    config = load_config()
    tickers = config["TICKERS"]
    feature_list = config["FEATURES"]
    macro_keys = config.get("MACRO", [])

    # Data loading and feature engineering
    cached_data = load_price_data(config["START_DATE"], config["END_DATE"], macro_keys)
    features, returns = compute_features(tickers, feature_list, cached_data, macro_keys)

    # Dataset preparation
    train_dataset, _, _ = prepare_main_datasets(features, returns, config)

    # Model training - returning trained model instance
    model = train_main_model(config, features, returns)
    model.to(DEVICE).eval()

    print("[⚙] Extracting feature weights and computing sensitivities...")
    weights = get_feature_weights(model)

    # Extract a representative sample from training data for sensitivity analysis
    first_batch_x, _ = train_dataset[0]
    first_batch_x = first_batch_x.unsqueeze(0).to(DEVICE)  # Add batch dimension

    sensitivities = compute_sensitivity(model, first_batch_x)

    print("[📊] Generating visualizations...")
    plot_weights_distribution(weights)
    plot_feature_correlation(features)

    print("[📝] Creating HTML report...")
    generate_html_report(features.columns.tolist(), weights, sensitivities)

if __name__ == "__main__":
    main()
