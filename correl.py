import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from load import load_config
from validate import TICKER_LIST, FEAT_LIST, MACRO_LIST
from feat import load_prices, comp_feat

START = "2017-01-01"
END = "2023-01-01"

def compute_feature_correlations(feat, ret, start_date, end_date, top_n=30, output_dir="correl_output"):
    print(f"[INFO] Running feature correlation analysis on test period from {start_date} to {end_date}")
    os.makedirs(output_dir, exist_ok=True)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    feat_test = feat[(feat.index >= start_date) & (feat.index <= end_date)]
    ret_test = ret[(ret.index >= start_date) & (ret.index <= end_date)]

    actual_start = feat_test.index.min().strftime("%Y-%m-%d")
    actual_end = feat_test.index.max().strftime("%Y-%m-%d")
    print(f"[INFO] Correlation calculated over period: {actual_start} to {actual_end}")
    
    top_features_per_asset = {}
    aggregate_corrs_by_feature = {}
    
    for asset in ret_test.columns:
        asset_corrs = []
        for feat_col in feat_test.columns:
            aligned = feat_test[[feat_col]].join(ret_test[[asset]], how="inner").dropna()
            if aligned.empty: 
                continue
            corr = np.corrcoef(aligned[feat_col], aligned[asset])[0, 1]
            asset_corrs.append((feat_col, corr))
            aggregate_corrs_by_feature.setdefault(feat_col, []).append(abs(corr))
        
        df = pd.DataFrame(asset_corrs, columns=["Feature", "Correlation"])
        df["AbsCorrelation"] = df["Correlation"].abs()
        df_sorted = df.sort_values("AbsCorrelation", ascending=False)
        top_features_per_asset[asset] = df_sorted.head(top_n)
    
    mean_abs_corrs = {feat: np.mean(corrs) for feat, corrs in aggregate_corrs_by_feature.items()}
    sorted_mean_abs = sorted(mean_abs_corrs.items(), key=lambda x: x[1], reverse=True)

    print("\n[INFO] Aggregate mean absolute correlations per feature across all assets:")
    for feat, mean_corr in sorted_mean_abs[:top_n]:
        print(f"{feat}: {mean_corr:.4f}")

    # --- New Section: Calculate Cross-Correlation for Features ---
    print("\n[INFO] Calculating cross-correlation between features...")
    cross_corr_results = compute_cross_correlation(feat_test)
    
    # Output the cross-correlation results to a file
    cross_corr_results.to_csv(os.path.join(output_dir, "cross_correlation_matrix.csv"))
    print(f"[INFO] Cross-correlation matrix saved to {output_dir}/cross_correlation_matrix.csv")

    # --- Heatmap of Cross-Correlation Matrix ---
    print("\n[INFO] Generating heatmap of the cross-correlation matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_corr_results, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    heatmap_path = os.path.join(output_dir, "cross_correlation_heatmap.png")
    plt.title("Cross-Correlation Heatmap of Features")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"[INFO] Heatmap saved to {heatmap_path}")

    return top_features_per_asset, sorted_mean_abs, cross_corr_results

def compute_cross_correlation(feat_test):
    """
    Computes the cross-correlation matrix between features.
    This function computes the correlation between all pairs of features on the same dates.
    """
    corr_matrix = feat_test.corr()  # Pandas function computes Pearson correlation
    return corr_matrix

def main():
    config = load_config()
    start_date = pd.to_datetime(START)
    end_date = pd.to_datetime(END)

    feat_names = FEAT_LIST
    macro_keys = MACRO_LIST
    print("[INFO] Loading prices and computing features/returns...")
    
    cached_data = load_prices(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), macro_keys)
    
    feat, ret = comp_feat(TICKER_LIST, feat_names, cached_data, macro_keys)
    ret = ret.loc[feat.index]

    top_features_per_asset, sorted_mean_abs, cross_corr_results = compute_feature_correlations(feat, ret, start_date, end_date)

    top_30_features = [feat for feat, _ in sorted_mean_abs[:30]]
    top_macros = [f for f in top_30_features if f in MACRO_LIST]
    top_feats = [f for f in top_30_features if f not in MACRO_LIST]

    def simplify_feat_name(feat_name):
        base = re.match(r"(.+?)_(\d+|[A-Z]+)", feat_name)
        return base.group(1) if base else feat_name

    simplified_feats = set(simplify_feat_name(f) for f in top_feats)

    print(f'"MACRO": "{",".join(top_macros)}",')
    print(f'"FEAT": "{",".join(sorted(simplified_feats))}",')

    return top_macros, simplified_feats

if __name__ == "__main__":
    main()
