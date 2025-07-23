import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def compute_feature_correlations(
    feat,
    ret,
    split_date,
    top_n=20,
    output_dir="correlation_test_output"
):
    print(f"[INFO] Running feature correlation analysis on test period from {split_date}")
    os.makedirs(output_dir, exist_ok=True)

    split_date = pd.to_datetime(split_date)
    feat_test = feat[feat.index >= split_date]
    ret_test = ret[ret.index >= split_date]

    actual_start = feat_test.index.min().strftime("%Y-%m-%d")
    actual_end = feat_test.index.max().strftime("%Y-%m-%d")
    print(f"[INFO] Correlation calculated over period: {actual_start} to {actual_end}")

    top_features_per_asset = {}
    aggregate_corrs_by_feature = {}

    # === Static correlations per asset ===
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

        top_df = df_sorted.head(top_n)
        top_features_per_asset[asset] = top_df

        print(f"\n[INFO] Top {top_n} features for {asset}:")
        print(top_df[["Feature", "Correlation"]].to_string(index=False))

    # === Aggregate summary ===
    mean_abs_corrs = {feat: np.mean(corrs) for feat, corrs in aggregate_corrs_by_feature.items()}
    sorted_mean_abs = sorted(mean_abs_corrs.items(), key=lambda x: x[1], reverse=True)

    print("\n[INFO] Aggregate mean absolute correlations per feature across all assets:")
    for feat, mean_corr in sorted_mean_abs[:top_n]:
        print(f"{feat}: {mean_corr:.4f}")

    print(f"\n[INFO] Correlation analysis complete.")
    return top_features_per_asset
