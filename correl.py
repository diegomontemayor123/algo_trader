import os
import pandas as pd
import numpy as np
from load import load_config
from validate import TICKER_LIST, FEAT_LIST, MACRO_LIST
from feat import load_prices, comp_feat  

def compute_feature_correlations(feat, ret, split_date, top_n=20, window=545, output_dir="correlation_test_output"):
    print(f"[INFO] Running feature correlation analysis on test period from {split_date}")
    os.makedirs(output_dir, exist_ok=True)

    split_date = pd.to_datetime(split_date)
    feat_test = feat[(feat.index >= split_date - pd.Timedelta(days=window)) & 
                     (feat.index <= split_date)]
    ret_test = ret[(ret.index >= split_date - pd.Timedelta(days=window)) & 
                   (ret.index <= split_date)]


    actual_start = feat_test.index.min().strftime("%Y-%m-%d")
    actual_end = feat_test.index.max().strftime("%Y-%m-%d")
    print(f"[INFO] Correlation calculated over period: {actual_start} to {actual_end}")

    top_features_per_asset = {}
    aggregate_corrs_by_feature = {}

    # Static correlations per asset
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

        df_sorted.to_csv(os.path.join(output_dir, f"{asset}_all_corr.csv"), index=False)

    mean_abs_corrs = {feat: np.mean(corrs) for feat, corrs in aggregate_corrs_by_feature.items()}
    sorted_mean_abs = sorted(mean_abs_corrs.items(), key=lambda x: x[1], reverse=True)

    print("\n[INFO] Aggregate mean absolute correlations per feature across all assets:")
    for feat, mean_corr in sorted_mean_abs[:top_n]:
        print(f"{feat}: {mean_corr:.4f}")
    return top_features_per_asset

def main():
    config = load_config()
    split_date = config["SPLIT"]
    start = config["START"]
    end = config["END"]
    feat_names = config["FEAT"] #FEAT_LIST
    macro_keys = config["MACRO"] #MACRO_LIST
    print("[INFO] Loading prices and computing features/returns...")
    cached_data = load_prices(start, end, macro_keys)
    feat, ret = comp_feat(TICKER_LIST, feat_names, cached_data, macro_keys)
    ret = ret.loc[feat.index]
    compute_feature_correlations(feat, ret, split_date)

if __name__ == "__main__":
    main()
