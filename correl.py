import os, re
import pandas as pd
import numpy as np
from load import load_config
from validate import TICKER_LIST, FEAT_LIST, MACRO_LIST
from feat import load_prices, comp_feat  

def compute_feature_correlations(feat, ret, split_date, top_n=30, window=400, output_dir="correl_output"):
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

    return top_features_per_asset, sorted_mean_abs  # <-- Return both

def main():
    config = load_config()
    split_date = pd.to_datetime(config["SPLIT"])
    start = pd.to_datetime(config["START"])
    end = pd.to_datetime(config["END"])
    window = 365  # Or load from config if you want

    # Calculate actual_start as split_date - window days, to avoid skewing
    actual_start = max(start, split_date - pd.Timedelta(days=window))

    feat_names = FEAT_LIST
    macro_keys = MACRO_LIST
    print("[INFO] Loading prices and computing features/returns...")
    
    cached_data = load_prices(actual_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), macro_keys)
    
    feat, ret = comp_feat(TICKER_LIST, feat_names, cached_data, macro_keys)
    ret = ret.loc[feat.index]

    top_features_per_asset, sorted_mean_abs = compute_feature_correlations(feat, ret, split_date, window=window)

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
