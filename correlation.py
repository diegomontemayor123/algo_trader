def compute_feature_correlations(feat, ret, split_date, top_n=10, output_dir="correlation_test_output", rolling_window=60, stride=5):
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

        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_df, x="Correlation", y="Feature", hue="Feature", palette="coolwarm", orient="h", legend=False)
        plt.title(f"Top {top_n} Correlated Features for {asset}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{asset}_correlation.png"))
        plt.close()

        df_sorted.to_csv(os.path.join(output_dir, f"{asset}_all_corr.csv"), index=False)

    mean_abs_corrs = {feat: np.mean(corrs) for feat, corrs in aggregate_corrs_by_feature.items()}
    sorted_mean_abs = sorted(mean_abs_corrs.items(), key=lambda x: x[1], reverse=True)

    print("\n[INFO] Aggregate mean absolute correlations per feature across all assets:")
    for feat, mean_corr in sorted_mean_abs[:top_n]:
        print(f"{feat}: {mean_corr:.4f}")

    # === Rolling window correlation across all assets ===
    print(f"\n[INFO] Calculating rolling window correlations across ALL ASSETS (window = {rolling_window} days, stride = {stride})...")

    rolling_corrs_list = []
    range_iter = range(0, len(feat_test) - rolling_window + 1, stride)

    for feat_col in feat_test.columns:
        rolling_series = []
        for i in range_iter:
            window_feat = feat_test[feat_col].iloc[i:i+rolling_window]
            corrs = []
            for asset in ret_test.columns:
                window_ret = ret_test[asset].iloc[i:i+rolling_window]
                valid = window_feat.notna() & window_ret.notna()
                if valid.sum() < rolling_window * 0.5:
                    rolling_series.append(np.nan)
                    continue
                corr = np.corrcoef(window_feat[valid], window_ret[valid])[0, 1]
                if np.isfinite(corr):
                    corrs.append(corr)
            rolling_series.append(np.mean(corrs) if corrs else np.nan)

        # Pad to full index using stride-aware alignment
        padded_series = pd.Series(np.nan, index=feat_test.index)
        rolling_dates = feat_test.index[::stride][:len(rolling_series)]
        padded_series.loc[rolling_dates] = rolling_series
        padded_series.name = feat_col
        rolling_corrs_list.append(padded_series)

    rolling_corrs = pd.concat(rolling_corrs_list, axis=1)

    # Identify most frequent top-N features over time
    top_features_over_time = rolling_corrs.abs().apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
    top_features_flat = pd.Series([feat for sublist in top_features_over_time.dropna() for feat in sublist])

    if top_features_flat.empty:
        print("[WARN] No valid rolling correlations detected. Skipping rolling correlation plot.")
        return top_features_per_asset

    top_features_freq = top_features_flat.value_counts().head(top_n).index.tolist()

    # Debug print
    print(f"\n[DEBUG] Top features by frequency in rolling correlation:")
    for feat_col in top_features_freq:
        valid_points = rolling_corrs[feat_col].notna().sum()
        print(f"{feat_col}: {valid_points} valid points")

    # Plot only features with valid data
    top_features_freq = [c for c in top_features_freq if rolling_corrs[c].notna().sum() > 0]
    if not top_features_freq:
        print("[WARN] No top features had valid values to plot.")
        return top_features_per_asset

    # Plot rolling correlation
    plt.figure(figsize=(15, 7))
    for feat_col in top_features_freq:
        plt.plot(rolling_corrs.index, rolling_corrs[feat_col], label=feat_col)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Rolling Window ({rolling_window} days, stride = {stride}) Correlation with ALL ASSET Returns - Top {top_n} Features Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean Correlation")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ALL_ASSETS_rolling_correlation_top{top_n}.png"))
    plt.close()

    print(f"[INFO] Rolling window correlation plot saved to {output_dir}")
    print(f"\n[INFO] Correlation analysis complete. Results saved in: {output_dir}")
    return top_features_per_asset
