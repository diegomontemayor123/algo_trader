import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def compute_best_features_by_rolling_corr(
    feat: pd.DataFrame,
    ret: pd.DataFrame,
    windows: List[int] = [30, 60, 90],
    top_n: int = 10,
    plot: bool = True,
    output_dir: str = "csv/rolling_corrs"
):
    """
    Compute rolling correlations for multiple windows, then average mean absolute rolling correlation
    per feature per target to find best features.

    Args:
        feat (pd.DataFrame): Feature DataFrame indexed by date.
        ret (pd.DataFrame): Returns DataFrame indexed by date, multiple tickers as columns.
        windows (List[int]): List of rolling window sizes.
        top_n (int): Number of top features to plot per target.
        plot (bool): Whether to plot rolling correlations of top features.
        output_dir (str): Directory to save CSV outputs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.concat([feat, ret], axis=1).dropna()

    for target in ret.columns:
        if target not in df.columns:
            print(f"[Warning] Target '{target}' not found in data. Skipping.")
            continue

        # Store mean abs rolling correlations for all windows here:
        mean_abs_corrs_per_window = []

        for window in windows:
            rolling_corrs = {}
            for col in feat.columns:
                if df[col].dtype != 'object':
                    corr_series = df[col].rolling(window).corr(df[target])
                    rolling_corrs[col] = corr_series

            corr_df = pd.DataFrame(rolling_corrs)
            mean_abs_corrs = corr_df.abs().mean()
            mean_abs_corrs_per_window.append(mean_abs_corrs)

            # Save rolling correlations for this window (optional)
            output_path = f"{output_dir}/rolling_corrs_{target}_window{window}.csv"
            corr_df.to_csv(output_path)
            print(f"[Saved] Rolling correlations for {target} (window={window}) → {output_path}")

        # Average mean abs correlations across all windows
        avg_mean_abs_corr = pd.concat(mean_abs_corrs_per_window, axis=1).mean(axis=1)
        avg_mean_abs_corr = avg_mean_abs_corr.sort_values(ascending=False)

        # Save summary CSV of best features
        summary_path = f"{output_dir}/best_features_{target}.csv"
        avg_mean_abs_corr.to_csv(summary_path, header=["avg_mean_abs_rolling_corr"])
        print(f"[Saved] Best features summary for {target} → {summary_path}")

        if plot:
            top_features = avg_mean_abs_corr.head(top_n).index
            plt.figure(figsize=(14, 6))
            for col in top_features:
                # Plot rolling correlations for all windows overlaid
                for window in windows:
                    corr_series = df[col].rolling(window).corr(df[target])
                    plt.plot(corr_series.index, corr_series, label=f"{col} (window={window})")
            plt.title(f"Rolling Correlations of Top {top_n} Features with {target}")
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.ylabel("Correlation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
