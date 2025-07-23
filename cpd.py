import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import os


def detect_change_points(series: pd.Series, model: str = "rbf", penalty: int = 10, min_size: int = 20) -> list:
    """
    Detect change points in a univariate time series using Ruptures.
    
    Parameters:
    - series: pd.Series - Time-indexed series (e.g., equity curve, returns).
    - model: str - Model type for ruptures ('l2', 'rbf', 'linear', etc.).
    - penalty: int - Penalty value to control sensitivity.
    - min_size: int - Minimum segment length between change points.
    
    Returns:
    - List of index positions where change points occur.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    series = series.dropna()
    algo = rpt.Pelt(model=model, min_size=min_size).fit(series.values)
    result = algo.predict(pen=penalty)
    return result


def plot_change_points(series: pd.Series, change_points: list, title="Change Point Detection", save_path=None):
    """
    Plot the time series and overlay the detected change points.
    
    Parameters:
    - series: pd.Series - Time series data.
    - change_points: list - List of indices or positions from CPD output.
    - title: str - Title for the plot.
    - save_path: str - If provided, saves the plot to this path.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Series must be a pandas Series.")

    plt.figure(figsize=(12, 5))
    rpt.display(series.values, change_points, figsize=(12, 4))
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def get_change_dates(series: pd.Series, change_indices: list) -> list:
    """
    Convert index positions to datetime values.
    
    Parameters:
    - series: pd.Series - Time-indexed data series.
    - change_indices: list - Indices returned from CPD.
    
    Returns:
    - List of pd.Timestamp corresponding to change points.
    """
    return [series.index[i] for i in change_indices if i < len(series)]
