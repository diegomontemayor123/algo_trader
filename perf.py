import numpy as np
import pandas as pd

def calc_perf_metrics(eq_curve):
    eq_curve = pd.Series(eq_curve).dropna()
    if len(eq_curve) < 2:
        print("Not enough data points to calc metrics.")
        return {'cagr': 0.0, 'sharpe': 0.0, 'max_down': 0.0}
    ret = eq_curve.pct_change().dropna()
    if ret.empty or eq_curve.iloc[0] <= 0:
        print("Invalid ret or initial capital.")
        return {'cagr': 0.0, 'sharpe': 0.0, 'max_down': 0.0}
    total_ret = eq_curve.iloc[-1] / eq_curve.iloc[0]
    years = len(ret) / 252
    if years <= 0:
        print("Invalid time span (years <= 0).")
        return {'cagr': 0.0, 'sharpe': 0.0, 'max_down': 0.0}
    try:cagr = total_ret ** (1 / years) - 1
    except Exception as e:
        print(f"Error calcing CAGR: {e}")
        cagr = 0.0
    std_ret = ret.std()
    if std_ret == 0 or np.isnan(std_ret):
        print("SD of ret is zero or NaN — Sharpe set to 0")
        sharpe = 0.0
    else:
        sharpe = ret.mean() / std_ret * np.sqrt(252)
    peak_values = np.maximum.accumulate(eq_curve)
    downs = (eq_curve - peak_values) / peak_values
    max_down = downs.min()
    return {'cagr': float(cagr),'sharpe': float(sharpe),'max_down': float(max_down)}
