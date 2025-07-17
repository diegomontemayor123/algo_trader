import numpy as np
import logging
import pandas as pd

def calculate_performance_metrics(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        logging.warning("[Performance] Not enough data points to calculate metrics.")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    returns = equity_curve.pct_change().dropna()
    if returns.empty or equity_curve.iloc[0] <= 0:
        logging.warning("[Performance] Invalid returns or initial capital.")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(returns) / 252
    if years <= 0:
        logging.warning("[Performance] Invalid time span (years <= 0).")
        return {'cagr': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    try:
        cagr = total_return ** (1 / years) - 1
    except Exception as e:
        logging.error(f"[Performance] Error calculating CAGR: {e}")
        cagr = 0.0
    std_returns = returns.std()
    if std_returns == 0 or np.isnan(std_returns):
        logging.warning("[Performance] Std dev of returns is zero or NaN — Sharpe set to 0")
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = returns.mean() / std_returns * np.sqrt(252)
    peak_values = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak_values) / peak_values
    max_drawdown = drawdowns.min()
    return {'cagr': float(cagr),'sharpe_ratio': float(sharpe_ratio),'max_drawdown': float(max_drawdown)}
