import os
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
start = '2015-01-01'
end = "2024-01-01"
filename = "price_data.csv"

leverage = 1.0
allow_short = False
initial_cash = 100

# === Data Load (Cached) ===
if os.path.exists(filename):
    prices = pd.read_csv(filename, index_col=0, parse_dates=True).round(2)
else:
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"].round(2)
    prices.to_csv(filename)

# === Equal-weighted Benchmark ===
equal_weights = np.ones(len(tickers)) / len(tickers)
daily_returns = prices.pct_change().fillna(0)
benchmark_returns = daily_returns.dot(equal_weights)
benchmark_equity = (1 + benchmark_returns).cumprod() * initial_cash
benchmark_equity.name = "Benchmark"

# === Returns and Covariance ===
daily_ret = prices.pct_change()
expected_ret = daily_ret.rolling(21).mean().rolling(126).mean().dropna()
cov = daily_ret.rolling(252).cov().dropna()
valid_dates = expected_ret.index.intersection(cov.index.get_level_values(0).unique())

# === Optimization ===
def neg_sharpe(w, mu, Sigma):
    port_ret = w @ mu
    port_vol = np.sqrt(w @ Sigma.values @ w)
    return -port_ret / port_vol if port_vol != 0 else np.inf

def optimize_weights(mu, Sigma):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, leverage)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - leverage}]
    res = minimize(neg_sharpe, w0, args=(mu.values, Sigma), bounds=bounds, constraints=constraints)
    return res.x if res.success else w0

# === Backtest ===
weight_list = []
for date in valid_dates:
    Sigma = cov.loc[date]
    if not isinstance(Sigma, pd.DataFrame):
        continue
    Sigma = Sigma.loc[tickers, tickers]
    mu = expected_ret.loc[date, tickers]
    if Sigma.shape != (len(tickers), len(tickers)):
        continue
    if mu.isnull().any() or Sigma.isnull().values.any():
        continue
    w = optimize_weights(mu, Sigma)
    weight_list.append(pd.Series(w, index=tickers, name=date))
weights = pd.DataFrame(weight_list).sort_index().ffill().fillna(0).round(4)
def simulate_cash_portfolio(weights, prices, initial_cash=100):
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = (weights.shift(1).fillna(0) * daily_returns).sum(axis=1)
    equity = (1 + strategy_returns).cumprod() * initial_cash
    return equity.rename("strategy_equity")
equity = simulate_cash_portfolio(weights, prices, initial_cash).round(4)

# === Metrics & Output ===
def compute_metrics(equity):
    returns = equity.pct_change().dropna().squeeze()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annualized_return = float((1 + total_return) ** (252 / len(returns)) - 1)
    annualized_vol = float(returns.std() * np.sqrt(252))
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    max_drawdown = float((equity / equity.cummax() - 1).min())
    return {"total_return": total_return,"annualized_return": annualized_return,"annualized_vol": annualized_vol,
        "sharpe": sharpe,"max_drawdown": max_drawdown}

plt.figure(figsize=(12, 6))
plt.plot(equity, label="Strategy")
plt.plot(benchmark_equity, label="Benchmark")
plt.title("Equity Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("equity_curve.png")
plt.show()
print("\n--- Performance Summary ---")
strat_metrics = compute_metrics(equity)
bench_metrics = compute_metrics(benchmark_equity)

for k in strat_metrics:
    print(f"{k.capitalize().replace('_', ' ')}:")
    print(f"  Strategy: {strat_metrics[k]:.2%}")
    print(f"  Benchmark: {bench_metrics[k]:.2%}")
