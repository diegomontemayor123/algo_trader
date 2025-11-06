# Algo Trader

## Overview

**Algo Trader** is an AI-driven algorithmic trading system designed to optimize stock portfolios using a Transformer-based neural network model. The system is focused on maximizing the Sharpe ratio, minimizing risk, and achieving robust out-of-sample performance on historical stock data. It integrates feature engineering, model training, backtesting, and performance evaluation for end-to-end algorithmic trading.

The primary objectives of **Algo Trader** are to:

1. Optimize portfolio weights for a basket of stocks.
2. Maximize the Sharpe ratio while controlling for volatility and leverage.
3. Test and validate the model using walk-forward testing with real historical data to compare against and equal-weight benchmark.

## Features

* **Transformer Model for Portfolio Optimization**: Uses a custom Transformer architecture for efficient learning of temporal dependencies in financial time series data.
* **Differentiable Sharpe Loss**: A custom loss function that directly optimizes the Sharpe ratio during model training.
* **Backtesting**: Full walk-forward backtesting and evaluation using key metrics like CAGR, Sharpe ratio, and maximum drawdown.
* **GPU Acceleration**: Training and backtesting are GPU-accelerated for faster execution.
* **Hyperparameter Optimization**: Customizable model configuration, including the number of layers, learning rate, and other hyperparameters.
* **Visual Performance Metrics**: Generates visual reports comparing the performance of the strategy against a benchmark.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/diegomontemayor123/algo_trader.git
cd algo_trader
```

### Step 2: Install dependencies

Make sure to install the required Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare data

Before training, make sure you have historical stock price data and any required macroeconomic data. This can be loaded using the `load_prices` function in the code.

---

## Usage

### Step 1: Train the Model

The model is trained using the `train()` function, which leverages the `train_model()` function to train the model on the provided features and return series.

### Step 2: Backtest the Model

After training, you can run a backtest on the trained model using the `run_btest()` function, which will simulate trading with historical data and compute performance metrics.

### Step 3: Performance Evaluation

After running the backtest, the performance metrics will be displayed. These include the Sharpe ratio, maximum drawdown, and CAGR of both the strategy and a benchmark.

Example output:

```
Sharpe Ratio: Strat: 20.55%, Bench: 12.23%
Max Drawdown: Strat: -12.40%, Bench: -23.55%
CAGR: Strat: 15.65%, Bench: 10.87%
```

---

## Key Components

### 1. **TransformerTrader**

This class defines the core Transformer model used for portfolio optimization. It is a sequence-to-sequence model designed to predict the optimal weight allocations for a portfolio based on input features and historical returns.

### 2. **DifferentiableSharpeLoss**

A custom loss function that directly optimizes the Sharpe ratio. It penalizes excessive volatility and aims to maintain a positive mean return over time.

### 3. **train_model** and **train**

These functions handle the training loop, which involves:

* Initializing the model.
* Loading data.
* Running through multiple epochs to optimize the portfolio weights.
* Early stopping to avoid overfitting.

### 4. **Backtesting (run_btest)**

This function simulates the performance of the model over historical data, generating key metrics such as Sharpe ratio, CAGR, and maximum drawdown. It also compares the model's performance against a benchmark strategy.

---

## Model Hyperparameters

You can adjust various hyperparameters in the `config` dictionary to fine-tune the model, including:

* `INIT_LR`: Learning rate for the Adam optimizer.
* `BATCH`: Batch size used for training.
* `DECAY`: Weight decay for regularization.
* `LAYERS`: Number of Transformer encoder layers.
* `DROPOUT`: Dropout rate for regularization.
* `MAX_HEADS`: Maximum number of attention heads in the Transformer model.
* `TICK`: List of stock tickers used in the portfolio.
* `MACRO`: List of macroeconomic features (e.g., GDP, inflation) to incorporate.

---

## License

This project is provided for educational purposes only and is not intended for live trading without further modifications and risk management.

## Contributing

Feel free to fork the repository, report issues, or submit pull requests for improvements or bug fixes. Please follow the existing code style and structure.

---

This version avoids the fabricated Python code snippets and sticks strictly to the features, installation, and usage steps based on the actual code you provided. Let me know if you'd like to make any further edits!
