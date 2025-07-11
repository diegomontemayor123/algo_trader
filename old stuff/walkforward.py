import json
import torch
import random
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from compute_features import compute_features, normalize_features
from model import (
    MarketDataset, split_train_validation, create_sequences,
    create_model, train_model_with_validation, calculate_performance_metrics,
    INITIAL_CAPITAL, START_DATE, END_DATE, TICKERS, DEVICE, SEED
)

TEST_ONLY = True
MODEL_PATH = "walk_trained_model.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_walkforward_test_with_validation(config, plot=False, weights_csv_path="walkforward_daily_pfo_weights.csv"):
    logging.info("[Walk-Forward] Starting walk-forward test with validation...")

    FEATURES = config["FEATURES"]
    features, returns = compute_features(TICKERS, START_DATE, END_DATE, FEATURES)
    features.index = pd.to_datetime(features.index)
    returns.index = pd.to_datetime(returns.index)
    # Drop last date to avoid NaNs in returns
    if not returns.empty:
        last_date = returns.index[-1]
        features = features.loc[features.index < last_date]
        returns = returns.loc[returns.index < last_date]

    nan_rows = returns[returns.isna().any(axis=1)]
    if not nan_rows.empty:
        logging.warning("[Walk-Forward] Found NaNs in returns at the following dates:")
        for dt, row in nan_rows.iterrows():
            logging.warning(f"  Date: {dt.date()}, NaN in columns: {row.index[row.isna()].tolist()}")

    if features.empty or returns.empty:
        logging.warning("[Walk-Forward] Feature or return data is empty. Returning flat portfolio.")
        idx = features.index if len(features) > 0 else pd.DatetimeIndex([pd.Timestamp.today()])
        return pd.Series([INITIAL_CAPITAL], index=[idx[0]])

    all_dates = features.index
    SPLIT_DATE = pd.Timestamp(config["SPLIT_DATE"])
    test_start_index = all_dates.get_indexer([SPLIT_DATE], method="bfill")[0]
    step = config["WALKFWD_STEP"]
    window = int(config["WALKFWD_WNDW"])
    model_input_dim = features.shape[1]

    portfolio_values = [INITIAL_CAPITAL]
    benchmark_values = [INITIAL_CAPITAL]

    daily_weights = []
    asset_names = returns.columns

    daily_portfolio_dates = []
    daily_portfolio_values = []
    daily_benchmark_values = []

    # This will hold the last weights to carry forward
    last_weights = None

    for i in range(test_start_index, len(all_dates) - config["LOOKBACK"] - config["PREDICT_DAYS"], step):
        start_idx = max(i - window, 0)
        train_start = all_dates[start_idx]
        train_end = all_dates[i]
        test_start = all_dates[i + config["LOOKBACK"]]
        test_end = all_dates[min(i + config["LOOKBACK"] + step, len(all_dates) - 1)]

        if train_end >= test_start:
            logging.warning(f"[Walk-Forward] Skipping due to data leakage: train_end={train_end}, test_start={test_start}")
            continue

        logging.info(f"\n[Walk-Forward] Training: {train_start.date()} to {train_end.date()}")
        logging.info(f"[Walk-Forward] Testing: {test_start.date()} to {test_end.date()}")

        train_mask = (features.index >= train_start) & (features.index < train_end)
        train_features = features.loc[train_mask]
        train_returns = returns.loc[train_mask]

        if len(train_features) < config["LOOKBACK"] + config["PREDICT_DAYS"]:
            logging.warning("[Walk-Forward] Insufficient training data, skipping this period.")
            continue

        if TEST_ONLY:
            logging.info("[Walk-Forward] TEST_ONLY mode active: loading model from disk.")
            model = create_model(model_input_dim)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        else:
            sequences, targets, _ = create_sequences(train_features, train_returns, 0, len(train_features))
            if len(sequences) == 0:
                logging.warning(f"[Walk-Forward] No valid sequences for {train_start.date()}–{train_end.date()}, skipping...")
                continue

            train_seq, train_tgt, val_seq, val_tgt = split_train_validation(sequences, targets)
            train_dataset = MarketDataset(torch.tensor(train_seq), torch.tensor(train_tgt))
            val_dataset = MarketDataset(torch.tensor(val_seq), torch.tensor(val_tgt))
            train_loader = DataLoader(train_dataset, batch_size=min(config["BATCH_SIZE"], len(train_dataset)), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=min(config["BATCH_SIZE"], len(val_dataset)), shuffle=False)

            model = create_model(model_input_dim)
            model = train_model_with_validation(model, train_loader, val_loader, epochs=config["EPOCHS"])
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"[Walk-Forward] Saved model to {MODEL_PATH}")

        test_mask = (features.index >= test_start) & (features.index <= test_end)
        test_features = features.loc[test_mask]
        test_returns = returns.loc[test_mask]

        if len(test_features) < config["LOOKBACK"] + 1:
            logging.warning("[Walk-Forward] Not enough test data, skipping this period.")
            continue

        model.eval()
        with torch.no_grad():
            # For each day in test window (except last LOOKBACK days), generate weights & update portfolio daily
            for j in range(len(test_features) - config["LOOKBACK"]):
                current_date = test_features.index[j + config["LOOKBACK"]]
                input_window = test_features.iloc[j:j + config["LOOKBACK"]].values.astype(np.float32)
                norm_window = normalize_features(input_window)
                input_tensor = torch.tensor(norm_window).unsqueeze(0).to(DEVICE)

                # Generate new weights only on the first day of this test segment, otherwise carry forward last weights
                if j == 0:
                    raw_weights = model(input_tensor).cpu().numpy().flatten()
                    weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
                    scale = min(config["MAX_LEVERAGE"] / weight_sum, 1.0)
                    last_weights = raw_weights * scale
                # else: use last_weights from previous day

                # Use last_weights for portfolio return calculation
                weights = last_weights

                period_returns = test_returns.loc[current_date].values

                if np.isnan(period_returns).any():
                    nan_cols = test_returns.columns[np.isnan(period_returns)].tolist()
                    logging.warning(
                        f"[Walk-Forward] NaN detected in returns at date {current_date.date()}, columns: {nan_cols}, skipping step."
                    )
                    continue

                portfolio_return = np.dot(weights, period_returns)
                benchmark_return = np.mean(period_returns)

                if np.isnan(portfolio_return) or np.isinf(portfolio_return):
                    logging.warning(f"[Walk-Forward] Invalid portfolio return at date {current_date.date()}, skipping this step.")
                    logging.warning(f"weights: {weights}")
                    logging.warning(f"period_returns: {period_returns}")
                    continue

                new_portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
                new_benchmark_value = benchmark_values[-1] * (1 + benchmark_return)

                portfolio_values.append(new_portfolio_value)
                benchmark_values.append(new_benchmark_value)

                daily_weights.append(pd.Series(weights, index=asset_names, name=current_date))

                daily_portfolio_dates.append(current_date)
                daily_portfolio_values.append(new_portfolio_value)
                daily_benchmark_values.append(new_benchmark_value)

    if not daily_weights:
        logging.warning("[Walk-Forward] No daily weights generated. Possibly insufficient data.")
        return pd.Series([INITIAL_CAPITAL])

    # Create a full business day date range covering all test dates
    full_dates = pd.date_range(start=daily_portfolio_dates[0], end=daily_portfolio_dates[-1], freq='B')

    # Portfolio values forward fill for missing dates (if any)
    pf_df = pd.DataFrame({"PortfolioValue": daily_portfolio_values}, index=pd.DatetimeIndex(daily_portfolio_dates))
    pf_df = pf_df.reindex(full_dates).ffill()
    daily_portfolio_dates = pf_df.index.tolist()
    daily_portfolio_values = pf_df["PortfolioValue"].tolist()

    # Benchmark values forward fill for missing dates (optional)
    benchmark_df = pd.DataFrame({"BenchmarkValue": daily_benchmark_values}, index=pd.DatetimeIndex(daily_portfolio_dates))
    benchmark_df = benchmark_df.reindex(full_dates).ffill()
    daily_benchmark_values = benchmark_df["BenchmarkValue"].tolist()

    weights_df = pd.DataFrame(daily_weights)
    weights_df["total_exposure"] = weights_df.abs().sum(axis=1)
    weights_df.index.name = "Date"
    weights_df.sort_index(inplace=True)
    weights_df.to_csv(weights_csv_path)
    logging.info(f"[Walk-Forward] Saved daily portfolio weights to {weights_csv_path}")

    performance_df = pd.DataFrame({
        "Date": pd.DatetimeIndex(daily_portfolio_dates),
        "PortfolioValue": daily_portfolio_values,
        "BenchmarkValue": daily_benchmark_values
    })
    performance_df.set_index("Date", inplace=True)
    performance_df.sort_index(inplace=True)
    performance_csv_path = "walkforward_daily_performance.csv"
    performance_df.to_csv(performance_csv_path)
    logging.info(f"[Walk-Forward] Saved daily portfolio performance to {performance_csv_path}")

    portfolio_metrics = calculate_performance_metrics(portfolio_values)
    benchmark_metrics = calculate_performance_metrics(benchmark_values)

    logging.info("\n[Walk-Forward] === Performance Summary ===")
    for metric_name in portfolio_metrics:
        logging.info(f"  {metric_name.replace('_', ' ').title()}:")
        logging.info(f"    Strategy: {portfolio_metrics[metric_name]:.2%}")
        logging.info(f"    Benchmark: {benchmark_metrics[metric_name]:.2%}")

    if plot:
        plt.figure(figsize=(12, 6))
        test_dates = pd.DatetimeIndex([w.name for w in daily_weights])
        assert len(test_dates) == len(portfolio_values) - 1, \
            f"Date and portfolio length mismatch: {len(test_dates)} vs {len(portfolio_values)-1}"
        plt.plot(test_dates, portfolio_values[1:], label="Strategy Portfolio", linewidth=2)
        plt.plot(test_dates, benchmark_values[1:], label="Equal Weight Benchmark", linewidth=2)
        plt.title("Portfolio Performance Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("walkforward_performance.png", dpi=300)
        plt.close()
        logging.info("[Walk-Forward] Saved performance plot to walkforward_performance.png")

    return pd.Series(portfolio_values[1:], index=pd.DatetimeIndex([w.name for w in daily_weights]))


if __name__ == "__main__":
    with open("best_hyperparameters.json", "r") as f:
        raw_params = json.load(f)
    if isinstance(raw_params.get("FEATURES"), str):
        raw_params["FEATURES"] = [f.strip() for f in raw_params["FEATURES"].split(",") if f.strip()]
    run_walkforward_test_with_validation(raw_params, plot=True)
