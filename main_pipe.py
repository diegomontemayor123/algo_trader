import os
import torch
import optuna
import numpy as np
from torch.utils.data import DataLoader

from walkforward import run_walkforward_test_with_validation
from backtest import run_backtest
from compute_features import compute_features, normalize_features
from tech import (
    create_model as base_create_model,
    train_model_with_validation,
    create_sequences,
    split_train_validation,
    MarketDataset,
    calculate_performance_metrics,
    TransformerTrader,
    calculate_heads
)

# --- Configuration defaults ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INITIAL_CAPITAL = 1_000_000
MAX_LEVERAGE = 3.0
LOOKBACK = 60
PREDICT_DAYS = 9
BATCH_SIZE = 64
EPOCHS = 5
WALKFORWARD_STEP_SIZE = 30
WALKFORWARD_TRAIN_WINDOW = 252 * 2
SPLIT_DATE = '2023-01-01'
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START_DATE = '2012-01-01'
END_DATE = '2025-06-01'
FEATURES = ['open', 'high', 'low', 'close', 'volume']
LAYER_COUNT = 6
DROPOUT = 0.1
DECAY = 0.01  # Default weight decay


# Updated create_model that accepts dropout, layer_count, and lookback dynamically
def create_model(input_dim, dropout=None, layer_count=None, lookback=None):
    heads = calculate_heads(input_dim)
    effective_dropout = dropout if dropout is not None else DROPOUT
    effective_layers = layer_count if layer_count is not None else LAYER_COUNT
    effective_lookback = lookback if lookback is not None else LOOKBACK

    print(f"[Model] Creating TransformerTrader with input_dim={input_dim}, heads={heads}, "
          f"layers={effective_layers}, dropout={effective_dropout}, lookback={effective_lookback}")

    model = TransformerTrader(
        input_dim=input_dim,
        num_heads=heads,
        num_layers=effective_layers,
        dropout=effective_dropout,
        seq_len=effective_lookback
    )
    return model.to(DEVICE)


def objective(trial):
    # Hyperparameter suggestions
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    layer_count = trial.suggest_int("layer_count", 2, 8)
    lookback = trial.suggest_int("lookback", 50, 120)
    max_leverage = trial.suggest_float("max_leverage", 1.0, 3.0)
    batch_size = trial.suggest_int("batch_size", 40, 100)
    predict_days = trial.suggest_int("predict_days", 1, 10)
    decay = trial.suggest_float("decay", 0.0, 0.1)
    epochs = 20  # Fixed epochs for tuning

    features = FEATURES

    # Model constructor closure using tuned params
    def create_model_with_params(input_dim):
        return create_model(
            input_dim=input_dim,
            dropout=dropout,
            layer_count=layer_count,
            lookback=lookback
        )

    # Wrapper for training to pass weight decay param
    def train_with_decay(*args, **kwargs):
        return train_model_with_validation(*args, weight_decay=decay, **kwargs)

    try:
        results = run_walkforward_test_with_validation(
            compute_features=compute_features,
            create_sequences=create_sequences,
            split_train_validation=split_train_validation,
            MarketDataset=MarketDataset,
            create_model=create_model_with_params,
            train_model_with_validation=train_with_decay,
            normalize_features=normalize_features,
            calculate_performance_metrics=calculate_performance_metrics,
            SPLIT_DATE=SPLIT_DATE,
            WALKFORWARD_STEP_SIZE=WALKFORWARD_STEP_SIZE,
            WALKFORWARD_TRAIN_WINDOW=WALKFORWARD_TRAIN_WINDOW,
            LOOKBACK=lookback,
            PREDICT_DAYS=predict_days,
            BATCH_SIZE=batch_size,
            INITIAL_CAPITAL=INITIAL_CAPITAL,
            MAX_LEVERAGE=max_leverage,
            DEVICE=DEVICE,
            TICKERS=TICKERS,
            START_DATE=START_DATE,
            END_DATE=END_DATE,
            FEATURES=features,
            epochs=epochs,
        )

        if not results:
            return -float("inf")

        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])

        # Custom objective: maximize Sharpe, penalize large drawdown
        score = avg_sharpe - 0.5 * abs(avg_drawdown)

        trial.set_user_attr("sharpe", avg_sharpe)
        trial.set_user_attr("drawdown", avg_drawdown)
        return score

    except Exception as e:
        print(f"[Trial Error] {e}")
        return -float("inf")


def main():
    print("[Optuna] Starting hyperparameter tuning...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    print("\n[Optuna] Best hyperparameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save trials to CSV for further analysis
    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)
    print("[Optuna] Trials saved to optuna_trials.csv")

    best_params = study.best_params

    # Extract best hyperparameters with fallbacks
    best_dropout = best_params.get("dropout", DROPOUT)
    best_layer_count = best_params.get("layer_count", LAYER_COUNT)
    best_lookback = best_params.get("lookback", LOOKBACK)
    best_max_leverage = best_params.get("max_leverage", MAX_LEVERAGE)
    best_batch_size = best_params.get("batch_size", BATCH_SIZE)
    best_predict_days = best_params.get("predict_days", PREDICT_DAYS)
    best_decay = best_params.get("decay", DECAY)

    print("\n[Pipeline] Running final walk-forward test with best hyperparameters...")

    def create_best_model(input_dim):
        return create_model(
            input_dim=input_dim,
            dropout=best_dropout,
            layer_count=best_layer_count,
            lookback=best_lookback
        )

    def train_with_decay(*args, **kwargs):
        return train_model_with_validation(*args, weight_decay=best_decay, **kwargs)

    final_results = run_walkforward_test_with_validation(
        compute_features=compute_features,
        create_sequences=create_sequences,
        split_train_validation=split_train_validation,
        MarketDataset=MarketDataset,
        create_model=create_best_model,
        train_model_with_validation=train_with_decay,
        normalize_features=normalize_features,
        calculate_performance_metrics=calculate_performance_metrics,
        SPLIT_DATE=SPLIT_DATE,
        WALKFORWARD_STEP_SIZE=WALKFORWARD_STEP_SIZE,
        WALKFORWARD_TRAIN_WINDOW=WALKFORWARD_TRAIN_WINDOW,
        LOOKBACK=best_lookback,
        PREDICT_DAYS=best_predict_days,
        BATCH_SIZE=best_batch_size,
        INITIAL_CAPITAL=INITIAL_CAPITAL,
        MAX_LEVERAGE=best_max_leverage,
        DEVICE=DEVICE,
        TICKERS=TICKERS,
        START_DATE=START_DATE,
        END_DATE=END_DATE,
        FEATURES=FEATURES,
        epochs=EPOCHS,
    )

    print("\n[Evaluation] Final Walk-Forward Results:")
    if final_results:
        for i, result in enumerate(final_results):
            print(f"  Period {i + 1}: Sharpe={result['sharpe_ratio']:.3f}, "
                  f"Max Drawdown={result['max_drawdown']:.2%}, "
                  f"CAGR={result['cagr']:.2%}")
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in final_results])
        avg_cagr = np.mean([r["cagr"] for r in final_results])
        avg_dd = np.mean([r["max_drawdown"] for r in final_results])
        print(f"\n[Summary] Avg Sharpe: {avg_sharpe:.3f}, Avg CAGR: {avg_cagr:.2%}, Avg Max Drawdown: {avg_dd:.2%}")
    else:
        print("[Warning] No walk-forward results returned.")

    print("\n[Pipeline] Execution complete.")


if __name__ == "__main__":
    main()
