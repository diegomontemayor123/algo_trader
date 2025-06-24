import torch
import numpy as np
from torch.utils.data import DataLoader


def run_walkforward_test_with_validation(compute_features,create_sequences,split_train_validation,MarketDataset,create_model,train_model_with_validation,normalize_features,calculate_performance_metrics,SPLIT_DATE,WALKFORWARD_STEP_SIZE,WALKFORWARD_TRAIN_WINDOW,LOOKBACK,PREDICT_DAYS,BATCH_SIZE,INITIAL_CAPITAL,MAX_LEVERAGE,DEVICE,TICKERS,START_DATE,END_DATE,FEATURES
):
    print("[Walk-Forward] Starting walk-forward test with validation...")
    features, returns = compute_features(TICKERS,START_DATE,END_DATE,FEATURES)
    all_dates = features.index
    test_start_index = all_dates.get_indexer([SPLIT_DATE], method="bfill")[0]
    walkforward_results = []
    model_input_dim = features.shape[1]
    for step in range(test_start_index, len(all_dates) - LOOKBACK - PREDICT_DAYS, WALKFORWARD_STEP_SIZE):
        train_start_date = all_dates[step - WALKFORWARD_TRAIN_WINDOW] if step - WALKFORWARD_TRAIN_WINDOW > 0 else all_dates[0]
        train_end_date = all_dates[step]
        test_start_date = all_dates[step + LOOKBACK]
        test_end_date = all_dates[min(step + LOOKBACK + WALKFORWARD_STEP_SIZE, len(all_dates) - 1)]
        print(f"\n[Walk-Forward] Training: {train_start_date.date()} to {train_end_date.date()}")
        print(f"[Walk-Forward] Testing: {test_start_date.date()} to {test_end_date.date()}")
        train_mask = (features.index >= train_start_date) & (features.index < train_end_date)
        train_features = features.loc[train_mask]
        train_returns = returns.loc[train_mask]
        if len(train_features) < LOOKBACK + PREDICT_DAYS:
            print("[Walk-Forward] Insufficient training data, skipping...")
            continue
        train_start_idx = 0
        train_end_idx = len(train_features)
        sequences, targets = create_sequences(train_features, train_returns, train_start_idx, train_end_idx)
        if len(sequences) == 0:
            print("[Walk-Forward] No valid sequences created, skipping...")
            continue
        train_seq, train_tgt, val_seq, val_tgt = split_train_validation(sequences, targets)
        print(f"[Walk-Forward] Train samples: {len(train_seq)}, Validation samples: {len(val_seq)}")
        train_dataset = MarketDataset(torch.tensor(train_seq), torch.tensor(train_tgt))
        val_dataset = MarketDataset(torch.tensor(val_seq), torch.tensor(val_tgt))
        train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)
        # Train model for this window
        model = create_model(model_input_dim)
        trained_model = train_model_with_validation(model, train_loader, val_loader, epochs=5)  # Fewer epochs for walk-forward
        # Test on out-of-sample period
        test_mask = (features.index >= test_start_date) & (features.index <= test_end_date)
        test_features = features.loc[test_mask]
        test_returns = returns.loc[test_mask]
        portfolio_values = [INITIAL_CAPITAL]
        trained_model.eval()
        with torch.no_grad():
            for i in range(len(test_features) - LOOKBACK):
                feature_window = test_features.iloc[i:i + LOOKBACK].values.astype(np.float32)
                normalized_features = normalize_features(feature_window)
                input_tensor = torch.tensor(normalized_features).unsqueeze(0).to(DEVICE)
                raw_weights = trained_model(input_tensor).cpu().numpy().flatten()
                weight_sum = np.sum(np.abs(raw_weights)) + 1e-6
                scaling_factor = min(MAX_LEVERAGE / weight_sum, 1.0)  # never scale up, only down if needed
                final_weights = raw_weights * scaling_factor
                period_returns = test_returns.iloc[i + LOOKBACK].values
                portfolio_return = np.dot(final_weights, period_returns)
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        if len(portfolio_values) > 1:
            period_metrics = calculate_performance_metrics(portfolio_values)
            print(f"[Walk-Forward] Period Performance:")
            print(f"  CAGR: {period_metrics['cagr']:.2%}")
            print(f"  Sharpe: {period_metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {period_metrics['max_drawdown']:.2%}")
            walkforward_results.append({
                'start_date': test_start_date,
                'end_date': test_end_date,
                'cagr': period_metrics['cagr'],
                'sharpe_ratio': period_metrics['sharpe_ratio'],
                'max_drawdown': period_metrics['max_drawdown']
            })
    if walkforward_results:
        avg_cagr = np.mean([r['cagr'] for r in walkforward_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in walkforward_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in walkforward_results])
        print(f"\n[Walk-Forward] === Aggregate Results Across All Periods ===")
        print(f"  Average CAGR: {avg_cagr:.2%}")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  Average Max Drawdown: {avg_drawdown:.2%}")
        print(f"  Number of Test Periods: {len(walkforward_results)}")
    else:
        print("[Walk-Forward] No valid test periods completed")