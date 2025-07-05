import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import TransformerTrader
from .data import MarketDataset
from .utils import get_data_splits, evaluate_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def test(model, X_test, y_test, criterion, device):
    model.eval()
    X_test = torch.tensor(np.array(X_test)).to(device)
    y_test = torch.tensor(np.array(y_test)).to(device)
    with torch.no_grad():
        output = model(X_test)
        loss = criterion(output, y_test).item()
    return loss, output.cpu().numpy()

def run_walkforward(features, returns, walkforward_splits, input_dim=84, heads=14, device='cuda'):
    criterion = torch.nn.MSELoss()
    results = []

    for i, (train_idx, val_idx, test_idx) in enumerate(walkforward_splits):
        print(f"\n[Walk-Forward] Period {i+1}:")
        print(f"Training: {features.index[train_idx[0]]} to {features.index[train_idx[-1]]}")
        print(f"Validation: {features.index[val_idx[0]]} to {features.index[val_idx[-1]]}")
        print(f"Testing: {features.index[test_idx[0]]} to {features.index[test_idx[-1]]}")

        X_train, y_train = features.iloc[train_idx].values, returns.iloc[train_idx].values
        X_val, y_val = features.iloc[val_idx].values, returns.iloc[val_idx].values
        X_test, y_test = features.iloc[test_idx].values, returns.iloc[test_idx].values

        train_dataset = MarketDataset(torch.tensor(np.array(X_train), dtype=torch.float32),
                                      torch.tensor(np.array(y_train), dtype=torch.float32))
        val_dataset = MarketDataset(torch.tensor(np.array(X_val), dtype=torch.float32),
                                    torch.tensor(np.array(y_val), dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        model = TransformerTrader(input_dim=input_dim, heads=heads, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        max_epochs = 5
        patience = 2

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)

            print(f"[Training] Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("[Training] Early stopping triggered.")
                    break

        # Load best model for testing
        model.load_state_dict(best_model_state)

        # Test evaluation
        test_loss, test_preds = test(model, X_test, y_test, criterion, device)
        print(f"[Testing] Test Loss: {test_loss:.4f}")

        results.append({
            'period': i + 1,
            'train_range': (features.index[train_idx[0]], features.index[train_idx[-1]]),
            'val_range': (features.index[val_idx[0]], features.index[val_idx[-1]]),
            'test_range': (features.index[test_idx[0]], features.index[test_idx[-1]]),
            'test_loss': test_loss,
            'test_predictions': test_preds
        })

    print("\n=== Walk-Forward Test Summary ===")
    for res in results:
        print(f"Period {res['period']}: Test Loss={res['test_loss']:.4f} | Test Range={res['test_range'][0]} to {res['test_range'][1]}")

    return results
