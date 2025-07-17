import torch, multiprocessing, logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import DifferentiableSharpeLoss, TransformerLRScheduler, create_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model_with_validation(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["INIT_LR"], weight_decay=config["DECAY"])
    total_steps = config["EPOCHS"] * len(train_loader)
    learning_warmup_steps = int(total_steps * config["WARMUP_FRAC"])
    print(f"[Scheduler] Total steps: {total_steps}, warmup: {learning_warmup_steps}")
    learning_scheduler = TransformerLRScheduler(optimizer, d_model=model.mlp_head[0].in_features, warmup_steps=learning_warmup_steps)
    loss_function = DifferentiableSharpeLoss(        return_penalty=config["RETURN_PENALTY"],
        drawdown_penalty=config["DRAWDOWN_PENALTY"],move_penalty=config["MOVE_PENALTY"])
    best_val_loss = float('inf')
    patience_counter = 0
    lrs = []
    for epoch in range(config["EPOCHS"]):
        model.train()
        train_losses = []
        for batch_idx, (batch_features, batch_returns) in enumerate(train_loader):
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            batch_returns = batch_returns.to(DEVICE, non_blocking=True)
            raw_weights = model(batch_features)
            abs_sum = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
            scaling_factor = torch.clamp(config["MAX_LEVERAGE"] / abs_sum, max=1.0)
            normalized_weights = raw_weights #* scaling_factor
            avg_raw_weight = raw_weights.mean().item()
            print(f"[Debug] Avg Raw Weight: {avg_raw_weight:.6f}")
            avg_norm_weight = normalized_weights.mean().item()
            print(f"[Debug] Avg Normalized Weight: {avg_norm_weight:.6f}")
            loss = loss_function(normalized_weights, batch_returns, model=model)
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("[Train] NaN or Inf loss detected — skipping model.")
                return None
            optimizer.zero_grad()
            loss.backward()
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm
            print(f"[Grad] Total grad norm: {total_grad_norm:.6f}")
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[Warning] Skipping retraining chunk due to NaNs in gradients: {name}")
                    return None
            optimizer.step()
            learning_scheduler.step()
            print(f"[LR] Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            lrs.append(optimizer.param_groups[0]['lr'])
            train_losses.append(loss.item())

        model.eval()
        val_portfolio_returns = []
        with torch.no_grad():
            for batch_features, batch_returns in val_loader:
                batch_features = batch_features.to(DEVICE, non_blocking=True)
                batch_returns = batch_returns.to(DEVICE, non_blocking=True)
                raw_weights = model(batch_features)
                weight_sum = torch.sum(torch.abs(raw_weights), dim=1, keepdim=True) + 1e-6
                scaling_factor = torch.clamp(config["MAX_LEVERAGE"] / weight_sum, max=1.0)
                normalized_weights = raw_weights #* scaling_factor
                portfolio_returns = (normalized_weights * batch_returns).sum(dim=1)
                val_portfolio_returns.extend(portfolio_returns.cpu().numpy())
        val_returns_array = np.array(val_portfolio_returns)
        mean_ret = val_returns_array.mean()
        std_ret = val_returns_array.std() + 1e-6
        avg_val_loss = -(mean_ret / std_ret)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["EARLY_STOP_PATIENCE"]:
                break
    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule During Training')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('img/learning_rate_schedule.png')
    plt.close()
    return model

def train_main_model(config, features, returns):
    from data_prep import prepare_main_datasets
    train_dataset, val_dataset, _ = prepare_main_datasets(features, returns, config)
    num_workers = min(2, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=num_workers)
    model = create_model(train_dataset[0][0].shape[1], config)
    trained_model = train_model_with_validation(model, train_loader, val_loader, config)
    return trained_model