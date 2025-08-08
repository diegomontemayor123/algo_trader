import torch, multiprocessing
import numpy as np
from torch.utils.data import DataLoader
from model import DifferentiableSharpeLoss, create_model
np.seterr(all='raise')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EPOCHS = 20

def train_model(model, train_loader, val_loader, config, asset_sd):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["INIT_LR"], weight_decay=config["DECAY"])
    loss_func = DifferentiableSharpeLoss(return_pen=config["RETURN_PEN"],return_exp=config["RETURN_EXP"],exp_exp=config["EXP_EXP"],exp_pen=config["EXP_PEN"],sd_pen=config["SD_PEN"],sd_exp=config["SD_EXP"],)
    best_val_loss = float('inf')
    patience_counter = 0; lrs = []
    for epoch in range(MAX_EPOCHS):
        model.train(); train_loss = []
        for batch_idx, (batch_feat, batch_ret) in enumerate(train_loader):
            batch_feat = batch_feat.to(DEVICE, non_blocking=True)
            batch_ret = batch_ret.to(DEVICE, non_blocking=True)
            norm_weight = model(batch_feat)
            loss = loss_func(norm_weight, batch_ret, asset_sd=asset_sd, model=model,epoch=epoch,batch_idx=batch_idx)
            if torch.isnan(loss) or torch.isinf(loss): print("[Train] NaN or Inf loss detected — skipping model.");return None
            optimizer.zero_grad()
            loss.backward()
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[Train] Skipping retraining chunk due to NaNs in gradients: {name}");return None
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            train_loss.append(loss.item())
        model.eval()
        val_pfo_ret = []
        with torch.no_grad():
            for batch_feat, batch_ret in val_loader:
                batch_feat = batch_feat.to(DEVICE, non_blocking=True)
                batch_ret = batch_ret.to(DEVICE, non_blocking=True)
                norm_weight = model(batch_feat)
                pfo_ret = (norm_weight * batch_ret).sum(dim=1)
                val_pfo_ret.extend(pfo_ret.cpu().numpy())
        val_ret_array = np.array(val_pfo_ret)
        mean_ret = val_ret_array.mean();std_ret = val_ret_array.std() + 1e-6
        avg_val_loss = -(mean_ret / std_ret)
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["EARLY_FAIL"]: break
    return model

def train(config, feat, ret):
    from prep import prep_data
    train_data, val_data, _ = prep_data(feat, ret, config)
    num_workers = min(2, multiprocessing.cpu_count())
    train_loader = DataLoader(train_data, batch_size=config["BATCH"], shuffle=True, num_workers=num_workers,pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=config["BATCH"], shuffle=False, num_workers=num_workers,pin_memory=True, persistent_workers=True)
    model = create_model(train_data[0][0].shape[1], config) 
    asset_sd = torch.tensor(train_data.ret.std(dim=0).cpu().numpy().astype(np.float32), device=DEVICE)
    model0 = train_model(model, train_loader, val_loader, config, asset_sd=asset_sd)
    return model0