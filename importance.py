import torch
import numpy as np
from collections import defaultdict

def importance(model, feat_data, ret_data, scaler, lback, time_decay=None, top_n=None):
    model.eval()
    device = next(model.parameters()).device
    feat_scores = defaultdict(list); grad_scores = defaultdict(list); dates = []
    
    for i in range(lback, len(feat_data)):
        date = feat_data.index[i]
        x = scaler.transform(feat_data.iloc[i-lback:i].values.astype(np.float32))
        x_tensor = torch.tensor(x).unsqueeze(0).to(device).requires_grad_(True)
        weights = model(x_tensor)
        target_ret = torch.tensor(ret_data.iloc[i].values).unsqueeze(0).to(device)
        pfo_ret = (weights * target_ret).sum()
        pfo_ret.backward()
        grad_norm = x_tensor.grad.abs().mean(dim=1).squeeze().detach().cpu().numpy()

        if hasattr(model, 'feat_attention'):
            with torch.no_grad():
                attn_weights = model.feat_attention(x_tensor.mean(dim=1)).squeeze().cpu().numpy()
            feat_scores[date] = attn_weights

        grad_scores[date] = grad_norm
        dates.append(date)

    n_dates = len(dates)
    time_weights = np.array([time_decay**(n_dates-1-i) for i in range(n_dates)])
    time_weights /= time_weights.sum()

    if feat_scores:
        feat_imp = (np.vstack([feat_scores[d] for d in dates]).T * time_weights).sum(axis=1)
    else:
        feat_imp = None

    grad_matrix = np.vstack([grad_scores[d] for d in dates])  
    weighted_grad_imp = (grad_matrix.T * time_weights).sum(axis=1)

    feature_names = np.array(feat_data.columns)

    # --- Select Top-N Features ---
    if top_n is not None:
        # Use gradient_importance as ranking (can also merge with attention if needed)
        scores = weighted_grad_imp if feat_imp is None else (weighted_grad_imp + feat_imp) / 2
        top_idx = np.argsort(scores)[::-1][:top_n]

        feature_names = feature_names[top_idx]
        weighted_grad_imp = weighted_grad_imp[top_idx]
        feat_imp = feat_imp[top_idx] if feat_imp is not None else None

    return {
        'attention_importance': feat_imp,
        'gradient_importance': weighted_grad_imp,
        'feature_names': feature_names,
        'time_weights': time_weights,
        'dates': dates
    }

def get_transformer_layer_importance(model, feat_data, scaler, lback, layer_idx=0):
    model.eval()
    attentions = []
    def hook(module, input):
        if hasattr(module, 'self_attn'):
            with torch.no_grad(): 
                x = input[0]
                attn_output, attn_weights = module.self_attn(x, x, x, average_attn_weights=False)
                attentions.append(attn_weights.cpu().numpy())
    handle = model.transformer_encoder.layers[layer_idx].register_forward_hook(hook)
    try:
        for i in range(lback, len(feat_data)): 
            x = scaler.transform(feat_data.iloc[i-lback:i].values.astype(np.float32))
            x_tensor = torch.tensor(x).unsqueeze(0)
            with torch.no_grad(): _ = model(x_tensor)                
        if attentions:
            avg_attention = np.mean(attentions, axis=0).mean(axis=1).mean(axis=0)  # [seq_len]
            return avg_attention
    finally:
        handle.remove()
    return None

def track_importance_during_training(model, feat_train, ret_train, scaler, config, top_n=None):
    return importance(model, feat_train, ret_train, scaler, config["LBACK"], time_decay=config["IMPDECAY"], top_n=top_n)
