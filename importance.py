import torch
import numpy as np
import pandas as pd
from collections import defaultdict

def analyze_feat_importance(model, feat_data, ret_data, scaler, lback, time_decay=0.95):
    """Extract feature importance with time weighting (recent = higher weight)"""
    model.eval()
    device = next(model.parameters()).device
    feat_scores = defaultdict(list); grad_scores = defaultdict(list); dates = []
    
    for i in range(lback, len(feat_data)):
        date = feat_data.index[i]
        x = scaler.transform(feat_data.iloc[i-lback:i].values.astype(np.float32))
        x_tensor = torch.tensor(x).unsqueeze(0).to(device).requires_grad_(True)
        
        # Forward pass
        weights = model(x_tensor)
        target_ret = torch.tensor(ret_data.iloc[i].values).unsqueeze(0).to(device)
        pfo_ret = (weights * target_ret).sum()
        
        # Backward pass for gradients
        pfo_ret.backward()
        grad_norm = x_tensor.grad.abs().mean(dim=1).squeeze().detach().cpu().numpy()
        
        # Attention weights (if using feat_attention)
        if hasattr(model, 'feat_attention'):
            with torch.no_grad():
                attn_weights = model.feat_attention(x_tensor.mean(dim=1)).squeeze().cpu().numpy()
            feat_scores[date] = attn_weights
            
        grad_scores[date] = grad_norm
        dates.append(date)
    
    # Time-weighted aggregation (recent dates get higher weight)
    n_dates = len(dates)
    time_weights = np.array([time_decay**(n_dates-1-i) for i in range(n_dates)])
    time_weights /= time_weights.sum()
    
    # Aggregate scores
    if feat_scores:
        feat_matrix = np.vstack([feat_scores[d] for d in dates])
        weighted_feat_imp = (feat_matrix.T * time_weights).sum(axis=1)
    else:
        weighted_feat_imp = None
        
    grad_matrix = np.vstack([grad_scores[d] for d in dates])  
    weighted_grad_imp = (grad_matrix.T * time_weights).sum(axis=1)
    
    return {
        'attention_importance': weighted_feat_imp,
        'gradient_importance': weighted_grad_imp,
        'feature_names': feat_data.columns,
        'time_weights': time_weights,
        'dates': dates
    }

def get_transformer_layer_importance(model, feat_data, scaler, lback, layer_idx=0):
    """Extract attention patterns from transformer layers"""
    model.eval()
    all_attentions = []
    
    # Hook to capture attention weights
    attentions = []
    def hook(module, input, output):
        if hasattr(module, 'self_attn'):
            # Get attention weights from transformer layer
            with torch.no_grad():
                x = input[0]
                attn_output, attn_weights = module.self_attn(x, x, x, average_attn_weights=False)
                attentions.append(attn_weights.cpu().numpy())
    
    # Register hook on specific transformer layer
    handle = model.transformer_encoder.layers[layer_idx].register_forward_hook(hook)
    
    try:
        for i in range(lback, min(lback + 100, len(feat_data))):  # Sample recent 100 days
            x = scaler.transform(feat_data.iloc[i-lback:i].values.astype(np.float32))
            x_tensor = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                _ = model(x_tensor)
                
        # Average attention across time steps and heads
        if attentions:
            avg_attention = np.mean(attentions, axis=0).mean(axis=1).mean(axis=0)  # [seq_len]
            return avg_attention
    finally:
        handle.remove()
        
    return None

# Add to your training loop after model creation:
def track_importance_during_training(model, feat_train, ret_train, scaler, config, top_n=5):
    """Call this after each chunk to track evolving importance"""
    importance = analyze_feat_importance(model, feat_train, ret_train, scaler, 
                                       config["LBACK"], time_decay=0.95)
    
    if importance['attention_importance'] is not None:
        feat_df = pd.DataFrame({
            'feature': importance['feature_names'],
            'attention_score': importance['attention_importance'],
            'gradient_score': importance['gradient_importance']
        }).sort_values('attention_score', ascending=False)
        
        print(f"[Importance] Top {top_n} features by attention:")
        for _, row in feat_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['attention_score']:.4f}")
            
    return importance