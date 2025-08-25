

def zscore_shrink(local_seq, global_seq, anchor_seq, alpha=None, beta=None):
    local_z = (local_seq - local_seq.mean(axis=0, keepdims=True)) / (local_seq.std(axis=0, keepdims=True) + 1e-8)
    anchor_z = (anchor_seq - anchor_seq.mean(axis=0, keepdims=True)) / (anchor_seq.std(axis=0, keepdims=True) + 1e-8)
    return alpha * local_z + (1 - alpha) * (1 - beta) * global_seq + beta * anchor_z

def zscore_shrink(local_seq, global_seq, anchor_seq, alpha=None, beta=None):
    local_z = (local_seq - local_seq.mean(axis=0, keepdims=True)) / (local_seq.std(axis=0, keepdims=True) + 1e-8)
    anchor_seq = np.broadcast_to(anchor_seq, local_seq.shape)
    return alpha * local_z + (1 - alpha) * (1-beta) * global_seq + beta * anchor_seq
