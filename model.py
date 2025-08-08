import torch, sys
import numpy as np
import torch.nn as nn
from feat import *
np.seterr(all='raise')
INITIAL_CAPITAL = 100 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerTrader(nn.Module):
    def __init__(self, dimen, num_heads, num_layers, dropout, seq_len, TICK, feat_attent):
        super().__init__()
        self.seq_len = seq_len ; self.feat_attent = feat_attent
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dimen))
        self.feat_attention = nn.Sequential(nn.Linear(dimen, dimen), nn.Tanh(), nn.Linear(dimen, dimen), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(d_model=dimen,nhead=num_heads,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(nn.Linear(dimen, 64),nn.PReLU(),nn.Dropout(dropout),nn.Linear(64, len(TICK)))
        print(f"[Model] MLP head output dim: {len(TICK)}")
    def forward(self, x):
        if self.feat_attent: x = x * self.feat_attention(x.mean(dim=1)).unsqueeze(1) 
        x = x + self.pos_embedding
        encoded = self.transformer_encoder(x)
        last_hidden = encoded[:, -1, :]
        return self.mlp_head(last_hidden)

def calc_heads(dimen, max_heads):
    if dimen % max_heads != 0:
        for heads in range(max_heads, 0, -1):
            if dimen % heads == 0: return heads
    else: return max_heads

def create_model(dimen, config):
    heads = calc_heads(dimen, config["MAX_HEADS"])
    print(f"[Model] Creating TransformerTrader with dimen={dimen}, heads={heads}, device={DEVICE}")
    return TransformerTrader(dimen=dimen,num_heads=heads,num_layers=config["LAYERS"],dropout=config["DROPOUT"],seq_len=config["LBACK"],TICK=config["TICK"],feat_attent=config["ATTENT"]).to(DEVICE, non_blocking=True)

def split_train_val(sequences, targets, valid_ratio):
    total_samples = len(sequences)
    val_size = int(total_samples * valid_ratio)
    train_size = total_samples - val_size
    return (sequences[:train_size], targets[:train_size], sequences[train_size:], targets[train_size:])

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, return_pen,return_exp, exp_pen, exp_exp, sd_pen,sd_exp):
        super().__init__()
        self.return_pen = return_pen;self.return_exp = return_exp;self.exp_exp = exp_exp;self.exp_pen = exp_pen;self.sd_pen = sd_pen;self.sd_exp=sd_exp
    def forward(self, pfo_weight, target_ret, asset_sd=None, model=None, epoch = 0,batch_idx=0):
        ret = (pfo_weight * target_ret).sum(dim=1);mean_ret = torch.mean(ret)
        if ret.numel() > 1 and not torch.isnan(ret).all():
            sd_ret = torch.std(ret, unbiased=False) + 1e-10
            if sd_ret < 1e-6:print("SD - ret too low (<1e-6), skip batch.");return None  
        else:print("ret invalid, skip batch.");return None 
        loss = -(self.return_pen * torch.sign(mean_ret) * mean_ret.abs().pow(self.return_exp))
        loss += self.sd_pen*sd_ret.pow(self.sd_exp) 
        loss += self.exp_pen*(pfo_weight * asset_sd).sum(dim=1).abs().pow(self.exp_exp).mean() 
        #print(f"-Epoch/Batch: {epoch} / {batch_idx}")
        #print(f"-Mean/SD/Exp Pen: {-self.return_pen * mean_ret.pow(self.return_exp) :.6f} / {(self.sd_pen * sd_ret.pow(self.sd_exp)):.6f} / {(self.exp_pen*(pfo_weight * asset_sd).sum(dim=1).abs().pow(self.exp_exp).mean()):.6f} ")
        #print(f"Loss/Mean/SD: {loss:.6f} / {mean_ret:.6f} / {sd_ret:.6f}")
        return loss


if __name__ == "__main__":
    from test import run_btest
    print(f"Configured TICK: {config['TICK']} (count: {len(config['TICK'])})")
    TICK = config["TICK"]; feat_list = config["FEAT"]; macro_keys = config["MACRO"]
    if isinstance(macro_keys, str): macro_keys = [k.strip() for k in macro_keys.split(",") if k.strip()]
    results = run_btest(device=DEVICE,initial_capital=INITIAL_CAPITAL, split=config["SPLIT"],lback=config["LBACK"],comp_feat=comp_feat,
                        norm_feat=norm_feat,TICK=TICK,start=config["START"],end=config["END"], feat=feat_list,macro_keys=macro_keys,test_chunk=config["TEST_CHUNK"],
                        plot=True,config=config,)     
    pfo_sharpe = results["pfo"].get("sharpe", float('nan'));max_down = results["pfo"].get("max_down", float('nan'))
    cagr = results["pfo"].get("cagr", float('nan'));bench_sharpe = results["bench"].get("sharpe", float('nan'))
    bench_down = results["bench"].get("max_down", float('nan'));bench_cagr = results["bench"].get("cagr", float('nan'))
    weight = pd.read_csv("csv/weight.csv", index_col="Date", parse_dates=True)
    exp_delta = weight.loc[(weight["total"] < 100), "total"].sum()

    print(f"\nSharpe Ratio: Strat: {pfo_sharpe * 100:.6f}%");print(f"Sharpe Ratio: Bench: {bench_sharpe * 100:.6f}%")
    print(f"Max Down: Strat: {max_down * 100:.6f}%");print(f"Max Down: Bench: {bench_down * 100:.6f}%")
    print(f"CAGR: Strat: {cagr * 100:.6f}%");print(f"CAGR: Bench: {bench_cagr * 100:.6f}%\n")
    print(f"Total Exp Delta: {exp_delta:.4f}");print("Avg Bench Outperf thru Chunks:")
    for k, v in results["perf_outperf"].items(): print(f"{k}: {v * 100:.6f}%")
    sys.stdout.flush()
