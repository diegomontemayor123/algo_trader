from __future__ import annotations
import os, math, warnings, json
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchLoader
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize

try: import xgboost as xgb
except: xgb = None

warnings.filterwarnings("ignore")

TICKERS = ['JPM','MSFT','NVDA','AVGO','LLY','COST','MA','XOM','UNH','AMZN','CAT','ADBE']
MACRO_TICKERS = ['^VIX','^TNX','^GSPC','GLD','TLT','DX-Y.NYB']
SECTOR_MAP = {'JPM':'XLF','MSFT':'XLK','NVDA':'XLK','AVGO':'XLK','LLY':'XLV','COST':'XLP','MA':'XLF','XOM':'XLE','UNH':'XLV','AMZN':'XLY','CAT':'XLI','ADBE':'XLK'}
START, END, SPLIT = '2012-01-01','2020-12-31','2017-01-01'
REBALANCE_FREQ, TARGET_HORIZON_DAYS, RETRAIN_FREQ_DAYS, SEED = 'W-FRI', 5, 20, 42
np.random.seed(SEED); torch.manual_seed(SEED)

def zscore(s): m,sd = s.mean(),s.std(); return s*0 if sd==0 or np.isnan(sd) or pd.isna(sd) else (s-m)/(getattr(sd,'iloc')[0] if hasattr(sd,'iloc') and len(sd)>0 else sd+1e-12)
def winsorize(s,z=3.0): m,sd = s.mean(),s.std(); return s if sd==0 or np.isnan(sd) else s.clip(m-z*sd,m+z*sd)
def decay_weights(n,hl=20.0): return np.exp(-np.log(2)*np.arange(n-1,-1,-1)/max(1e-9,hl))

class PriceLoader:
    def __init__(self,cache_path='csv/institutional_data.csv'): self.cache_path=cache_path; os.makedirs('csv',exist_ok=True)
    def load(self,start=START,end=END):
        end_dt,all_tickers = pd.to_datetime(end)+pd.Timedelta(days=30),sorted(set(TICKERS+MACRO_TICKERS+list(SECTOR_MAP.values())))
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path,index_col=0,parse_dates=True)
                if hasattr(df.index,'min') and len(df)>0 and df.index.min()<=pd.to_datetime(start) and df.index.max()>=end_dt: return df.loc[start:end]
            except: pass
        cols = {}
        for t in all_tickers:
            try:
                d = yf.download(t,start=pd.to_datetime(start)-pd.Timedelta(days=800),end=end_dt,auto_adjust=True,progress=False)
                if not d.empty: cols.update({(t,'close'):d['Close'],(t,'vol'):d.get('Volume',pd.Series(1e6,index=d.index)),(t,'high'):d.get('High',d['Close']),(t,'low'):d.get('Low',d['Close'])})
            except: continue
        if not cols: raise RuntimeError('No data downloaded')
        df = pd.concat(cols,axis=1).sort_index().ffill()
        try: df.to_csv(self.cache_path)
        except: pass
        return df.loc[start:end]

class FeatureEngine:
    def __init__(self, windows=[5,10,20,60,120]):
        self.windows = windows
        self.feature_columns = None

    def _tech_block(self, close, vol, high, low):
        r = close.pct_change()
        f = pd.DataFrame(index=close.index)
        for w in self.windows:
            roll = r.rolling(w)
            for w in self.windows:
                f[f'ret_{w}'] = close.pct_change(w)
                f[f'vol_{w}']  = roll.std().reindex(close.index).fillna(0.0)
                f[f'mom_{w}']  = (roll.mean() / (roll.std() + 1e-12)).reindex(close.index).fillna(0.0)
                f[f'rev_{w}']  = (-roll.sum()).reindex(close.index).fillna(0.0)
                f[f'skew_{w}'] = roll.skew().reindex(close.index).fillna(0.0)
                f[f'kurt_{w}'] = roll.kurt().reindex(close.index).fillna(0.0)
                delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down + 1e-12)
        f['rsi'] = 100 - (100 / (1 + rs))
        exp1, exp2 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = exp1 - exp2
        f['macd'] = macd - macd.ewm(span=9).mean()
        ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
        f['bb_pos'] = (close - ma20) / (2 * sd20 + 1e-12)
        f['dollar_vol'] = (close*vol).rolling(20).mean().reindex(close.index).fillna(0.0)
        f['amihud']     = (r.abs()/(close*vol + 1e-12)).rolling(20).mean().reindex(close.index).fillna(0.0)
        f['hl_ratio']   = ((high - low)/(close + 1e-12)).reindex(close.index).fillna(0.0)
        return f

    def build(self, data):
        feats = []
        for t in TICKERS:
            if (t,'close') not in data: continue
            p,v,h,l = data[(t,'close')].ffill(), data[(t,'vol')].ffill(), data[(t,'high')].ffill(), data[(t,'low')].ffill()
            f = self._tech_block(p,v,h,l).fillna(0.0)
            feats.append(f.add_suffix(f'_{t}'))
        X = pd.concat(feats, axis=1) if feats else pd.DataFrame(index=data.index)
        X['dow'] = X.index.dayofweek
        X['month'] = X.index.month
        X['qtr'] = X.index.quarter
        X = X.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        X = X.apply(winsorize).apply(zscore).fillna(0.0)
        if self.feature_columns is None: self.feature_columns = list(X.columns)
        return X

class TargetBuilder:
    def __init__(self, horizon=TARGET_HORIZON_DAYS):
        self.h = horizon

    def build(self, data):
        targets = {}
        for t in TICKERS:
            if (t,'close') not in data: continue
            series = data[(t,'close')].pct_change(self.h).shift(-self.h)
            tmp = pd.Series(0.0, index=data.index)
            tmp.iloc[:len(series)] = series.fillna(0.0).values.ravel()
            targets[t] = tmp
        return pd.DataFrame(targets, index=data.index)


class RegimeDetector:
    def __init__(self,lookback=60): self.lookback=lookback
    def detect(self,returns):
        if returns.empty or len(returns)<self.lookback: return pd.Series(0,index=returns.index,dtype=int)
        try:
            vol,breadth = returns.rolling(self.lookback).std().mean(axis=1),(returns>0).sum(axis=1)/max(1,returns.shape[1])
            reg = pd.Series(0,index=returns.index)
            reg[vol>vol.quantile(0.85)],reg[(vol>vol.quantile(0.95))|(breadth<0.2)] = 1,2
            return reg.ffill().fillna(0).astype(int)
        except: return pd.Series(0,index=returns.index,dtype=int)

class NeuralAlpha(nn.Module):
    def __init__(self,in_dim,out_dim): super().__init__(); h=256; self.net=nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h//2,h//4),nn.ReLU(),nn.Linear(h//4,out_dim))
    def forward(self,x): return torch.tanh(self.net(x))

class EnsembleAlpha:
    def __init__(self): self.models,self.scalers,self.selected_features,self.neural = {r:{} for r in ['default','r1','r2']},{},[],{}
    def _select_features(self,X,Y,k=120):
        valid = Y.dropna().index; Xv,Yv = X.loc[valid],Y.loc[valid]
        if len(Xv)<200: return list(X.columns)[:min(k,X.shape[1])]
        try: mi,order = mutual_info_regression(Xv.fillna(0),Yv.values,random_state=SEED),pd.Series(mutual_info_regression(Xv.fillna(0),Yv.values,random_state=SEED),index=X.columns).sort_values(ascending=False)
        except: return list(X.columns)[:min(k,X.shape[1])]
        selected,corr = [],Xv.corr().abs()
        for f in order.index:
            if not selected or corr.loc[f,selected].max()<0.75: selected.append(f)
            if len(selected)>=k: break
        return selected
    def fit(self,X,Y,regimes=None):
        try: y_selector=Y.mean(axis=1); self.selected_features=self._select_features(X,y_selector); Xsel=X[self.selected_features].fillna(0)
        except: return
        keys = ['default'] if regimes is None else ['default','r1','r2']
        for key in keys:
            try:
                if key=='default': idx=Xsel.index.intersection(Y.index)
                else: rcode={'r1':1,'r2':2}[key]; idx=regimes[regimes==rcode].index.intersection(Xsel.index).intersection(Y.index); 
                if len(idx)<150: continue
                Xk,Yk,scaler = Xsel.loc[idx],Y.loc[idx],RobustScaler(); X_scaled=scaler.fit_transform(Xk); self.scalers[key]=scaler; mdl_bucket={}
                for t in Y.columns:
                    try:
                        y,models = Yk[t].fillna(0).values,[]
                        gbm=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=SEED); gbm.fit(X_scaled,y); models.append(('gbm',gbm))
                        sgd=SGDRegressor(loss='huber',penalty='elasticnet',alpha=1e-4,random_state=SEED,max_iter=2000); sgd.fit(X_scaled,y); models.append(('sgd',sgd))
                        if xgb: xg=xgb.XGBRegressor(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=SEED,reg_lambda=1.0); xg.fit(X_scaled,y,verbose=False); models.append(('xgb',xg))
                        mdl_bucket[t] = models
                    except: continue
                net=NeuralAlpha(Xsel.shape[1],Y.shape[1]); Xten,Yten=torch.tensor(X_scaled,dtype=torch.float32),torch.tensor(Yk.fillna(0).values,dtype=torch.float32)
                ds,dl,opt,loss_fn = TensorDataset(Xten,Yten),TorchLoader(TensorDataset(Xten,Yten),batch_size=max(8,min(128,len(ds)//4)),shuffle=True),torch.optim.Adam(net.parameters(),lr=1e-3),nn.MSELoss()
                best_loss,patience,counter = float('inf'),8,0
                for epoch in range(100):
                    net.train(); running=0.0
                    for xb,yb in dl: opt.zero_grad(); pred=net(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step(); running+=loss.item()*len(xb)
                    epoch_loss = running/len(ds)
                    if epoch_loss+1e-6<best_loss: best_loss,counter,best_state = epoch_loss,0,{k:v.cpu().clone() for k,v in net.state_dict().items()}
                    else: counter+=1; 
                    if counter>=patience: break
                try: net.load_state_dict(best_state)
                except: pass
                self.models[key] = {'per_asset':mdl_bucket,'stacker':Ridge(alpha=1.0),'neural':net}
                tscv,oof_preds,oof_y = TimeSeriesSplit(n_splits=5),[],[]
                Xarr,Yarr = X_scaled,Yk.fillna(0).values
                for tr,vl in tscv.split(Xarr):
                    try:
                        Xtr,Xvl,Ytr,Yvl = Xarr[tr],Xarr[vl],Yarr[tr],Yarr[vl]; P=[]
                        for ai,t in enumerate(Yk.columns): ytr=Ytr[:,ai]; mdl=GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=SEED); mdl.fit(Xtr,ytr); P.append(mdl.predict(Xvl))
                        P = np.vstack(P).T
                        with torch.no_grad(): net.eval(); Pn=net(torch.tensor(Xvl,dtype=torch.float32)).numpy()
                        Z = np.hstack([P,Pn]); oof_preds.append(Z); oof_y.append(Yvl)
                    except: continue
                if oof_preds: Zfull,Yfull = np.vstack(oof_preds),np.vstack(oof_y); self.models[key]['stacker'].fit(Zfull,Yfull)
            except: continue
    def predict(self,X,regime_code=0):
        if not self.selected_features: return pd.Series(0.0,index=TICKERS)
        key = {0:'default',1:'r1',2:'r2'}.get(regime_code,'default')
        if key not in self.models or not self.models[key]: key='default'
        scaler = self.scalers.get(key)
        if scaler is None: return pd.Series(0.0,index=TICKERS)
        try: Xsel,Xsc,mdlpack = X[self.selected_features].fillna(0),scaler.transform(X[self.selected_features].fillna(0)),self.models[key]
        except: return pd.Series(0.0,index=TICKERS)
        P = []
        for t in TICKERS:
            if t not in mdlpack['per_asset']: P.append(np.zeros((Xsc.shape[0],))); continue
            try: preds=[mdl.predict(Xsc) for name,mdl in mdlpack['per_asset'][t]]; P.append(np.average(np.vstack(preds),axis=0,weights=decay_weights(len(preds),hl=5)) if preds else np.zeros((Xsc.shape[0],)))
            except: P.append(np.zeros((Xsc.shape[0],)))
        P = np.vstack(P).T
        try:
            with torch.no_grad(): Np=mdlpack['neural'](torch.tensor(Xsc,dtype=torch.float32)).numpy()
            Z,Yhat = np.hstack([P,Np]),mdlpack['stacker'].predict(np.hstack([P,Np]))
            return pd.Series(np.tanh(Yhat[-1]),index=TICKERS)
        except: return pd.Series(np.tanh(np.mean(P,axis=1)[-1]) if len(P)>0 else 0.0,index=TICKERS)

class PortfolioConstructor:
    def __init__(self,target_vol=0.15,max_w=0.12,sector_cap=0.4,tc_bps=8.0): self.target_vol,self.max_w,self.sector_cap,self.tc_bps=target_vol,max_w,sector_cap,tc_bps
    def _black_litterman(self,cov,market_weights,views,tau=0.05):
        try: lam,pi,P,Omega = 3.0,lam*cov@market_weights,np.eye(len(views)),np.diag(np.maximum(1e-6,np.diag(P@(tau*cov)@P.T))); return np.linalg.inv(np.linalg.inv(tau*cov)+P.T@np.linalg.inv(Omega)@P)@(np.linalg.inv(tau*cov)@pi+P.T@np.linalg.inv(Omega)@views)
        except: return views
    def optimize(self,alpha,returns,current=None,regime=0):
        if not alpha: return {}
        try:
            symbols,a = [s for s,v in alpha.items() if not np.isnan(v)],np.array([alpha[s] for s in [s for s,v in alpha.items() if not np.isnan(v)]])
            rets = returns[symbols].dropna()
            cov = np.eye(len(symbols))*(0.04*(1+0.5*regime)) if len(rets)<60 else LedoitWolf().fit(rets.values).covariance_*(1.0+0.5*regime)
            mktw,bl_mu = np.ones(len(symbols))/len(symbols),self._black_litterman(cov,np.ones(len(symbols))/len(symbols),a)
            def obj(w): pr,pv,sr = float(w@bl_mu),math.sqrt(max(1e-12,w@cov@w)),pr/(pv+1e-12); return -(min(sr,1.0 if regime==0 else (0.8 if regime==1 else 0.6))-(self.tc_bps/1e4)*np.sum(np.abs(w-np.array([current.get(s,0.0) for s in symbols]))) if current else 0)
            bounds,cons,x0 = [(0.0,self.max_w if regime<2 else self.max_w*0.8) for _ in symbols],({'type':'eq','fun':lambda w:np.sum(w)-1.0},),np.ones(len(symbols))/len(symbols)
            res = minimize(obj,x0,bounds=bounds,constraints=cons,method='SLSQP')
            w,wdict = res.x if res.success else x0,dict(zip(symbols,res.x if res.success else x0))
            sector_weights = {}
            for s,val in list(wdict.items()): sec=SECTOR_MAP.get(s,'OTHER'); sector_weights[sec]=sector_weights.get(sec,0.0)+val
            for sec,sw in sector_weights.items():
                if sw>self.sector_cap: scale=self.sector_cap/sw; [setattr(wdict,s,wdict[s]*scale) or wdict.update({s:wdict[s]*scale}) for s in symbols if SECTOR_MAP.get(s,'OTHER')==sec]
            total = sum(max(0.0,v) for v in wdict.values())
            if total>0: wdict = {s:max(0.0,wdict[s])/total for s in wdict}
            port_vol,lev = math.sqrt(np.dot(np.array([wdict[s] for s in symbols]),cov@np.array([wdict[s] for s in symbols]))),min(1.0,self.target_vol/max(1e-6,math.sqrt(np.dot(np.array([wdict[s] for s in symbols]),cov@np.array([wdict[s] for s in symbols])))))
            return {k:v*lev for k,v in wdict.items() if v*lev>0.005}
        except: return {}

@dataclass
class BacktestResult:
    total_return: float; ann_return: float; vol: float; sharpe: float; calmar: float; max_dd: float; win_rate: float; equity_curve: pd.Series; final_weights: Dict[str,float]

class Backtester:
    def __init__(self,slippage_bps=2.0,tc_bps=8.0): self.slip,self.tc=slippage_bps,tc_bps
    def run(self,prices,features,returns,regimes,model,builder,start=SPLIT,end=END,retrain_freq_days=RETRAIN_FREQ_DAYS):
        dates,eq,equity,weights,last_train,tb,Y = pd.date_range(pd.to_datetime(start),pd.to_datetime(end),freq=REBALANCE_FREQ),1.0,[],{},None,TargetBuilder(TARGET_HORIZON_DAYS),TargetBuilder(TARGET_HORIZON_DAYS).build(prices)
        for i,d in enumerate(dates):
            if d not in features.index or d not in prices.index: continue
            if last_train is None or (d-last_train).days>=retrain_freq_days:
                try: train_end,train_start = d-BDay(1),d-BDay(1)-BDay(750); Xtr,Ytr,Rtr,Regtr = features.loc[train_start:train_end],Y.loc[train_start:train_end],returns.loc[train_start:train_end],regimes.loc[train_start:train_end]; 
                except: continue
                if len(Xtr)>300 and len(Ytr)>300: model.fit(Xtr,Ytr,Regtr); last_train=d
            try: reg,xwin,preds = regimes.loc[:d].iloc[-1] if len(regimes.loc[:d]) else 0,features.loc[:d].iloc[-1:],model.predict(features.loc[:d].iloc[-1:],regimes.loc[:d].iloc[-1] if len(regimes.loc[:d]) else 0)
            except: continue
            sig = {k:float(v) for k,v in preds.clip(-1,1).to_dict().items() if abs(v)>0.05}
            if not sig: equity.append((d,eq)); continue
            try: ret_hist,target = pd.concat({t:prices[(t,'close')] for t in TICKERS if (t,'close') in prices},axis=1).pct_change(),builder.optimize(sig,pd.concat({t:prices[(t,'close')] for t in TICKERS if (t,'close') in prices},axis=1).pct_change().loc[:d],current=weights,regime=int(reg))
            except: continue
            if not target: equity.append((d,eq)); continue
            turnover,tc_cost,slip_cost = sum(abs(target.get(t,0.0)-weights.get(t,0.0)) for t in set(list(target.keys())+list(weights.keys()))),(self.tc/1e4)*sum(abs(target.get(t,0.0)-weights.get(t,0.0)) for t in set(list(target.keys())+list(weights.keys()))),(self.slip/1e4)*sum(abs(target.get(t,0.0)-weights.get(t,0.0)) for t in target)
            next_d = dates[i+1] if i+1<len(dates) else d+BDay(TARGET_HORIZON_DAYS)
            try: rets,period_ret = returns.loc[d:next_d,list(target.keys())],float((returns.loc[d:next_d,list(target.keys())]*pd.Series(target)).sum(axis=1).sum()); eq*=(1.0+period_ret-tc_cost-slip_cost); equity.append((d,eq)); weights=target
            except: equity.append((d,eq))
        if not equity: raise RuntimeError('No equity points generated')
        curve,daily = pd.Series({d:v for d,v in equity}).sort_index(),pd.Series({d:v for d,v in equity}).sort_index().pct_change().dropna()
        ann,vol,sharpe = daily.mean()*252,daily.std()*np.sqrt(252),daily.mean()*252/(daily.std()*np.sqrt(252)+1e-12)
        rollmax,drawdown,max_dd = curve.cummax(),(curve/curve.cummax()-1.0),float((curve/curve.cummax()-1.0).min())
        calmar,win_rate = (ann/(abs(max_dd)+1e-12)) if max_dd<0 else np.inf,float((daily>0).mean())
        return BacktestResult(total_return=float(curve.iloc[-1]-1.0),ann_return=float(ann),vol=float(vol),sharpe=float(sharpe),calmar=float(calmar),max_dd=float(max_dd),win_rate=win_rate,equity_curve=curve,final_weights=weights)

class HybridInstitutionalTrader:
    def __init__(self): self.loader,self.fe,self.regimes,self.ensemble,self.portfolio,self.bt = PriceLoader(),FeatureEngine(),RegimeDetector(),EnsembleAlpha(),PortfolioConstructor(),Backtester()
    def run(self,start=START,end=END,split=SPLIT):
        print("Loading data..."); data = self.loader.load(start,end)
        print("Building features..."); close_cols,prices_simple,returns,X = [(t,'close') for t in TICKERS if (t,'close') in data],pd.concat({t:data[(t,'close')] for t in TICKERS if (t,'close') in data},axis=1),pd.concat({t:data[(t,'close')] for t in TICKERS if (t,'close') in data},axis=1).pct_change(),self.fe.build(data)
        print("Detecting regimes..."); regs = self.regimes.detect(returns.fillna(0))
        print("Running backtest..."); res = self.bt.run(data,X,returns,regs,self.ensemble,self.portfolio,start=split,end=end)
        print(json.dumps({'TotalReturn_%':round(res.total_return*100,2),'AnnReturn_%':round(res.ann_return*100,2),'Vol_%':round(res.vol*100,2),'Sharpe':round(res.sharpe,2),'Calmar':round(res.calmar,2),'MaxDD_%':round(res.max_dd*100,2),'WinRate_%':round(res.win_rate*100,2)},indent=2))
        print('\nFinal Weights:'); [print(f'{k}: {v:.2%}') for k,v in sorted(res.final_weights.items(),key=lambda x:x[1],reverse=True) if v>0.01]
        return {'summary':res}
if __name__ == '__main__': HybridInstitutionalTrader().run()