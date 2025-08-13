from __future__ import annotations
import os, math, warnings, json
from dataclasses import dataclass
from typing import Dict, List, Optional
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

try:
    import xgboost as xgb
except Exception:
    xgb = None

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
TICKERS = ['JPM','MSFT','NVDA','AVGO','LLY','COST','MA','XOM','UNH','AMZN','CAT','ADBE']
# Use robust Yahoo tickers; DX-Y.NYB often works better than DXY=X; also try DX=F fallback dynamically.
MACRO_TICKERS = ['^VIX','^TNX','^GSPC','GLD','TLT','DX-Y.NYB']
SECTOR_MAP = {'JPM':'XLF','MSFT':'XLK','NVDA':'XLK','AVGO':'XLK','LLY':'XLV','COST':'XLP','MA':'XLF','XOM':'XLE','UNH':'XLV','AMZN':'XLY','CAT':'XLI','ADBE':'XLK'}
START, END, SPLIT = '2012-01-01','2020-12-31','2017-01-01'
REBALANCE_FREQ = 'W-FRI'
TARGET_HORIZON_DAYS = 5
RETRAIN_FREQ_DAYS = 20
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - m) / (sd + 1e-12)

def winsorize(s: pd.Series, z: float = 3.0) -> pd.Series:
    m, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s
    return s.clip(m - z*sd, m + z*sd)

def decay_weights(n: int, hl: float = 20.0) -> np.ndarray:
    return np.exp(-np.log(2) * np.arange(n-1, -1, -1) / max(1e-9, hl))

def rolling_beta(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    # beta = Cov(x,y)/Var(y) over rolling window
    cov = x.rolling(w).cov(y)
    var = y.rolling(w).var()
    return cov / (var + 1e-12)

def rolling_z(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return (s - mu) / (sd + 1e-12)

# -----------------------------
# Data Loader
# -----------------------------
class PriceLoader:
    def __init__(self, cache_path: str = 'csv/institutional_data.csv'):
        self.cache_path = cache_path
        os.makedirs('csv', exist_ok=True)

    def _safe_download(self, t: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if not d.empty:
                return d
        except Exception:
            pass
        # DX-Y.NYB fallback
        if t == 'DX-Y.NYB':
            try:
                d = yf.download('DX=F', start=start, end=end, auto_adjust=True, progress=False)
                if not d.empty:
                    return d
            except Exception:
                pass
        return None

    def load(self, start=START, end=END) -> pd.DataFrame:
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=30)
        start_dl = pd.to_datetime(start) - pd.Timedelta(days=800)
        all_tickers = sorted(set(TICKERS + MACRO_TICKERS + list(SECTOR_MAP.values())))

        if os.path.exists(self.cache_path):
            df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.isna()]
            if not df.empty and df.index.min() <= pd.to_datetime(start) and df.index.max() >= end_dt:
                return df.loc[start:end]

        cols: Dict[tuple, pd.Series] = {}
        for t in all_tickers:
            d = self._safe_download(t, start_dl, end_dt)
            if d is None or d.empty:
                print(f"Warning: failed to download {t}")
                continue
            cols[(t,'close')] = d['Close']
            cols[(t,'vol')]   = d['Volume'] if 'Volume' in d else pd.Series(1e6, index=d.index)
            cols[(t,'high')]  = d['High'] if 'High' in d else d['Close']
            cols[(t,'low')]   = d['Low'] if 'Low' in d else d['Close']

        if not cols:
            raise RuntimeError('No data downloaded')
        df = pd.concat(cols, axis=1).sort_index()
        df = df.ffill()
        df.to_csv(self.cache_path)
        return df.loc[start:end]

# -----------------------------
# Feature Engineering
# -----------------------------
class FeatureEngine:
    def __init__(self, windows: List[int] = [5,10,20,60,120]):
        self.windows = windows

    def _tech_block(self, close: pd.Series, vol: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
        r = close.pct_change()
        f = pd.DataFrame(index=close.index)
        for w in self.windows:
            f[f'ret_{w}'] = close.pct_change(w)
            roll = r.rolling(w)
            f[f'vol_{w}'] = roll.std()
            f[f'mom_{w}'] = roll.mean() / (roll.std() + 1e-12)
            f[f'rev_{w}'] = -roll.sum()
            f[f'skew_{w}'] = roll.skew()
            f[f'kurt_{w}'] = roll.kurt()
        # RSI
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down + 1e-12)
        f['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        exp1, exp2 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = exp1 - exp2
        f['macd'] = macd - macd.ewm(span=9).mean()
        # BB pos
        ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
        f['bb_pos'] = (close - ma20) / (2*sd20 + 1e-12)
        # Liquidity & range
        f['dollar_vol'] = (close * vol).rolling(20).mean()
        f['amihud'] = (r.abs() / (close * vol + 1e-12)).rolling(20).mean()
        f['hl_ratio'] = (high - low) / (close + 1e-12)
        return f

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = []
        for t in TICKERS:
            if (t,'close') not in data:
                continue
            p = data[(t,'close')].ffill()
            v = data[(t,'vol')].ffill()
            h = data[(t,'high')].ffill()
            l = data[(t,'low')].ffill()
            f = self._tech_block(p,v,h,l)
            # sector features (safe, per-ticker Series)
            sec = SECTOR_MAP.get(t)
            if sec and (sec,'close') in data:
                sec_ret = data[(sec,'close')].pct_change()
                r_t = p.pct_change()
                beta_series = rolling_beta(r_t[[t]], sec_ret, 60)
                # Beta calculation block (already in your code)
                if isinstance(beta_series, pd.DataFrame):
                    beta_series = beta_series.iloc[:, 0]  # take the single column as Series
                f[(t, 'sector_beta')] = beta_series

                # Alpha calculation block (mirroring beta structure)
                alpha_series = r_t.rolling(20).mean() - sec_ret.rolling(20).mean()
                if isinstance(alpha_series, pd.DataFrame):
                    alpha_series = alpha_series.iloc[:, 0]  # take the single column as Series
                f[(t, 'sector_alpha')] = alpha_series

            feats.append(f.add_suffix(f'_{t}'))

        # Macro features
        for m in MACRO_TICKERS:
            if (m,'close') in data:
                mret = data[(m,'close')].pct_change()
                f = pd.DataFrame(index=data.index)
                f[f'{m}_ret'] = mret
                f[f'{m}_vol'] = mret.rolling(20).std()
                f[f'{m}_z60'] = rolling_z(mret, 60)
                feats.append(f)

        X = pd.concat(feats, axis=1)
        # Cross-sectional features
        valid = [t for t in TICKERS if (t,'close') in data]
        if valid:
            closes = pd.concat({t: data[(t,'close')] for t in valid}, axis=1)
            X['breadth'] = (closes.pct_change() > 0).sum(axis=1) / len(valid)
            X['xsec_mom20'] = closes.pct_change(20).mean(axis=1)
        X['dow'] = X.index.dayofweek
        X['month'] = X.index.month
        X['qtr'] = X.index.quarter

        # Clean & scale per-column with robust z
        X = X.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        X = X.apply(winsorize).apply(zscore).fillna(0.0)
        return X

# -----------------------------
# Regime Detector (vol/corr/breadth)
# -----------------------------
class RegimeDetector:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def detect(self, returns: pd.DataFrame) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype='int')
        vol = returns.rolling(self.lookback).std().mean(axis=1)
        # realized average cross-correlation
        def avg_corr(window: pd.DataFrame) -> float:
            if window.shape[0] < 5:
                return 0.0
            c = window.corr()
            if c.shape[0] < 2:
                return 0.0
            iu = np.triu_indices_from(c, k=1)
            vals = c.values[iu]
            return float(np.nanmean(vals))
        corr = returns.rolling(self.lookback).apply(lambda w: avg_corr(pd.DataFrame(w, columns=returns.columns)), raw=False)
        breadth = (returns > 0).sum(axis=1) / max(1, returns.shape[1])
        reg = pd.Series(0, index=returns.index)
        reg[(vol > vol.quantile(0.85)) | (corr > corr.quantile(0.85))] = 1
        reg[(vol > vol.quantile(0.95)) | (corr > corr.quantile(0.92)) | (breadth < 0.2)] = 2
        return reg.fillna(method='ffill').fillna(0).astype(int)

# -----------------------------
# Targets
# -----------------------------
class TargetBuilder:
    def __init__(self, horizon: int = TARGET_HORIZON_DAYS):
        self.h = horizon

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        Y = {}
        for t in TICKERS:
            if (t,'close') in data:
                ret = data[(t,'close')].pct_change(self.h)
                Y[t] = ret.shift(-self.h)  # future return over next h days
        return pd.DataFrame(Y).dropna(how='all')

# -----------------------------
# Neural Model (multi-output per-asset)
# -----------------------------
class NeuralAlpha(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h//2, h//4), nn.ReLU(),
            nn.Linear(h//4, out_dim)
        )

    def forward(self, x):
        return torch.tanh(self.net(x))

# -----------------------------
# ML Ensemble (per-asset + neural) with regime specialization
# -----------------------------
class EnsembleAlpha:
    def __init__(self):
        self.models: Dict[str, Dict[str, object]] = {r: {} for r in ['default','r1','r2']}
        self.scalers: Dict[str, RobustScaler] = {}
        self.selected_features: List[str] = []
        self.neural: Dict[str, NeuralAlpha] = {}

    def _select_features(self, X: pd.DataFrame, Y: pd.Series, k: int = 120) -> List[str]:
        valid = Y.dropna().index
        Xv, Yv = X.loc[valid], Y.loc[valid]
        if len(Xv) < 200:
            return list(X.columns)[:min(k, X.shape[1])]
        mi = mutual_info_regression(Xv.fillna(0), Yv.values, random_state=SEED)
        order = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        selected: List[str] = []
        corr = Xv.corr().abs()
        for f in order.index:
            if not selected:
                selected.append(f)
            else:
                if corr.loc[f, selected].max() < 0.75:
                    selected.append(f)
            if len(selected) >= k:
                break
        return selected

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, regimes: Optional[pd.Series] = None):
        # Use cross-asset mean as feature selector target to avoid peeking per-asset
        y_selector = Y.mean(axis=1)
        self.selected_features = self._select_features(X, y_selector)
        Xsel = X[self.selected_features].fillna(0)

        # Define regime keys
        keys = ['default']
        if regimes is not None:
            keys = ['default','r1','r2']
        for key in keys:
            if key == 'default':
                idx = Xsel.index.intersection(Y.index)
            else:
                rcode = {'r1':1,'r2':2}[key]
                idx = regimes[regimes==rcode].index.intersection(Xsel.index).intersection(Y.index)
                if len(idx) < 150:
                    continue

            Xk, Yk = Xsel.loc[idx], Y.loc[idx]
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(Xk)
            self.scalers[key] = scaler

            # Per-asset tabular models
            mdl_bucket: Dict[str, object] = {}
            for t in Y.columns:
                y = Yk[t].fillna(0).values
                models = []
                # GBM
                gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)
                gbm.fit(X_scaled, y)
                models.append(('gbm', gbm))
                # SGD Huber
                sgd = SGDRegressor(loss='huber', penalty='elasticnet', alpha=1e-4, random_state=SEED, max_iter=2000)
                sgd.fit(X_scaled, y)
                models.append(('sgd', sgd))
                # XGB (optional)
                if xgb is not None:
                    xg = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, reg_lambda=1.0)
                    xg.fit(X_scaled, y, verbose=False)
                    models.append(('xgb', xg))
                mdl_bucket[t] = models

            # Neural multi-output
            net = NeuralAlpha(Xsel.shape[1], Y.shape[1])
            Xten = torch.tensor(X_scaled, dtype=torch.float32)
            Yten = torch.tensor(Yk.fillna(0).values, dtype=torch.float32)
            ds = TensorDataset(Xten, Yten)
            dl = TorchLoader(ds, batch_size=128, shuffle=True)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            # simple early stopping
            best_loss, patience, counter = float('inf'), 8, 0
            best_state = None
            for epoch in range(100):
                net.train()
                running = 0.0
                for xb,yb in dl:
                    opt.zero_grad()
                    pred = net(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    running += loss.item()*len(xb)
                epoch_loss = running/len(ds)
                if epoch_loss + 1e-6 < best_loss:
                    best_loss = epoch_loss
                    counter = 0
                    best_state = {k:v.cpu().clone() for k,v in net.state_dict().items()}
                else:
                    counter += 1
                    if counter >= patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)

            self.models[key] = {
                'per_asset': mdl_bucket,
                'stacker': Ridge(alpha=1.0),  # will be fit below
                'neural': net
            }

            # Fit stacker on out-of-fold preds for robustness
            tscv = TimeSeriesSplit(n_splits=5)
            oof_preds = []
            oof_y = []
            Xarr = X_scaled
            Yarr = Yk.fillna(0).values  # shape: [n, assets]
            for tr, vl in tscv.split(Xarr):
                Xtr, Xvl = Xarr[tr], Xarr[vl]
                Ytr, Yvl = Yarr[tr], Yarr[vl]
                # Train quick per-asset clones (re-using above could leak); keep it light
                P = []
                for ai, t in enumerate(Yk.columns):
                    ytr = Ytr[:,ai]
                    mdl = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)
                    mdl.fit(Xtr, ytr)
                    P.append(mdl.predict(Xvl))
                P = np.vstack(P).T  # [n_vl, assets]
                # neural preds
                with torch.no_grad():
                    net.eval()
                    Pn = net(torch.tensor(Xvl, dtype=torch.float32)).numpy()
                # combine features to stack
                Z = np.hstack([P, Pn])  # [n_vl, 2*assets]
                oof_preds.append(Z)
                oof_y.append(Yvl)
            Zfull = np.vstack(oof_preds)
            Yfull = np.vstack(oof_y)
            stacker = self.models[key]['stacker']
            stacker.fit(Zfull, Yfull)

    def predict(self, X: pd.DataFrame, regime_code: int = 0) -> pd.Series:
        if not self.selected_features:
            return pd.Series(0.0, index=TICKERS)
        key = {0:'default',1:'r1',2:'r2'}.get(regime_code,'default')
        if key not in self.models or not self.models[key]:
            key = 'default'
        scaler = self.scalers.get(key)
        if scaler is None:
            return pd.Series(0.0, index=TICKERS)
        Xsel = X[self.selected_features].fillna(0)
        Xsc = scaler.transform(Xsel)
        mdlpack = self.models[key]

        # per-asset preds
        P = []
        for t in TICKERS:
            if t not in mdlpack['per_asset']:
                P.append(np.zeros((Xsc.shape[0],)))
                continue
            preds = []
            for name, mdl in mdlpack['per_asset'][t]:
                try:
                    preds.append(mdl.predict(Xsc))
                except Exception:
                    continue
            if preds:
                w = decay_weights(len(preds), hl=5)
                preds = np.average(np.vstack(preds), axis=0, weights=w)
            else:
                preds = np.zeros((Xsc.shape[0],))
            P.append(preds)
        P = np.vstack(P).T  # [n, assets]

        # neural preds
        with torch.no_grad():
            Np = mdlpack['neural'](torch.tensor(Xsc, dtype=torch.float32)).numpy()
        Z = np.hstack([P, Np])  # [n, 2*assets]
        Yhat = mdlpack['stacker'].predict(Z)  # [n, assets]
        # Take last row predictions
        last = Yhat[-1]
        return pd.Series(np.tanh(last), index=TICKERS)

# -----------------------------
# Portfolio Construction: Black-Litterman + volatility targeting + turnover costs
# -----------------------------
class PortfolioConstructor:
    def __init__(self, target_vol: float = 0.15, max_w: float = 0.12, sector_cap: float = 0.4, tc_bps: float = 8.0):
        self.target_vol = target_vol
        self.max_w = max_w
        self.sector_cap = sector_cap
        self.tc_bps = tc_bps

    def _black_litterman(self, cov: np.ndarray, market_weights: np.ndarray, views: np.ndarray, tau: float = 0.05) -> np.ndarray:
        # Simple BL: combine equilibrium returns (pi) with views
        lam = 3.0  # risk aversion
        pi = lam * cov @ market_weights
        P = np.eye(len(views))
        Omega = np.diag(np.maximum(1e-6, np.diag(P @ (tau*cov) @ P.T)))
        post = np.linalg.inv(np.linalg.inv(tau*cov) + P.T @ np.linalg.inv(Omega) @ P) @ (np.linalg.inv(tau*cov) @ pi + P.T @ np.linalg.inv(Omega) @ views)
        return post

    def optimize(self, alpha: Dict[str, float], returns: pd.DataFrame, current: Optional[Dict[str, float]] = None, regime: int = 0) -> Dict[str,float]:
        if not alpha:
            return {}
        symbols = [s for s,v in alpha.items() if not np.isnan(v)]
        a = np.array([alpha[s] for s in symbols])
        rets = returns[symbols].dropna()
        if len(rets) < 60:
            cov = np.eye(len(symbols)) * (0.04 * (1 + 0.5*regime))
        else:
            cov = LedoitWolf().fit(rets.values).covariance_
            cov *= (1.0 + 0.5*regime)  # crisis inflate
        mktw = np.ones(len(symbols)) / len(symbols)
        bl_mu = self._black_litterman(cov, mktw, a)

        # objective: maximize sharpe with tc penalty & Kelly cap
        def obj(w):
            pr = float(w @ bl_mu)
            pv = math.sqrt(max(1e-12, w @ cov @ w))
            sr = pr / (pv + 1e-12)
            if current is not None:
                cw = np.array([current.get(s,0.0) for s in symbols])
                turnover = np.sum(np.abs(w - cw))
                sr -= (self.tc_bps/1e4) * turnover  # penalize
            # Kelly fraction cap: discourage extreme leverage in high regimes
            kcap = 1.0 if regime == 0 else (0.8 if regime == 1 else 0.6)
            return -(min(sr, kcap))

        bounds = [(0.0, self.max_w if regime < 2 else self.max_w*0.8) for _ in symbols]
        cons = ({'type':'eq','fun': lambda w: np.sum(w) - 1.0},)
        x0 = np.ones(len(symbols))/len(symbols)
        try:
            res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP')
            w = res.x if res.success else x0
        except Exception:
            w = x0
        # sector cap
        wdict = dict(zip(symbols, w))
        sector_weights: Dict[str,float] = {}
        for s, val in list(wdict.items()):
            sec = SECTOR_MAP.get(s,'OTHER')
            sector_weights[sec] = sector_weights.get(sec,0.0) + val
        for sec, sw in sector_weights.items():
            if sw > self.sector_cap:
                scale = self.sector_cap / sw
                for s in symbols:
                    if SECTOR_MAP.get(s,'OTHER') == sec:
                        wdict[s] *= scale
        # renormalize
        total = sum(max(0.0,v) for v in wdict.values())
        if total > 0:
            for s in wdict:
                wdict[s] = max(0.0, wdict[s]) / total
        # vol target scaling (ex-post applied as cash allocation)
        wvec = np.array([wdict[s] for s in symbols])
        port_vol = math.sqrt(float(wvec @ cov @ wvec))
        lev = min(1.0, self.target_vol / max(1e-6, port_vol))
        for s in wdict:
            wdict[s] *= lev
        # drop dust
        return {k:v for k,v in wdict.items() if v > 0.005}

# -----------------------------
# Backtester
# -----------------------------
@dataclass
class BacktestResult:
    total_return: float
    ann_return: float
    vol: float
    sharpe: float
    calmar: float
    max_dd: float
    win_rate: float
    equity_curve: pd.Series
    final_weights: Dict[str,float]

class Backtester:
    def __init__(self, slippage_bps: float = 2.0, tc_bps: float = 8.0):
        self.slip = slippage_bps
        self.tc = tc_bps

    def run(self,
            prices: pd.DataFrame,
            features: pd.DataFrame,
            returns: pd.DataFrame,
            regimes: pd.Series,
            model: EnsembleAlpha,
            builder: PortfolioConstructor,
            start: str = SPLIT,
            end: str = END,
            retrain_freq_days: int = RETRAIN_FREQ_DAYS) -> BacktestResult:
        dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq=REBALANCE_FREQ)
        eq = 1.0
        equity = []
        weights: Dict[str,float] = {}
        last_train = None
        tb = TargetBuilder(TARGET_HORIZON_DAYS)
        Y = tb.build(prices)

        # Precompute simple returns for portfolio computation
        price_panel = prices[[ (t,'close') for t in TICKERS if (t,'close') in prices]].droplevel(1, axis=1)
        rets_full = price_panel.pct_change()

        for i, d in enumerate(dates):
            if d not in features.index or d not in price_panel.index:
                continue
            # retrain on rolling window of ~3 years
            if last_train is None or (d - last_train).days >= retrain_freq_days:
                train_end = d - BDay(1)
                train_start = train_end - BDay(750)
                Xtr = features.loc[train_start:train_end]
                Ytr = Y.loc[train_start:train_end]
                Regtr = regimes.loc[train_start:train_end]
                if len(Xtr) > 300 and len(Ytr) > 300:
                    model.fit(Xtr, Ytr, Regtr)
                    last_train = d

            # predict using last row feature
            reg = regimes.loc[:d].iloc[-1] if len(regimes.loc[:d]) else 0
            xwin = features.loc[:d].iloc[-1:]
            preds = model.predict(xwin, reg)
            # form signals
            sig = preds.clip(-1,1).to_dict()
            sig = {k: float(v) for k,v in sig.items() if abs(v) > 0.05}
            if not sig:
                equity.append((d, eq))
                continue
            # optimize
            target = builder.optimize(sig, rets_full.loc[:d], current=weights, regime=int(reg))
            if not target:
                equity.append((d, eq))
                continue
            # apply costs
            turnover = sum(abs(target.get(t,0.0) - weights.get(t,0.0)) for t in set(list(target.keys()) + list(weights.keys())))
            tc_cost = (self.tc/1e4) * turnover
            slip_cost = (self.slip/1e-4) * 0.0  # placeholder if you want to model slippage per trade

            # hold until next rebalance
            next_d = dates[i+1] if i+1 < len(dates) else d + BDay(TARGET_HORIZON_DAYS)
            per = rets_full.loc[d:next_d, list(target.keys())]
            if per.empty:
                equity.append((d, eq))
                continue
            # daily portfolio returns then compound over period
            daily_port = (per * pd.Series(target)).sum(axis=1)
            gross = float((1.0 + daily_port).prod() - 1.0)
            eq *= (1.0 + gross - tc_cost - slip_cost)
            equity.append((d, eq))
            weights = target

        if not equity:
            raise RuntimeError('No equity points generated')
        curve = pd.Series({d:v for d,v in equity}).sort_index()
        daily = curve.pct_change().dropna()
        ann = daily.mean()*252
        vol = daily.std()*np.sqrt(252)
        sharpe = ann/(vol+1e-12)
        rollmax = curve.cummax()
        drawdown = (curve/rollmax - 1.0)
        max_dd = float(drawdown.min())
        calmar = (ann/(abs(max_dd)+1e-12)) if max_dd < 0 else np.inf
        win_rate = float((daily>0).mean())
        return BacktestResult(total_return=float(curve.iloc[-1]-1.0), ann_return=float(ann), vol=float(vol), sharpe=float(sharpe), calmar=float(calmar), max_dd=float(max_dd), win_rate=win_rate, equity_curve=curve, final_weights=weights)

# -----------------------------
# Orchestrator
# -----------------------------
class HybridInstitutionalTrader:
    def __init__(self):
        self.loader = PriceLoader()
        self.fe = FeatureEngine()
        self.regimes = RegimeDetector()
        self.ensemble = EnsembleAlpha()
        self.portfolio = PortfolioConstructor()
        self.bt = Backtester()

    def run(self, start=START, end=END, split=SPLIT) -> Dict[str,object]:
        data = self.loader.load(start, end)
        # flat price frame for convenience
        prices_simple = pd.concat({t: data[(t,'close')] for t in TICKERS if (t,'close') in data}, axis=1)
        returns = prices_simple.pct_change()
        X = self.fe.build(data)
        regs = self.regimes.detect(returns.fillna(0))

        res = self.bt.run(data, X, returns, regs, self.ensemble, self.portfolio, start=split, end=end)
        # Print concise summary
        print(json.dumps({
            'TotalReturn_%': round(res.total_return*100,2),
            'AnnReturn_%': round(res.ann_return*100,2),
            'Vol_%': round(res.vol*100,2),
            'Sharpe': round(res.sharpe,2),
            'Calmar': round(res.calmar,2),
            'MaxDD_%': round(res.max_dd*100,2),
            'WinRate_%': round(res.win_rate*100,2),
        }, indent=2))
        print('\nFinal Weights:')
        for k,v in sorted(res.final_weights.items(), key=lambda x: x[1], reverse=True):
            if v>0.01:
                print(f'{k}: {v:.2%}')
        return {
            'summary': res,
        }

if __name__ == '__main__':
    trader = HybridInstitutionalTrader()
    trader.run()
