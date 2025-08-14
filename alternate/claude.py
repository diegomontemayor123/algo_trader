from __future__ import annotations
import warnings, os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import RobustScaler
from scipy.optimize import minimize
import xgboost as xgb
warnings.filterwarnings("ignore")

TICKERS = ['JPM', 'MSFT', 'NVDA', 'AVGO', 'LLY', 'COST', 'MA', 'XOM', 'UNH', 'AMZN', 'CAT', 'ADBE']
MACRO_TICKERS = ['^VIX', '^TNX', '^GSPC', 'GLD', 'TLT', 'DX-Y.NYB']
SECTOR_MAP = {'JPM':'XLF', 'MSFT':'XLK', 'NVDA':'XLK', 'AVGO':'XLK', 'LLY':'XLV', 'COST':'XLP', 'MA':'XLF', 'XOM':'XLE', 'UNH':'XLV', 'AMZN':'XLY', 'CAT':'XLI', 'ADBE':'XLK'}
START, END, SPLIT = '2015-01-01', '2019-01-01', '2017-01-01'

def winsorize(s, z=3.0): m, sd = s.mean(), s.std(); return s.clip(m - z*sd, m + z*sd) if sd > 0 else s
def zscore(s): return (s - s.mean()) / (s.std() + 1e-8) if s.std() > 1e-8 else s * 0
def decay_weights(n, hl=30): return np.exp(-np.log(2) * np.arange(n-1, -1, -1) / hl)

class DataLoader:
    def __init__(self, cache_path="csv/enhanced_data.csv"): self.cache_path = cache_path; os.makedirs("csv", exist_ok=True)
        
    def load_data(self, start=START, end=END):
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=30)
        start_dt = pd.to_datetime(start)
        all_tickers = TICKERS + MACRO_TICKERS + list(set(SECTOR_MAP.values()))
        
        if os.path.exists(self.cache_path):
            data = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                try:
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.index = pd.to_datetime(data.index)
                    
                    if (data.index.min() <= start_dt) and (data.index.max() >= end_dt):
                        return data.loc[start:end]
                except (TypeError, ValueError) as e:
                    print(f"Warning: Issue with cached data dates, reloading: {e}")
                    pass
        
        raw = {}
        for ticker in all_tickers:
            try:
                df = yf.download(ticker, start=start_dt - pd.Timedelta(days=400), end=end_dt, auto_adjust=True)
                if not df.empty:
                    raw[ticker] = df['Close']
                    raw[f'{ticker}_vol'] = df['Volume'] if 'Volume' in df else pd.Series(1e6, index=df.index)
                    raw[f'{ticker}_high'] = df['High']
                    raw[f'{ticker}_low'] = df['Low']
            except Exception as e:
                print(f"Warning: Failed to download {ticker}: {e}")
                continue
        
        if not raw:
            raise ValueError("No data was successfully downloaded")
        
        data = pd.concat(raw, axis=1).fillna(method='ffill')
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        data.to_csv(self.cache_path)
        return data.loc[start:end]

class FeatureEngine:
    def __init__(self, lookback_windows=[5, 10, 20, 60, 120]): self.windows = lookback_windows
    
    def create_features(self, data):
        features = []
        
        for ticker in TICKERS:
            if ticker not in data.columns: continue
            p = data[ticker].fillna(method='ffill')
            r = p.pct_change()
            v = data.get(f'{ticker}_vol', pd.Series(1e6, index=p.index)).fillna(method='ffill')
            h = data.get(f'{ticker}_high', p).fillna(method='ffill')
            l = data.get(f'{ticker}_low', p).fillna(method='ffill')
            
            f = pd.DataFrame(index=p.index)
            for w in self.windows:
                f[f'ret_{w}'] = p.pct_change(w)
                f[f'vol_{w}'] = r.rolling(w).std()
                f[f'mom_{w}'] = (r.rolling(w).mean() / (r.rolling(w).std() + 1e-8))
                f[f'rev_{w}'] = -r.rolling(w).sum()
                f[f'skew_{w}'] = r.rolling(w).skew()
            
            f['rsi'] = self.rsi(p, 14)
            f['bb_pos'] = self.bb_pos(p, 20)
            f['macd'] = self.macd(p)
            f['hl_ratio'] = (h - l) / (p + 1e-8)
            f['dollar_vol'] = (p * v).rolling(20).mean()
            f['amihud'] = (r.abs() / (p * v + 1e-8)).rolling(20).mean()
            
            # Fixed sector calculations with error handling
            sector_etf = SECTOR_MAP.get(ticker)
            if sector_etf and sector_etf in data.columns:
                try:
                    sector_ret = data[sector_etf].pct_change()
                    if isinstance(r, pd.Series) and isinstance(sector_ret, pd.Series):
                        f['sector_beta'] = r.rolling(60).corr(sector_ret)
                        f['sector_alpha'] = r.rolling(20).mean() - sector_ret.rolling(20).mean()
                    else:
                        f['sector_beta'] = 0.0
                        f['sector_alpha'] = 0.0
                except Exception as e:
                    print(f"Error calculating sector features for {ticker}: {e}")
                    f['sector_beta'] = 0.0
                    f['sector_alpha'] = 0.0
            
            features.append(f.add_suffix(f'_{ticker}'))
        
        for macro in MACRO_TICKERS:
            if macro in data.columns:
                m_ret = data[macro].pct_change()
                m_vol = m_ret.rolling(20).std()
                m_rolling_mean = m_ret.rolling(60).mean()
                m_rolling_std = m_ret.rolling(60).std()
                
                m_ret = m_ret.squeeze() if hasattr(m_ret, 'squeeze') else m_ret
                m_vol = m_vol.squeeze() if hasattr(m_vol, 'squeeze') else m_vol
                m_rolling_mean = m_rolling_mean.squeeze() if hasattr(m_rolling_mean, 'squeeze') else m_rolling_mean
                m_rolling_std = m_rolling_std.squeeze() if hasattr(m_rolling_std, 'squeeze') else m_rolling_std
                
                m_zscore = (m_ret - m_rolling_mean) / (m_rolling_std + 1e-8)
                
                macro_feat = pd.DataFrame({
                    'ret': m_ret, 
                    'vol': m_vol, 
                    'zscore': m_zscore
                }, index=data.index)
                features.append(macro_feat.add_suffix(f'_{macro}'))
        
        X = pd.concat(features, axis=1)
        tnx_val = data['^TNX'].iloc[:, 0] if '^TNX' in data.columns and len(data['^TNX'].shape) > 1 else data.get('^TNX', 0.03)
        vix_val = data['^VIX'].iloc[:, 0] if '^VIX' in data.columns and len(data['^VIX'].shape) > 1 else data.get('^VIX', 20)
        X['vix_term'] = tnx_val - vix_val / 100
        
        X['market_breadth'] = sum((data[t].pct_change() > 0).iloc[:, 0] if len(data[t].shape) > 1 else (data[t].pct_change() > 0) for t in TICKERS if t in data.columns) / len(TICKERS)
        X['cross_momentum'] = sum(data[t].pct_change(20).iloc[:, 0] if len(data[t].shape) > 1 else data[t].pct_change(20) for t in TICKERS if t in data.columns) / len(TICKERS) 
        X['dow'] = X.index.dayofweek
        X['month'] = X.index.month
        X['qtr'] = X.index.quarter
        return X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def rsi(self, p, w=14): delta = p.diff(); return 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(w).mean() / (-delta.where(delta < 0, 0)).rolling(w).mean()))
    def bb_pos(self, p, w=20): ma = p.rolling(w).mean(); return (p - ma) / (2 * p.rolling(w).std())
    def macd(self, p, f=12, s=26, sig=9): exp1, exp2 = p.ewm(span=f).mean(), p.ewm(span=s).mean(); return (exp1 - exp2) - (exp1 - exp2).ewm(span=sig).mean()

class RegimeDetector:
    def __init__(self):
        self.states = ['low_vol', 'high_vol', 'crisis']
        self.lookback = 60
        
    def detect(self, returns):
        if len(returns) < self.lookback: return pd.Series(0, index=returns.index)
        
        vol = returns.std(axis=1).rolling(self.lookback).mean()
        breadth = (returns > 0).sum(axis=1) / returns.shape[1]
        
        regime = pd.Series(0, index=returns.index)
        regime[vol > vol.quantile(0.8)] = 1
        regime[(vol > vol.quantile(0.95)) | (breadth < 0.2)] = 2
        
        return regime

class NeuralAlpha(nn.Module):
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.ReLU(),
            nn.Linear(hidden_dim//4, len(TICKERS))
        )
        
    def forward(self, x): return torch.tanh(self.net(x))

class EnhancedMLStrategy:
    def __init__(self):
        self.models = {}
        if xgb: self.models['xgb'] = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
        self.models['gbm'] = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
        self.models['sgd'] = SGDRegressor(loss='huber', penalty='elasticnet', alpha=1e-4, random_state=42)
        
        self.neural_model = None
        self.scalers = {}
        self.selected_features = []
        self.regime_models = {0: {}, 1: {}, 2: {}}
        self.feature_columns = None  # Store the expected feature columns
        
    def select_features(self, X, y, max_features=80):
        if len(X) < 100: return list(X.columns)[:max_features]
        
        valid_idx = y.notna() & X.notna().all(axis=1)
        X_clean, y_clean = X[valid_idx], y[valid_idx]
        if len(X_clean) < 50: return list(X.columns)[:max_features]
        
        mi_scores = mutual_info_regression(X_clean.fillna(0), y_clean, random_state=42)
        feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        selected, corr_matrix = [], X_clean.corr().abs()
        for feature in feature_scores.index:
            if len(selected) == 0 or corr_matrix.loc[feature, selected].max() < 0.75:
                selected.append(feature)
            if len(selected) >= max_features: break
        return selected
    
    def align_features(self, X):
        """Ensure feature consistency between training and prediction"""
        if self.feature_columns is None:
            return X
        
        # Create a DataFrame with all expected features, filled with zeros
        aligned_X = pd.DataFrame(0.0, index=X.index, columns=self.feature_columns)
        
        # Fill in the values for features that exist in X
        for col in X.columns:
            if col in self.feature_columns:
                aligned_X[col] = X[col]
        
        return aligned_X
    
    def fit_regime_models(self, X, y, regimes):
        for regime in [0, 1, 2]:
            regime_mask = (regimes == regime)
            if regime_mask.sum() < 50: continue
            
            X_regime, y_regime = X[regime_mask], y[regime_mask]
            valid_idx = y_regime.notna() & X_regime.notna().all(axis=1)
            if valid_idx.sum() < 30: continue
            
            X_clean, y_clean = X_regime[valid_idx], y_regime[valid_idx]
            X_aligned = self.align_features(X_clean)
            X_selected = X_aligned[self.selected_features].fillna(0)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            self.scalers[regime] = scaler
            self.regime_models[regime] = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'fit'): 
                        # Clone the model to avoid conflicts
                        if name == 'xgb' and xgb:
                            regime_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
                        elif name == 'gbm':
                            regime_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
                        elif name == 'sgd':
                            regime_model = SGDRegressor(loss='huber', penalty='elasticnet', alpha=1e-4, random_state=42)
                        else:
                            continue
                        
                        regime_model.fit(X_scaled, y_clean)
                        self.regime_models[regime][name] = regime_model
                except Exception as e:
                    print(f"Error fitting {name} model for regime {regime}: {e}")
                    continue
    
    def fit(self, X, y, regimes=None):
        valid_idx = y.notna() & X.notna().all(axis=1)
        X_clean, y_clean = X[valid_idx], y[valid_idx]
        if len(X_clean) < 100: return
        
        # Store the feature columns for consistency
        self.feature_columns = list(X_clean.columns)
        self.selected_features = self.select_features(X_clean, y_clean)
        
        if regimes is not None:
            regimes_clean = regimes[valid_idx]
            self.fit_regime_models(X_clean, y_clean, regimes_clean)
        
        X_selected = X_clean[self.selected_features].fillna(0)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers['default'] = scaler
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'): model.fit(X_scaled, y_clean)
            except Exception as e:
                print(f"Error fitting {name} model: {e}")
                continue
        
        try:
            self.neural_model = NeuralAlpha(len(self.selected_features))
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_clean.values)
            
            optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
            for epoch in range(100):
                pred = self.neural_model(X_tensor)
                loss = nn.MSELoss()(pred.mean(dim=1), y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        except Exception as e:
            print(f"Error training neural model: {e}")
            pass
    
    def predict(self, X, current_regime=0):
        if not self.selected_features or self.feature_columns is None: 
            return pd.Series(0.0, index=X.index)
        
        # Align features to match training data
        X_aligned = self.align_features(X)
        X_selected = X_aligned[self.selected_features].fillna(0)
        
        regime_key = current_regime if current_regime in self.scalers else 'default'
        if regime_key not in self.scalers: 
            return pd.Series(0.0, index=X.index)
        
        try:
            X_scaled = self.scalers[regime_key].transform(X_selected)
        except Exception as e:
            print(f"Error scaling features: {e}")
            return pd.Series(0.0, index=X.index)
        
        predictions = []
        
        models_to_use = self.regime_models.get(current_regime, {}) if current_regime in self.regime_models else self.models
        
        for model in models_to_use.values():
            try:
                if hasattr(model, 'predict'): 
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
            except Exception as e:
                print(f"Error in model prediction: {e}")
                continue
        
        if self.neural_model:
            try:
                with torch.no_grad():
                    neural_pred = self.neural_model(torch.FloatTensor(X_scaled)).mean(dim=1).numpy()
                    predictions.append(neural_pred)
            except Exception as e:
                print(f"Error in neural model prediction: {e}")
                pass
        
        if not predictions: 
            return pd.Series(0.0, index=X.index)
        
        weights = decay_weights(len(predictions), hl=5)[:len(predictions)]
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return pd.Series(np.tanh(ensemble_pred), index=X.index)

class DynamicPortfolioOptimizer:
    def __init__(self): self.target_vol, self.max_weight, self.tc_bps = 0.15, 0.12, 8.0
    
    def black_litterman_priors(self, returns_data, market_caps=None):
        if market_caps is None: market_caps = np.ones(len(TICKERS))
        cov_matrix = returns_data.cov().values
        market_weights = market_caps / market_caps.sum()
        risk_aversion = 3.0
        return risk_aversion * cov_matrix @ market_weights
    
    def optimize(self, signals, returns_data, current_weights=None, regime=0, market_caps=None):
        if not signals: return {}
        
        symbols = list(signals.keys())
        alpha_views = np.array([signals[s] for s in symbols])
        
        if len(returns_data) > 120:
            returns_matrix = returns_data[symbols].fillna(0)
            cov_matrix = LedoitWolf().fit(returns_matrix.values).covariance_
            
            vol_scale = 1.0 + 0.5 * regime
            cov_matrix *= vol_scale
        else:
            cov_matrix = np.eye(len(symbols)) * (0.04 * (1 + regime))
        
        prior_returns = self.black_litterman_priors(returns_data[symbols], market_caps)
        combined_returns = 0.3 * prior_returns + 0.7 * alpha_views
        
        def objective(w):
            port_ret = np.dot(w, combined_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            
            if current_weights is not None:
                current_w = np.array([current_weights.get(s, 0) for s in symbols])
                turnover = np.sum(np.abs(w - current_w))
                tc_penalty = (self.tc_bps / 1e4) * turnover
                port_ret -= tc_penalty
            
            return -(port_ret / (port_vol + 1e-6))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: self.target_vol**2 - np.dot(w.T, np.dot(cov_matrix, w))}
        ]
        
        max_single_weight = self.max_weight * (0.8 if regime == 2 else 1.0)
        bounds = [(0, max_single_weight) for _ in range(len(symbols))]
        x0 = np.ones(len(symbols)) / len(symbols)
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = dict(zip(symbols, result.x))
                return {k: v for k, v in weights.items() if v > 0.005}
        except: pass
        
        signal_weights = np.maximum(alpha_views, 0)
        signal_weights = signal_weights / (signal_weights.sum() + 1e-6)
        signal_weights = np.minimum(signal_weights, max_single_weight)
        signal_weights = signal_weights / signal_weights.sum()
        return dict(zip(symbols, signal_weights))

class RiskManager:
    def __init__(self): self.max_var, self.max_corr, self.max_sector = 0.025, 0.3, 0.4
    
    def calculate_var(self, weights, returns_data, confidence=0.05):
        if not weights or len(returns_data) < 60: return 0.0
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        ret_matrix = returns_data[symbols].fillna(0)
        port_returns = (ret_matrix * w).sum(axis=1)
        return np.percentile(port_returns, confidence * 100)
    
    def apply_limits(self, weights, returns_data, regime=0):
        if not weights: return weights
        
        current_var = abs(self.calculate_var(weights, returns_data))
        if current_var > self.max_var:
            scale_factor = self.max_var / (current_var + 1e-6)
            weights = {k: v * scale_factor for k, v in weights.items()}
        
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = SECTOR_MAP.get(ticker, 'OTHER')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        for sector, sector_weight in sector_weights.items():
            if sector_weight > self.max_sector:
                scale_factor = self.max_sector / sector_weight
                for ticker, weight in weights.items():
                    if SECTOR_MAP.get(ticker) == sector:
                        weights[ticker] *= scale_factor
        
        if regime == 2:
            leverage_limit = 0.6
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight > leverage_limit:
                scale = leverage_limit / total_weight
                weights = {k: v * scale for k, v in weights.items()}
        
        return {k: v for k, v in weights.items() if abs(v) > 0.005}

class HybridAlphaTrader:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.ml_strategy = EnhancedMLStrategy()
        self.portfolio_optimizer = DynamicPortfolioOptimizer()
        self.risk_manager = RiskManager()
        
    def backtest(self, start=START, end=END, split=SPLIT, retrain_freq=20):
        print("Loading data...")
        data = self.data_loader.load_data(start, end)
        print("Creating features...")
        features = self.feature_engine.create_features(data)
        returns_data = data[TICKERS].pct_change()
        regimes = self.regime_detector.detect(returns_data.fillna(0))
        
        split_date = pd.to_datetime(split)
        test_data = data[split_date:]
        test_features = features[split_date:]
        test_returns = returns_data[split_date:]
        test_regimes = regimes[split_date:]
        
        if len(test_data) < 50: return {'error': 'Insufficient test data'}
        
        print("Starting backtest...")
        portfolio_values, current_weights = [], None
        last_retrain = None
        
        test_dates = pd.date_range(split_date, end, freq='W-FRI')
        for i, date in enumerate(test_dates):
            if date not in test_data.index: continue
            
            train_end = date - BDay(1)
            train_start = max(pd.to_datetime(start), train_end - BDay(500))
            
            if last_retrain is None or (date - last_retrain).days >= retrain_freq:
                print(f"Retraining models for {date.strftime('%Y-%m-%d')}...")
                train_features = features.loc[train_start:train_end]
                train_returns = returns_data.loc[train_start:train_end]
                train_regimes = regimes.loc[train_start:train_end]
                
                if len(train_features) > 200:
                    future_returns = train_returns.shift(-5).mean(axis=1)
                    self.ml_strategy.fit(train_features, future_returns, train_regimes)
                    last_retrain = date
            
            current_regime = test_regimes.get(date, 0)
            window_data = data.loc[train_start:date]
            window_features = features.loc[train_start:date]
            
            if len(window_features) < 100: continue
            
            latest_features = window_features.iloc[-1:]
            predictions = self.ml_strategy.predict(latest_features, current_regime)
            
            if predictions.empty or abs(predictions.iloc[0]) < 1e-6: continue
            
            signals = {ticker: max(-1, min(1, predictions.iloc[0])) for ticker in TICKERS}
            signals = {k: v for k, v in signals.items() if abs(v) > 0.05}
            
            if signals:
                target_weights = self.portfolio_optimizer.optimize(
                    signals, window_data[TICKERS].pct_change(), current_weights, current_regime
                )
                target_weights = self.risk_manager.apply_limits(target_weights, window_data[TICKERS].pct_change(), current_regime)
                
                next_date = test_dates[i + 1] if i + 1 < len(test_dates) else date + BDay(7)
                if next_date <= test_data.index[-1]:
                    period_returns = test_returns.loc[date:next_date, list(target_weights.keys())]
                    if not period_returns.empty and len(period_returns) > 0:
                        portfolio_return = sum(target_weights[ticker] * period_returns[ticker].sum() 
                                             for ticker in target_weights if ticker in period_returns.columns)
                        
                        tc_cost = 0
                        if current_weights:
                            turnover = sum(abs(target_weights.get(t, 0) - current_weights.get(t, 0)) for t in set(list(target_weights.keys()) + list(current_weights.keys())))
                            tc_cost = (self.portfolio_optimizer.tc_bps / 1e4) * turnover
                        
                        net_return = portfolio_return - tc_cost
                        portfolio_values.append((date, net_return))
                        current_weights = target_weights
        
        if not portfolio_values: return {'error': 'No valid trades'}
        
        print("Calculating performance metrics...")
        equity_curve = pd.Series(dict(portfolio_values)).cumsum()
        returns = equity_curve.diff().fillna(0)
        
        total_return = equity_curve.iloc[-1]
        vol = returns.std() * np.sqrt(52)
        sharpe = (returns.mean() * 52) / (vol + 1e-6)
        max_dd = (equity_curve - equity_curve.cummax()).min()
        calmar = (returns.mean() * 52) / (abs(max_dd) + 1e-6)
        
        return {
            'total_return': total_return, 'annualized_return': returns.mean() * 52,
            'volatility': vol, 'sharpe_ratio': sharpe, 'calmar_ratio': calmar,
            'max_drawdown': max_dd, 'equity_curve': equity_curve,
            'win_rate': (returns > 0).mean(), 'final_weights': current_weights or {}
        }

if __name__ == "__main__":
    trader = HybridAlphaTrader()
    results = trader.backtest()
    
    if 'error' not in results:
        print(f"Total Return: {results['total_return']:.2%}, Ann. Return: {results['annualized_return']:.2%}")
        print(f"Sharpe: {results['sharpe_ratio']:.2f}, Calmar: {results['calmar_ratio']:.2f}, Max DD: {results['max_drawdown']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}, Win Rate: {results['win_rate']:.2%}")
        print("Final Weights:", {k: f"{v:.2%}" for k, v in sorted(results['final_weights'].items(), key=lambda x: x[1], reverse=True)[:8]})
    else: print(f"Error: {results['error']}")