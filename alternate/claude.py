from __future__ import annotations
import warnings, os, numpy as np, pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf, torch, torch.nn as nn
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

w = lambda s, z=3.0: (lambda m, sd: s.clip(m-z*sd, m+z*sd) if sd > 0 else s)(s.mean(), s.std())
z = lambda s: (s - s.mean()) / (s.std() + 1e-8) if s.std() > 1e-8 else s * 0
dw = lambda n, hl=30: np.exp(-np.log(2) * np.arange(n-1, -1, -1) / hl)

class DataLoader:
    def __init__(self, cache="csv/data.csv"): self.cache = cache; os.makedirs("csv", exist_ok=True)
    
    def load_data(self, start=START, end=END):
        end_dt, start_dt = pd.to_datetime(end) + pd.Timedelta(days=30), pd.to_datetime(start)
        tickers = TICKERS + MACRO_TICKERS + list(set(SECTOR_MAP.values()))
        
        if os.path.exists(self.cache):
            data = pd.read_csv(self.cache, index_col=0, parse_dates=True)
            try:
                if (data.index.min() <= start_dt) and (data.index.max() >= end_dt):
                    return data.loc[start:end]
            except: pass
        
        raw = {}
        for t in tickers:
            try:
                df = yf.download(t, start=start_dt - pd.Timedelta(days=400), end=end_dt, auto_adjust=True)
                if not df.empty:
                    raw[t], raw[f'{t}_vol'], raw[f'{t}_high'], raw[f'{t}_low'] = df['Close'], df.get('Volume', pd.Series(1e6, index=df.index)), df['High'], df['Low']
            except: continue
        
        data = pd.concat(raw, axis=1).fillna(method='ffill')
        data.to_csv(self.cache)
        return data.loc[start:end]

class FeatureEngine:
    def __init__(self, windows=[5, 10, 20, 60, 120]): self.w = windows
    
    def create_features(self, data):
        features = []
        for t in TICKERS:
            if t not in data.columns: continue
            p, r = data[t].fillna(method='ffill'), data[t].pct_change()
            v, h, l = data.get(f'{t}_vol', pd.Series(1e6, index=p.index)).fillna(method='ffill'), data.get(f'{t}_high', p).fillna(method='ffill'), data.get(f'{t}_low', p).fillna(method='ffill')
            
            f = pd.DataFrame(index=p.index)
            for w in self.w:
                f[f'ret_{w}'], f[f'vol_{w}'], f[f'mom_{w}'], f[f'rev_{w}'], f[f'skew_{w}'] = p.pct_change(w), r.rolling(w).std(), r.rolling(w).mean()/(r.rolling(w).std()+1e-8), -r.rolling(w).sum(), r.rolling(w).skew()
            
            f['rsi'] = 100 - (100 / (1 + (lambda d: d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean())(p.diff())))
            f['bb_pos'] = (p - p.rolling(20).mean()) / (2 * p.rolling(20).std())
            f['macd'] = (lambda e1, e2: (e1 - e2) - (e1 - e2).ewm(span=9).mean())(p.ewm(span=12).mean(), p.ewm(span=26).mean())
            f['hl_ratio'], f['dollar_vol'], f['amihud'] = (h - l) / (p + 1e-8), (p * v).rolling(20).mean(), (r.abs() / (p * v + 1e-8)).rolling(20).mean()
            
            s = SECTOR_MAP.get(t)
            if s and s in data.columns:
                try:
                    sr = data[s].pct_change()
                    f['sector_beta'], f['sector_alpha'] = r.rolling(60).corr(sr), r.rolling(20).mean() - sr.rolling(20).mean()
                except: f['sector_beta'] = f['sector_alpha'] = 0.0
            features.append(f.add_suffix(f'_{t}'))
        
        for m in MACRO_TICKERS:
            if m in data.columns:
                mr, mv, mm, ms = data[m].pct_change(), data[m].pct_change().rolling(20).std(), data[m].pct_change().rolling(60).mean(), data[m].pct_change().rolling(60).std()
                features.append(pd.DataFrame({'ret': mr, 'vol': mv, 'zscore': (mr - mm) / (ms + 1e-8)}, index=data.index).add_suffix(f'_{m}'))
        
        X = pd.concat(features, axis=1)
        X['vix_term'] = data.get('^TNX', 0.03) - data.get('^VIX', 20) / 100
        X['market_breadth'] = sum((data[t].pct_change() > 0) for t in TICKERS if t in data.columns) / len(TICKERS)
        X['cross_momentum'] = sum(data[t].pct_change(20) for t in TICKERS if t in data.columns) / len(TICKERS)
        X['dow'], X['month'], X['qtr'] = X.index.dayofweek, X.index.month, X.index.quarter
        return X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

class RegimeDetector:
    def detect(self, returns, lb=60):
        if len(returns) < lb: return pd.Series(0, index=returns.index)
        vol, breadth = returns.std(axis=1).rolling(lb).mean(), (returns > 0).sum(axis=1) / returns.shape[1]
        regime = pd.Series(0, index=returns.index)
        regime[vol > vol.quantile(0.8)], regime[(vol > vol.quantile(0.95)) | (breadth < 0.2)] = 1, 2
        return regime

class NeuralAlpha(nn.Module):
    def __init__(self, n_features, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h//2, h//4), nn.ReLU(), nn.Linear(h//4, len(TICKERS)))
    def forward(self, x): return torch.tanh(self.net(x))

class EnhancedMLStrategy:
    def __init__(self):
        self.models = {'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42), 
                      'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42),
                      'sgd': SGDRegressor(loss='huber', penalty='elasticnet', alpha=1e-4, random_state=42)}
        self.neural_model, self.scalers, self.selected_features, self.regime_models, self.feature_columns = None, {}, [], {0:{}, 1:{}, 2:{}}, None
    
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
        if self.feature_columns is None: return X
        aligned_X = pd.DataFrame(0.0, index=X.index, columns=self.feature_columns)
        for col in X.columns:
            if col in self.feature_columns: aligned_X[col] = X[col]
        return aligned_X
    
    def fit_regime_models(self, X, y, regimes):
        for regime in [0, 1, 2]:
            regime_mask = (regimes == regime)
            if regime_mask.sum() < 50: continue
            X_regime, y_regime = X[regime_mask], y[regime_mask]
            valid_idx = y_regime.notna() & X_regime.notna().all(axis=1)
            if valid_idx.sum() < 30: continue
            
            X_clean, y_clean = X_regime[valid_idx], y_regime[valid_idx]
            X_aligned, X_selected = self.align_features(X_clean), self.align_features(X_clean)[self.selected_features].fillna(0)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            self.scalers[regime] = scaler
            
            for name in self.models:
                try:
                    if name == 'xgb': model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
                    elif name == 'gbm': model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
                    elif name == 'sgd': model = SGDRegressor(loss='huber', penalty='elasticnet', alpha=1e-4, random_state=42)
                    model.fit(X_scaled, y_clean)
                    self.regime_models[regime][name] = model
                except: continue
    
    def fit(self, X, y, regimes=None):
        valid_idx = y.notna() & X.notna().all(axis=1)
        X_clean, y_clean = X[valid_idx], y[valid_idx]
        if len(X_clean) < 100: return
        
        self.feature_columns, self.selected_features = list(X_clean.columns), self.select_features(X_clean, y_clean)
        
        if regimes is not None: self.fit_regime_models(X_clean, y_clean, regimes[valid_idx])
        
        X_selected = X_clean[self.selected_features].fillna(0)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers['default'] = scaler
        
        for model in self.models.values():
            try: model.fit(X_scaled, y_clean)
            except: continue
        
        try:
            self.neural_model = NeuralAlpha(len(self.selected_features))
            X_tensor, y_tensor = torch.FloatTensor(X_scaled), torch.FloatTensor(y_clean.values)
            optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
            for _ in range(100):
                pred = self.neural_model(X_tensor)
                loss = nn.MSELoss()(pred.mean(dim=1), y_tensor)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        except: pass
    
    def predict(self, X, current_regime=0):
        if not self.selected_features or self.feature_columns is None: return pd.Series(0.0, index=X.index)
        X_aligned, X_selected = self.align_features(X), self.align_features(X)[self.selected_features].fillna(0)
        regime_key = current_regime if current_regime in self.scalers else 'default'
        if regime_key not in self.scalers: return pd.Series(0.0, index=X.index)
        
        try: X_scaled = self.scalers[regime_key].transform(X_selected)
        except: return pd.Series(0.0, index=X.index)
        
        predictions, models_to_use = [], self.regime_models.get(current_regime, {}) if current_regime in self.regime_models else self.models
        
        for model in models_to_use.values():
            try: predictions.append(model.predict(X_scaled))
            except: continue
        
        if self.neural_model:
            try:
                with torch.no_grad(): predictions.append(self.neural_model(torch.FloatTensor(X_scaled)).mean(dim=1).numpy())
            except: pass
        
        if not predictions: return pd.Series(0.0, index=X.index)
        weights = dw(len(predictions), hl=5)[:len(predictions)]
        return pd.Series(np.tanh(np.average(predictions, axis=0, weights=weights)), index=X.index)

class DynamicPortfolioOptimizer:
    def __init__(self): self.target_vol, self.max_weight, self.tc_bps = 0.15, 0.12, 8.0
    
    def optimize(self, signals, returns_data, current_weights=None, regime=0, market_caps=None):
        if not signals: return {}
        symbols, alpha_views = list(signals.keys()), np.array([signals[s] for s in signals.keys()])
        
        if len(returns_data) > 120:
            returns_matrix = returns_data[symbols].fillna(0)
            cov_matrix = LedoitWolf().fit(returns_matrix.values).covariance_ * (1.0 + 0.5 * regime)
        else: cov_matrix = np.eye(len(symbols)) * (0.04 * (1 + regime))
        
        prior_returns = 3.0 * cov_matrix @ (np.ones(len(symbols))/len(symbols)) if market_caps is None else 3.0 * cov_matrix @ (market_caps/market_caps.sum())
        combined_returns = 0.3 * prior_returns + 0.7 * alpha_views
        
        def objective(w):
            port_ret = np.dot(w, combined_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if current_weights is not None:
                current_w = np.array([current_weights.get(s, 0) for s in symbols])
                port_ret -= (self.tc_bps / 1e4) * np.sum(np.abs(w - current_w))
            return -(port_ret / (port_vol + 1e-6))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: self.target_vol**2 - np.dot(w.T, np.dot(cov_matrix, w))}]
        max_single_weight = self.max_weight * (0.8 if regime == 2 else 1.0)
        bounds = [(0, max_single_weight) for _ in range(len(symbols))]
        
        try:
            result = minimize(objective, np.ones(len(symbols))/len(symbols), method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success: return {k: v for k, v in zip(symbols, result.x) if v > 0.005}
        except: pass
        
        signal_weights = np.maximum(alpha_views, 0)
        signal_weights = np.minimum(signal_weights / (signal_weights.sum() + 1e-6), max_single_weight)
        return dict(zip(symbols, signal_weights / signal_weights.sum()))

class RiskManager:
    def __init__(self): self.max_var, self.max_corr, self.max_sector = 0.025, 0.3, 0.4
    
    def calculate_var(self, weights, returns_data, confidence=0.05):
        if not weights or len(returns_data) < 60: return 0.0
        symbols, w = list(weights.keys()), np.array([weights[s] for s in weights.keys()])
        port_returns = (returns_data[symbols].fillna(0) * w).sum(axis=1)
        return np.percentile(port_returns, confidence * 100)
    
    def apply_limits(self, weights, returns_data, regime=0):
        if not weights: return weights
        
        current_var = abs(self.calculate_var(weights, returns_data))
        if current_var > self.max_var:
            weights = {k: v * self.max_var / (current_var + 1e-6) for k, v in weights.items()}
        
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = SECTOR_MAP.get(ticker, 'OTHER')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        for sector, sector_weight in sector_weights.items():
            if sector_weight > self.max_sector:
                scale_factor = self.max_sector / sector_weight
                for ticker, weight in weights.items():
                    if SECTOR_MAP.get(ticker) == sector: weights[ticker] *= scale_factor
        
        if regime == 2:
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight > 0.6: weights = {k: v * 0.6 / total_weight for k, v in weights.items()}
        
        return {k: v for k, v in weights.items() if abs(v) > 0.005}

class HybridAlphaTrader:
    def __init__(self):
        self.data_loader, self.feature_engine, self.regime_detector = DataLoader(), FeatureEngine(), RegimeDetector()
        self.ml_strategy, self.portfolio_optimizer, self.risk_manager = EnhancedMLStrategy(), DynamicPortfolioOptimizer(), RiskManager()
    
    def backtest(self, start=START, end=END, split=SPLIT, retrain_freq=20):
        data, features = self.data_loader.load_data(start, end), self.feature_engine.create_features(self.data_loader.load_data(start, end))
        returns_data, regimes = data[TICKERS].pct_change(), self.regime_detector.detect(data[TICKERS].pct_change().fillna(0))
        
        split_date = pd.to_datetime(split)
        test_data, test_features, test_returns, test_regimes = data[split_date:], features[split_date:], returns_data[split_date:], regimes[split_date:]
        if len(test_data) < 50: return {'error': 'Insufficient test data'}
        
        portfolio_values, current_weights, last_retrain = [], None, None
        
        for i, date in enumerate(pd.date_range(split_date, end, freq='W-FRI')):
            if date not in test_data.index: continue
            train_end, train_start = date - BDay(1), max(pd.to_datetime(start), date - BDay(500) - BDay(1))
            
            if last_retrain is None or (date - last_retrain).days >= retrain_freq:
                train_features, train_returns, train_regimes = features.loc[train_start:train_end], returns_data.loc[train_start:train_end], regimes.loc[train_start:train_end]
                if len(train_features) > 200:
                    future_returns = train_returns.shift(-5).mean(axis=1)
                    self.ml_strategy.fit(train_features, future_returns, train_regimes)
                    last_retrain = date
            
            current_regime = test_regimes.get(date, 0)
            window_data, window_features = data.loc[train_start:date], features.loc[train_start:date]
            if len(window_features) < 100: continue
            
            predictions = self.ml_strategy.predict(window_features.iloc[-1:], current_regime)
            if predictions.empty or abs(predictions.iloc[0]) < 1e-6: continue
            
            signals = {t: max(-1, min(1, predictions.iloc[0])) for t in TICKERS}
            signals = {k: v for k, v in signals.items() if abs(v) > 0.05}
            
            if signals:
                target_weights = self.portfolio_optimizer.optimize(signals, window_data[TICKERS].pct_change(), current_weights, current_regime)
                target_weights = self.risk_manager.apply_limits(target_weights, window_data[TICKERS].pct_change(), current_regime)
                
                test_dates = pd.date_range(split_date, end, freq='W-FRI')
                next_date = test_dates[i + 1] if i + 1 < len(test_dates) else date + BDay(7)
                if next_date <= test_data.index[-1]:
                    period_returns = test_returns.loc[date:next_date, list(target_weights.keys())]
                    if not period_returns.empty and len(period_returns) > 0:
                        portfolio_return = sum(target_weights[ticker] * period_returns[ticker].sum() for ticker in target_weights if ticker in period_returns.columns)
                        tc_cost = 0
                        if current_weights:
                            turnover = sum(abs(target_weights.get(t, 0) - current_weights.get(t, 0)) for t in set(list(target_weights.keys()) + list(current_weights.keys())))
                            tc_cost = (self.portfolio_optimizer.tc_bps / 1e4) * turnover
                        net_return = portfolio_return - tc_cost
                        portfolio_values.append((date, net_return))
                        current_weights = target_weights
        
        if not portfolio_values: return {'error': 'No valid trades'}
        
        equity_curve, returns = pd.Series(dict(portfolio_values)).cumsum(), pd.Series(dict(portfolio_values)).cumsum().diff().fillna(0)
        total_return, vol, sharpe = equity_curve.iloc[-1], returns.std() * np.sqrt(52), (returns.mean() * 52) / (returns.std() * np.sqrt(52) + 1e-6)
        max_dd, calmar = (equity_curve - equity_curve.cummax()).min(), (returns.mean() * 52) / (abs((equity_curve - equity_curve.cummax()).min()) + 1e-6)
        
        return {'total_return': total_return, 'annualized_return': returns.mean() * 52, 'volatility': vol, 'sharpe_ratio': sharpe, 'calmar_ratio': calmar,
                'max_drawdown': max_dd, 'equity_curve': equity_curve, 'win_rate': (returns > 0).mean(), 'final_weights': current_weights or {}}

if __name__ == "__main__":
    trader = HybridAlphaTrader()
    results = trader.backtest()
    
    if 'error' not in results:
        print(f"Total Return: {results['total_return']:.2%}, Ann. Return: {results['annualized_return']:.2%}")
        print(f"Sharpe: {results['sharpe_ratio']:.2f}, Calmar: {results['calmar_ratio']:.2f}, Max DD: {results['max_drawdown']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}, Win Rate: {results['win_rate']:.2%}")
        print("Final Weights:", {k: f"{v:.2%}" for k, v in sorted(results['final_weights'].items(), key=lambda x: x[1], reverse=True)[:8]})
    else: print(f"Error: {results['error']}")