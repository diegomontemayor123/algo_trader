import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA, FastICA
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingSignal:
    symbol: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: pd.Timestamp
    alpha_source: str
    risk_score: float

@dataclass
class MarketRegime:
    name: str
    volatility: str  # low, medium, high
    trend: str  # bull, bear, sideways
    correlation: str  # low, medium, high
    liquidity: str  # tight, normal, stressed

class AlphaStrategy(ABC):
    """Base class for alpha generation strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, TradingSignal]:
        pass
    
    @abstractmethod
    def update_parameters(self, regime: MarketRegime) -> None:
        pass

class MeanReversionStrategy(AlphaStrategy):
    """Statistical arbitrage using mean reversion with dynamic parameters"""
    
    def __init__(self, lookback: int = 20, threshold: float = 2.0):
        self.lookback = lookback
        self.threshold = threshold
        self.half_life = 10
        
    def calculate_half_life(self, price_series: pd.Series) -> float:
        """Ornstein-Uhlenbeck process half-life estimation"""
        lagged = price_series.shift(1).dropna()
        price_series = price_series[1:]
        delta = price_series - lagged
        
        # OLS regression: delta_P = lambda * P_lag + epsilon
        X = lagged.values.reshape(-1, 1)
        y = delta.values
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y)
        lambda_param = reg.coef_[0]
        
        if lambda_param >= 0:
            return np.inf
        
        half_life = -np.log(2) / lambda_param
        return max(1, min(252, half_life))  # Cap between 1 day and 1 year
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, TradingSignal]:
        signals = {}
        
        for symbol in data.columns:
            if len(data[symbol].dropna()) < self.lookback * 2:
                continue
                
            prices = data[symbol].dropna()
            
            # Dynamic half-life calculation
            if len(prices) > 50:
                self.half_life = self.calculate_half_life(prices[-50:])
            
            # Exponentially weighted moving average with dynamic decay
            alpha = 1 - np.exp(-1/self.half_life)
            ewma = prices.ewm(alpha=alpha).mean()
            
            # Kalman filter for better signal estimation
            from pykalman import KalmanFilter
            kf = KalmanFilter(n_dim_state=2)
            state_means, _ = kf.em(prices.values.reshape(-1, 1)).smooth()
            kalman_mean = state_means[-1, 0]
            
            # Combine EWMA and Kalman estimates
            combined_mean = 0.7 * ewma.iloc[-1] + 0.3 * kalman_mean
            
            # Dynamic volatility using GARCH-like estimation
            returns = prices.pct_change().dropna()
            vol = returns.rolling(window=min(len(returns), self.lookback)).std() * np.sqrt(252)
            current_vol = vol.iloc[-1] if len(vol) > 0 else returns.std() * np.sqrt(252)
            
            # Z-score with volatility adjustment
            deviation = prices.iloc[-1] - combined_mean
            z_score = deviation / (current_vol * prices.iloc[-1] / np.sqrt(252))
            
            # Signal strength with confidence
            signal_strength = np.tanh(z_score / self.threshold)
            confidence = min(1.0, abs(z_score) / self.threshold)
            
            # Risk score based on various factors
            risk_factors = {
                'volatility': min(1.0, current_vol / 0.3),  # High vol = high risk
                'liquidity': 0.1,  # Assume good liquidity for now
                'regime_stability': 0.2,  # Market regime risk
            }
            risk_score = np.mean(list(risk_factors.values()))
            
            signals[symbol] = TradingSignal(
                symbol=symbol,
                signal=-signal_strength,  # Negative because we're mean reverting
                confidence=confidence,
                timestamp=data.index[-1],
                alpha_source='mean_reversion',
                risk_score=risk_score
            )
            
        return signals
    
    def update_parameters(self, regime: MarketRegime) -> None:
        """Adjust parameters based on market regime"""
        if regime.volatility == 'high':
            self.threshold *= 1.2  # Require stronger signals
        elif regime.volatility == 'low':
            self.threshold *= 0.9
            
        if regime.trend == 'sideways':
            self.lookback = min(30, int(self.lookback * 1.1))
        else:
            self.lookback = max(10, int(self.lookback * 0.9))

class MomentumStrategy(AlphaStrategy):
    """Multi-timeframe momentum with regime awareness"""
    
    def __init__(self):
        self.timeframes = [5, 10, 20, 60, 120]  # Days
        self.min_momentum_threshold = 0.02
        
    def risk_adjusted_momentum(self, returns: pd.Series, window: int) -> float:
        """Calculate risk-adjusted momentum score"""
        if len(returns) < window:
            return 0.0
            
        period_return = returns.rolling(window).sum().iloc[-1]
        period_vol = returns.rolling(window).std().iloc[-1]
        
        if period_vol == 0:
            return 0.0
            
        # Sharpe-like ratio for momentum
        return period_return / (period_vol * np.sqrt(window))
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, TradingSignal]:
        signals = {}
        
        for symbol in data.columns:
            prices = data[symbol].dropna()
            if len(prices) < max(self.timeframes) * 2:
                continue
                
            returns = prices.pct_change().dropna()
            
            # Multi-timeframe momentum scores
            momentum_scores = []
            for tf in self.timeframes:
                score = self.risk_adjusted_momentum(returns, tf)
                weight = 1.0 / tf  # Shorter timeframes get higher weight
                momentum_scores.append(score * weight)
            
            # Weighted average momentum
            total_momentum = sum(momentum_scores) / sum(1.0/tf for tf in self.timeframes)
            
            # Trend consistency check
            price_changes = [prices.iloc[-tf] for tf in self.timeframes if len(prices) >= tf]
            if len(price_changes) >= 3:
                current_price = prices.iloc[-1]
                trend_consistency = sum(1 for p in price_changes if 
                                      (current_price > p and total_momentum > 0) or
                                      (current_price < p and total_momentum < 0)) / len(price_changes)
            else:
                trend_consistency = 0.5
            
            # Signal with trend filter
            signal_strength = np.tanh(total_momentum * 2)
            confidence = min(1.0, abs(total_momentum) * trend_consistency)
            
            # Only trade if momentum exceeds threshold
            if abs(total_momentum) < self.min_momentum_threshold:
                signal_strength *= 0.1
                confidence *= 0.1
            
            risk_score = 0.3  # Base risk for momentum
            
            signals[symbol] = TradingSignal(
                symbol=symbol,
                signal=signal_strength,
                confidence=confidence,
                timestamp=data.index[-1],
                alpha_source='momentum',
                risk_score=risk_score
            )
            
        return signals
    
    def update_parameters(self, regime: MarketRegime) -> None:
        if regime.trend == 'bear':
            self.min_momentum_threshold *= 1.3  # Require stronger signals in bear markets
        elif regime.trend == 'bull':
            self.min_momentum_threshold *= 0.8

class MLAlphaStrategy(AlphaStrategy):
    """Machine learning based alpha generation"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
            'elastic': ElasticNet(alpha=0.1, random_state=42),
            'huber': HuberRegressor()
        }
        self.feature_scaler = RobustScaler()
        self.is_fitted = False
        self.feature_importance = {}
        
    def create_features(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = pd.DataFrame(index=prices.index)
        
        # Price-based features
        features['rsi'] = self.rsi(prices, 14)
        features['bb_position'] = self.bollinger_position(prices, 20)
        features['price_mom_5'] = prices.pct_change(5)
        features['price_mom_20'] = prices.pct_change(20)
        
        # Return-based features
        features['vol_5'] = returns.rolling(5).std()
        features['vol_20'] = returns.rolling(20).std()
        features['skew_20'] = returns.rolling(20).skew()
        features['kurt_20'] = returns.rolling(20).kurt()
        
        # Technical indicators
        features['macd'] = self.macd(prices)
        features['volume_trend'] = np.arange(len(prices))  # Placeholder for volume trend
        
        # Cross-sectional features (will be filled in generate_signals)
        features['relative_strength'] = 0.0
        features['sector_momentum'] = 0.0
        
        return features.fillna(method='ffill').fillna(0)
    
    def rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Position within Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return (prices - ma) / (2 * std)
    
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, TradingSignal]:
        signals = {}
        
        # Create features for all symbols
        all_features = {}
        all_returns = {}
        
        for symbol in data.columns:
            prices = data[symbol].dropna()
            if len(prices) < 50:  # Need sufficient history
                continue
                
            returns = prices.pct_change().dropna()
            features = self.create_features(prices, returns)
            
            all_features[symbol] = features
            all_returns[symbol] = returns
        
        if not all_features:
            return signals
        
        # Add cross-sectional features
        for symbol in all_features:
            features = all_features[symbol]
            returns = all_returns[symbol]
            
            # Relative strength vs other symbols
            other_returns = [all_returns[s].iloc[-20:].mean() for s in all_returns if s != symbol]
            if other_returns:
                own_return = returns.iloc[-20:].mean()
                features.loc[features.index[-1], 'relative_strength'] = own_return - np.mean(other_returns)
        
        # Generate predictions if model is fitted
        for symbol in all_features:
            features = all_features[symbol]
            returns = all_returns[symbol]
            
            if not self.is_fitted and len(features) > 100:
                # Train models on this symbol's data
                self.train_models(features, returns)
            
            if self.is_fitted:
                # Get latest features
                latest_features = features.iloc[-1:].values
                
                # Ensemble prediction
                predictions = []
                for name, model in self.models.items():
                    if hasattr(model, 'predict'):
                        try:
                            pred = model.predict(latest_features)[0]
                            predictions.append(pred)
                        except:
                            continue
                
                if predictions:
                    ensemble_pred = np.mean(predictions)
                    prediction_std = np.std(predictions)
                    
                    # Convert prediction to signal
                    signal_strength = np.tanh(ensemble_pred * 10)  # Scale and bound
                    confidence = min(1.0, 1.0 / (1.0 + prediction_std))  # Lower std = higher confidence
                    
                    risk_score = min(1.0, prediction_std + 0.2)  # Higher uncertainty = higher risk
                    
                    signals[symbol] = TradingSignal(
                        symbol=symbol,
                        signal=signal_strength,
                        confidence=confidence,
                        timestamp=data.index[-1],
                        alpha_source='ml_ensemble',
                        risk_score=risk_score
                    )
        
        return signals
    
    def train_models(self, features: pd.DataFrame, returns: pd.Series):
        """Train ensemble of models"""
        # Prepare training data
        X = features.iloc[:-5]  # All but last 5 observations
        y = returns.shift(-1).iloc[:-5]  # Next day returns
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 50:
            return
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                continue
        
        self.is_fitted = True
    
    def update_parameters(self, regime: MarketRegime) -> None:
        # Could retrain models or adjust ensemble weights based on regime
        pass

class RegimeDetector:
    """Market regime detection using multiple indicators"""
    
    def __init__(self):
        self.lookback = 60
        self.vol_threshold_low = 0.15
        self.vol_threshold_high = 0.35
        
    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(market_data) < self.lookback:
            return MarketRegime("unknown", "medium", "sideways", "medium", "normal")
        
        # Calculate market-wide metrics
        returns = market_data.pct_change().dropna()
        recent_returns = returns.tail(self.lookback)
        
        # Volatility regime
        market_vol = recent_returns.std(axis=1).mean() * np.sqrt(252)
        if market_vol < self.vol_threshold_low:
            vol_regime = "low"
        elif market_vol > self.vol_threshold_high:
            vol_regime = "high"
        else:
            vol_regime = "medium"
        
        # Trend regime
        market_return = recent_returns.mean(axis=1).sum()
        if market_return > 0.05:  # 5% over lookback period
            trend_regime = "bull"
        elif market_return < -0.05:
            trend_regime = "bear"
        else:
            trend_regime = "sideways"
        
        # Correlation regime
        corr_matrix = recent_returns.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        if avg_correlation < 0.3:
            corr_regime = "low"
        elif avg_correlation > 0.7:
            corr_regime = "high"
        else:
            corr_regime = "medium"
        
        # Liquidity regime (simplified)
        liquidity_regime = "normal"  # Would need bid-ask spread data for proper implementation
        
        return MarketRegime(
            name=f"{trend_regime}_{vol_regime}_{corr_regime}",
            volatility=vol_regime,
            trend=trend_regime,
            correlation=corr_regime,
            liquidity=liquidity_regime
        )

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple objectives"""
    
    def __init__(self):
        self.max_weight = 0.1  # Max 10% in any single position
        self.max_sector_weight = 0.3  # Max 30% in any sector
        self.target_volatility = 0.15  # 15% annual volatility target
        
    def optimize_portfolio(self, signals: Dict[str, TradingSignal], 
                         covariance_matrix: pd.DataFrame,
                         current_portfolio: Dict[str, float] = None) -> Dict[str, float]:
        """Optimize portfolio weights using multiple objectives"""
        
        if not signals:
            return {}
        
        symbols = list(signals.keys())
        n_assets = len(symbols)
        
        # Expected returns from signals
        expected_returns = np.array([signals[s].signal * signals[s].confidence for s in symbols])
        
        # Risk adjustment
        risk_penalties = np.array([signals[s].risk_score for s in symbols])
        adjusted_returns = expected_returns * (1 - risk_penalties)
        
        # Covariance matrix (simplified - would use actual covariance)
        if covariance_matrix is None or set(symbols) != set(covariance_matrix.columns):
            # Create simplified covariance matrix
            base_vol = 0.2  # 20% annual volatility
            correlation = 0.3  # 30% correlation
            cov_matrix = np.full((n_assets, n_assets), base_vol**2 * correlation)
            np.fill_diagonal(cov_matrix, base_vol**2)
        else:
            cov_matrix = covariance_matrix.loc[symbols, symbols].values
        
        # Optimization objective: maximize Sharpe ratio with constraints
        def objective(weights):
            portfolio_return = np.dot(weights, adjusted_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = portfolio_return / portfolio_vol
            
            # Add penalty for high turnover if current portfolio exists
            turnover_penalty = 0
            if current_portfolio:
                current_weights = np.array([current_portfolio.get(s, 0) for s in symbols])
                turnover = np.sum(np.abs(weights - current_weights))
                turnover_penalty = 0.01 * turnover  # 1% penalty per unit turnover
            
            return -(sharpe - turnover_penalty)  # Negative because minimize
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Fully invested
            {'type': 'ineq', 'fun': lambda x: self.target_volatility**2 - 
             np.dot(x.T, np.dot(cov_matrix, x))}  # Volatility constraint
        ]
        
        # Bounds
        bounds = [(0, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = result.x
                return dict(zip(symbols, weights))
        except:
            pass
        
        # Fallback: equal risk contribution
        risk_contributions = np.array([signals[s].risk_score for s in symbols])
        inverse_risk = 1.0 / (risk_contributions + 0.01)  # Avoid division by zero
        weights = inverse_risk / np.sum(inverse_risk)
        
        return dict(zip(symbols, weights))

class InstitutionalAlphaTrader:
    """Main trading system orchestrating all components"""
    
    def __init__(self):
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'momentum': MomentumStrategy(),
            'ml_alpha': MLAlphaStrategy()
        }
        self.regime_detector = RegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        self.current_regime = None
        self.signal_history = []
        self.performance_history = []
        
    def generate_alpha_signals(self, market_data: pd.DataFrame) -> Dict[str, TradingSignal]:
        """Generate combined alpha signals from all strategies"""
        
        # Detect current market regime
        self.current_regime = self.regime_detector.detect_regime(market_data)
        
        # Update strategy parameters based on regime
        for strategy in self.strategies.values():
            strategy.update_parameters(self.current_regime)
        
        # Collect signals from all strategies
        all_signals = {}
        strategy_signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(market_data)
                strategy_signals[name] = signals
                
                # Combine signals (weighted average)
                for symbol, signal in signals.items():
                    if symbol not in all_signals:
                        all_signals[symbol] = []
                    all_signals[symbol].append(signal)
            except Exception as e:
                print(f"Strategy {name} failed: {e}")
                continue
        
        # Combine signals for each symbol
        combined_signals = {}
        for symbol, signal_list in all_signals.items():
            if not signal_list:
                continue
                
            # Weight strategies based on regime and historical performance
            weights = self._calculate_strategy_weights(signal_list)
            
            # Weighted average of signals
            combined_signal = sum(s.signal * w for s, w in zip(signal_list, weights))
            combined_confidence = sum(s.confidence * w for s, w in zip(signal_list, weights))
            combined_risk = sum(s.risk_score * w for s, w in zip(signal_list, weights))
            
            combined_signals[symbol] = TradingSignal(
                symbol=symbol,
                signal=combined_signal,
                confidence=combined_confidence,
                timestamp=market_data.index[-1],
                alpha_source='ensemble',
                risk_score=combined_risk
            )
        
        # Store signal history
        self.signal_history.append({
            'timestamp': market_data.index[-1],
            'regime': self.current_regime,
            'signals': combined_signals
        })
        
        return combined_signals
    
    def _calculate_strategy_weights(self, signal_list: List[TradingSignal]) -> List[float]:
        """Calculate weights for combining strategy signals"""
        
        # Base weights
        base_weights = {
            'mean_reversion': 0.4,
            'momentum': 0.3,
            'ml_alpha': 0.3
        }
        
        # Adjust based on regime
        if self.current_regime:
            if self.current_regime.volatility == 'high':
                base_weights['mean_reversion'] *= 1.2  # Mean reversion works better in high vol
                base_weights['momentum'] *= 0.8
            elif self.current_regime.trend != 'sideways':
                base_weights['momentum'] *= 1.3  # Momentum works better in trending markets
                base_weights['mean_reversion'] *= 0.7
        
        # Get weights for each signal
        weights = []
        for signal in signal_list:
            strategy_name = signal.alpha_source
            base_weight = base_weights.get(strategy_name, 0.33)
            
            # Adjust by confidence
            confidence_weight = base_weight * signal.confidence
            weights.append(confidence_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def construct_portfolio(self, signals: Dict[str, TradingSignal],
                          market_data: pd.DataFrame,
                          current_portfolio: Dict[str, float] = None) -> Dict[str, float]:
        """Construct optimal portfolio from alpha signals"""
        
        if not signals:
            return {}
        
        # Filter signals by minimum confidence and adjust for regime
        min_confidence = 0.3
        if self.current_regime and self.current_regime.volatility == 'high':
            min_confidence = 0.5  # Require higher confidence in volatile markets
        
        filtered_signals = {
            symbol: signal for symbol, signal in signals.items()
            if signal.confidence >= min_confidence and abs(signal.signal) > 0.1
        }
        
        if not filtered_signals:
            return current_portfolio or {}
        
        # Optimize portfolio
        optimal_weights = self.portfolio_optimizer.optimize_portfolio(
            filtered_signals, None, current_portfolio
        )
        
        return optimal_weights
    
    def get_trading_recommendations(self, market_data: pd.DataFrame,
                                  current_portfolio: Dict[str, float] = None) -> Dict:
        """Get complete trading recommendations"""
        
        # Generate alpha signals
        signals = self.generate_alpha_signals(market_data)
        
        # Construct optimal portfolio
        target_portfolio = self.construct_portfolio(signals, market_data, current_portfolio)
        
        # Calculate trades needed
        trades = {}
        if current_portfolio:
            for symbol in set(list(current_portfolio.keys()) + list(target_portfolio.keys())):
                current_weight = current_portfolio.get(symbol, 0)
                target_weight = target_portfolio.get(symbol, 0)
                trade_size = target_weight - current_weight
                
                if abs(trade_size) > 0.01:  # Only trade if change > 1%
                    trades[symbol] = trade_size
        else:
            trades = target_portfolio
        
        return {
            'signals': signals,
            'target_portfolio': target_portfolio,
            'trades': trades,
            'regime': self.current_regime,
            'timestamp': market_data.index[-1]
        }
    
    def backtest_performance(self, historical_data: pd.DataFrame,
                           rebalance_frequency: str = 'weekly') -> Dict:
        """Backtest the trading system"""
        
        # This would implement a comprehensive backtesting framework
        # For brevity, just return a placeholder structure
        
        return {
            'total_return': 0.15,  # 15% annual return
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'hit_rate': 0.58,
            'avg_holding_period': 12,  # days
            'turnover': 2.5  # annual turnover
        }

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.max_portfolio_var = 0.02  # 2% daily VaR limit
        self.max_individual_weight = 0.15  # 15% max position size
        self.max_sector_concentration = 0.40  # 40% max sector exposure
        self.leverage_limit = 1.0  # No leverage
        self.correlation_threshold = 0.8  # Flag highly correlated positions
        
    def calculate_portfolio_risk(self, weights: Dict[str, float], 
                               covariance_matrix: pd.DataFrame,
                               confidence_level: float = 0.05) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not weights or covariance_matrix is None:
            return {}
            
        symbols = list(weights.keys())
        weight_vector = np.array([weights.get(s, 0) for s in symbols])
        
        # Ensure covariance matrix matches
        if set(symbols).issubset(set(covariance_matrix.columns)):
            cov_matrix = covariance_matrix.loc[symbols, symbols].values
        else:
            # Fallback covariance matrix
            n = len(symbols)
            cov_matrix = np.eye(n) * 0.04 + np.ones((n, n)) * 0.01
        
        # Portfolio variance and volatility
        portfolio_var = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Value at Risk (parametric)
        var_multiplier = stats.norm.ppf(confidence_level)
        portfolio_var_estimate = portfolio_vol * var_multiplier
        
        # Component VaR
        marginal_var = np.dot(cov_matrix, weight_vector) / portfolio_vol
        component_var = weight_vector * marginal_var * var_multiplier
        
        # Concentration metrics
        herfindahl_index = np.sum(weight_vector ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_var': portfolio_var_estimate,
            'component_var': dict(zip(symbols, component_var)),
            'concentration_hhi': herfindahl_index,
            'effective_positions': effective_positions,
            'leverage': np.sum(np.abs(weight_vector))
        }
        
    def apply_risk_limits(self, target_weights: Dict[str, float],
                         covariance_matrix: pd.DataFrame = None) -> Dict[str, float]:
        """Apply risk limits to target portfolio"""
        
        if not target_weights:
            return {}
            
        adjusted_weights = target_weights.copy()
        
        # Individual position limits
        for symbol, weight in adjusted_weights.items():
            if abs(weight) > self.max_individual_weight:
                adjusted_weights[symbol] = np.sign(weight) * self.max_individual_weight
                
        # Renormalize after position limits
        total_weight = sum(abs(w) for w in adjusted_weights.values())
        if total_weight > self.leverage_limit:
            scale_factor = self.leverage_limit / total_weight
            adjusted_weights = {k: v * scale_factor for k, v in adjusted_weights.items()}
            
        # Check portfolio VaR limit
        if covariance_matrix is not None:
            risk_metrics = self.calculate_portfolio_risk(adjusted_weights, covariance_matrix)
            
            if risk_metrics.get('portfolio_var', 0) > self.max_portfolio_var:
                # Scale down entire portfolio to meet VaR limit
                current_var = risk_metrics['portfolio_var']
                scale_factor = np.sqrt(self.max_portfolio_var / current_var)
                adjusted_weights = {k: v * scale_factor for k, v in adjusted_weights.items()}
                
        return adjusted_weights

class AlternativeDataProcessor:
    """Process alternative data sources for enhanced alpha"""
    
    def __init__(self):
        self.sentiment_weight = 0.3
        self.macro_weight = 0.4
        self.fundamental_weight = 0.3
        
    def process_sentiment_data(self, sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        """Process sentiment data (news, social media, analyst ratings)"""
        
        # Normalize sentiment scores to [-1, 1]
        processed_sentiment = {}
        for symbol, score in sentiment_scores.items():
            # Apply non-linear transformation to emphasize extreme sentiments
            normalized_score = np.tanh(score)
            processed_sentiment[symbol] = normalized_score
            
        return processed_sentiment
    
    def process_macro_data(self, macro_indicators: Dict[str, float],
                          sector_exposures: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Process macro indicators and map to individual securities"""
        
        macro_signals = {}
        
        # Example macro indicators: GDP growth, inflation, interest rates, etc.
        # Map these to sector/security level impacts
        
        for symbol in sector_exposures:
            sector_exp = sector_exposures[symbol]
            macro_signal = 0
            
            # Weight macro indicators by sector exposure
            for indicator, value in macro_indicators.items():
                if indicator == 'gdp_growth':
                    # Cyclical sectors benefit from GDP growth
                    cyclical_exposure = sector_exp.get('technology', 0) + sector_exp.get('consumer_discretionary', 0)
                    macro_signal += value * cyclical_exposure * 0.5
                    
                elif indicator == 'interest_rates':
                    # Rate sensitive sectors
                    rate_sensitive = sector_exp.get('financials', 0) - sector_exp.get('utilities', 0) * 0.5
                    macro_signal += value * rate_sensitive * 0.3
                    
                elif indicator == 'inflation':
                    # Inflation hedges
                    inflation_hedge = sector_exp.get('materials', 0) + sector_exp.get('energy', 0)
                    macro_signal += value * inflation_hedge * 0.4
            
            macro_signals[symbol] = np.tanh(macro_signal)  # Bound between -1 and 1
            
        return macro_signals
    
    def process_fundamental_data(self, fundamental_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Process fundamental data for value/quality signals"""
        
        fundamental_signals = {}
        
        for symbol, metrics in fundamental_metrics.items():
            # Quality metrics
            roe = metrics.get('roe', 0)
            debt_to_equity = metrics.get('debt_to_equity', 1)
            current_ratio = metrics.get('current_ratio', 1)
            
            # Value metrics  
            pe_ratio = metrics.get('pe_ratio', 20)
            pb_ratio = metrics.get('pb_ratio', 3)
            ev_ebitda = metrics.get('ev_ebitda', 15)
            
            # Growth metrics
            earnings_growth = metrics.get('earnings_growth', 0)
            revenue_growth = metrics.get('revenue_growth', 0)
            
            # Combine into composite score
            quality_score = (min(roe / 0.15, 1) +  # Cap ROE benefit at 15%
                           max(0, (2 - debt_to_equity)) +  # Penalize high debt
                           min(current_ratio / 2, 1)) / 3  # Liquidity
            
            value_score = (max(0, (25 - pe_ratio) / 25) +  # Lower PE is better
                          max(0, (5 - pb_ratio) / 5) +    # Lower PB is better  
                          max(0, (20 - ev_ebitda) / 20)) / 3  # Lower EV/EBITDA is better
            
            growth_score = (min(earnings_growth / 0.2, 1) +  # Cap at 20% growth
                           min(revenue_growth / 0.15, 1)) / 2  # Cap at 15% growth
            
            # Weighted combination
            fundamental_score = (quality_score * 0.4 + 
                               value_score * 0.35 + 
                               growth_score * 0.25)
            
            # Convert to signal [-1, 1]
            fundamental_signals[symbol] = (fundamental_score - 0.5) * 2
            
        return fundamental_signals
    
    def combine_alternative_signals(self, sentiment: Dict[str, float] = None,
                                   macro: Dict[str, float] = None,
                                   fundamental: Dict[str, float] = None) -> Dict[str, float]:
        """Combine alternative data signals"""
        
        all_symbols = set()
        if sentiment:
            all_symbols.update(sentiment.keys())
        if macro:
            all_symbols.update(macro.keys())
        if fundamental:
            all_symbols.update(fundamental.keys())
            
        combined_signals = {}
        
        for symbol in all_symbols:
            signal_components = []
            weights = []
            
            if sentiment and symbol in sentiment:
                signal_components.append(sentiment[symbol])
                weights.append(self.sentiment_weight)
                
            if macro and symbol in macro:
                signal_components.append(macro[symbol])
                weights.append(self.macro_weight)
                
            if fundamental and symbol in fundamental:
                signal_components.append(fundamental[symbol])
                weights.append(self.fundamental_weight)
            
            if signal_components:
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Weighted combination
                combined_signal = sum(s * w for s, w in zip(signal_components, weights))
                combined_signals[symbol] = combined_signal
                
        return combined_signals

class TransactionCostModel:
    """Model transaction costs for realistic backtesting"""
    
    def __init__(self):
        self.fixed_cost = 0.0001  # 1bp fixed cost
        self.linear_cost = 0.0005  # 5bp linear impact
        self.quadratic_cost = 0.001  # Market impact coefficient
        
    def calculate_transaction_costs(self, trades: Dict[str, float],
                                  market_data: pd.DataFrame,
                                  volumes: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate realistic transaction costs"""
        
        costs = {}
        
        for symbol, trade_size in trades.items():
            if abs(trade_size) < 0.001:  # Skip tiny trades
                costs[symbol] = 0
                continue
                
            # Get recent price and volume data
            if symbol in market_data.columns:
                recent_prices = market_data[symbol].dropna().tail(5)
                avg_price = recent_prices.mean()
                price_volatility = recent_prices.pct_change().std()
            else:
                avg_price = 100  # Default
                price_volatility = 0.02  # Default 2% volatility
                
            # Volume-based liquidity (default if not provided)
            daily_volume = volumes.get(symbol, 1000000) if volumes else 1000000
            
            # Calculate trade value
            trade_value = abs(trade_size * avg_price)
            
            # Fixed cost component
            fixed_component = self.fixed_cost
            
            # Linear impact (bid-ask spread)
            linear_component = self.linear_cost * price_volatility
            
            # Quadratic impact (market impact)
            volume_participation = trade_value / (daily_volume * avg_price)
            quadratic_component = self.quadratic_cost * (volume_participation ** 0.6)
            
            # Total cost as percentage of trade value
            total_cost_pct = fixed_component + linear_component + quadratic_component
            total_cost = total_cost_pct * trade_value
            
            costs[symbol] = total_cost
            
        return costs

# Enhanced main trader class with all components
class EnhancedAlphaTrader(InstitutionalAlphaTrader):
    """Enhanced version with risk management, alternative data, and transaction costs"""
    
    def __init__(self):
        super().__init__()
        self.risk_manager = RiskManager()
        self.alt_data_processor = AlternativeDataProcessor()
        self.transaction_cost_model = TransactionCostModel()
        
        # Performance tracking
        self.daily_returns = []
        self.daily_positions = []
        self.transaction_costs = []
        
    def enhanced_signal_generation(self, market_data: pd.DataFrame,
                                  alternative_data: Dict = None) -> Dict[str, TradingSignal]:
        """Generate enhanced signals incorporating alternative data"""
        
        # Get base technical signals
        base_signals = self.generate_alpha_signals(market_data)
        
        if not alternative_data:
            return base_signals
            
        # Process alternative data
        alt_signals = {}
        
        if 'sentiment' in alternative_data:
            sentiment_signals = self.alt_data_processor.process_sentiment_data(
                alternative_data['sentiment']
            )
            
        if 'macro' in alternative_data and 'sector_exposures' in alternative_data:
            macro_signals = self.alt_data_processor.process_macro_data(
                alternative_data['macro'], 
                alternative_data['sector_exposures']
            )
            
        if 'fundamentals' in alternative_data:
            fundamental_signals = self.alt_data_processor.process_fundamental_data(
                alternative_data['fundamentals']
            )
            
        # Combine alternative signals
        combined_alt = self.alt_data_processor.combine_alternative_signals(
            sentiment_signals if 'sentiment' in alternative_data else None,
            macro_signals if 'macro' in alternative_data else None,
            fundamental_signals if 'fundamentals' in alternative_data else None
        )
        
        # Enhance base signals with alternative data
        enhanced_signals = {}
        for symbol in base_signals:
            base_signal = base_signals[symbol]
            alt_boost = combined_alt.get(symbol, 0)
            
            # Combine technical and alternative signals
            # Technical gets 70% weight, alternative gets 30%
            enhanced_signal_strength = base_signal.signal * 0.7 + alt_boost * 0.3
            enhanced_confidence = base_signal.confidence * (1 + abs(alt_boost) * 0.2)
            enhanced_confidence = min(enhanced_confidence, 1.0)
            
            enhanced_signals[symbol] = TradingSignal(
                symbol=symbol,
                signal=enhanced_signal_strength,
                confidence=enhanced_confidence,
                timestamp=base_signal.timestamp,
                alpha_source='enhanced_ensemble',
                risk_score=base_signal.risk_score
            )
            
        return enhanced_signals
    
    def construct_risk_adjusted_portfolio(self, signals: Dict[str, TradingSignal],
                                        market_data: pd.DataFrame,
                                        current_portfolio: Dict[str, float] = None,
                                        covariance_matrix: pd.DataFrame = None) -> Dict[str, float]:
        """Construct portfolio with comprehensive risk management"""
        
        # Get initial portfolio from optimizer
        initial_portfolio = self.construct_portfolio(signals, market_data, current_portfolio)
        
        if not initial_portfolio:
            return {}
            
        # Apply risk limits
        risk_adjusted_portfolio = self.risk_manager.apply_risk_limits(
            initial_portfolio, covariance_matrix
        )
        
        # Calculate transaction costs
        if current_portfolio:
            trades = {symbol: risk_adjusted_portfolio.get(symbol, 0) - current_portfolio.get(symbol, 0)
                     for symbol in set(list(current_portfolio.keys()) + list(risk_adjusted_portfolio.keys()))}
            
            transaction_costs = self.transaction_cost_model.calculate_transaction_costs(
                trades, market_data
            )
            
            # Adjust for transaction costs (simplified)
            # In practice, would re-optimize considering costs
            total_costs = sum(transaction_costs.values())
            if total_costs > 0.01:  # If costs > 1% of portfolio, reduce turnover
                turnover_reduction = min(0.5, 0.01 / total_costs)
                for symbol in trades:
                    if symbol in risk_adjusted_portfolio:
                        current_weight = current_portfolio.get(symbol, 0)
                        target_weight = risk_adjusted_portfolio[symbol]
                        adjusted_weight = current_weight + (target_weight - current_weight) * turnover_reduction
                        risk_adjusted_portfolio[symbol] = adjusted_weight
        
        return risk_adjusted_portfolio

# Example usage with enhanced features
if __name__ == "__main__":
    # Initialize enhanced trading system
    trader = EnhancedAlphaTrader()
    
    # Simulate market data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX']
    
    np.random.seed(42)
    price_data = {}
    for symbol in symbols:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = prices
    
    market_data = pd.DataFrame(price_data, index=dates)
    
    # Simulate alternative data
    alternative_data = {
        'sentiment': {symbol: np.random.normal(0, 0.5) for symbol in symbols},
        'macro': {
            'gdp_growth': 0.02,
            'interest_rates': 0.05,
            'inflation': 0.03
        },
        'sector_exposures': {
            symbol: {
                'technology': 0.8 if symbol in ['AAPL', 'MSFT', 'GOOGL'] else 0.1,
                'consumer_discretionary': 0.9 if symbol in ['AMZN', 'TSLA'] else 0.1,
                'communication': 0.9 if symbol in ['META', 'NFLX'] else 0.1,
                'financials': 0.1,
                'utilities': 0.1,
                'materials': 0.1,
                'energy': 0.1
            } for symbol in symbols
        },
        'fundamentals': {
            symbol: {
                'roe': np.random.uniform(0.1, 0.25),
                'debt_to_equity': np.random.uniform(0.3, 2.0),
                'current_ratio': np.random.uniform(1.0, 3.0),
                'pe_ratio': np.random.uniform(10, 30),
                'pb_ratio': np.random.uniform(1, 8),
                'ev_ebitda': np.random.uniform(8, 25),
                'earnings_growth': np.random.uniform(-0.1, 0.3),
                'revenue_growth': np.random.uniform(0, 0.2)
            } for symbol in symbols
        }
    }
    
    # Generate enhanced signals
    enhanced_signals = trader.enhanced_signal_generation(market_data, alternative_data)
    
    # Construct risk-adjusted portfolio
    portfolio = trader.construct_risk_adjusted_portfolio(enhanced_signals, market_data)
    
    # Calculate risk metrics
    risk_metrics = trader.risk_manager.calculate_portfolio_risk(portfolio, None)
    
    print("Enhanced Alpha Trading System Results")
    print("=" * 50)
    print(f"Market Regime: {trader.current_regime.name}")
    print(f"Volatility: {trader.current_regime.volatility}")
    print(f"Trend: {trader.current_regime.trend}")
    
    print("\\nTop Enhanced Signals:")
    sorted_signals = sorted(enhanced_signals.items(), 
                           key=lambda x: abs(x[1].signal * x[1].confidence), 
                           reverse=True)
    for symbol, signal in sorted_signals[:5]:
        print(f"{symbol}: Signal={signal.signal:.3f}, Confidence={signal.confidence:.3f}, Risk={signal.risk_score:.3f}")
    
    print("\\nOptimal Portfolio Allocation:")
    sorted_portfolio = sorted(portfolio.items(), key=lambda x: abs(x[1]), reverse=True)
    for symbol, weight in sorted_portfolio:
        if abs(weight) > 0.01:
            print(f"{symbol}: {weight:.2%}")
    
    print("\\nRisk Metrics:")
    print(f"Portfolio Volatility: {risk_metrics.get('portfolio_volatility', 0):.2%}")
    print(f"Portfolio VaR (95%): {risk_metrics.get('portfolio_var', 0):.2%}")
    print(f"Effective # Positions: {risk_metrics.get('effective_positions', 0):.1f}")
    print(f"Concentration (HHI): {risk_metrics.get('concentration_hhi', 0):.3f}")
    
    print("\\nSystem Features:")
    print("✓ Multi-strategy alpha generation (mean reversion, momentum, ML)")
    print("✓ Dynamic regime detection and parameter adjustment")
    print("✓ Alternative data integration (sentiment, macro, fundamentals)")
    print("✓ Advanced risk management with VaR limits")
    print("✓ Transaction cost modeling")
    print("✓ Robust portfolio optimization")
    print("✓ Real-time signal ensemble weighting")