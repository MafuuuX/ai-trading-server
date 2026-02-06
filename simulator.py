"""
Trade Simulation and Backtesting Engine for AI Trading Server

Provides simulation-based training mode with:
- Historical data backtesting
- Strategy evaluation across risk profiles
- Performance metrics calculation (Sharpe, Drawdown, Win Rate)
- Automatic parameter optimization
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from risk_profiles import RiskProfile, RiskManager, PROFILES, interpolate_profile

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a simulated trade"""
    ticker: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    is_long: bool
    position_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'signal_change', 'timeout'
    hold_duration_hours: float
    confidence: float
    expected_change: float


@dataclass
class SimulationResult:
    """Overall simulation result"""
    profile_name: str
    risk_level: int
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    
    # Core metrics
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time metrics
    avg_hold_duration_hours: float
    max_hold_duration_hours: float
    total_duration_days: int
    
    # Breakdown
    trades_by_ticker: Dict[str, int]
    pnl_by_ticker: Dict[str, float]
    monthly_returns: Dict[str, float]
    
    # Sample trades
    sample_trades: List[Dict]


class TradingSimulator:
    """
    Simulates trading strategies on historical data.
    Used for training mode before live trading.
    """
    
    def __init__(self, data_fetcher, model_predictor=None):
        self.data_fetcher = data_fetcher
        self.model_predictor = model_predictor
        self.results_path = Path("./data/simulation_results.json")
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.simulation_history: List[Dict] = []
        self._load_history()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        self.is_running = False
        self.current_simulation = None
        self.progress = 0.0
    
    def _load_history(self):
        """Load simulation history from disk"""
        try:
            if self.results_path.exists():
                with open(self.results_path, 'r') as f:
                    self.simulation_history = json.load(f)
                    logger.info(f"Loaded {len(self.simulation_history)} simulation results")
        except Exception as e:
            logger.warning(f"Could not load simulation history: {e}")
            self.simulation_history = []
    
    def _save_history(self):
        """Save simulation history to disk"""
        try:
            # Keep only last 50 simulations
            if len(self.simulation_history) > 50:
                self.simulation_history = self.simulation_history[-50:]
            
            with open(self.results_path, 'w') as f:
                json.dump(self.simulation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save simulation history: {e}")
    
    def run_simulation(
        self,
        tickers: List[str],
        profile: RiskProfile,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000.0,
        use_predictions: bool = False
    ) -> SimulationResult:
        """
        Run a trading simulation with the given parameters.
        
        Args:
            tickers: List of stock tickers to trade
            profile: Risk profile to use
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            initial_capital: Starting capital
            use_predictions: If True, use ML model predictions; else use simple strategy
        
        Returns:
            SimulationResult with all metrics
        """
        with self._lock:
            self.is_running = True
            self.progress = 0.0
        
        try:
            # Set default dates
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            logger.info(f"Starting simulation: {profile.name}, {start_date} to {end_date}, "
                       f"${initial_capital:,.0f}, {len(tickers)} tickers")
            
            # Fetch historical data for all tickers
            all_data = {}
            for i, ticker in enumerate(tickers):
                self.progress = (i / len(tickers)) * 30  # 0-30% for data loading
                try:
                    df = self.data_fetcher.fetch_historical_data(ticker, period="2y")
                    if df is not None and len(df) > 60:
                        all_data[ticker] = df
                except Exception as e:
                    logger.warning(f"Could not fetch data for {ticker}: {e}")
            
            if not all_data:
                raise ValueError("No data available for simulation")
            
            # Initialize simulation state
            capital = initial_capital
            trades: List[TradeResult] = []
            open_positions: Dict[str, Dict] = {}  # ticker -> position info
            daily_equity: List[float] = [initial_capital]
            
            # Get common date range
            min_len = min(len(df) for df in all_data.values())
            
            # Simulate day by day
            for day_idx in range(60, min_len):  # Start after 60 days for indicators
                self.progress = 30 + (day_idx / min_len) * 60  # 30-90% for simulation
                
                # Check exit conditions for open positions
                for ticker in list(open_positions.keys()):
                    pos = open_positions[ticker]
                    df = all_data[ticker]
                    current_price = float(df['Close'].iloc[day_idx])
                    
                    # Check stop-loss and take-profit
                    exit_reason = None
                    if pos['is_long']:
                        if current_price <= pos['stop_loss']:
                            exit_reason = 'stop_loss'
                        elif current_price >= pos['take_profit']:
                            exit_reason = 'take_profit'
                    else:
                        if current_price >= pos['stop_loss']:
                            exit_reason = 'stop_loss'
                        elif current_price <= pos['take_profit']:
                            exit_reason = 'take_profit'
                    
                    # Check timeout (max 10 days)
                    days_held = day_idx - pos['entry_day']
                    if days_held >= 10 and not exit_reason:
                        exit_reason = 'timeout'
                    
                    if exit_reason:
                        # Close position
                        if pos['is_long']:
                            pnl = (current_price - pos['entry_price']) * pos['shares']
                        else:
                            pnl = (pos['entry_price'] - current_price) * pos['shares']
                        
                        pnl_pct = (pnl / pos['position_value']) * 100
                        
                        trade = TradeResult(
                            ticker=ticker,
                            entry_time=pos['entry_time'],
                            exit_time=str(df.index[day_idx]) if hasattr(df, 'index') else f"Day {day_idx}",
                            entry_price=pos['entry_price'],
                            exit_price=current_price,
                            is_long=pos['is_long'],
                            position_size=pos['position_value'],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            exit_reason=exit_reason,
                            hold_duration_hours=days_held * 24,
                            confidence=pos['confidence'],
                            expected_change=pos['expected_change']
                        )
                        trades.append(trade)
                        capital += pos['position_value'] + pnl
                        del open_positions[ticker]
                
                # Look for new entries
                if len(open_positions) < profile.max_concurrent_trades:
                    for ticker, df in all_data.items():
                        if ticker in open_positions:
                            continue
                        if len(open_positions) >= profile.max_concurrent_trades:
                            break
                        
                        # Calculate signals (simple momentum strategy if no ML)
                        current_price = float(df['Close'].iloc[day_idx])
                        prev_price = float(df['Close'].iloc[day_idx - 1])
                        sma_20 = float(df['Close'].iloc[day_idx-20:day_idx].mean())
                        sma_50 = float(df['Close'].iloc[day_idx-50:day_idx].mean()) if day_idx >= 50 else sma_20
                        
                        # Calculate RSI
                        delta = df['Close'].iloc[day_idx-14:day_idx].diff()
                        gain = delta.where(delta > 0, 0).mean()
                        loss = (-delta.where(delta < 0, 0)).mean()
                        rs = gain / (loss + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Generate signal
                        if use_predictions and self.model_predictor:
                            try:
                                pred = self.model_predictor.predict(ticker, df.iloc[:day_idx+1])
                                expected_change = pred.get('predicted_change', 0)
                                confidence = pred.get('confidence', 0.5)
                            except:
                                expected_change = ((sma_20 / current_price) - 1) * 100
                                confidence = 0.6
                        else:
                            # Simple strategy: momentum + mean reversion
                            momentum = (current_price - sma_20) / sma_20 * 100
                            trend = 1 if sma_20 > sma_50 else -1
                            
                            if trend > 0 and rsi < 70 and momentum > 0:
                                expected_change = momentum * 0.5
                                confidence = 0.6 + (70 - rsi) / 200
                            elif trend < 0 and rsi > 30 and momentum < 0:
                                expected_change = momentum * 0.5
                                confidence = 0.6 + (rsi - 30) / 200
                            else:
                                expected_change = 0
                                confidence = 0.4
                        
                        # Check entry conditions
                        is_long = expected_change > 0
                        should_enter, reason = self._should_enter(
                            profile, expected_change, confidence,
                            len(open_positions), is_long
                        )
                        
                        if should_enter:
                            # Calculate position size
                            position_value = self._calc_position_size(
                                profile, capital, confidence
                            )
                            
                            if position_value > capital * 0.1:  # Max 10% check
                                position_value = capital * 0.1
                            
                            if position_value < 100:  # Min position
                                continue
                            
                            shares = position_value / current_price
                            
                            # Set stop-loss and take-profit
                            sl_pct = profile.stop_loss_default / 100
                            tp_pct = profile.take_profit_default / 100
                            
                            if is_long:
                                stop_loss = current_price * (1 - sl_pct)
                                take_profit = current_price * (1 + tp_pct)
                            else:
                                stop_loss = current_price * (1 + sl_pct)
                                take_profit = current_price * (1 - tp_pct)
                            
                            open_positions[ticker] = {
                                'entry_price': current_price,
                                'entry_time': str(df.index[day_idx]) if hasattr(df, 'index') else f"Day {day_idx}",
                                'entry_day': day_idx,
                                'shares': shares,
                                'position_value': position_value,
                                'is_long': is_long,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'confidence': confidence,
                                'expected_change': expected_change
                            }
                            capital -= position_value
                
                # Track daily equity
                equity = capital
                for ticker, pos in open_positions.items():
                    current_price = float(all_data[ticker]['Close'].iloc[day_idx])
                    if pos['is_long']:
                        equity += pos['shares'] * current_price
                    else:
                        equity += pos['position_value'] + (pos['entry_price'] - current_price) * pos['shares']
                daily_equity.append(equity)
            
            # Close remaining positions at market
            final_day = min_len - 1
            for ticker, pos in list(open_positions.items()):
                current_price = float(all_data[ticker]['Close'].iloc[final_day])
                if pos['is_long']:
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['shares']
                
                pnl_pct = (pnl / pos['position_value']) * 100
                days_held = final_day - pos['entry_day']
                
                trade = TradeResult(
                    ticker=ticker,
                    entry_time=pos['entry_time'],
                    exit_time="End of simulation",
                    entry_price=pos['entry_price'],
                    exit_price=current_price,
                    is_long=pos['is_long'],
                    position_size=pos['position_value'],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason='end_of_simulation',
                    hold_duration_hours=days_held * 24,
                    confidence=pos['confidence'],
                    expected_change=pos['expected_change']
                )
                trades.append(trade)
                capital += pos['position_value'] + pnl
            
            self.progress = 95
            
            # Calculate metrics
            result = self._calculate_metrics(
                profile, start_date, end_date, initial_capital,
                capital, trades, daily_equity
            )
            
            # Save to history
            self.simulation_history.append(asdict(result))
            self._save_history()
            
            self.progress = 100
            logger.info(f"Simulation complete: {result.total_trades} trades, "
                       f"{result.total_return_pct:.2f}% return, "
                       f"Sharpe: {result.sharpe_ratio:.2f}")
            
            return result
            
        finally:
            with self._lock:
                self.is_running = False
                self.current_simulation = None
    
    def _should_enter(self, profile: RiskProfile, expected_change: float,
                      confidence: float, current_trades: int, is_long: bool) -> Tuple[bool, str]:
        """Check if trade should be entered based on profile"""
        if current_trades >= profile.max_concurrent_trades:
            return False, "Max trades reached"
        
        if confidence < profile.min_confidence:
            return False, "Confidence too low"
        
        if is_long:
            if expected_change < profile.long_entry_threshold:
                return False, "Expected change too low"
        else:
            if expected_change > profile.short_entry_threshold:
                return False, "Expected change too high"
        
        return True, "OK"
    
    def _calc_position_size(self, profile: RiskProfile, capital: float,
                            confidence: float) -> float:
        """Calculate position size based on profile and confidence"""
        base_pct = profile.position_size_default
        
        if confidence >= profile.min_confidence:
            conf_factor = (confidence - profile.min_confidence) / (1.0 - profile.min_confidence)
            adjusted_pct = base_pct + (profile.position_size_max - base_pct) * conf_factor * 0.5
        else:
            adjusted_pct = profile.position_size_min
        
        final_pct = max(profile.position_size_min, min(profile.position_size_max, adjusted_pct))
        return capital * (final_pct / 100.0)
    
    def _calculate_metrics(self, profile: RiskProfile, start_date: str, end_date: str,
                          initial_capital: float, final_capital: float,
                          trades: List[TradeResult], daily_equity: List[float]) -> SimulationResult:
        """Calculate all simulation metrics"""
        total_trades = len(trades)
        
        if total_trades == 0:
            return SimulationResult(
                profile_name=profile.name,
                risk_level=profile.level,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                avg_trade_pnl=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                avg_hold_duration_hours=0,
                max_hold_duration_hours=0,
                total_duration_days=0,
                trades_by_ticker={},
                pnl_by_ticker={},
                monthly_returns={},
                sample_trades=[]
            )
        
        # Basic metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_trade_pnl = sum(t.pnl for t in trades) / total_trades
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Time metrics
        durations = [t.hold_duration_hours for t in trades]
        avg_hold_duration = sum(durations) / len(durations) if durations else 0
        max_hold_duration = max(durations) if durations else 0
        
        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(daily_equity)):
            if daily_equity[i-1] > 0:
                daily_returns.append((daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1])
        
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            sortino_ratio = (avg_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Max drawdown
        peak = daily_equity[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        for equity in daily_equity:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Breakdowns
        trades_by_ticker = {}
        pnl_by_ticker = {}
        for t in trades:
            trades_by_ticker[t.ticker] = trades_by_ticker.get(t.ticker, 0) + 1
            pnl_by_ticker[t.ticker] = pnl_by_ticker.get(t.ticker, 0) + t.pnl
        
        # Sample trades (first 10)
        sample_trades = [asdict(t) for t in trades[:10]]
        
        return SimulationResult(
            profile_name=profile.name,
            risk_level=profile.level,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_hold_duration_hours=avg_hold_duration,
            max_hold_duration_hours=max_hold_duration,
            total_duration_days=len(daily_equity),
            trades_by_ticker=trades_by_ticker,
            pnl_by_ticker=pnl_by_ticker,
            monthly_returns={},  # TODO: implement monthly breakdown
            sample_trades=sample_trades
        )
    
    def compare_profiles(self, tickers: List[str], initial_capital: float = 100000.0) -> Dict:
        """
        Run simulations for all predefined profiles and compare results.
        Used for automatic optimization.
        """
        results = {}
        
        for name, profile in PROFILES.items():
            try:
                result = self.run_simulation(
                    tickers=tickers,
                    profile=profile,
                    initial_capital=initial_capital
                )
                results[name] = asdict(result)
            except Exception as e:
                logger.error(f"Error simulating profile {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Find best profile
        best_profile = None
        best_sharpe = float('-inf')
        for name, result in results.items():
            if 'error' not in result and result.get('sharpe_ratio', 0) > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_profile = name
        
        return {
            "results": results,
            "best_profile": best_profile,
            "best_sharpe": best_sharpe,
            "recommendation": f"Based on Sharpe ratio, '{best_profile}' profile performed best"
        }
    
    def optimize_level(self, tickers: List[str], initial_capital: float = 100000.0) -> Dict:
        """
        Find optimal risk level (1-10) by testing multiple levels.
        """
        results = {}
        
        for level in [1, 3, 5, 7, 10]:
            profile = interpolate_profile(level)
            try:
                result = self.run_simulation(
                    tickers=tickers,
                    profile=profile,
                    initial_capital=initial_capital
                )
                results[level] = {
                    "sharpe": result.sharpe_ratio,
                    "return_pct": result.total_return_pct,
                    "max_drawdown": result.max_drawdown_pct,
                    "win_rate": result.win_rate
                }
            except Exception as e:
                results[level] = {"error": str(e)}
        
        # Find optimal level (maximize Sharpe, minimize drawdown)
        best_level = 5
        best_score = float('-inf')
        for level, metrics in results.items():
            if 'error' not in metrics:
                # Score = Sharpe - 0.5 * drawdown%
                score = metrics['sharpe'] - 0.5 * metrics['max_drawdown'] / 10
                if score > best_score:
                    best_score = score
                    best_level = level
        
        return {
            "results": results,
            "optimal_level": best_level,
            "recommendation": f"Optimal risk level: {best_level}/10"
        }
    
    def get_status(self) -> Dict:
        """Get current simulation status"""
        return {
            "is_running": self.is_running,
            "progress": self.progress,
            "current_simulation": self.current_simulation,
            "history_count": len(self.simulation_history),
            "last_simulation": self.simulation_history[-1] if self.simulation_history else None
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent simulation history"""
        return self.simulation_history[-limit:]


# Factory function
def create_simulator(data_fetcher, model_predictor=None) -> TradingSimulator:
    """Create a trading simulator instance"""
    return TradingSimulator(data_fetcher, model_predictor)
