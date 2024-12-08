"""
Forex Trading Bot V2 - Backtesting Module.

This module provides backtesting capabilities for the trading strategy.
It loads historical data, applies strategy rules, and calculates performance metrics.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from src.core.mt5 import MT5Handler

class BacktestResult:
    """Container for backtest results and metrics."""

    def __init__(self):
        self.trades: List[Dict] = []
        self.metrics: Dict = {}
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def calculate_metrics(self) -> None:
        """Calculate all performance metrics from trade history."""
        if not self.trades:
            return

        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] < 0]

        # Basic metrics
        total_trades = len(self.trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        # Profit metrics
        gross_profit = sum(t['profit'] for t in winning_trades)
        gross_loss = sum(t['profit'] for t in losing_trades)
        net_profit = gross_profit + gross_loss

        # Calculate averages
        avg_win = gross_profit / num_winning if num_winning > 0 else 0
        avg_loss = gross_loss / num_losing if num_losing > 0 else 0

        # Win rate and profit factor
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown()

        # Store all metrics
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'profit_per_trade': net_profit / total_trades if total_trades > 0 else 0,
            'largest_win': max(t['profit'] for t in winning_trades) if winning_trades else 0,
            'largest_loss': min(t['profit'] for t in losing_trades) if losing_trades else 0,
            'consecutive_wins': self._calculate_max_consecutive(True),
            'consecutive_losses': self._calculate_max_consecutive(False)
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]
        max_dd = 0.0

        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_max_consecutive(self, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_streak = current_streak = 0

        for trade in self.trades:
            is_win = trade['profit'] > 0

            if is_win == wins:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

class StrategyBacktester:
    """Main backtesting engine."""

    def __init__(
        self,
        strategy_path: str,
        initial_balance: float = 10000,
        lot_size: float = 0.1,
        commission: float = 7,
        slippage_pips: float = 1
    ):
        """Initialize backtester with strategy and parameters."""
        self.strategy = self._load_strategy(strategy_path)
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.commission = commission
        self.slippage_pips = slippage_pips
        self.mt5_handler = MT5Handler()
        self.logger = self._setup_logging()

    def _load_strategy(self, strategy_path: str) -> Dict:
        """Load strategy configuration from JSON file."""
        try:
            with open(strategy_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading strategy file: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for backtesting."""
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data from MT5."""
        try:
            # Convert timeframe string to MT5 timeframe constant
            tf_map = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15}
            tf = tf_map.get(timeframe)

            if not tf:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get data from MT5
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            if rates is None:
                raise ValueError(f"No data received for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Add pip value column
            pip_value = 0.0001 if symbol[-3:] != 'JPY' else 0.01
            df['pip_value'] = pip_value

            return df

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise

    def _apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply strategy indicators to the data."""
        try:
            # Moving Averages
            fast_ma = self.strategy['indicators']['moving_averages']['fast_ma']
            slow_ma = self.strategy['indicators']['moving_averages']['slow_ma']

            df['fast_ema'] = df['close'].ewm(span=fast_ma['period'], adjust=False).mean()
            df['slow_ema'] = df['close'].ewm(span=slow_ma['period'], adjust=False).mean()

            # RSI
            rsi_settings = self.strategy['indicators']['rsi']
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_settings['period']).mean()
            avg_loss = loss.rolling(window=rsi_settings['period']).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Volume
            vol_settings = self.strategy['indicators']['volume']
            df['volume_ma'] = df['tick_volume'].rolling(
                window=vol_settings['volume_ma']['period']
            ).mean()

            return df

        except Exception as e:
            self.logger.error(f"Error applying indicators: {e}")
            raise

    def _check_signal(self, row: pd.Series) -> Optional[str]:
        """Check for trading signals based on strategy rules."""
        try:
            # Buy Signal
            if (row['fast_ema'] > row['slow_ema'] and  # EMAs crossed up
                row['rsi'] < 50 and                     # RSI below centerline
                row['tick_volume'] > row['volume_ma'] * 1.5):  # Volume confirmation
                return 'BUY'

            # Sell Signal
            if (row['fast_ema'] < row['slow_ema'] and  # EMAs crossed down
                row['rsi'] > 50 and                     # RSI above centerline
                row['tick_volume'] > row['volume_ma'] * 1.5):  # Volume confirmation
                return 'SELL'

            return None

        except Exception as e:
            self.logger.error(f"Error checking signals: {e}")
            raise

    def _calculate_profit(
        self,
        entry_price: float,
        exit_price: float,
        trade_type: str,
        pip_value: float
    ) -> float:
        """Calculate profit/loss for a trade including costs."""
        # Calculate raw profit/loss
        multiplier = 1 if trade_type == 'BUY' else -1
        price_diff = (exit_price - entry_price) * multiplier

        # Convert to pips
        pips = price_diff / pip_value

        # Calculate profit including costs
        profit = (pips * 10 * self.lot_size) - self.commission

        return profit

    def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Run backtest for specified period and return results."""
        self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")

        try:
            # Initialize results
            result = BacktestResult()
            current_balance = self.initial_balance
            open_position = None

            # Get and prepare data
            df = self._get_historical_data(symbol, timeframe, start_date, end_date)
            df = self._apply_indicators(df)

            # Main backtest loop
            for index, row in df.iterrows():
                # Skip first few rows until we have indicator values
                if row.isnull().any():
                    continue

                # Check for signal
                signal = self._check_signal(row)

                # Handle open position
                if open_position:
                    # Calculate current profit
                    current_profit = self._calculate_profit(
                        open_position['entry_price'],
                        row['close'],
                        open_position['type'],
                        row['pip_value']
                    )

                    # Check for exit (opposite signal or take profit/stop loss)
                    if (signal and signal != open_position['type']) or \
                       abs(current_profit) > self.initial_balance * 0.02:  # 2% risk per trade

                        # Record trade
                        result.trades.append({
                            'entry_time': open_position['entry_time'],
                            'exit_time': row['time'],
                            'type': open_position['type'],
                            'entry_price': open_position['entry_price'],
                            'exit_price': row['close'],
                            'profit': current_profit
                        })

                        # Update balance
                        current_balance += current_profit
                        open_position = None

                # Open new position if we have a signal and no open position
                elif signal:
                    open_position = {
                        'type': signal,
                        'entry_price': row['close'],
                        'entry_time': row['time']
                    }

                # Record equity point
                result.equity_curve.append(current_balance)
                result.timestamps.append(row['time'])

            # Close any remaining position
            if open_position:
                final_profit = self._calculate_profit(
                    open_position['entry_price'],
                    df.iloc[-1]['close'],
                    open_position['type'],
                    df.iloc[-1]['pip_value']
                )

                result.trades.append({
                    'entry_time': open_position['entry_time'],
                    'exit_time': df.iloc[-1]['time'],
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'exit_price': df.iloc[-1]['close'],
                    'profit': final_profit
                })

                current_balance += final_profit
                result.equity_curve.append(current_balance)
                result.timestamps.append(df.iloc[-1]['time'])

            # Calculate final metrics
            result.calculate_metrics()

            self.logger.info(f"Backtest completed. Total trades: {len(result.trades)}")
            return result

        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            raise

def main():
    """Example usage of backtester."""
    # Setup paths
    strategy_path = Path(__file__).parent.parent / "config" / "strategy.json"

    # Initialize backtester
    backtester = StrategyBacktester(
        strategy_path=str(strategy_path),
        initial_balance=10000,
        lot_size=0.1
    )

    # Run backtest
    result = backtester.run_backtest(
        symbol="EURUSD",
        timeframe="M5",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 1)
    )

    # Print results
    print("\nBacktest Results:")
    print(f"Total Trades: {result.metrics['total_trades']}")
    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
    print(f"Net Profit: ${result.metrics['net_profit']:.2f}")
    print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Average Win: ${result.metrics['average_win']:.2f}")
    print(f"Average Loss: ${result.metrics['average_loss']:.2f}")
    print(f"Risk/Reward Ratio: {result.metrics['risk_reward_ratio']:.2f}")
    print(f"Largest Win: ${result.metrics['largest_win']:.2f}")
    print(f"Largest Loss: ${result.metrics['largest_loss']:.2f}")
    print(f"Maximum Consecutive Wins: {result.metrics['consecutive_wins']}")