"""Backtesting Module for Forex Trading Bot V2.

This module provides backtesting capabilities for trading strategies.
It uses historical data from MT5 to simulate trading and analyze strategy performance.

Features:
1. Historical data management using MT5Handler
2. Trade simulation and tracking
3. Performance analysis and reporting
4. Risk metrics calculation

Author: mazelcar
Created: December 2024
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.core.mt5 import MT5Handler
from src.strategy.base import Strategy

class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(self,
                 strategy: Strategy,
                 initial_balance: float = 10000,
                 commission: float = 0.0,
                 spread: float = 0.0):
        """Initialize backtester.

        Args:
            strategy: Trading strategy instance
            initial_balance: Starting account balance
            commission: Trade commission in currency units
            spread: Additional spread to add to historical data
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.commission = commission
        self.spread = spread

        # Initialize MT5 handler for data
        self.mt5_handler = MT5Handler()

        # Performance tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self.current_positions: List[Dict] = []

        # Results
        self.results: Dict = {}

    def run(self,
            symbol: str,
            timeframe: str,
            start_date: datetime,
            end_date: datetime,
            ) -> Dict:
        """Run backtest over specified period.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary containing backtest results
        """
        try:
            # Get historical data
            print(f"Fetching historical data for {symbol}...")
            data = self.mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is None or len(data) == 0:
                raise ValueError(f"No historical data available for {symbol}")

            print(f"Retrieved {len(data)} data points")

            # Add symbol column - ADDED THIS LINE
            data['symbol'] = symbol

            # Add spread to data
            if self.spread > 0:
                data['ask'] = data['close'] + self.spread
                data['bid'] = data['close']
            else:
                data['ask'] = data['close']
                data['bid'] = data['close']

            # Run simulation
            return self._run_simulation(data)

        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            return {}

    def _run_simulation(self, data: pd.DataFrame) -> Dict:
        """Run the actual backtest simulation."""
        current_equity = self.initial_balance
        running_drawdown = 0
        max_drawdown = 0
        peak_equity = self.initial_balance

        print("Starting backtest simulation...")
        print(f"Initial data shape: {data.shape}")

        # Calculate minimum required window size
        min_window = max(
            self.strategy.slow_ema_period * 2,  # For EMA calculation
            30,  # For trend analysis
            self.strategy.rsi_period * 2,  # For RSI
            self.strategy.volume_period * 2,  # For volume analysis
            20  # Minimum baseline
        )

        for index in range(len(data)):
            # Get current window of data for analysis
            current_window = data.iloc[max(0, index-min_window):index+1]

            # Skip if not enough data points
            if len(current_window) < min_window:
                continue

            try:
                # Update strategy market condition
                self.strategy.update_market_condition(current_window)

                # Generate signals only if we have enough data
                signal = self.strategy.generate_signals(current_window)

                # Process open positions
                current_equity = self._process_positions(
                    current_window.iloc[-1],
                    current_equity
                )

                # Update drawdown calculations
                if current_equity > peak_equity:
                    peak_equity = current_equity

                current_drawdown = (peak_equity - current_equity) / peak_equity
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown

                # Handle new signals
                if signal and signal.get('type') in ['BUY', 'SELL']:
                    # Validate signal
                    if self.strategy.validate_signal(signal, current_window.iloc[-1]):
                        self._process_signal(
                            signal,
                            current_window.iloc[-1],
                            current_equity
                        )
            except Exception as e:
                print(f"Error at bar {index}: {str(e)}")
                continue

            # Update equity curve
            self.equity_curve.append(current_equity)

        return self._calculate_results(data)

    def _process_positions(self, current_bar: pd.Series, equity: float) -> float:
        """Process open positions.

        Args:
            current_bar: Current price bar
            equity: Current equity

        Returns:
            Updated equity
        """
        remaining_positions = []

        for position in self.current_positions:
            # Check if stop loss or take profit hit
            if position['type'] == 'BUY':
                if current_bar['low'] <= position['sl']:
                    # Stop loss hit
                    profit = (position['sl'] - position['entry_price']) * position['size']
                    equity += profit - self.commission
                    position['exit_price'] = position['sl']
                    position['exit_time'] = current_bar.name
                    position['profit'] = profit
                    self.trades.append(position)
                    continue
                elif current_bar['high'] >= position['tp']:
                    # Take profit hit
                    profit = (position['tp'] - position['entry_price']) * position['size']
                    equity += profit - self.commission
                    position['exit_price'] = position['tp']
                    position['exit_time'] = current_bar.name
                    position['profit'] = profit
                    self.trades.append(position)
                    continue
            else:  # SELL position
                if current_bar['high'] >= position['sl']:
                    # Stop loss hit
                    profit = (position['entry_price'] - position['sl']) * position['size']
                    equity += profit - self.commission
                    position['exit_price'] = position['sl']
                    position['exit_time'] = current_bar.name
                    position['profit'] = profit
                    self.trades.append(position)
                    continue
                elif current_bar['low'] <= position['tp']:
                    # Take profit hit
                    profit = (position['entry_price'] - position['tp']) * position['size']
                    equity += profit - self.commission
                    position['exit_price'] = position['tp']
                    position['exit_time'] = current_bar.name
                    position['profit'] = profit
                    self.trades.append(position)
                    continue

            remaining_positions.append(position)

        self.current_positions = remaining_positions
        return equity

    def _process_signal(self, signal: Dict, current_bar: pd.Series, equity: float) -> None:
        """Process a new trading signal.

        Args:
            signal: Trading signal
            current_bar: Current price bar
            equity: Current equity
        """
        # Calculate position size
        size = self.strategy.calculate_position_size(
            {'balance': equity},
            pd.DataFrame([current_bar])  # Create DataFrame from current bar
        )


        # Get entry price
        entry_price = current_bar['ask'] if signal['type'] == 'BUY' else current_bar['bid']


        # Calculate stop loss and take profit
        sl = self.strategy.calculate_stop_loss(
            pd.DataFrame([current_bar]),
            signal
        )
        tp = self.strategy.calculate_take_profit(
            pd.DataFrame([current_bar]),
            signal
        )

        if sl is None or tp is None:
            return

        # Create position
        position = {
            'type': signal['type'],
            'size': size,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_time': current_bar.name,
            'signal_strength': signal.get('strength', 0)
        }

        self.current_positions.append(position)

    def _calculate_results(self, data: pd.DataFrame) -> Dict:
        """Calculate backtest results and statistics.

        Args:
            data: Historical price data

        Returns:
            Dictionary of backtest results
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'return_pct': 0.0,
                'sharpe_ratio': 0.0
            }

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

        # Calculate Sharpe Ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0

        # Maximum Drawdown
        peak = 0
        max_dd = 0

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_profit': gross_profit - gross_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'final_equity': self.equity_curve[-1],
            'return_pct': (self.equity_curve[-1] - self.initial_balance) / self.initial_balance * 100
        }

    def save_results(self, filepath: str) -> None:
        """Save backtest results to file.

        Args:
            filepath: Path to save results
        """
        try:
            results = {
                'strategy_name': self.strategy.name,
                'test_period': {
                    'start': str(min(t['entry_time'] for t in self.trades)) if self.trades else None,
                    'end': str(max(t['exit_time'] for t in self.trades)) if self.trades else None
                },
                'initial_balance': self.initial_balance,
                'metrics': self.results,
                'trades': [
                    {
                        'entry_time': str(t['entry_time']),
                        'exit_time': str(t['exit_time']),
                        'type': t['type'],
                        'profit': t['profit'],
                        'entry_price': t['entry_price'],
                        'exit_price': t['exit_price'],
                        'size': t['size']
                    }
                    for t in self.trades
                ],
                'equity_curve': self.equity_curve
            }

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {filepath}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def plot_results(self) -> None:
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot equity curve
            ax1.plot(self.equity_curve)
            ax1.set_title('Equity Curve')
            ax1.grid(True)

            # Plot drawdown
            equity_series = pd.Series(self.equity_curve)
            running_max = equity_series.cummax()
            drawdown = (running_max - equity_series) / running_max * 100

            ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
            ax2.set_title('Drawdown %')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib is required for plotting")
        except Exception as e:
            print(f"Error plotting results: {e}")