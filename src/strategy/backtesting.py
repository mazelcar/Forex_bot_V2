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

from src.audit import setup_audit_logging
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
        """Run backtest over specified period."""
        try:
            print("\n" + "="*50)
            print("Starting Backtest Run")
            print("="*50)

            # Get historical data
            print(f"\nFetching historical data for {symbol}...")
            data = self.mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is None or len(data) == 0:
                raise ValueError(f"No historical data available for {symbol}")

            print(f"Retrieved {len(data)} data points")
            print(f"Data range: {data['time'].min()} to {data['time'].max()}")

            # Add symbol column
            data['symbol'] = symbol

            print("\nStarting Step 1.1: Basic Data Validation...")
            # Use the validator through the strategy
            basic_validation = self.strategy.data_validator.validate_basic_data(data)
            if not basic_validation['overall_pass']:
                print("Basic validation failed")
                if 'error_messages' in basic_validation:
                    for msg in basic_validation['error_messages']:
                        print(f"Error: {msg}")
                return {}

            print("\nStarting Step 1.2: Test Period Validation...")
            print(f"Validating period from {start_date} to {end_date}")
            # Test period validation
            period_validation = self.strategy.data_validator.validate_test_period(
                data, start_date, end_date
            )
            if not period_validation['overall_pass']:
                print("Period validation failed")
                if 'error_messages' in period_validation:
                    for msg in period_validation['error_messages']:
                        print(f"Error: {msg}")
                return {}

            print("\nAll validations passed, proceeding with simulation...")
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
            import traceback
            print(traceback.format_exc())
            return {}

    def _run_simulation(self, data: pd.DataFrame) -> Dict:
        """Run the actual backtest simulation."""
        current_equity = self.initial_balance
        running_drawdown = 0
        max_drawdown = 0
        peak_equity = self.initial_balance

        print("Starting backtest simulation...")
        print(f"Initial data shape: {data.shape}")

        # Calculate minimum required window size AFTER market condition analysis
        self.strategy.update_market_condition(data)  # Update market condition first
        self.strategy._adjust_parameters()  # Let strategy adjust its parameters

        # Now calculate the window size with adjusted parameters
        min_window = max(
            self.strategy.slow_ema_period * 2,  # For EMA calculation
            30,  # For trend analysis
            self.strategy.rsi_period * 2,  # For RSI
            self.strategy.volume_period * 2,  # For volume analysis
            20  # Minimum baseline
        )

        print(f"\nRequired window size after parameter adjustment: {min_window}")

        if len(data) < min_window:
            print(f"Insufficient data for simulation: need {min_window}, have {len(data)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'total_profit': 0,
            }

        # Process data in windows
        for index in range(min_window, len(data)):
            try:
                # Get current window of data for analysis
                current_window = data.iloc[max(0, index-min_window):index+1]

                if len(current_window) < min_window:
                    continue

                # Update strategy market condition
                self.strategy.update_market_condition(current_window)

                # Generate signals
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
        try:
            # Log entry conditions
            print(f"\nProcessing signal: {signal['type']} at {current_bar.name}")
            print(f"Current equity: ${equity:.2f}")

            # Validate signal again before processing
            if not self.strategy.validate_signal(signal, current_bar):
                print("Signal failed final validation")
                return

            # Calculate position size with error handling
            try:
                size = self.strategy.calculate_position_size(
                    {'balance': equity},
                    pd.DataFrame([current_bar])
                )
                print(f"Calculated position size: {size:.2f} lots")
            except Exception as e:
                print(f"Position size calculation failed: {e}")
                return

            # Get entry price with spread consideration
            if signal['type'] == 'BUY':
                entry_price = current_bar['ask'] if 'ask' in current_bar else current_bar['close'] + self.spread
            else:  # SELL
                entry_price = current_bar['bid'] if 'bid' in current_bar else current_bar['close'] - self.spread

            # Calculate stop loss and take profit
            try:
                sl = self.strategy.calculate_stop_loss(
                    pd.DataFrame([current_bar]),
                    signal
                )
                tp = self.strategy.calculate_take_profit(
                    pd.DataFrame([current_bar]),
                    signal
                )

                print(f"Entry: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            except Exception as e:
                print(f"SL/TP calculation failed: {e}")
                return

            # Validate SL/TP levels
            if sl is None or tp is None:
                print("Invalid SL/TP levels")
                return

            # Validate risk parameters
            risk_amount = abs(entry_price - sl) * size
            risk_percent = risk_amount / equity
            if risk_percent > 0.02:  # Max 2% risk per trade
                print(f"Risk too high: {risk_percent*100:.2f}%")
                return

            # Create and log position
            position = {
                'type': signal['type'],
                'size': size,
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'entry_time': current_bar.name,
                'signal_strength': signal.get('strength', 0),
                'market_condition': self.strategy.current_market_condition.copy()
            }

            print(f"Opening {position['type']} position:")
            print(f"Size: {position['size']:.2f} lots")
            print(f"Entry: {position['entry_price']:.5f}")
            print(f"SL: {position['sl']:.5f}")
            print(f"TP: {position['tp']:.5f}")

            # Add position to current positions
            self.current_positions.append(position)

        except Exception as e:
            print(f"Error processing signal: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _calculate_results(self, data: pd.DataFrame) -> Dict:
        """Calculate final backtest results."""
        results = {
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['profit'] > 0]),
            'losing_trades': len([t for t in self.trades if t['profit'] <= 0]),
            'gross_profit': sum([t['profit'] for t in self.trades if t['profit'] > 0]),
            'gross_loss': sum([t['profit'] for t in self.trades if t['profit'] <= 0]),
            'total_profit': sum([t['profit'] for t in self.trades]),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            if results['losing_trades'] > 0:
                results['profit_factor'] = abs(results['gross_profit'] / results['gross_loss'])
            else:
                results['profit_factor'] = float('inf') if results['gross_profit'] > 0 else 0

        return results

    def audit_calculations(self) -> None:
        """Audit strategy calculations."""
        print("\n=== Starting Calculation Audit ===")
        print("Setting up logging...")

        logger, log_dir, timestamp = setup_audit_logging()
        logger.info("Starting Calculation Audit")

        try:
            # Setup test environment
            from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
            from src.core.mt5 import MT5Handler

            strategy_config = str(Path("config/strategy.json"))
            strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
            logger.info("Strategy instance created")

            mt5_handler = MT5Handler()
            logger.info("MT5Handler instance created")

            # Get larger sample for better calculation testing
            symbol = "EURUSD"
            timeframe = "M5"
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)  # 24 hours of data

            data = mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is not None:
                data['symbol'] = symbol
                logger.info(f"Retrieved {len(data)} data points")

                # Enhanced Signal Strength Testing
                logger.info("\n=== DETAILED SIGNAL STRENGTH ANALYSIS ===")

                # Test Case 1: Strong Bullish Signal
                logger.info("\nTest Case 1: Strong Bullish Signal")
                test_crossover_strong = {
                    'type': 'BULLISH',
                    'strength': 0.8,
                    'momentum': 'Up',
                    'spread': 2.5
                }
                test_rsi_strong = 45.0
                test_volume_strong = {
                    'above_average': True,
                    'high_volume': True,
                    'volume_ratio': 1.5
                }

                logger.info("Input Parameters:")
                logger.info(f"Crossover: {test_crossover_strong}")
                logger.info(f"RSI: {test_rsi_strong}")
                logger.info(f"Volume: {test_volume_strong}")

                strength_strong = strategy._calculate_signal_strength(
                    test_crossover_strong,
                    test_rsi_strong,
                    test_volume_strong,
                    data['close'].iloc[-1],
                    data['close'].iloc[-2]
                )
                logger.info(f"Calculated Strength: {strength_strong}")

                # Test Case 2: Weak Bullish Signal
                logger.info("\nTest Case 2: Weak Bullish Signal")
                test_crossover_weak = {
                    'type': 'BULLISH',
                    'strength': 0.3,
                    'momentum': 'Up',
                    'spread': 1.0
                }
                test_rsi_weak = 65.0
                test_volume_weak = {
                    'above_average': False,
                    'high_volume': False,
                    'volume_ratio': 0.8
                }

                logger.info("Input Parameters:")
                logger.info(f"Crossover: {test_crossover_weak}")
                logger.info(f"RSI: {test_rsi_weak}")
                logger.info(f"Volume: {test_volume_weak}")

                strength_weak = strategy._calculate_signal_strength(
                    test_crossover_weak,
                    test_rsi_weak,
                    test_volume_weak,
                    data['close'].iloc[-1],
                    data['close'].iloc[-2]
                )
                logger.info(f"Calculated Strength: {strength_weak}")

                # Test Case 3: Real Market Conditions
                logger.info("\nTest Case 3: Current Market Conditions")

                # Generate real signal
                real_signal = strategy.generate_signals(data)
                logger.info("\nReal Signal Generated:")
                logger.info(f"Signal Type: {real_signal.get('type', 'NONE')}")
                logger.info(f"Signal Strength: {real_signal.get('strength', 0)}")

                if real_signal and real_signal['type'] != 'NONE':
                    logger.info("\nValidating Real Signal:")
                    is_valid = strategy.validate_signal(real_signal, data.iloc[-1])
                    logger.info(f"Signal Valid: {is_valid}")

                    # Log validation components
                    logger.info("\nValidation Components:")
                    if 'spread' in data.columns:
                        logger.info(f"Current Spread: {data['spread'].iloc[-1]}")
                    logger.info(f"Signal Strength Threshold: {strategy.config['signal_strength']['levels']['moderate']['ema_conditions']['min_separation_pips']}")

                    # Log market conditions
                    market_condition = strategy._analyze_market_condition(data)
                    logger.info("\nMarket Conditions:")
                    logger.info(f"Phase: {market_condition.get('phase', 'unknown')}")
                    logger.info(f"Volatility: {market_condition.get('volatility', 0)}")
                    logger.info(f"Trend Strength: {market_condition.get('trend_strength', 0)}")

                # Signal Strength Distribution Analysis
                logger.info("\n=== Signal Strength Distribution Analysis ===")
                test_strengths = []
                test_conditions = [
                    {'crossover': 0.2, 'rsi': 35, 'volume': 0.8},
                    {'crossover': 0.4, 'rsi': 45, 'volume': 1.0},
                    {'crossover': 0.6, 'rsi': 55, 'volume': 1.2},
                    {'crossover': 0.8, 'rsi': 65, 'volume': 1.4},
                    {'crossover': 1.0, 'rsi': 75, 'volume': 1.6}
                ]

                for cond in test_conditions:
                    test_crossover = {
                        'type': 'BULLISH',
                        'strength': cond['crossover'],
                        'momentum': 'Up',
                        'spread': 2.0
                    }
                    test_volume = {
                        'above_average': cond['volume'] > 1.0,
                        'high_volume': cond['volume'] > 1.3,
                        'volume_ratio': cond['volume']
                    }

                    strength = strategy._calculate_signal_strength(
                        test_crossover,
                        cond['rsi'],
                        test_volume,
                        data['close'].iloc[-1],
                        data['close'].iloc[-2]
                    )
                    test_strengths.append(strength)

                    logger.info(f"\nTest Condition:")
                    logger.info(f"Crossover Strength: {cond['crossover']}")
                    logger.info(f"RSI: {cond['rsi']}")
                    logger.info(f"Volume Ratio: {cond['volume']}")
                    logger.info(f"Calculated Strength: {strength}")

                logger.info("\nStrength Distribution Summary:")
                logger.info(f"Min Strength: {min(test_strengths)}")
                logger.info(f"Max Strength: {max(test_strengths)}")
                logger.info(f"Average Strength: {sum(test_strengths)/len(test_strengths)}")

            logger.info("Calculation audit completed")
            print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

        except Exception as e:
            logger.error(f"Calculation audit failed: {str(e)}")
            raise

    def _analyze_session_performance(self) -> Dict:
        """Analyze trading performance by market session."""
        try:
            if not self.trades:
                return {}

            # Convert trades to DataFrame for analysis
            trades_df = pd.DataFrame(self.trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            # Add session information
            trades_df['hour'] = trades_df['entry_time'].dt.hour

            # Define sessions
            session_ranges = {
                'Sydney': (21, 6),    # 21:00 - 06:00
                'Tokyo': (23, 8),     # 23:00 - 08:00
                'London': (8, 17),    # 08:00 - 17:00
                'New York': (13, 22), # 13:00 - 22:00
            }

            # Categorize trades by session
            def get_session(hour):
                sessions = []
                for session, (start, end) in session_ranges.items():
                    if start > end:  # Session crosses midnight
                        if hour >= start or hour < end:
                            sessions.append(session)
                    else:
                        if start <= hour < end:
                            sessions.append(session)
                return ', '.join(sessions) if sessions else 'Off-Session'

            trades_df['session'] = trades_df['hour'].apply(get_session)

            # Calculate metrics by session
            session_metrics = {}
            for session in trades_df['session'].unique():
                session_trades = trades_df[trades_df['session'] == session]

                if len(session_trades) > 0:
                    winning_trades = len(session_trades[session_trades['profit'] > 0])
                    total_trades = len(session_trades)

                    session_metrics[session] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                        'total_profit': session_trades['profit'].sum(),
                        'avg_profit': session_trades['profit'].mean(),
                        'largest_win': session_trades['profit'].max(),
                        'largest_loss': session_trades['profit'].min(),
                    }

            return session_metrics
        except Exception as e:
            print(f"Error analyzing session performance: {str(e)}")
            return {}

    def _analyze_time_slots(self) -> Dict:
        """Analyze trading performance by hourly time slots."""
        try:
            if not self.trades:
                return {}

            trades_df = pd.DataFrame(self.trades)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['hour'] = trades_df['entry_time'].dt.hour

            # Analyze 4-hour blocks
            trades_df['time_block'] = trades_df['hour'] // 4 * 4
            time_slot_metrics = {}

            for block in sorted(trades_df['time_block'].unique()):
                block_trades = trades_df[trades_df['time_block'] == block]
                block_end = block + 4
                slot_name = f"{block:02d}:00-{block_end:02d}:00"

                if len(block_trades) > 0:
                    winning_trades = len(block_trades[block_trades['profit'] > 0])
                    total_trades = len(block_trades)

                    time_slot_metrics[slot_name] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                        'total_profit': block_trades['profit'].sum(),
                        'avg_profit': block_trades['profit'].mean(),
                    }

            return time_slot_metrics
        except Exception as e:
            print(f"Error analyzing time slots: {str(e)}")
            return {}

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