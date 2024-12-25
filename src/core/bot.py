"""Forex Trading Bot V2 - Bot Orchestrator.

This module contains the `ForexBot` class that serves as the central orchestrator
for the trading system. The bot is responsible for:

1. Managing core components
2. Coordinating trading operations
3. Maintaining system state

The `ForexBot` class is the main entry point for the Forex Trading Bot V2 system.
It initializes the necessary components (like the dashboard and MT5 handler) and runs
the main bot loop. The main loop updates the dashboard with the latest system data,
fetches data to identify support/resistance levels, checks for trade signals based
on bounce candles, and places or manages trades accordingly.

Author: mazelcar
Created: December 2024
"""

import time
from datetime import datetime, timedelta

from src.core.dashboard import Dashboard
from src.core.mt5 import MT5Handler


def is_within_allowed_hours(current_hour, allowed_hours=range(12,17)):
        return current_hour in allowed_hours

class ForexBot:
    """Core bot orchestrator for the trading system.

    This class runs a loop that:
    - Fetches current market status and account info
    - Updates the dashboard
    - Periodically fetches H1 data to identify support/resistance levels
    - Periodically fetches the latest M5 bars to look for bounce candles
    - Places trades if conditions are met and manages open trades
    """

    def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
        """Initialize the ForexBot with its configuration and components."""
        self.mode = mode
        self.running = False
        self._setup_logging(debug)

        # Initialize components
        self.mt5_handler = MT5Handler(debug=debug)
        self.dashboard = Dashboard()

        # Track open trade if any (a simple example)
        self.open_trade = None

    def _setup_logging(self, debug: bool) -> None:
        """Set up logging configuration."""
        if debug:
            self.dashboard.log_level = 'DEBUG'

    def run(self) -> None:
        """Run the main bot execution loop."""
        self.running = True
        symbol = "EURUSD"
        allowed_hours = range(12,17)  # Allowed trading hours (New York local time)
        utc_to_ny_offset = 5  # Example offset from UTC to NY time (no DST)

        try:
            while self.running:
                market_status = self.mt5_handler.get_market_status()

                # Update dashboard with the latest information
                real_data = {
                    'account': self.mt5_handler.get_account_info(),
                    'positions': self.mt5_handler.get_positions(),
                    'market': {
                        'status': market_status['overall_status'],
                        'session': ', '.join([
                            market for market, is_open
                            in market_status['status'].items()
                            if is_open
                        ]) or 'All Markets Closed'
                    },
                    'system': {
                        'mt5_connection': 'OK' if self.mt5_handler.connected else 'ERROR',
                        'signal_system': 'OK',
                        'risk_manager': 'OK'
                    }
                }

                signal_type = "NONE"
                signal_strength = 0.0
                signal_reasons = []

                self.dashboard.update(real_data)

                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                h1_data = self.mt5_handler.get_historical_data(
                    symbol=symbol,
                    timeframe="H1",
                    start_date=start_date,
                    end_date=end_date
                )

                # Manage open trades if any
                if self.open_trade is not None:
                    m5_data_for_trade = self.mt5_handler.get_historical_data(
                        symbol=symbol,
                        timeframe="M5",
                        start_date=end_date - timedelta(minutes=10),
                        end_date=end_date
                    )
                    if m5_data_for_trade is not None and not m5_data_for_trade.empty:
                        last_bar = m5_data_for_trade.iloc[-1]
                        trade = self.open_trade
                        hit_stop = (trade['type'] == 'BUY' and last_bar['low'] <= trade['sl'])
                        hit_take_profit = (trade['type'] == 'BUY' and last_bar['high'] >= trade['tp'])

                        if hit_stop or hit_take_profit:
                            exit_price = trade['sl'] if hit_stop else trade['tp']
                            profit = (exit_price - trade['entry_price']) / last_bar['pip_value'] * trade['position_size'] * 10.0
                            print(f"Trade closed. Profit: {profit:.2f}")
                            self.open_trade = None

                if self.open_trade is None:
                    m5_data = self.mt5_handler.get_historical_data(
                        symbol=symbol,
                        timeframe="M5",
                        start_date=end_date - timedelta(hours=1),
                        end_date=end_date
                    )

                    if m5_data is not None and len(m5_data) > 0:
                        last_bar = m5_data.iloc[-1]

                        # Convert UTC to NY time:
                        bar_time_utc = last_bar['time']
                        ny_time = bar_time_utc - timedelta(hours=utc_to_ny_offset)
                        ny_hour = ny_time.hour

                        if ny_hour not in allowed_hours:
                            signal_reasons.append("Outside allowed trading hours")


                    else:
                        signal_type = "NONE"
                        signal_strength = 0.0
                        signal_reasons.append("Insufficient M5 data")
                else:
                    signal_type = "NONE"
                    signal_strength = 0.0
                    signal_reasons.append("Open trade in progress, not looking for new signals")

                # Record the signal
                self.signal_manager.record_signal(signal_type, signal_strength, signal_reasons)

                # Write the last 8 hours of signals to a single file, overwriting it each time
                self.signal_manager.write_historical_signals_to_file(last_hours=8, filename="historical_signals.txt")

                current_signal = self.signal_manager.get_current_signal()
                if current_signal is None:
                    real_data['signal'] = {"type": "NONE", "reasons": ["No signal recorded"]}
                else:
                    real_data['signal'] = current_signal

                self.dashboard.update(real_data)

                time.sleep(60)

        except KeyboardInterrupt:
            self.running = False
        finally:
            print("\nBot stopped")


    def stop(self) -> None:
        """Stop the bot execution gracefully."""
        self.running = False

    def __del__(self):
        """Cleanup when bot is destroyed."""
        if hasattr(self, 'mt5_handler'):
            del self.mt5_handler
