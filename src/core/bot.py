"""Forex Trading Bot V2 - Bot Orchestrator.

This module contains the `ForexBot` class that serves as the central orchestrator
for the trading system. The bot is responsible for:

1. Managing core components
2. Coordinating trading operations
3. Maintaining system state

The `ForexBot` class is the main entry point for the bot's execution. It initializes
the necessary components, such as the dashboard, and runs the main bot loop. The
bot loop updates the dashboard with the latest system data, which can include
account information, open positions, market status, and overall system health.

Author: mazelcar
Created: December 2024
"""

import time
from src.core.dashboard import Dashboard
from src.core.mt5 import MT5Handler


class ForexBot:
   """Core bot orchestrator for the trading system.

   This class serves as the main entry point for the Forex Trading Bot V2 system.
   It is responsible for initializing the necessary components, running the main
   bot loop, and handling shutdown and cleanup.

   Attributes:
       mode (str): The operation mode of the bot ('auto' or 'manual').
       running (bool): Flag indicating whether the bot is currently running.
       dashboard (Dashboard): Instance of the dashboard component.
       mt5_handler (MT5Handler): Instance of the MT5 handler component.
   """

   def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
       """Initialize the ForexBot with its configuration and components."""
       self.mode = mode
       self.running = False
       self._setup_logging(debug)

       # Initialize components
       self.mt5_handler = MT5Handler(debug=debug)
       self.dashboard = Dashboard()

   def _setup_logging(self, debug: bool) -> None:
       """Set up logging configuration."""
       if debug:
           self.dashboard.log_level = 'DEBUG'

   def run(self) -> None:
    """Run the main bot execution loop."""
    self.running = True

    try:
        while self.running:
            # Get real data from MT5
            market_status = self.mt5_handler.get_market_status()

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
                    'signal_system': 'OK',  # Will be updated later
                    'risk_manager': 'OK'  # Will be updated later
                }
            }

            # Update dashboard with current data
            self.dashboard.update(real_data)

            # Control update frequency
            time.sleep(1)

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