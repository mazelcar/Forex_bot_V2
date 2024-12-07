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

The bot can operate in two modes: "auto" and "manual". In "auto" mode, the bot
will execute trades automatically based on its trading strategy. In "manual" mode,
the bot will require user confirmation before executing any trades.

The bot also supports debug logging, which can provide detailed operational
information for troubleshooting and development purposes.

Author: mazelcar
Created: December 2024
"""

import time
from src.core.dashboard import Dashboard  # Fixed absolute import


class ForexBot:
    """Core bot orchestrator for the trading system.

    This class serves as the main entry point for the Forex Trading Bot V2 system.
    It is responsible for initializing the necessary components, running the main
    bot loop, and handling shutdown and cleanup.

    Attributes:
        mode (str): The operation mode of the bot ('auto' or 'manual').
        running (bool): Flag indicating whether the bot is currently running.
        dashboard (Dashboard): Instance of the dashboard component.
        test_data (dict): Placeholder data for the bot's state.
    """

    def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
        """Initialize the ForexBot with its configuration and components.

        Args:
            mode (str): Operation mode ('auto' or 'manual').
            debug (bool): Flag to enable debug-level logging.
        """
        self.mode = mode
        self.running = False
        self._setup_logging(debug)

        # Initialize components
        self.dashboard = Dashboard()

        # Initialize placeholder data
        self.test_data = {
            'account': {
                'balance': 10000.00,
                'equity': 10000.00,
                'profit': 0.00
            },
            'positions': [],
            'market': {
                'status': 'OPEN',
                'session': 'London'
            },
            'system': {
                'mt5_connection': 'OK',
                'signal_system': 'OK',
                'risk_manager': 'OK'
            }
        }

    def _setup_logging(self, debug: bool) -> None:
        """Set up logging configuration.

        Args:
            debug (bool): Enable debug logging if True.
        """
        if debug:
            self.dashboard.log_level = 'DEBUG'

    def run(self) -> None:
        """Run the main bot execution loop.

        This method runs the central bot loop, which updates the bot's internal
        state and the dashboard with the latest data. The loop continues until
        the bot is explicitly stopped (e.g., by a keyboard interrupt).

        The bot loop performs the following steps:
        1. Update the bot's internal data (currently using placeholder data)
        2. Update the dashboard with the current bot state
        3. Control the update frequency by sleeping for 1 second

        Raises:
            KeyboardInterrupt: When the user interrupts the bot (e.g., Ctrl+C).
        """
        self.running = True

        try:
            while self.running:
                # Update data (will be real data later)
                self.test_data['positions'] = []  # No positions for now

                # Update dashboard with current data
                self.dashboard.update(self.test_data)

                # Control update frequency
                time.sleep(1)

        except KeyboardInterrupt:
            self.running = False
        finally:
            print("\nBot stopped")

    def stop(self) -> None:
        """Stop the bot execution gracefully.

        This method provides a clean way to stop the bot's execution
        from outside the main loop.
        """
        self.running = False
