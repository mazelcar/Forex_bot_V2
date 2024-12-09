"""Forex Trading Bot V2 - System Entry Point.

This module serves as the primary entry point for the Forex Trading Bot V2 system.
It provides a clean command-line interface for launching and controlling the trading
bot while maintaining minimal responsibilities and clear separation of concerns.

Key Responsibilities:
    - Command Line Interface: Provides a user-friendly CLI for bot configuration
    - Bot Instantiation: Creates and initializes the main bot orchestrator
    - Error Handling: Manages top-level exceptions and provides clean shutdown
    - Path Management: Ensures proper Python path setup for imports

The module explicitly does NOT handle:
    - Trading Logic: Delegated to the bot orchestrator
    - Configuration Management: Handled by the bot's internal systems
    - Market Interactions: Managed by specialized components
    - State Management: Maintained by the bot orchestrator

Usage Examples:
    1. Run in automated mode (default):
        python main.py

    2. Run in manual mode with trade confirmation:
        python main.py --mode manual

    3. Run with debug logging enabled:
        python main.py --debug

    4. Run in manual mode with debug logging:
        python main.py --mode manual --debug

    # ADDED: New usage examples for audit mode
    5. Run MT5 module audit:
        python main.py --audit mt5

    6. Run dashboard audit:
        python main.py --audit dashboard

    7. Run full system audit:
        python main.py --audit all

Command Line Arguments:
    --mode:  Operation mode ('auto' or 'manual')
            - auto: Fully automated trading (default)
            - manual: Requires trade confirmation

    --debug: Enable debug level logging
            - Provides detailed operational information
            - Includes component state transitions
            - Shows detailed error information

    # ADDED: New audit argument documentation
    --audit: Run system audit checks
            - mt5: Audit MT5 module
            - dashboard: Audit dashboard module
            - all: Audit all modules

Dependencies:
    - Python 3.8+
    - MetaTrader5: For trading operations
    - Standard library: sys, argparse, pathlib

Project Structure:
    main.py                 # This file - system entry
    src/
        core/
            bot.py         # Main orchestrator
            dashboard.py   # Critical monitoring

Error Handling:
    - KeyboardInterrupt: Clean shutdown on Ctrl+C
    - SystemExit: Managed shutdown with status code
    - Exception: Catches and logs unexpected errors

Author: mazelcar
Created: December 2024
Version: 2.0.0
License: MIT
"""

import sys
import argparse
from pathlib import Path
from typing import NoReturn

# Add project root to Python path for reliable imports
PROJECT_ROOT = str(Path(__file__).parent.absolute())
sys.path.append(PROJECT_ROOT)

from src.core.bot import ForexBot

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments for the trading bot.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - mode (str): Operation mode ('auto' or 'manual')
            - debug (bool): Debug logging flag
            # ADDED: New return value documentation
            - audit (str): Module to audit (mt5, dashboard, or all)
    """
    parser = argparse.ArgumentParser(
        description='Forex Trading Bot V2 - Advanced Automated Trading System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['auto', 'manual'],
        default='auto',
        help='Trading operation mode (auto: fully automated, manual: requires confirmation)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug level logging for detailed operational information'
    )

    # ADDED: New audit argument
    parser.add_argument(
        '--audit',
        choices=['mt5', 'dashboard', 'strategy', 'backtest', 'base', 'run_backtest', 'calculations', 'all'],
        help='Run audit on specified module(s)'
    )

    return parser.parse_args()

def main() -> NoReturn:
    """Primary entry point for the Forex Trading Bot V2 system.

    This function serves as the main entry point and orchestrates the high-level
    flow of the application. It maintains minimal responsibilities while ensuring
    proper bot initialization, execution, and shutdown handling.

    # ADDED: New functionality documentation
    Supports two operational modes:
    1. Normal bot operation with auto/manual trading
    2. Audit mode for system testing and verification
    """
    args = parse_arguments()

    # ADDED: Handle audit mode
    if args.audit:
        from src.audit import run_audit
        try:
            run_audit(args.audit)
            sys.exit(0)
        except Exception as e:
            print(f"\nAudit failed: {str(e)}")
            sys.exit(1)

    try:
        bot = ForexBot(
            mode=args.mode,
            debug=args.debug
        )
        bot.run()

    except KeyboardInterrupt:
        print("\nShutdown signal received - initiating graceful shutdown...")
        sys.exit(0)
    except (RuntimeError, ConnectionError, ValueError) as e:
        print(f"\nFatal error occurred: {str(e)}")
        print("See logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()