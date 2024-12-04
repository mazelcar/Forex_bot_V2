#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forex Trading Bot V2 - System Entry Point

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

Command Line Arguments:
    --mode:  Operation mode ('auto' or 'manual')
            - auto: Fully automated trading (default)
            - manual: Requires trade confirmation
    
    --debug: Enable debug level logging
            - Provides detailed operational information
            - Includes component state transitions
            - Shows detailed error information

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
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

from src.core.bot import ForexBot

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments for bot configuration.

    This function sets up the argument parser and defines the valid command line
    options for configuring the bot's operation. It provides a user-friendly
    interface for controlling bot behavior while maintaining strict argument
    validation.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - mode (str): Operation mode ('auto' or 'manual')
            - debug (bool): Debug logging flag

    Command Line Arguments:
        --mode:  Operation mode selection
                - auto: Fully automated trading (default)
                - manual: Requires trade confirmation
        
        --debug: Enable detailed debug logging

    Example:
        args = parse_arguments()
        print(f"Mode: {args.mode}")        # 'auto' or 'manual'
        print(f"Debug: {args.debug}")      # True or False

    Notes:
        - The function uses argparse's built-in validation
        - Invalid arguments will trigger help display
        - Default values are clearly shown in help
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
    
    return parser.parse_args()

def main() -> NoReturn:
    """
    Primary entry point for the Forex Trading Bot V2 system.

    This function serves as the main entry point and orchestrates the high-level
    flow of the application. It maintains minimal responsibilities while ensuring
    proper bot initialization, execution, and shutdown handling.

    Responsibilities:
        1. Process command line arguments
        2. Initialize the trading bot
        3. Start bot execution
        4. Handle shutdown and cleanup

    Process Flow:
        1. Parse command line arguments
        2. Create bot instance with configuration
        3. Start main bot loop
        4. Handle interruption and cleanup
        5. Exit with appropriate status

    Error Handling:
        - KeyboardInterrupt: Clean shutdown on Ctrl+C
        - Exception: Logs error and exits with status 1
        - SystemExit: Managed exit with status code

    Exit Codes:
        0: Clean shutdown (including Ctrl+C)
        1: Error during execution

    Example:
        This function is typically called by the __main__ block:
            if __name__ == "__main__":
                main()

    Notes:
        - The function never returns (hence NoReturn type hint)
        - All trading functionality is delegated to the bot
        - Maintains clean separation of concerns
    """
    # Process command line arguments
    args = parse_arguments()
    
    try:
        # Initialize and run bot with configuration
        bot = ForexBot(
            mode=args.mode,
            debug=args.debug
        )
        
        # Start bot execution - ForexBot handles all core functionality
        bot.run()
        
    except KeyboardInterrupt:
        # Handle clean shutdown on Ctrl+C
        print("\nShutdown signal received - initiating graceful shutdown...")
        sys.exit(0)
    except Exception as e:
        # Handle unexpected errors
        print(f"\nFatal error occurred: {str(e)}")
        print("See logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()