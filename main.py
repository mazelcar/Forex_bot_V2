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
        choices=['mt5', 'trading_hours', 'bar_timestamps', 'dashboard', 'strategy', 'backtest', 'base', 'run_backtest', 'calculations', 'all'],
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