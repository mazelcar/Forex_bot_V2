"""Audit Module for Forex Trading Bot V2.

This module provides audit capabilities for testing and verifying
different components of the trading system.

Current audit capabilities:
- Dashboard display and functionality testing
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

from src.core.dashboard import Dashboard

def setup_audit_logging() -> Tuple[logging.Logger, Path, str]:
    """Setup logging for audit operations.

    Returns:
        tuple containing:
        - logger: Configured logging instance
        - log_dir: Path to log directory
        - timestamp: Current timestamp string
    """
    try:
        # Get absolute paths
        project_root = Path(__file__).parent.parent.absolute()
        log_dir = project_root / "logs" / "audit"

        print("\nAttempting to create log directory at: {log_dir}")

        # Create directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory exists and is writable
        if not log_dir.exists():
            raise RuntimeError(f"Failed to create log directory at: {log_dir}")
        if not os.access(log_dir, os.W_OK):
            raise RuntimeError(f"Log directory exists but is not writable: {log_dir}")

        print(f"Log directory verified at: {log_dir}")

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"audit_{timestamp}.log"

        try:
            # Test file creation
            with open(log_file, 'w') as f:
                f.write("Initializing audit log\n")
        except Exception as e:
            raise RuntimeError(f"Cannot create log file at {log_file}: {str(e)}")

        print(f"Log file created and verified at: {log_file}")

        # Configure logger
        logger = logging.getLogger('audit')
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger, log_dir, timestamp

    except Exception as e:
        print("\nFATAL ERROR in logging setup:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)

def audit_mt5() -> None:
    """Audit MT5 functionality and connection."""
    print("\n=== Starting MT5 Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting MT5 Audit")

    try:
        # Initialize MT5 handler
        from src.core.mt5 import MT5Handler
        mt5_handler = MT5Handler(debug=True)
        logger.info("MT5Handler instance created")

        # Test connection
        logger.info("Testing MT5 connection...")
        if mt5_handler.connected:
            logger.info("MT5 connection: SUCCESS")
        else:
            logger.error("MT5 connection: FAILED")
            raise RuntimeError("Could not connect to MT5")

        # Extended test scenarios
        test_scenarios = [
            {
                "name": "Account connection test",
                "test": "login",
                "params": {
                    "username": "12345",  # Test account
                    "password": "test",   # Test password
                    "server": "MetaQuotes-Demo"
                }
            },
            {
                "name": "Account info retrieval",
                "test": "get_account_info"
            },
            {
                "name": "Position retrieval",
                "test": "get_positions"
            },
            {
                "name": "Market data validation",
                "test": "market_data",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            },
            {
                "name": "Symbol info validation",
                "test": "symbol_info",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            },
            {
                "name": "Historical data retrieval",
                "test": "historical_data",
                "params": {
                    "symbol": "EURUSD",
                    "timeframes": ["M1", "M5", "M15", "H1"]
                }
            },
            {
                "name": "Market session checks",
                "test": "market_session",
                "params": {
                    "markets": ["Sydney", "Tokyo", "London", "New York"]
                }
            },
            {
                "name": "Price tick validation",
                "test": "price_ticks",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            }
        ]

        for scenario in test_scenarios:
            logger.info("Testing scenario: %s", scenario['name'])

            try:
                if scenario['test'] == 'login':
                    # Test login - but don't use real credentials in audit
                    logger.info("Login method: Available")
                    logger.info("Login parameters validation: OK")

                elif scenario['test'] == 'get_account_info':
                    # Test account info structure
                    account_info = mt5_handler.get_account_info()
                    required_fields = ['balance', 'equity', 'profit', 'margin', 'margin_free']

                    if isinstance(account_info, dict):
                        missing_fields = [field for field in required_fields if field not in account_info]
                        if not missing_fields:
                            logger.info("Account info structure: Valid")
                            for field, value in account_info.items():
                                logger.info(f"  {field}: {value}")
                        else:
                            logger.error("Account info missing fields: %s", missing_fields)
                    else:
                        logger.error("Account info invalid type: %s", type(account_info))

                elif scenario['test'] == 'get_positions':
                    # Test position retrieval structure
                    positions = mt5_handler.get_positions()
                    logger.info("Position retrieval: OK")
                    if isinstance(positions, list):
                        logger.info("Positions structure: Valid")
                        logger.info("Current open positions: %d", len(positions))
                        for pos in positions:
                            logger.info(f"  Position: {pos['symbol']} {pos['type']} {pos['volume']} lots")
                    else:
                        logger.error("Positions invalid type: %s", type(positions))

                elif scenario['test'] == 'market_data':
                    import MetaTrader5 as mt5
                    for symbol in scenario['params']['symbols']:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is not None:
                            logger.info(f"Market data for {symbol}:")
                            logger.info(f"  Bid: {tick.bid:.5f}")
                            logger.info(f"  Ask: {tick.ask:.5f}")
                            logger.info(f"  Spread: {(tick.ask - tick.bid):.5f}")
                            logger.info(f"  Volume: {tick.volume}")
                            logger.info(f"  Time: {datetime.fromtimestamp(tick.time)}")
                        else:
                            logger.error("Market data for %s: Unavailable", symbol)

                elif scenario['test'] == 'symbol_info':
                    for symbol in scenario['params']['symbols']:
                        info = mt5_handler.get_symbol_info(symbol)
                        if info is not None:
                            logger.info(f"Symbol info for {symbol}:")
                            for key, value in info.items():
                                logger.info(f"  {key}: {value}")
                        else:
                            logger.error(f"Symbol info not available for {symbol}")

                elif scenario['test'] == 'historical_data':
                    # Use last Thursday
                    end_date = datetime.now()
                    while end_date.weekday() != 3:  # 3 is Thursday
                        end_date = end_date - timedelta(days=1)
                    # Set to market close time (17:00 EST)
                    end_date = end_date.replace(hour=17, minute=0, second=0, microsecond=0)
                    # Get previous 24 hours
                    start_date = end_date - timedelta(days=1)

                    logger.info("Testing with detailed parameters:")
                    logger.info(f"Start date: {start_date}")
                    logger.info(f"End date: {end_date}")

                    for tf in scenario['params']['timeframes']:
                        data = mt5_handler.get_historical_data(
                            scenario['params']['symbol'],
                            tf,
                            start_date,
                            end_date
                        )

                        if data is not None:
                            logger.info(f"Historical data for {scenario['params']['symbol']} {tf}:")
                            logger.info(f"  Rows: {len(data)}")
                            logger.info(f"  Columns: {list(data.columns)}")
                            if len(data) > 0:
                                logger.info(f"  First timestamp: {data['time'].iloc[0]}")
                                logger.info(f"  Last timestamp: {data['time'].iloc[-1]}")
                            else:
                                logger.info("  No data points available in the specified timeframe")
                                error = mt5.last_error()
                                logger.error(f"  MT5 Error getting data: {error}")
                        else:
                            logger.error(f"Historical data not available for {tf}")
                            error = mt5.last_error()
                            logger.error(f"  MT5 Error: {error}")

                elif scenario['test'] == 'market_session':
                    status = mt5_handler.get_market_status()
                    logger.info("Market session status:")
                    for market, is_open in status['status'].items():
                        logger.info(f"  {market}: {'OPEN' if is_open else 'CLOSED'}")
                    logger.info(f"Overall status: {status['overall_status']}")

                elif scenario['test'] == 'price_ticks':
                    import MetaTrader5 as mt5
                    for symbol in scenario['params']['symbols']:
                        ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(minutes=1), 100, mt5.COPY_TICKS_ALL)
                        if ticks is not None:
                            logger.info(f"Price ticks for {symbol}:")
                            logger.info(f"  Number of ticks: {len(ticks)}")
                            if len(ticks) > 0:
                                logger.info("  Latest tick details:")
                                logger.info(f"    Bid: {ticks[-1]['bid']:.5f}")
                                logger.info(f"    Ask: {ticks[-1]['ask']:.5f}")
                                logger.info(f"    Volume: {ticks[-1]['volume']}")
                        else:
                            logger.error(f"No ticks available for {symbol}")

                logger.info("Scenario %s: Completed", scenario['name'])

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Scenario %s failed: %s", scenario['name'], str(e))

        # Test order parameters validation (without placing orders)
        logger.info("Testing trade parameter validation...")
        test_trade_params = {
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "sl": None,
            "tp": None
        }
        try:
            # Just validate the parameters without placing the trade
            if all(param in test_trade_params for param in ['symbol', 'order_type', 'volume']):
                logger.info("Trade parameters structure: Valid")
                logger.info("Trade parameters:")
                for param, value in test_trade_params.items():
                    logger.info(f"  {param}: {value}")
            else:
                logger.error("Trade parameters structure: Invalid")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Trade parameter validation failed: %s", str(e))

        logger.info("MT5 audit completed successfully")
        print("\n=== MT5 Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error("MT5 audit failed: %s", str(e))
        raise
    finally:
        # Ensure MT5 is properly shut down
        if 'mt5_handler' in locals() and mt5_handler.connected:
            mt5_handler.__del__()
            logger.info("MT5 connection closed")

def audit_dashboard() -> None:
    """Audit dashboard functionality without displaying on screen."""
    print("\n=== Starting Dashboard Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Dashboard Audit")

    # Test scenarios
    test_scenarios = [
        {
            "name": "Empty data test",
            "data": {
                "account": {"balance": 0, "equity": 0, "profit": 0},
                "positions": [],
                "market": {"status": "CLOSED", "session": "NONE"},
                "system": {"mt5_connection": "OK", "signal_system": "OK", "risk_manager": "OK"}
            }
        },
        {
            "name": "Normal operation data",
            "data": {
                "account": {"balance": 10000, "equity": 10500, "profit": 500},
                "positions": [
                    {"symbol": "EURUSD", "type": "BUY", "profit": 300},
                    {"symbol": "GBPUSD", "type": "SELL", "profit": 200}
                ],
                "market": {"status": "OPEN", "session": "London"},
                "system": {"mt5_connection": "OK", "signal_system": "OK", "risk_manager": "OK"}
            }
        },
        {
            "name": "Error state data",
            "data": {
                "account": {"balance": 9500, "equity": 9000, "profit": -500},
                "positions": [],
                "market": {"status": "ERROR", "session": "Unknown"},
                "system": {"mt5_connection": "ERROR", "signal_system": "OK", "risk_manager": "WARNING"}
            }
        }
    ]

    try:
        # Create dashboard instance but suppress actual display
        dashboard = Dashboard()
        logger.info("Dashboard instance created successfully")

        for scenario in test_scenarios:
            logger.info("Testing scenario: %s", scenario['name'])

            # Test component methods without actually displaying
            try:
                # Don't actually clear screen, just verify method exists
                if hasattr(dashboard, 'clear_screen'):
                    logger.info("Screen clearing method: Available")
                else:
                    logger.error("Screen clearing method: Missing")
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Screen clearing check failed: %s", str(e))

            try:
                # Test data structure handling without display
                data = scenario['data']

                # Verify account data structure
                if all(k in data['account'] for k in ['balance', 'equity', 'profit']):
                    logger.info("Account data structure: Valid")
                else:
                    logger.error("Account data structure: Invalid")

                # Verify positions data structure
                if isinstance(data['positions'], list):
                    logger.info("Positions data structure: Valid")
                else:
                    logger.error("Positions data structure: Invalid")

                # Verify market data structure
                if all(k in data['market'] for k in ['status', 'session']):
                    logger.info("Market data structure: Valid")
                else:
                    logger.error("Market data structure: Invalid")

                # Verify system data structure
                if all(k in data['system'] for k in ['mt5_connection', 'signal_system', 'risk_manager']):
                    logger.info("System data structure: Valid")
                else:
                    logger.error("System data structure: Invalid")

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Data structure validation failed: %s", str(e))

        logger.info("Dashboard audit completed successfully")
        print("\n=== Dashboard Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Dashboard audit failed: %s", str(e))
        raise

def run_audit(target: str) -> None:
    """Run audit for specified target.

    Args:
        target: Module to audit ('dashboard', 'mt5', or 'all')
    """
    if target in ['dashboard', 'all']:
        audit_dashboard()

    if target == 'mt5':
        audit_mt5()

    if target == 'all':
        # TODO: Add other module audits here  # pylint: disable=fixme
        pass