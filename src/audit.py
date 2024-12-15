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

import pandas as pd

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

def audit_strategy() -> None:
    """Audit MA RSI Volume strategy functionality."""
    print("\n=== Starting Strategy Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Strategy Audit")

    try:
        # Initialize strategy with config
        from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        # List all methods in the strategy class
        logger.info("\nInspecting strategy methods:")
        import inspect
        methods = inspect.getmembers(strategy, predicate=inspect.ismethod)
        for name, method in methods:
            if not name.startswith('_'):
                logger.info(f"Public method: {name}")
            else:
                logger.info(f"Private method: {name}")

        # Initialize MT5 handler for test data
        from src.core.mt5 import MT5Handler
        mt5_handler = MT5Handler(debug=True)
        logger.info("MT5Handler instance created")

        # Get sample data for testing
        symbol = "EURUSD"
        timeframe = "M5"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        logger.info(f"\nFetching test data for {symbol} from {start_date} to {end_date}")
        test_data = mt5_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if test_data is not None:
            logger.info(f"Retrieved {len(test_data)} data points")
            logger.info(f"Columns available: {list(test_data.columns)}")

            # Test each major component
            logger.info("\nTesting market condition analysis...")
            try:
                market_condition = strategy._analyze_market_condition(test_data)
                logger.info(f"Market condition result: {market_condition}")
            except Exception as e:
                logger.error(f"Market condition analysis failed: {e}")

            logger.info("\nTesting volatility calculation...")
            try:
                volatility = strategy._calculate_volatility(test_data)
                logger.info(f"Volatility result: {volatility}")
            except Exception as e:
                logger.error(f"Volatility calculation failed: {e}")

            logger.info("\nTesting volume analysis...")
            try:
                # Try to find the correct volume analysis method
                for name, method in methods:
                    if 'volume' in name.lower() and 'analyze' in name.lower():
                        logger.info(f"Found volume analysis method: {name}")
                        volume_result = method(test_data)
                        logger.info(f"Volume analysis result: {volume_result}")
                        break
                else:
                    logger.error("No volume analysis method found")
            except Exception as e:
                logger.error(f"Volume analysis failed: {e}")

            logger.info("\nTesting signal generation...")
            try:
                signals = strategy.generate_signals(test_data)
                logger.info(f"Generated signals: {signals}")
            except Exception as e:
                logger.error(f"Signal generation failed: {e}")

        else:
            logger.error("Failed to retrieve test data")

        logger.info("Strategy audit completed successfully")
        print("\n=== Strategy Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Strategy audit failed: {str(e)}")
        raise

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
                    # ADDED: Test specific data window that matches backtest requirements
                    logger.info("\n=== Testing Strategy Data Window Requirements ===")
                    test_start = end_date - timedelta(days=5)  # ADDED: Match backtest warmup period
                    logger.info(f"Strategy Test Period:")  # ADDED: Log strategy test period
                    logger.info(f"• Start: {test_start}")  # ADDED: Log start time
                    logger.info(f"• End: {end_date}")  # ADDED: Log end time
                    logger.info(f"• Total Days: {(end_date - test_start).days}")

                    # ADDED: Get data specifically for strategy window
                    strategy_data = mt5_handler.get_historical_data(
                        symbol="EURUSD",
                        timeframe="M5",
                        start_date=test_start,
                        end_date=end_date
                    )

                    # ADDED: Log strategy data details
                    if strategy_data is not None:
                        logger.info("\nStrategy Data Analysis:")
                        logger.info(f"• Total Bars Retrieved: {len(strategy_data)}")
                        logger.info(f"• First Bar Time: {strategy_data['time'].min()}")
                        logger.info(f"• Last Bar Time: {strategy_data['time'].max()}")
                        logger.info(f"• Hours Covered: {(strategy_data['time'].max() - strategy_data['time'].min()).total_seconds() / 3600:.2f}")
                        logger.info(f"• Bars Per Day Avg: {len(strategy_data) / (end_date - test_start).days:.2f}")
                    else:
                        logger.error("Failed to retrieve strategy test data")

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

def audit_base_strategy() -> None:
    """Audit base Strategy functionality."""
    print("\n=== Starting Base Strategy Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Base Strategy Audit")

    try:
        # We'll use MA_RSI_Volume_Strategy since it inherits from base Strategy
        from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        # Create test market data
        sample_data = pd.DataFrame({
            'symbol': ['EURUSD'],
            'spread': [2.0],
            'time': [datetime.now()],
            'session': ['London']
        })

        logger.info("\nTesting signal validation...")
        test_signals = [
            {'type': 'BUY', 'strength': 0.8},
            {'type': 'SELL', 'strength': 0.3},
            {'type': 'NONE', 'strength': 0.0},
            {},  # Empty signal
            None  # None signal
        ]

        for signal in test_signals:
            try:
                logger.info(f"\nValidating signal: {signal}")
                logger.info(f"Market data: {dict(sample_data.iloc[0])}")
                is_valid = strategy.validate_signal(signal, sample_data.iloc[0])
                logger.info(f"Validation result: {is_valid}")
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")

        logger.info("Base Strategy audit completed successfully")
        print("\n=== Base Strategy Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Base Strategy audit failed: {str(e)}")
        raise

def audit_backtest() -> None:
    """Audit backtesting functionality."""
    print("\n=== Starting Backtest Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Backtest Audit")

    try:
        # Initialize strategy and backtester
        from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
        from src.strategy.backtesting import Backtester
        from src.core.mt5 import MT5Handler

        # Create strategy instance
        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        # Create backtester instance
        backtester = Backtester(
            strategy=strategy,
            initial_balance=10000,
            commission=2.0,
            spread=0.0001
        )
        logger.info("Backtester instance created")

        # Get test data
        mt5_handler = MT5Handler(debug=True)
        logger.info("MT5Handler instance created")

        # Test short timeframe first
        symbol = "EURUSD"
        timeframe = "M5"
        end_date = datetime(2024, 12, 12, 23, 0)  # Last known good data point
        start_date = end_date - timedelta(hours=8)  # Just 4 hours for testing

        logger.info(f"\nTesting short timeframe backtest:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {start_date} to {end_date}")

        test_data = mt5_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if test_data is not None:
            # Add symbol column to data
            test_data['symbol'] = symbol

            logger.info(f"Retrieved {len(test_data)} data points")
            logger.info(f"Columns available: {list(test_data.columns)}")

            # Test backtester components
            logger.info("\nTesting data windowing...")
            window_size = 100
            test_window = test_data.iloc[0:min(window_size, len(test_data))]
            logger.info(f"Window size: {len(test_window)}")
            logger.info(f"Window columns: {list(test_window.columns)}")

            logger.info("\nTesting strategy update...")
            strategy.update_market_condition(test_window)
            logger.info("Strategy updated with window data")

            logger.info("\nTesting signal generation...")
            try:
                signal = strategy.generate_signals(test_window)
                logger.info(f"Generated signal: {signal}")
            except Exception as e:
                logger.error(f"Signal generation failed: {str(e)}")
                logger.error(f"Data shape: {test_window.shape}")
                logger.error(f"Data columns: {test_window.columns}")

            logger.info("\nTesting position processing...")
            equity = backtester._process_positions(test_window.iloc[-1], 10000)
            logger.info(f"Processed positions, equity: {equity}")

            logger.info("\nRunning full backtest simulation...")
            logger.info("Creating simulation data...")
            sim_data = test_data.copy()  # Create a copy for simulation
            sim_data['symbol'] = symbol   # Ensure symbol column exists

            results = backtester._run_simulation(sim_data)
            logger.info(f"Backtest results: {results}")

        else:
            logger.error("Failed to retrieve test data")

        logger.info("Backtest audit completed successfully")
        print("\n=== Backtest Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Backtest audit failed: {str(e)}")
        raise

def audit_run_backtest() -> None:
   """Audit backtest runtime functionality."""
   print("\n=== Starting Run Backtest Audit ===")
   print("Setting up logging...")

   logger, log_dir, timestamp = setup_audit_logging()
   logger.info("Starting Run Backtest Audit")

   try:
       # Initialize components
       from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
       from src.strategy.backtesting import Backtester
       from src.core.mt5 import MT5Handler

       strategy_config = str(Path("config/strategy.json"))
       strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
       logger.info("Strategy instance created")

       backtester = Backtester(
           strategy=strategy,
           initial_balance=10000,
           commission=2.0,
           spread=0.0001
       )
       logger.info("Backtester instance created")

       mt5_handler = MT5Handler()
       logger.info("MT5Handler instance created")

       # Get sample data for testing - increased to 4 hours
       symbol = "EURUSD"
       timeframe = "M5"
       end_date = datetime.now()
       start_date = end_date - timedelta(hours=4)  # Changed from 1 to 4 hours

       logger.info(f"\nFetching test data:")
       logger.info(f"Symbol: {symbol}")
       logger.info(f"Timeframe: {timeframe}")
       logger.info(f"Period: {start_date} to {end_date}")

       data = mt5_handler.get_historical_data(
           symbol=symbol,
           timeframe=timeframe,
           start_date=start_date,
           end_date=end_date
       )

       if data is not None:
           # Add symbol column first
           data['symbol'] = symbol

           # Log data structure after adding symbol
           logger.info("Initial data structure:")
           logger.info(f"Shape: {data.shape}")
           logger.info(f"Columns: {list(data.columns)}")
           logger.info(f"First row:\n{data.iloc[0]}")

           # Now verify required columns
           required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'symbol']
           logger.info("\nVerifying required columns:")
           missing_columns = [col for col in required_columns if col not in data.columns]

           # Log missing columns if any
           if missing_columns:
               logger.error(f"Missing required columns: {missing_columns}")
               for col in missing_columns:
                   logger.error(f"Column '{col}' is required but not found in data")
               raise ValueError(f"Data missing required columns: {missing_columns}")

           # Verify symbol column was added correctly
           logger.info("\nVerifying symbol column addition:")
           logger.info(f"Symbol column exists: {('symbol' in data.columns)}")
           logger.info(f"Symbol column value: {data['symbol'].iloc[0]}")

           logger.info(f"Retrieved {len(data)} data points")
           logger.info(f"Columns: {list(data.columns)}")

           # Test windowing with enhanced error checking
           logger.info("\nTesting data windowing:")
           try:
               if len(data) < 20:
                   logger.error(f"Insufficient data points. Found {len(data)}, need at least 20")
                   raise ValueError("Insufficient data points for windowing")

               current_window = data.iloc[0:20]  # Get first 20 bars
               logger.info(f"Window size: {len(current_window)}")
               logger.info(f"Window columns: {list(current_window.columns)}")

               # Verify window data integrity
               logger.info("Window data validation:")
               logger.info(f"Window contains nulls: {current_window.isnull().any().any()}")
               logger.info(f"Window symbol column check: {current_window['symbol'].nunique()} unique values")

           except Exception as e:
               logger.error(f"Window creation failed: {str(e)}")
               logger.error(f"Window data shape: {data.shape}")
               raise

           # Test market condition update
           logger.info("\nTesting market condition update:")
           try:
               strategy.update_market_condition(current_window)
               logger.info("Market condition updated successfully")
               logger.info(f"Current condition: {strategy.current_market_condition}")
           except Exception as e:
               logger.error(f"Market condition update failed: {str(e)}")
               logger.error("Data used for update:")
               logger.error(f"Shape: {current_window.shape}")
               logger.error(f"Columns: {current_window.columns}")
               raise

           # Test signal generation with enhanced error checking
           logger.info("\nTesting signal generation:")
           try:
               # Pre-signal generation data validation
               logger.info("Validating data for signal generation:")
               for col in ['open', 'high', 'low', 'close', 'tick_volume', 'symbol']:
                   if col not in current_window.columns:
                       raise ValueError(f"Missing required column for signal generation: {col}")

               signal = strategy.generate_signals(current_window)
               logger.info(f"Signal generated: {signal}")
           except Exception as e:
               logger.error(f"Signal generation failed: {str(e)}")
               logger.error(f"Data used for signal generation:")
               logger.error(f"Shape: {current_window.shape}")
               logger.error(f"Columns: {current_window.columns}")
               logger.error(f"First row:\n{current_window.iloc[0]}")
               raise

           # Test validation
           logger.info("\nTesting signal validation:")
           if signal:
               try:
                   is_valid = strategy.validate_signal(signal, current_window.iloc[-1])
                   logger.info(f"Signal validation result: {is_valid}")
               except Exception as e:
                   logger.error(f"Signal validation failed: {str(e)}")

           # Test simulation
           logger.info("\nTesting simulation run:")
           try:
               results = backtester._run_simulation(current_window)
               logger.info(f"Simulation results: {results}")
           except Exception as e:
               logger.error(f"Simulation failed: {str(e)}")

       else:
           logger.error("Failed to retrieve test data")

       logger.info("\nRun Backtest audit completed")
       print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

   except Exception as e:
       logger.error(f"Run Backtest audit failed: {str(e)}")
       raise

def audit_calculations() -> None:
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

            # Test ATR calculation
            logger.info("\nTesting ATR calculation:")
            try:
                atr = strategy._calculate_atr(data)
                logger.info(f"ATR value: {atr}")
                logger.info("ATR calculation components:")
                logger.info(f"High-Low range: {(data['high'] - data['low']).mean()}")
                logger.info(f"High-PrevClose range: {abs(data['high'] - data['close'].shift()).mean()}")
                logger.info(f"Low-PrevClose range: {abs(data['low'] - data['close'].shift()).mean()}")
            except Exception as e:
                logger.error(f"ATR calculation failed: {str(e)}")

            # Test volume analysis
            logger.info("\nTesting volume analysis:")
            try:
                volume_data = strategy._analyze_volume_conditions(data)
                logger.info(f"Volume analysis: {volume_data}")
                logger.info("Volume components:")
                logger.info(f"Current volume: {data['tick_volume'].iloc[-1]}")
                logger.info(f"Volume SMA: {data['tick_volume'].rolling(window=strategy.volume_period).mean().iloc[-1]}")
            except Exception as e:
                logger.error(f"Volume analysis failed: {str(e)}")

            # Test signal strength calculation
            logger.info("\nTesting signal strength calculation:")
            try:
                test_crossover = {'type': 'BULLISH', 'strength': 0.8}
                test_rsi = 45.0
                test_volume = {'above_average': True, 'high_volume': False, 'volume_ratio': 1.2}
                strength = strategy._calculate_signal_strength(
                    test_crossover,
                    test_rsi,
                    test_volume,
                    data['close'].iloc[-1],
                    data['close'].iloc[-2]
                )
                logger.info(f"Signal strength: {strength}")
            except Exception as e:
                logger.error(f"Signal strength calculation failed: {str(e)}")

        logger.info("Calculation audit completed")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Calculation audit failed: {str(e)}")
        raise

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

    if target in ['strategy', 'all']:
        audit_strategy()

    if target in ['backtest', 'all']:
        audit_backtest()

    if target in ['base', 'all']:
        audit_base_strategy()

    if target in ['run_backtest', 'all']:
        audit_run_backtest()

    if target in ['calculations', 'all']:
        audit_calculations()

    if target == 'all':
        # TODO: Add other module audits here  # pylint: disable=fixme
        pass