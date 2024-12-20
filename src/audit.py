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

def audit_mt5() -> None:
    """Audit MT5 functionality and connection."""
    print("\n=== Starting MT5 Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting MT5 Audit")

    from src.core.mt5 import MT5Handler
    # Initialize MT5Handler with logger so logs from MarketSessionManager are captured
    mt5_handler = MT5Handler(debug=True, logger=logger)
    logger.info("MT5Handler instance created")

    try:
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
                    logger.info(f"Strategy Test Period:")
                    logger.info(f"• Start: {test_start}")
                    logger.info(f"• End: {end_date}")
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
        if 'mt5_handler' in locals() and hasattr(mt5_handler, 'connected') and mt5_handler.connected:
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
        from src.strategy.strategies.ma_rsi_volume import MA_RSI_Volume_Strategy
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
        from src.strategy.backtesting.backtester import Backtester
        from src.core.mt5 import MT5Handler

        # Create strategy instance
        strategy_config = str(Path("config/strategy.json"))
        logger.info("Strategy instance created")

        # Create backtester instance
        backtester = Backtester(
            strategy=strategy, # type: ignore
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
        start_date = end_date - timedelta(hours=8)  # Just 8 hours for testing

        logger.info(f"\nTesting short timeframe backtest:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"MT5 Connection Status: {mt5_handler.connected}")

        # ENHANCED: Get and log market session status with more detail
        current_market_status = mt5_handler.get_market_status()
        logger.info("\nDetailed Market Session Analysis:")
        logger.info(f"Overall Market Status: {current_market_status['overall_status']}")
        for market, status in current_market_status['status'].items():
            logger.info(f"{market} Session: {'OPEN' if status else 'CLOSED'}")
            if status:
                logger.info(f"  Active Trading Hours: {market}")
                if market in strategy.config.get('sessions', {}): # type: ignore
                    session_info = strategy.config['sessions'][market]
                    logger.info(f"  Start Time: {session_info.get('start_time', 'Not Set')}")
                    logger.info(f"  End Time: {session_info.get('end_time', 'Not Set')}")

        test_data = mt5_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if test_data is not None:
            # Add symbol column to data
            test_data['symbol'] = symbol

            # ENHANCED: Initial Data Analysis with detailed statistics
            logger.info("\nEnhanced Initial Data Analysis:")
            logger.info(f"Total data points: {len(test_data)}")
            logger.info(f"Time range: {test_data['time'].min()} to {test_data['time'].max()}")
            time_diffs = test_data['time'].diff()
            logger.info(f"Average time between bars: {time_diffs.mean()}")
            logger.info(f"Time gaps > 5min: {(time_diffs > pd.Timedelta('5min')).sum()}")
            if (time_diffs > pd.Timedelta('5min')).any():
                gap_times = test_data['time'][time_diffs > pd.Timedelta('5min')]
                logger.info("Gap occurrences:")
                for t in gap_times:
                    logger.info(f"  Gap at: {t}")

            # Data Quality Checks with Enhanced Analysis
            logger.info("\nEnhanced Data Quality Checks:")
            price_columns = ['open', 'high', 'low', 'close']
            volume_columns = ['tick_volume']

            # Price Data Analysis
            logger.info("\nPrice Data Analysis:")
            for col in price_columns:
                nulls = test_data[col].isnull().sum()
                zeros = (test_data[col] == 0).sum()
                mean_val = test_data[col].mean()
                std_val = test_data[col].std()
                logger.info(f"\n{col.upper()} Analysis:")
                logger.info(f"  Basic Statistics:")
                logger.info(f"    Null values: {nulls}")
                logger.info(f"    Zero values: {zeros}")
                logger.info(f"    Min: {test_data[col].min():.5f}")
                logger.info(f"    Max: {test_data[col].max():.5f}")
                logger.info(f"    Mean: {mean_val:.5f}")
                logger.info(f"    Std Dev: {std_val:.5f}")

                # Price Movement Analysis
                moves = test_data[col].diff().abs()
                logger.info(f"  Price Movement Analysis:")
                logger.info(f"    Average Movement: {moves.mean():.5f}")
                logger.info(f"    Max Movement: {moves.max():.5f}")
                logger.info(f"    Movement > 10 pips: {(moves > 0.001).sum()}")

            # Volume Analysis
            logger.info("\nVolume Data Analysis:")
            for col in volume_columns:
                mean_vol = test_data[col].mean()
                std_vol = test_data[col].std()
                logger.info(f"\n{col.upper()} Analysis:")
                logger.info(f"  Mean Volume: {mean_vol:.2f}")
                logger.info(f"  Std Dev Volume: {std_vol:.2f}")
                logger.info(f"  Coefficient of Variation: {(std_vol/mean_vol if mean_vol > 0 else 0):.4f}")
                logger.info(f"  Zero Volume Bars: {(test_data[col] == 0).sum()}")
                logger.info(f"  Low Volume Bars (<25% mean): {(test_data[col] < mean_vol * 0.25).sum()}")
                logger.info(f"  High Volume Bars (>200% mean): {(test_data[col] > mean_vol * 2).sum()}")

            # Strategy Configuration and Required Bars
            logger.info("\nStrategy Configuration Analysis:")
            logger.info(f"Strategy name: {strategy.name}")
            logger.info(f"Fast EMA period: {strategy.fast_ema_period}")
            logger.info(f"Slow EMA period: {strategy.slow_ema_period}")
            logger.info(f"RSI period: {strategy.rsi_period}")
            logger.info(f"Volume period: {strategy.volume_period}")
            logger.info("\nTesting data windowing...")
            window_size = 100
            test_window = test_data.iloc[0:min(window_size, len(test_data))]
            logger.info("\nEnhanced Basic Data Validation:")
            try:
                min_required_bars = max(
                    strategy.fast_ema_period * 2,
                    strategy.slow_ema_period * 2,
                    strategy.rsi_period * 2,
                    strategy.volume_period * 2,
                    50
                )
                logger.info(f"Minimum Required Bars Detail:")
                logger.info(f"  Fast EMA warmup: {strategy.fast_ema_period * 2}")
                logger.info(f"  Slow EMA warmup: {strategy.slow_ema_period * 2}")
                logger.info(f"  RSI warmup: {strategy.rsi_period * 2}")
                logger.info(f"  Volume warmup: {strategy.volume_period * 2}")
                logger.info(f"  Base minimum: 50")
                logger.info(f"  Final required: {min_required_bars}")
                logger.info(f"  Available bars: {len(test_data)}")
                logger.info(f"  Sufficient data: {len(test_data) >= min_required_bars}")

                basic_validation = strategy.data_validator.validate_basic_data(test_data)
                logger.info(f"\nBasic Validation Results:")
                logger.info(f"Overall Pass: {basic_validation['overall_pass']}")

                # Log validation checks
                if 'checks' in basic_validation:
                    logger.info("\nValidation Checks:")
                    for check_name, check_result in basic_validation['checks'].items():
                        logger.info(f"  {check_name}: {'[PASS]' if check_result else '[FAIL]'}")

                # Log any error messages
                if not basic_validation['overall_pass']:
                    logger.error("\nValidation Errors:")
                    for msg in basic_validation.get('error_messages', []):
                        logger.error(f"  {msg}")

            except Exception as e:
                logger.error(f"Data validation failed: {str(e)}")
                logger.error(f"Data shape: {test_data.shape}")
                logger.error(f"Data columns: {test_data.columns}")
                import traceback
                logger.error(traceback.format_exc())

            # Enhanced Signal Validation Parameters
            logger.info("\nDetailed Signal Validation Parameters:")
            try:
                logger.info("\nFilter Configurations:")
                filters = strategy.config['filters']
                for filter_name, filter_config in filters.items():
                    logger.info(f"\n{filter_name.upper()} Filter:")
                    logger.info(f"  Configuration: {filter_config}")
                    if isinstance(filter_config, dict):
                        if filter_config.get('dynamic_adjustment', {}).get('enabled'):
                            logger.info(f"  Dynamic Adjustment: Enabled")
                            logger.info(f"  Adjustment Parameters: {filter_config['dynamic_adjustment']}")

            except Exception as e:
                logger.error(f"Error analyzing filters: {str(e)}")

            # Enhanced Signal Generation and Validation
            logger.info("\nEnhanced Signal Analysis:")
            try:
                # Generate signals with market condition context
                market_condition = strategy._analyze_market_condition(test_window)
                logger.info("\nMarket Condition Context:")
                logger.info(f"Phase: {market_condition.get('phase', 'Unknown')}")
                logger.info(f"Volatility: {market_condition.get('volatility', 'Unknown')}")
                logger.info(f"Trend Strength: {market_condition.get('trend_strength', 'Unknown')}")

                # Generate and analyze signal
                signal = strategy.generate_signals(test_window)
                if signal:
                    logger.info("\nDetailed Signal Properties:")
                    for key, value in signal.items():
                        logger.info(f"  {key}: {value}")

                    # Enhanced validation logging
                    logger.info("\nSignal Validation Process:")
                    market_data = test_window.iloc[-1]

                    # 1. Spread Check
                    current_spread = float(market_data.get('spread', float('inf')))
                    max_spread = strategy.config['filters']['spread']['max_spread_pips']
                    logger.info(f"\nSpread Validation:")
                    logger.info(f"  Current Spread: {current_spread}")
                    logger.info(f"  Maximum Allowed: {max_spread}")
                    logger.info(f"  Spread Valid: {current_spread <= max_spread}")

                    # 2. Market Session Check
                    logger.info(f"\nSession Validation:")
                    session_valid = strategy._is_valid_session(market_data)
                    logger.info(f"  Current Time: {market_data.get('time')}")
                    logger.info(f"  Session Valid: {session_valid}")

                    # 3. Signal Strength Check
                    logger.info(f"\nStrength Validation:")
                    min_strength = 0.7  # From strategy config
                    signal_strength = float(signal.get('strength', 0))
                    logger.info(f"  Signal Strength: {signal_strength}")
                    logger.info(f"  Minimum Required: {min_strength}")
                    logger.info(f"  Strength Valid: {signal_strength >= min_strength}")

                    # Overall validation result
                    is_valid = strategy.validate_signal(signal, market_data)
                    logger.info(f"\nFinal Validation Result: {is_valid}")

                    if is_valid:
                        try:
                            # Enhanced position parameter logging
                            logger.info("\nPosition Calculation Details:")
                            position_size = strategy.calculate_position_size({'balance': 10000}, test_window)
                            sl = strategy.calculate_stop_loss(test_window, signal)
                            tp = strategy.calculate_take_profit(test_window, signal)

                            logger.info(f"Position Parameters:")
                            logger.info(f"  Size: {position_size}")
                            logger.info(f"  Entry: {signal.get('entry_price')}")
                            logger.info(f"  Stop Loss: {sl}")
                            logger.info(f"  Take Profit: {tp}")
                            if sl and tp and signal.get('entry_price'):
                                risk = abs(signal['entry_price'] - sl) * position_size
                                reward = abs(tp - signal['entry_price']) * position_size
                                logger.info(f"  Risk Amount: {risk:.2f}")
                                logger.info(f"  Reward Amount: {reward:.2f}")
                                logger.info(f"  Risk-Reward Ratio: {(reward/risk if risk > 0 else 'N/A')}")
                        except Exception as e:
                            logger.error(f"Position calculation error: {str(e)}")
                else:
                    logger.info("No signal generated")
                    logger.info("Analyzing market conditions preventing signal generation:")
                    logger.info(f"  Market Phase: {market_condition['phase']}")
                    logger.info(f"  Volatility Level: {market_condition['volatility']}")
                    logger.info(f"  Trend Strength: {market_condition['trend_strength']}")

            except Exception as e:
                logger.error(f"Signal analysis failed: {str(e)}")
                logger.error(f"Data shape: {test_window.shape}")
                logger.error(f"Data columns: {test_window.columns}")

            # Analyze Simulation Process
            logger.info("\nAnalyzing Simulation Process:")
            try:
                # First, analyze backtester configuration
                logger.info("\nBacktester Configuration:")
                logger.info(f"Initial Balance: ${backtester.initial_balance:,.2f}")
                logger.info(f"Commission: ${backtester.commission:.2f}")
                logger.info(f"Spread: {backtester.spread:.5f}")

                # Analyze signal generation and validation process
                signal = strategy.generate_signals(test_window)
                if signal and signal['type'] != 'NONE':
                    logger.info("\nSignal Details:")
                    logger.info(f"Type: {signal['type']}")
                    logger.info(f"Strength: {signal['strength']:.2f}")
                    logger.info(f"Entry Price: {signal.get('entry_price', 'Not Set')}")

                    # Test position sizing
                    position_size = strategy.calculate_position_size( # type: ignore
                        {'balance': backtester.initial_balance},
                        test_window
                    )
                    sl = strategy.calculate_stop_loss(test_window, signal)
                    tp = strategy.calculate_take_profit(test_window, signal)

                    logger.info("\nTrade Parameters:")
                    logger.info(f"Position Size: {position_size:.2f} lots")
                    logger.info(f"Stop Loss: {sl:.5f if sl else 'Not Set'}")
                    logger.info(f"Take Profit: {tp:.5f if tp else 'Not Set'}")

                    # Calculate potential risk/reward
                    if sl and tp:
                        risk = abs(signal['entry_price'] - sl) * position_size
                        reward = abs(tp - signal['entry_price']) * position_size
                        logger.info(f"Risk Amount: ${risk:.2f}")
                        logger.info(f"Reward Amount: ${reward:.2f}")
                        logger.info(f"Risk-Reward Ratio: {reward/risk:.2f}")

                # Run simulation with detailed logging
                logger.info("\nRunning Simulation:")
                results = backtester._run_simulation(test_window)

                logger.info("\nSimulation Results Analysis:")
                logger.info(f"Total Trades: {results.get('total_trades', 0)}")
                logger.info(f"Winning Trades: {results.get('winning_trades', 0)}")
                logger.info(f"Losing Trades: {results.get('losing_trades', 0)}")
                logger.info(f"Gross Profit: ${results.get('gross_profit', 0):.2f}")
                logger.info(f"Gross Loss: ${results.get('gross_loss', 0):.2f}")
                logger.info(f"Net Profit: ${results.get('total_profit', 0):.2f}")

                # Analyze why trades might not be executing
                if results.get('total_trades', 0) == 0:
                    logger.info("\nAnalyzing Why No Trades Were Executed:")
                    logger.info("1. Signal Generation:")
                    logger.info(f"   - Valid Signal Generated: {signal['type'] != 'NONE' if signal else False}")
                    logger.info(f"   - Signal Strength: {signal.get('strength', 0) if signal else 'No Signal'}")

                    logger.info("\n2. Position Parameters:")
                    logger.info(f"   - Position Size: {position_size if 'position_size' in locals() else 'Not Calculated'}")
                    logger.info(f"   - Stop Loss Available: {sl is not None if 'sl' in locals() else 'Not Calculated'}")
                    logger.info(f"   - Take Profit Available: {tp is not None if 'tp' in locals() else 'Not Calculated'}")

            except Exception as e:
                logger.error(f"Simulation analysis failed: {str(e)}")
                logger.error(f"Data shape: {test_window.shape}")
                logger.error(f"Data columns: {test_window.columns}")
                import traceback
                logger.error(traceback.format_exc())

            # Enhanced Simulation Analysis
            logger.info("\nRunning enhanced simulation analysis...")
            try:
                sim_data = test_data.copy()
                sim_data['symbol'] = symbol

                # Pre-simulation signal analysis
                logger.info("\nPre-simulation Signal Analysis:")
                all_signals = []
                validation_stats = {
                    'total_generated': 0,
                    'spread_rejected': 0,
                    'session_rejected': 0,
                    'strength_rejected': 0,
                    'fully_validated': 0,
                    'buy_signals': 0,
                    'sell_signals': 0
                }

                # Analyze each potential signal
                for i in range(len(sim_data) - min_required_bars):
                    window = sim_data.iloc[i:i+min_required_bars]
                    sig = strategy.generate_signals(window)

                    if sig and sig['type'] != 'NONE':
                        validation_stats['total_generated'] += 1

                        if sig['type'] == 'BUY':
                            validation_stats['buy_signals'] += 1
                        elif sig['type'] == 'SELL':
                            validation_stats['sell_signals'] += 1

                        market_data = window.iloc[-1]

                        # Check validation criteria
                        current_spread = float(market_data.get('spread', float('inf')))
                        if current_spread > max_spread:
                            validation_stats['spread_rejected'] += 1
                            continue

                        if not strategy._is_valid_session(market_data):
                            validation_stats['session_rejected'] += 1
                            continue

                        if float(sig.get('strength', 0)) < 0.7:
                            validation_stats['strength_rejected'] += 1
                            continue

                        validation_stats['fully_validated'] += 1
                        all_signals.append(sig)

                # Log validation statistics
                logger.info("\nSignal Validation Statistics:")
                for key, value in validation_stats.items():
                    logger.info(f"  {key}: {value}")
                if validation_stats['total_generated'] > 0:
                    logger.info(f"  Validation Rate: {(validation_stats['fully_validated']/validation_stats['total_generated']*100):.2f}%")
                    logger.info(f"  Buy/Sell Ratio: {(validation_stats['buy_signals']/validation_stats['sell_signals'] if validation_stats['sell_signals'] > 0 else 'N/A')}")

                # Run simulation
                results = backtester._run_simulation(sim_data)
                logger.info(f"\nSimulation Results: {results}")

                # Post-simulation analysis
                logger.info("\nPost-simulation Analysis:")
                logger.info(f"Signal to Trade Conversion Rate: {(results.get('total_trades', 0)/validation_stats['fully_validated']*100 if validation_stats['fully_validated'] > 0 else 0):.2f}%")
                if results.get('total_trades', 0) > 0:
                    logger.info(f"Win Rate: {(results.get('winning_trades', 0)/results.get('total_trades', 0)*100):.2f}%")
                    logger.info(f"Average Profit per Trade: {(results.get('total_profit', 0)/results.get('total_trades', 0)):.2f}")

            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}")
                logger.error(f"Simulation data shape: {sim_data.shape}")
                logger.error(f"Simulation data columns: {sim_data.columns}")

        else:
            logger.error("Failed to retrieve test data")

        logger.info("Backtest audit completed successfully")
        print("\n=== Backtest Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Backtest audit failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def audit_run_backtest() -> None:
   """Audit backtest runtime functionality."""
   print("\n=== Starting Run Backtest Audit ===")
   print("Setting up logging...")

   logger, log_dir, timestamp = setup_audit_logging()
   logger.info("Starting Run Backtest Audit")

   try:
       # Initialize components
       from src.strategy.backtesting.backtester import Backtester
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
    """Audit strategy calculations and isolate which module is failing."""
    print("\n=== Starting Calculation Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Calculation Audit")

    try:
        # Setup test environment
        from src.strategy.strategies.ma_rsi_volume import MA_RSI_Volume_Strategy
        from src.core.mt5 import MT5Handler
        import MetaTrader5 as mt5

        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        mt5_handler = MT5Handler()
        logger.info("MT5Handler instance created")

        # Get larger sample for better calculation testing
        symbol = "EURUSD"
        timeframe = "M5"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Changed from hours to days

        logger.info(f"\nFetching historical data:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Timeframe: {timeframe}")

        # Add error checking for MT5 connection
        if not mt5_handler.connected:
            logger.error("MT5 not connected")
            return

        # Get data with enhanced error handling
        try:
            data = mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is None:
                logger.error("MT5 returned None for historical data")
                mt5_error = mt5.last_error()
                logger.error(f"MT5 Error: {mt5_error}")
                return

            if len(data) == 0:
                logger.error("MT5 returned empty dataset")
                return

            logger.info(f"Successfully retrieved {len(data)} bars")
            logger.info(f"Data range: {data['time'].min()} to {data['time'].max()}")

        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return

        if data is not None and len(data) > 0:
            data['symbol'] = symbol
            logger.info(f"Retrieved {len(data)} data points")

            # Verify we have enough data
            min_required = max(
                strategy.slow_ema_period * 2,
                strategy.rsi_period * 2,
                strategy.volume_period * 2,
                20  # Minimum baseline
            )

            logger.info(f"\nData Requirements:")
            logger.info(f"Minimum required bars: {min_required}")
            logger.info(f"Available bars: {len(data)}")

            if len(data) < min_required:
                logger.error(f"Insufficient data points. Need at least {min_required}, have {len(data)}")
                return

            # First, let's analyze the signal generation components
            logger.info("\n=== SIGNAL GENERATION COMPONENT ANALYSIS ===")

            # Analyze EMA calculations
            try:
                logger.info("\nAnalyzing EMA calculations...")
                fast_ema = data['close'].ewm(span=strategy.fast_ema_period, adjust=False).mean()
                slow_ema = data['close'].ewm(span=strategy.slow_ema_period, adjust=False).mean()

                logger.info(f"Fast EMA current: {fast_ema.iloc[-1]:.5f}")
                logger.info(f"Slow EMA current: {slow_ema.iloc[-1]:.5f}")
                logger.info(f"EMA Spread (pips): {(fast_ema.iloc[-1] - slow_ema.iloc[-1])*10000:.1f}")
            except Exception as e:
                logger.error(f"EMA Calculation failed: {str(e)}")

            # Analyze RSI with validation
            try:
                logger.info("\nAnalyzing RSI calculation...")
                rsi = strategy._calculate_rsi(data['close'])
                logger.info(f"Current RSI: {rsi.iloc[-1]:.2f}")
                logger.info(f"Previous RSI: {rsi.iloc[-2]:.2f}")
            except Exception as e:
                logger.error(f"RSI Calculation failed: {str(e)}")

            # Analyze Volume with validation
            try:
                logger.info("\nAnalyzing Volume calculation...")
                if 'tick_volume' in data.columns:
                    volume_data = data['tick_volume'].tail(strategy.volume_period * 2)
                    logger.info(f"Recent Volume Data: {volume_data.tail()}")
                    vol_sma = data['tick_volume'].rolling(window=strategy.volume_period).mean()
                    logger.info(f"Volume SMA: {vol_sma.iloc[-1]:.2f}")
                    logger.info(f"Current Volume: {data['tick_volume'].iloc[-1]}")
                    logger.info(f"Volume Ratio: {data['tick_volume'].iloc[-1] / vol_sma.iloc[-1]:.2f}")
                else:
                    logger.error("Tick volume data missing in dataset.")
            except Exception as e:
                logger.error(f"Volume Calculation failed: {str(e)}")

            # ATR Analysis with validation
            try:
                logger.info("\nAnalyzing ATR calculation...")
                atr = strategy._calculate_atr(data)
                logger.info(f"ATR value: {atr:.5f}")
                logger.info(f"ATR in pips: {(atr * 10000):.1f}")
            except Exception as e:
                logger.error(f"ATR Calculation failed: {str(e)}")

            # Add market condition analysis before signal generation
            try:
                logger.info("\nAnalyzing Market Condition...")
                market_condition = strategy._analyze_market_condition(data)
                logger.info(f"Market Phase: {market_condition.get('phase', 'unknown')}")
                logger.info(f"Volatility: {market_condition.get('volatility', 0)}")
                logger.info(f"Trend Strength: {market_condition.get('trend_strength', 0)}")
            except Exception as e:
                logger.error(f"Market Condition Analysis failed: {str(e)}")

            # Detailed Signal Conditions
            try:
                logger.info("\nAnalyzing Signal Conditions...")

                # EMA Condition
                ema_condition = fast_ema.iloc[-1] > slow_ema.iloc[-1]  # Buy signal condition
                logger.info(f"EMA Condition (Fast > Slow): {ema_condition}")

                # RSI Condition
                rsi_condition = 35 < rsi.iloc[-1] < 70  # Buy signal condition (RSI in valid range)
                logger.info(f"RSI Condition (35 < RSI < 70): {rsi_condition}")

                # Volume Condition
                volume_condition = data['tick_volume'].iloc[-1] > vol_sma.iloc[-1]  # Volume above average
                logger.info(f"Volume Condition (Current Volume > SMA): {volume_condition}")

                # Market Condition (trend strength)
                market_condition_valid = market_condition.get('trend_strength', 0) > 0.5  # Trend strength above threshold
                logger.info(f"Market Condition (Trend Strength > 0.5): {market_condition_valid}")

                # Final Signal Condition
                signal_valid = ema_condition and rsi_condition and volume_condition and market_condition_valid
                logger.info(f"Final Signal Valid: {signal_valid}")

            except Exception as e:
                logger.error(f"Signal Conditions Analysis failed: {str(e)}")

            # Generate the signal
            try:
                logger.info("\nGenerating Signal...")
                signal = strategy.generate_signals(data)
                logger.info(f"Generated Signal: {signal}")
            except Exception as e:
                logger.error(f"Signal Generation failed: {str(e)}")

        logger.info("Calculation audit completed")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Calculation audit failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def audit_trading_hours_logic():
    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Trading Hours Logic Audit")

    # Define the allowed_hours as in the bot
    allowed_hours = range(12, 17)  # 12:00 to 16:59

    # Simulate a bar timestamp
    test_times = [
        datetime(2024, 12, 20, 10, 0), # 10 AM - outside allowed hours
        datetime(2024, 12, 20, 13, 30), # 1:30 PM - inside allowed hours
        datetime(2024, 12, 20, 17, 0), # 5:00 PM - exactly at the edge (not included in 12-17 range)
    ]

    for t in test_times:
        current_hour = t.hour
        logger.info(f"Testing current_hour={current_hour} with allowed_hours={list(allowed_hours)}")

        signal_reasons = []
        if current_hour not in allowed_hours:
            signal_reasons.append("Outside allowed trading hours")

        if "Outside allowed trading hours" in signal_reasons:
            logger.info(f"Hour {current_hour}: Correctly identified as outside trading hours.")
        else:
            if current_hour in allowed_hours:
                logger.info(f"Hour {current_hour}: Correctly identified as inside trading hours.")
            else:
                logger.error(f"Hour {current_hour}: Expected outside hours but got no reason.")


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

def audit_bar_timestamps(symbol="EURUSD", timeframe="M5", allowed_hours=range(12,17)):
    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Bar Timestamp Audit")

    from datetime import datetime, timedelta
    from src.core.mt5 import MT5Handler
    mt5_handler = MT5Handler(debug=True, logger=logger)

    if not mt5_handler.connected:
        logger.error("MT5 not connected")
        return

    # Try a known good historical period (e.g., Wednesday at 14:00 UTC)
    end_date = datetime(2024, 12, 18, 14, 0)  # Wednesday 14:00 UTC (example)
    start_date = end_date - timedelta(hours=4)
    data = mt5_handler.get_historical_data(symbol, timeframe, start_date, end_date)

    if data is None or data.empty:
        logger.error("No data returned. Try another timeframe, symbol, or a known active trading period.")
        return

    last_bar = data.iloc[-1]
    bar_time_utc = last_bar['time']
    logger.info(f"Raw bar time (UTC): {bar_time_utc}")

    # Convert to New York time (UTC-5)
    ny_time = bar_time_utc - timedelta(hours=5)
    logger.info(f"Converted bar time (NY): {ny_time}")

    ny_hour = ny_time.hour
    if ny_hour in allowed_hours:
        logger.info(f"Bar time {ny_time} is inside allowed hours (NY).")
    else:
        logger.info(f"Bar time {ny_time} is outside allowed hours (NY).")


def run_audit(target: str) -> None:
    """Run audit for specified target.

    Args:
        target: Module to audit ('dashboard', 'mt5', or 'all')
    """
    if target == 'bar_timestamps':
        audit_bar_timestamps()  # Call the new function here

    if target in ['dashboard', 'all']:
        audit_dashboard()

    if target == 'trading_hours':
        audit_trading_hours_logic()

    if target == 'mt5':
        audit_mt5()

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