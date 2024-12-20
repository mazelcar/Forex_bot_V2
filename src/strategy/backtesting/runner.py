# src/strategy/backtesting/runner.py

from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime
from src.strategy.validation.data_validator import DataValidator
from src.core.mt5 import MT5Handler
from src.strategy.manager import StrategyManager

class BacktestRunner:
    """Handles backtest execution with MT5 data loading and strategy integration."""

    def __init__(self, validator: DataValidator, mt5_handler: MT5Handler, manager: StrategyManager):
        self.validator = validator
        self.mt5_handler = mt5_handler
        self.manager = manager
        self.data: Optional[pd.DataFrame] = None
        self.results = {}
        self.signals = []
        self.trades = []  # Store executed trades
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.symbol: Optional[str] = None
        self.timeframe: Optional[str] = None

    def load_data(self, symbol: str, timeframe: str, start_date: datetime,
                  end_date: datetime) -> bool:
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date

        try:
            mt5_data = self.mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_volume=True
            )

            if mt5_data is None:
                print("Failed to get data from MT5")
                return False

            if len(mt5_data) == 0:
                print("No data returned from MT5")
                return False

            self.data = mt5_data
            if 'symbol' not in self.data.columns:
                self.data['symbol'] = symbol

            print(f"Loaded {len(self.data)} bars of {symbol} {timeframe} data")
            return True

        except Exception as e:
            print(f"Error loading data from MT5: {str(e)}")
            return False

    def can_proceed(self) -> bool:
        if self.data is None:
            print("No data loaded")
            return False

        # Basic data validation
        validation_results = self.validator.validate_basic_data(self.data)
        if not validation_results['passed']:
            print("\nData validation failed. Details:")
            for msg in validation_results['messages']:
                print(f"- {msg}")
            return False

        # Test period coverage checks
        period_results = self.validator.validate_test_period_coverage(
            self.data,
            self.start_date,
            self.end_date
        )
        if not period_results['passed']:
            messages = period_results.get('messages', [])
            later_error = [m for m in messages if "Data starts later than requested period." in m]
            if later_error:
                for m in later_error:
                    if "Actual start:" in m:
                        part = m.split("Actual start: ")[1].strip()
                        from datetime import datetime
                        try:
                            actual_start = datetime.fromisoformat(part)
                            print(f"\nAdjusting requested start_date to {actual_start} due to coverage issue.")
                            self.start_date = actual_start
                            period_results = self.validator.validate_test_period_coverage(
                                self.data,
                                self.start_date,
                                self.end_date
                            )
                            if not period_results['passed']:
                                print("\nTest period coverage validation failed:")
                                for msg in period_results['messages']:
                                    print(f"- {msg}")
                                return False
                            else:
                                break
                        except ValueError:
                            print("\nFailed to parse actual start from message. Cannot fix coverage issue.")
                            return False
            else:
                print("\nTest period coverage validation failed:")
                for msg in period_results['messages']:
                    print(f"- {msg}")
                return False

        # Volume data quality checks
        volume_results = self.validator.validate_volume_quality(self.data)
        if not volume_results['passed']:
            print("\nVolume quality validation failed:")
            for msg in volume_results['messages']:
                print(f"- {msg}")
            return False

        # Weekend/Holiday checks
        holidays = self.validator.config.get('holidays', {})
        weekend_holiday_results = self.validator.validate_weekend_holiday_handling(self.data, holidays)
        if not weekend_holiday_results['passed']:
            print("\nWeekend/Holiday validation failed:")
            for msg in weekend_holiday_results['messages']:
                print(f"- {msg}")
            return False

        # Indicator warmup validation (Day 7)
        from src.strategy.validation.indicator_validator import IndicatorValidator
        # Assuming same config dict is used for indicator settings
        indicator_validator = IndicatorValidator(self.validator.config)
        warmup_results = indicator_validator.validate_indicator_warmup(self.data)
        if not warmup_results['passed']:
            print("\nIndicator warmup validation failed:")
            for msg in warmup_results['messages']:
                print(f"- {msg}")
            return False

        calc_results = indicator_validator.calculate_and_verify_indicators(self.data)
        if not calc_results['passed']:
            print("\nIndicator calculation verification failed:")
            for msg in calc_results['messages']:
                print(f"- {msg}")
            return False

        no_lookahead_results = indicator_validator.validate_no_lookahead(self.data)
        if not no_lookahead_results['passed']:
            print("\nNo look-ahead bias validation failed:")
            for msg in no_lookahead_results['messages']:
                print(f"- {msg}")
            return False

        # If all validations pass
        return True

    def _run_simulation(self, strategy_name: str, account_info: Dict[str, Any]) -> None:
        if self.data is None or self.data.empty:
            print("No data to simulate.")
            return

        # Assume position_size and pip_value are from data or config
        position_size = 0.1
        pip_value = self.data['pip_value'].iloc[0] if 'pip_value' in self.data.columns else 0.0001

        # We'll store opened trades as we go
        # For simplicity, open and close trades right after receiving signals
        # In real scenario, you'd have logic to manage open trades, stops, etc.

        for i in range(len(self.data)-1):
            row = self.data.iloc[i]
            next_row = self.data.iloc[i+1]

            signal = self.manager.execute_strategy(strategy_name, row, account_info)
            self._process_signal(signal)

            if signal.get("type") == "BUY":
                entry_time = row['time']
                entry_price = row['close']
                exit_time = next_row['time']
                exit_price = next_row['close']

                # Profit calculation for demonstration:
                # Profit in pips = (exit_price - entry_price) / pip_value
                # Actual profit = pips * position_size * (some money per pip)
                # We simplify: profit in currency = (exit_price - entry_price) / pip_value * position_size * 10 just as example
                # Adjust to real formula if needed.
                pips = (exit_price - entry_price) / pip_value
                profit = pips * position_size * 10.0  # arbitrary scaling for demonstration

                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit': profit
                }
                self.trades.append(trade)

        print("Simulation complete. Signals generated:")
        for s in self.signals:
            print(s)

    def _process_signal(self, signal: Dict[str, Any]) -> None:
        """Process the strategy's signal. For now, just store it."""
        if signal.get("type") != "NONE":
            self.signals.append(signal)


# Example test code (not strictly required if just building up):
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from src.strategy.strategies.strategy_template import StrategyTemplate

    # Setup components
    config = {}
    validator = DataValidator(config)
    mt5_handler = MT5Handler(debug=True)
    manager = StrategyManager()

    # Register a simple strategy
    basic_strategy = StrategyTemplate(config={})
    manager.register_strategy("basic", basic_strategy)

    runner = BacktestRunner(validator, mt5_handler, manager)

    symbol = "EURUSD"
    timeframe = "M5"
    start_date = datetime(2024, 12, 18, 14, 35)
    end_date = datetime.now()

    print(f"\nTesting with {symbol} {timeframe}")
    print(f"Period: {start_date} to {end_date}")

    print("\nLoading data from MT5...")
    load_success = runner.load_data(symbol, timeframe, start_date, end_date)
    print(f"Data loading {'successful' if load_success else 'failed'}")

    if load_success:
        print("\nFirst few bars:")
        print(runner.data.head())

        print("\nChecking if backtest can proceed...")
        can_proceed = runner.can_proceed()
        print(f"Can proceed with backtest: {can_proceed}")

        if can_proceed:
            print("\nData validation passed. Ready for backtesting!")
            account_info = {"balance": 10000}
            runner._run_simulation("basic", account_info)  # Use runner instead of self here

            # After simulation completes, we have runner.trades with the trades
            from src.strategy.analysis.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(runner.trades)  # Use runner.trades instead of self.trades
            metrics = analyzer.compute_metrics()

            print("\nPerformance Summary:")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Average Win: {metrics['avg_win']:.2f}")
            print(f"Average Loss: {metrics['avg_loss']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}")
