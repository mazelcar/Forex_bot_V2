import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import redirect_stdout

from src.strategy.validation.data_validator import DataValidator
from src.core.mt5 import MT5Handler
from src.strategy.manager import StrategyManager
from src.strategy.analysis.performance_analyzer import PerformanceAnalyzer

def identify_support_resistance_zones(higher_tf_data: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify support and resistance levels from higher timeframe data (H1).
    Steps:
    - Find swing highs/lows: a swing high is where 'high' > highs of bars around it.
    - Similarly for swing lows.
    - Keep only those levels touched multiple times.
    - Optional: confirm support levels are near/above a short MA for bullish alignment.

    This is a simplified example. Adjust as needed.
    """

    # Ensure data sorted by time
    higher_tf_data = higher_tf_data.sort_values('time').reset_index(drop=True)

    # Compute a short MA for trend confirmation (e.g., 20-period EMA)
    higher_tf_data['ema_short'] = higher_tf_data['close'].ewm(span=20, adjust=False).mean()

    swing_highs = []
    swing_lows = []
    lookback = 2  # how many bars before/after to confirm a swing

    # Identify swings
    for i in range(lookback, len(higher_tf_data)-lookback):
        h = higher_tf_data.loc[i, 'high']
        l = higher_tf_data.loc[i, 'low']
        # Check if it's a swing high
        if all(h > higher_tf_data.loc[i-j, 'high'] for j in range(1, lookback+1)) and \
           all(h > higher_tf_data.loc[i+j, 'high'] for j in range(1, lookback+1)):
            swing_highs.append(h)
        # Check if it's a swing low
        if all(l < higher_tf_data.loc[i-j, 'low'] for j in range(1, lookback+1)) and \
           all(l < higher_tf_data.loc[i+j, 'low'] for j in range(1, lookback+1)):
            swing_lows.append(l)

    # Filter levels by requiring multiple touches (simplified: keep if occurs at least twice)
    # In reality, you'd cluster these swings into zones and require multiple near touches.
    # For now, just pick the median of swing highs/lows as representative levels.
    if len(swing_lows) > 2:
        support_levels = [np.median(swing_lows)]
    else:
        support_levels = [1.05]  # fallback dummy

    if len(swing_highs) > 2:
        resistance_levels = [np.median(swing_highs)]
    else:
        resistance_levels = [1.06]  # fallback dummy

    # Filter support levels to only those near or above EMA to align with some trend notion:
    support_levels = [lvl for lvl in support_levels if lvl < higher_tf_data['ema_short'].mean()]

    # If no supports found, fallback
    if not support_levels:
        support_levels = [1.045]

    # Similarly, we could filter resistance levels, but let's skip for now.
    if not resistance_levels:
        resistance_levels = [1.065]

    return support_levels, resistance_levels

def is_bounce_candle(row: pd.Series, level: float, direction: str = 'support') -> bool:
    """
    Check if there's a bullish bounce at support or a bearish bounce at resistance.
    """
    tolerance = 0.0005  # Example tolerance
    if direction == 'support':
        near_support = abs(row['low'] - level) <= tolerance
        bullish_candle = row['close'] > row['open']
        return near_support and bullish_candle
    else:
        near_resistance = abs(row['high'] - level) <= tolerance
        bearish_candle = row['close'] < row['open']
        return near_resistance and bearish_candle

class BacktestRunner:
    """Handles backtest execution with MT5 data loading and strategy integration."""
    def __init__(self, validator: DataValidator, mt5_handler: MT5Handler, manager: StrategyManager):
        self.validator = validator
        self.mt5_handler = mt5_handler
        self.manager = manager
        self.data: Optional[pd.DataFrame] = None
        self.results = {}
        self.signals = []
        self.trades = []
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

        validation_results = self.validator.validate_basic_data(self.data)
        if not validation_results['passed']:
            print("\nData validation failed. Details:")
            for msg in validation_results['messages']:
                print(f"- {msg}")
            return False

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

        volume_results = self.validator.validate_volume_quality(self.data)
        if not volume_results['passed']:
            print("\nVolume quality validation failed:")
            for msg in volume_results['messages']:
                print(f"- {msg}")
            return False

        holidays = self.validator.config.get('holidays', {})
        weekend_holiday_results = self.validator.validate_weekend_holiday_handling(self.data, holidays)
        if not weekend_holiday_results['passed']:
            print("\nWeekend/Holiday validation failed:")
            for msg in weekend_holiday_results['messages']:
                print(f"- {msg}")
            return False

        from src.strategy.validation.indicator_validator import IndicatorValidator
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

        return True

    def _run_simulation(self, strategy_name: str, account_info: dict, output_file: str = "backtest_output.txt") -> None:
        if self.data is None or self.data.empty:
            print("No data to simulate.")
            return

        # Load higher timeframe data for better S/R (use H1 for example)
        # In reality, you'd fetch H1 data similarly:
        h1_data = self.mt5_handler.get_historical_data(
            symbol=self.symbol,
            timeframe='H1',
            start_date=self.start_date - timedelta(days=5),  # extra data for S/R calc
            end_date=self.end_date,
            include_volume=False
        )

        if h1_data is None or len(h1_data) == 0:
            print("Failed to load H1 data for S/R calculation. Falling back to simple levels.")
            support_levels = [1.05]
            resistance_levels = [1.06]
        else:
            support_levels, resistance_levels = identify_support_resistance_zones(h1_data)

        open_trade = None

        with open(output_file, 'w') as f, redirect_stdout(f):
            for i in range(len(self.data)):
                row = self.data.iloc[i]

                # Filter out trades outside the most liquid hours (e.g., 12:00 to 16:59 UTC)
                # If current hour not in range(12,17), skip opening new trades
                current_hour = row['time'].hour
                allowed_hours = range(12,17)

                # Manage open trades first
                if open_trade is not None:
                    hit_stop = (open_trade['type'] == 'BUY' and row['low'] <= open_trade['sl'])
                    hit_take_profit = (open_trade['type'] == 'BUY' and row['high'] >= open_trade['tp'])

                    if hit_stop or hit_take_profit:
                        exit_price = open_trade['sl'] if hit_stop else open_trade['tp']
                        profit = (exit_price - open_trade['entry_price']) / row['pip_value'] * open_trade['position_size'] * 10.0
                        trade = {
                            'entry_time': open_trade['entry_time'],
                            'entry_price': open_trade['entry_price'],
                            'exit_time': row['time'],
                            'exit_price': exit_price,
                            'profit': profit
                        }
                        self.trades.append(trade)
                        open_trade = None

                # Consider new trades if no open trade and time is allowed
                if open_trade is None and i < len(self.data)-1 and current_hour in allowed_hours:
                    # Check for a bullish bounce at support with adjusted SL/TP
                    for s_level in support_levels:
                        if is_bounce_candle(row, s_level, direction='support'):
                            # Adjusted stop and target for better R:R
                            sl = s_level - 0.0004  # tighter stop
                            tp = row['close'] + 0.0012  # slightly bigger TP
                            open_trade = {
                                'type': 'BUY',
                                'entry_time': row['time'],
                                'entry_price': row['close'],
                                'sl': sl,
                                'tp': tp,
                                'position_size': 0.1
                            }
                            break

                    # If desired, add SELL logic at resistance as well with similar logic:
                    # for r_level in resistance_levels:
                    #     if is_bounce_candle(row, r_level, direction='resistance'):
                    #         sl = r_level + 0.0004
                    #         tp = row['close'] - 0.0012
                    #         open_trade = {
                    #             'type': 'SELL',
                    #             'entry_time': row['time'],
                    #             'entry_price': row['close'],
                    #             'sl': sl,
                    #             'tp': tp,
                    #             'position_size': 0.1
                    #         }
                    #         break

            print("Simulation complete. Signals generated:")
            for s in self.signals:
                print(s)

            print(f"\nTotal Trades Executed: {len(self.trades)}")

            if self.trades:
                analyzer = PerformanceAnalyzer(self.trades)
                metrics = analyzer.compute_metrics()

                final_equity = 10000 + sum(t['profit'] for t in self.trades)
                net_profit = final_equity - 10000

                print("\n=== Backtest Results ===")
                print(f"Initial Balance: 10,000")
                print(f"Final Net Profit: {net_profit:.2f}")
                print(f"Win Rate: {metrics['win_rate']:.2f}%")
                print(f"Average Win: {metrics['avg_win']:.2f}")
                print(f"Average Loss: {metrics['avg_loss']:.2f}")
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.2f}\n")

                print("Performance Summary:")
                print(f"Win Rate: {metrics['win_rate']:.2f}%")
                print(f"Average Win: {metrics['avg_win']:.2f}")
                print(f"Average Loss: {metrics['avg_loss']:.2f}")
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.2f}")
            else:
                print("No trades were executed during the backtest.")

    def _process_signal(self, signal: Dict[str, Any]) -> None:
        if signal.get("type") != "NONE":
            self.signals.append(signal)


if __name__ == "__main__":
    from src.strategy.strategies.strategy_template import StrategyTemplate
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
    # Test for 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

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
            account_info = {
                "balance": 10000
            }

            runner._run_simulation("basic", account_info, output_file="backtest_output.txt")

            analyzer = PerformanceAnalyzer(runner.trades)
            metrics = analyzer.compute_metrics()

            print("\nPerformance Summary:")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Average Win: {metrics['avg_win']:.2f}")
            print(f"Average Loss: {metrics['avg_loss']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}")
