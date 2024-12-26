import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

import pytz

# Import your local modules
from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.data_validator import DataValidator
from src.strategy.report_writer import ReportWriter
from src.strategy.report_writer import analyze_trades







# -------------------------------------------------------
# Setup logging (Avoid multiple handlers)
# -------------------------------------------------------
def get_logger(name="runner", logfile="runner_debug.log"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = get_logger(name="runner", logfile="runner_debug.log")





def load_market_news(news_file="config/market_news.json") -> List[Dict]:
    """Attempt to load market news JSON. If unavailable, return an empty list."""
    import json
    if not os.path.exists(news_file):
        logger.warning(f"Market news file not found: {news_file}. Proceeding without it.")
        return []

    try:
        with open(news_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading market_news.json: {str(e)}")
        return []


def load_data(symbol="EURUSD", timeframe="H1", days=None, start_date=None, end_date=None, max_retries=3) -> pd.DataFrame:
    mt5 = MT5Handler(debug=True)

    print(f"Attempting to load {symbol} {timeframe} data...")  # Debug print

    if days and not start_date and not end_date:
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days)
        print(f"Date range: {start_date} to {end_date}")  # Debug print

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")  # Debug print
            df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)

            if df is None:
                print(f"MT5 returned None on attempt {attempt + 1}")
                continue

            if df.empty:
                print(f"MT5 returned empty DataFrame on attempt {attempt + 1}")
                continue

            print(f"Retrieved {len(df)} bars")  # Debug print
            return df

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if start_date:
                start_date -= timedelta(days=5)
                print(f"Retrying with new start date: {start_date}")  # Debug print

    print("Failed to load data after all attempts")  # Debug print
    return pd.DataFrame()


def validate_data_for_backtest(df: pd.DataFrame, timeframe: str = "M15") -> bool:
    """
    Enhanced data validation checking for data quality issues.
    """
    if df.empty:
        print("ERROR: No data loaded.")
        return False

    # Basic datetime checks
    current_time = datetime.now(pytz.UTC)
    df_time_max = pd.to_datetime(df['time'].max())
    if not df_time_max.tzinfo:
        df_time_max = pytz.UTC.localize(df_time_max)

    if df_time_max > current_time:
        print(f"ERROR: Data contains future dates! Max date: {df_time_max}")
        return False

    # Timezone consistency
    df['time'] = pd.to_datetime(df['time'])
    if not df['time'].dt.tz:
        df['time'] = df['time'].dt.tz_localize(pytz.UTC)

    # Column validation
    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False

    # Calculate dynamic spread thresholds
    df['spread'] = df['high'] - df['low']
    median_spread = df['spread'].median()
    spread_std = df['spread'].std()

    # Flag spreads that are more than 5 standard deviations from median
    df['is_extreme_spread'] = df['spread'] > (median_spread + 5 * spread_std)

    # Only invalid if not extreme spread and OHLC invalid
    invalid_prices = df[
        ~df['is_extreme_spread'] &  # Not an extreme spread event
        ((df['high'] < df['low']) |
         (df['close'] > df['high']) |
         (df['close'] < df['low']) |
         (df['open'] > df['high']) |
         (df['open'] < df['low']))
    ]

    if not invalid_prices.empty:
        print("\nTruly invalid price data:")
        print(invalid_prices[['time', 'open', 'high', 'low', 'close', 'spread']].head())
        return False

    # Zero price check
    if (df[['open', 'high', 'low', 'close']] == 0).any().any():
        print("ERROR: Found zero prices in data")
        return False

    # Date analysis
    date_min = df["time"].min()
    date_max = df["time"].max()
    date_range = date_max - date_min
    total_days = date_range.days

    print(f"\nData Range Analysis:")
    print(f"Start: {date_min}")
    print(f"End: {date_max}")
    print(f"Total days: {total_days}")

    # Bar calculation
    bars_per_day = {
        "M15": 4 * 24,
        "M5": 12 * 24,
        "H1": 24
    }.get(timeframe, 24)

    expected_bars = total_days * bars_per_day * (5/7)  # Excluding weekends
    actual_bars = len(df)
    completeness = (actual_bars / expected_bars) * 100

    print(f"\nBar Count Analysis:")
    print(f"Expected bars (excluding weekends): {expected_bars:.0f}")
    print(f"Actual bars: {actual_bars}")
    print(f"Data completeness: {completeness:.1f}%")

    # Gap analysis
    df_sorted = df.sort_values("time").reset_index(drop=True)
    time_diffs = df_sorted["time"].diff()

    expected_diff = pd.Timedelta(**{
        'minutes': int(timeframe[1:]) if timeframe.startswith('M') else 15,
        'hours': int(timeframe[1:]) if timeframe.startswith('H') else 0
    })

    weekend_gaps = time_diffs[
        (df_sorted['time'].dt.dayofweek == 0) &
        (time_diffs > pd.Timedelta(days=1))
    ]

    unexpected_gaps = time_diffs[
        (time_diffs > expected_diff * 1.5) &
        ~((df_sorted['time'].dt.dayofweek == 0) &
          (time_diffs > pd.Timedelta(days=1)))
    ]

    print(f"\nGap Analysis:")
    print(f"Weekend gaps detected: {len(weekend_gaps)}")
    print(f"Unexpected gaps: {len(unexpected_gaps)}")

    if len(unexpected_gaps) > 0:
        print("\nLargest unexpected gaps:")
        largest_gaps = unexpected_gaps.nlargest(3)
        for idx in largest_gaps.index:
            if idx > 0:
                gap_start = df_sorted.loc[idx-1, 'time']
                print(f"Gap of {time_diffs[idx]} at {gap_start}")

    # Quality checks
    if completeness < 90:
        print("ERROR: Data completeness below 90%")
        return False

    if len(unexpected_gaps) > total_days * 0.1:
        print("ERROR: Too many unexpected gaps in data")
        return False

    print("\nValidation passed: Data quality checks complete")
    return True


def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def resample_m15_to_h1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    Resample from M15 to H1 bars, removing NaNs.
    """
    df_m15["time"] = pd.to_datetime(df_m15["time"])
    df_m15.set_index("time", inplace=True)
    df_m15.sort_index(inplace=True)

    df_h1_resampled = df_m15.resample("1h").agg({  # Changed from "1H" to "1h"
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tick_volume": "sum",
})
    df_h1_resampled.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df_h1_resampled.reset_index(inplace=True)
    return df_h1_resampled


def check_h1_data_or_resample(
    symbol: str,
    h1_start: datetime,
    h1_end: datetime,
    threshold=0.9
) -> pd.DataFrame:
    """
    Fetch H1 data. If completeness < threshold,
    fallback to M15 => resample => H1.
    """
    df_h1 = load_data(
        symbol=symbol,
        timeframe="H1",
        start_date=h1_start,
        end_date=h1_end
    )
    if df_h1.empty:
        logger.warning("No H1 data returned. Will try fallback to M15 resampling.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    # Validate to measure completeness
    try:
        validate_data_for_backtest(df_h1, timeframe="H1")
    except ValueError as e:
        logger.warning(f"Validation error on H1: {str(e)}. Fallback to M15.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    # Calculate completeness ratio
    df_h1_min = pd.to_datetime(df_h1["time"].min())
    df_h1_max = pd.to_datetime(df_h1["time"].max())
    day_span = (df_h1_max - df_h1_min).days
    expected_bars = day_span * 24 * (5/7)
    actual_bars = len(df_h1)
    completeness = actual_bars / (expected_bars if expected_bars > 0 else 1e-9)

    if completeness < threshold:
        logger.warning(
            f"H1 data completeness {completeness:.1%} < {threshold:.1%}. "
            "Falling back to M15 resampling."
        )
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    return df_h1


def fallback_resample_from_m15(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    logger.info("Attempting fallback: fetch M15 data and resample to H1.")
    df_m15 = load_data(symbol=symbol, timeframe="M15", start_date=start, end_date=end)
    if df_m15.empty:
        logger.error("M15 fallback data also empty. Returning empty.")
        return pd.DataFrame()  # still empty
    # Resample
    df_h1_resampled = resample_m15_to_h1(df_m15)
    try:
        validate_data_for_backtest(df_h1_resampled, timeframe="H1")
    except:
        logger.warning("Resampled H1 data is also incomplete or invalid.")
    return df_h1_resampled


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    """
    Simple backtest engine that:
    - Iterates over the DF
    - Opens trades if signal is triggered
    - Exits trades if conditions are met
    - Returns final trades & balance
    """
    if df.empty:
        logger.warning("DataFrame empty in run_backtest. Returning no trades.")
        return {"Trades": [], "final_balance": initial_balance}

    trades: list["SR_Bounce_Strategy.Trade"] = []
    balance = initial_balance
    active_trade: Optional["SR_Bounce_Strategy.Trade"] = None

    logger.debug("Starting backtest run...")

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]

        if len(current_segment) < 5:
            continue

        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                trades.append(active_trade)
                logger.debug(f"New trade opened: {new_trade.type} at {new_trade.entry_price}")
        else:
            should_close, fill_price, pnl = strategy.exit_trade(current_segment, active_trade)
            if should_close:
                balance += pnl
                last_bar = current_segment.iloc[-1]
                active_trade.close_i = last_bar.name
                active_trade.close_time = last_bar["time"]
                active_trade.close_price = fill_price
                active_trade.pnl = pnl
                logger.debug(f"Trade closed: PnL={pnl:.2f}")
                active_trade = None

    # If a trade is still open at the end, force close
    if active_trade:
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])
        if active_trade.type == "BUY":
            pnl = (last_close - active_trade.entry_price) * 10000.0 * active_trade.size
        else:  # SELL
            pnl = (active_trade.entry_price - last_close) * 10000.0 * active_trade.size

        balance += pnl
        active_trade.close_i = last_bar.name
        active_trade.close_time = last_bar["time"]
        active_trade.close_price = last_close
        active_trade.pnl = pnl
        logger.debug(f"Final trade forcibly closed: PnL={pnl:.2f}")
        active_trade = None

    return {
        "Trades": [t.to_dict() for t in trades],
        "final_balance": balance
    }


def main():
    print("\nStarting backtest with enhanced data validation...")

    symbol = "EURUSD"
    timeframe = "M15"
    days = 180

    print(f"\nLoading {timeframe} data for {symbol}...")
    df = load_data(symbol, timeframe, days=days)
    if df.empty:
        print("ERROR: No valid data loaded. Exiting main().")
        return

    # If data passes validation, continue with backtest
    train_df, test_df = split_data_for_backtest(df, 0.8)
    print(f"Train/Test split: {len(train_df)} / {len(test_df)} bars")

    # Prepare H1 data for S/R
    test_start = pd.to_datetime(test_df['time'].min())
    test_end = pd.to_datetime(test_df['time'].max())

    # e.g., 45 days prior to test start
    h1_start = test_start - timedelta(days=45)

    print(f"\nFetching H1 data from {h1_start} to {test_end} ...")
    df_h1 = check_h1_data_or_resample(symbol, h1_start, test_end, threshold=0.90)
    if df_h1.empty:
        print("Failed to load or resample H1 data. Exiting.")
        return
    validate_data_for_backtest(df_h1, "H1")

    # Instantiate strategy with default or config
    strategy = SR_Bounce_Strategy(config_file=None)  # or specify a config_file

    # Use 2 weeks of data for weekly S/R
    strategy.update_weekly_levels(df_h1, weeks=2, weekly_buffer=0.00075)

    # Example: check training bounces (optional)
    print("\n--- Checking for bounces in the TRAINING SET ---")
    bounce_count = 0
    for i in range(len(train_df)):
        current_segment = train_df.iloc[: i + 1]
        sig = strategy.generate_signals(current_segment)
        if sig["type"] != "NONE":
            print(f"Bounce found at index={i}, time={current_segment.iloc[-1]['time']}, signal={sig}")
            bounce_count += 1
    print(f"Total bounces detected in training set: {bounce_count}")

    # Run backtest on test_df
    backtest_result = run_backtest(strategy, test_df, initial_balance=10000.0)
    trades = backtest_result["Trades"]
    final_balance = backtest_result["final_balance"]

    stats = analyze_trades(trades, 10000.0)

    print("\n--- BACKTEST COMPLETE (Console Summary) ---")
    print(f"Total Trades: {stats['count']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")

    # Write report
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"BACKTEST_REPORT_{symbol}_{now_str}.md")

    with ReportWriter(report_file) as writer:
        writer.write_data_overview(test_df)
        writer.write_trades_section(trades)
        writer.write_stats_section(stats, final_balance)
        writer.write_monthly_breakdown({})
        writer.write_sr_levels([], [])

    print(f"\nDetailed report written to: {report_file}")
    # Signal Generation Stats printing
    print("\n--- Signal Generation Stats ---")
    print(f"Potential signals filtered due to volume: {strategy.signal_generator.signal_stats['volume_filtered']}")
    print(f"First bounces recorded: {strategy.signal_generator.signal_stats['first_bounce_recorded']}")
    print(f"Second bounces filtered (low volume): {strategy.signal_generator.signal_stats['second_bounce_low_volume']}")
    print(f"Signals that missed by tolerance: {strategy.signal_generator.signal_stats['tolerance_misses']}")
    print(f"Signals generated: {strategy.signal_generator.signal_stats['signals_generated']}")

    # Data Quality Stats
    validator = DataValidator()
    df, quality_metrics = validator.validate_and_clean_data(df, timeframe)
    if quality_metrics.get('true_gaps_detected', 0) > 0:
        print(f"WARNING: Found {quality_metrics['true_gaps_detected']} unexpected gaps during trading hours")
    print(f"Weekend gaps: {quality_metrics.get('weekend_gaps', 0)}")
    print(f"Session gaps: {quality_metrics.get('session_gaps', 0)}")


if __name__ == "__main__":
    main()
