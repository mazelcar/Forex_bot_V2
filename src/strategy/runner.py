import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.report_writer import ReportWriter
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.report_writer import analyze_trades

logging.basicConfig(
    filename="runner_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.debug("Entering load data")

def load_data(symbol="EURUSD", timeframe="H1", days=None, start_date=None, end_date=None) -> pd.DataFrame:
    mt5 = MT5Handler(debug=True)
    if days and not start_date and not end_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

    # Add retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)

            # Validate data completeness
            if df is not None and not df.empty:
                time_diffs = df['time'].diff()
                expected_diff = pd.Timedelta('1 hour') if timeframe == "H1" else pd.Timedelta('15 minutes')
                gaps = time_diffs[time_diffs > expected_diff * 2]

                completeness = 1 - (len(gaps) / len(df))
                if completeness >= 0.95:
                    return df.sort_values("time").reset_index(drop=True)

            # If data is incomplete, adjust time range and retry
            if start_date:
                start_date -= pd.Timedelta(days=5)
            logging.warning(f"Retry {attempt + 1}: Adjusting time range for better data completeness")

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")

    raise ValueError("Failed to load complete dataset after maximum retries")


def validate_data_for_backtest(df: pd.DataFrame, timeframe: str = "M15") -> None:
    if df.empty:
        raise ValueError("No data loaded.")

    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data not valid: missing columns {missing_cols}")

    date_range = df["time"].max() - df["time"].min()
    print(f"\nData Range Analysis:")
    print(f"Start: {df['time'].min()}")
    print(f"End: {df['time'].max()}")
    print(f"Total days: {date_range.days}")

    # ---------------------------------------------------------
    # Dynamically compute expected_bars based on timeframe
    # ---------------------------------------------------------
    if timeframe == "M15":
        bars_per_day = 24 * 4  # 4 bars per hour
    elif timeframe == "M5":
        bars_per_day = 24 * 12
    elif timeframe == "H1":
        bars_per_day = 24
    else:
        # Fallback or extended logic for other TFs
        bars_per_day = 24 * 4  # old default

    expected_bars = date_range.days * bars_per_day * (5/7)  # ignoring weekends
    actual_bars = len(df)
    completeness = (actual_bars / expected_bars) * 100

    print(f"\nBar Count Analysis:")
    print(f"Expected bars (excluding weekends): {expected_bars:.0f}")
    print(f"Actual bars: {actual_bars}")
    print(f"Data completeness: {completeness:.1f}%")

    # Gap detection remains the same:
    time_diffs = df["time"].diff()
    if timeframe.startswith("M"):  # e.g. M15, M5
        max_allowed_gap = pd.Timedelta(minutes=1.1 * int(timeframe[1:]))
    elif timeframe.startswith("H"):
        max_allowed_gap = pd.Timedelta(hours=1.1)  # allow ~1.1h gap
    else:
        max_allowed_gap = pd.Timedelta(minutes=16)  # old default

    gaps = time_diffs[time_diffs > max_allowed_gap]
    if not gaps.empty:
        print(f"\nFound {len(gaps)} gaps in data larger than 1 bar")
        print("Largest gaps:")
        for idx in gaps.nlargest(3).index:
            print(f"Gap of {time_diffs[idx]} at {df['time'][idx]}")

    print("\nValidation passed: Data quality checks complete")



def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    if df.empty:
        return {"Trades": [], "final_balance": initial_balance}

    trades: list["SR_Bounce_Strategy.Trade"] = []
    balance = initial_balance
    active_trade: Optional["SR_Bounce_Strategy.Trade"] = None

    logging.debug("Starting backtest run...")

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]

        if len(current_segment) < 5:
            continue

        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                trades.append(active_trade)
                logging.debug(f"New trade opened: {new_trade.type} at {new_trade.entry_price}")
        else:
            should_close, fill_price, pnl = strategy.exit_trade(current_segment, active_trade)
            if should_close:
                balance += pnl
                last_bar = current_segment.iloc[-1]
                active_trade.close_i = last_bar.name
                active_trade.close_time = last_bar["time"]
                active_trade.close_price = fill_price
                active_trade.pnl = pnl
                logging.debug(f"Trade closed: PnL={pnl:.2f}")
                active_trade = None

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
        logging.debug(f"Final trade closed: PnL={pnl:.2f}")
        active_trade = None

    return {
        "Trades": [t.to_dict() for t in trades],
        "final_balance": balance
    }

def fetch_and_resample_m15_to_h1():
    """
    Fetch 2 years of M15 data, resample to H1 bars, and validate completeness.
    """
    # 1) Load M15 data (2 years)
    days_needed = 730  # ~2 years
    symbol = "EURUSD"
    timeframe = "M15"

    print(f"\n--- Fetching {days_needed} days of {symbol} {timeframe} data ---")
    df_m15 = load_data(symbol=symbol, timeframe=timeframe, days=days_needed)
    if df_m15.empty:
        print("No M15 data returned. Exiting fetch_and_resample_m15_to_h1.")
        return

    # First validate the M15 data
    print(f"Fetched {len(df_m15)} M15 bars. Validating original M15 dataset...")
    validate_data_for_backtest(df_m15, timeframe="M15")  # <--- Validate M15 first

    # 2) Convert 'time' column to a proper DateTime index for resampling
    df_m15["time"] = pd.to_datetime(df_m15["time"])
    df_m15.set_index("time", inplace=True)
    df_m15.sort_index(inplace=True)  # Ensure ascending time order

    # 3) Resample from M15 to H1
    print("\n--- Resampling M15 -> H1 ---")
    df_h1_resampled = df_m15.resample("1H").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",  # sum volumes over each hour
    })

    # 4) Remove rows that are all NaN (e.g., partial or missing intervals)
    df_h1_resampled.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # 5) Restore the 'time' column as normal
    df_h1_resampled.reset_index(inplace=True)

    print(f"Resampled dataset shape: {df_h1_resampled.shape}.")
    print(f"Date range: {df_h1_resampled['time'].min()} -> {df_h1_resampled['time'].max()}")

    # 6) Validate the new H1 DataFrame
    print("\n--- Validating Resampled H1 Data ---")
    validate_data_for_backtest(df_h1_resampled, timeframe="H1")  # <--- Now we can validate H1

    # (Optional) Save the resampled H1 data to CSV
    output_file = "EURUSD_H1_resampled.csv"
    df_h1_resampled.to_csv(output_file, index=False)
    print(f"\nSaved resampled H1 data to CSV: {output_file}")



def check_training_bounces(strategy: SR_Bounce_Strategy, df_train: pd.DataFrame):
    print("\n--- Checking for bounces in the TRAINING SET ---")
    bounce_count = 0
    bounce_details = []

    for i in range(len(df_train)):
        current_segment = df_train.iloc[: i + 1]
        sig = strategy.generate_signals(current_segment)
        if sig["type"] != "NONE":
            bar_time = current_segment.iloc[-1]["time"]
            print(f"Bounce found at index={i}, time={bar_time}, signal={sig}")
            bounce_count += 1
            bounce_details.append({
                'time': bar_time,
                'type': sig["type"],
                'level': sig.get('level', 0.0)
            })

    print(f"Total bounces detected in training set: {bounce_count}")
    return bounce_details


# [ADDED LINES] Function to Fetch 2 Years of Data (H1 & M15) -------------------
def fetch_long_term_data():
    """
    Fetch 2 years of H1 and M15 data for EURUSD and validate completeness.
    (Optional) Saves CSV for offline use.
    """
    # Fetch 2 years of H1 data
    print("\n--- Fetching ~2 years of EURUSD H1 data ---")
    df_h1_2years = load_data(symbol="EURUSD", timeframe="H1", days=730)
    print(f"\nFetched {len(df_h1_2years)} H1 bars over ~2 years.")
    validate_data_for_backtest(df_h1_2years)

    # Fetch 2 years of M15 data
    print("\n--- Fetching ~2 years of EURUSD M15 data ---")
    df_m15_2years = load_data(symbol="EURUSD", timeframe="M15", days=730)
    print(f"\nFetched {len(df_m15_2years)} M15 bars over ~2 years.")
    validate_data_for_backtest(df_m15_2years)

    # (Optional) Save to CSV
    df_h1_2years.to_csv("EURUSD_H1_2years.csv", index=False)
    df_m15_2years.to_csv("EURUSD_M15_2years.csv", index=False)
    print("\nSaved H1 and M15 data to CSV files: EURUSD_H1_2years.csv, EURUSD_M15_2years.csv")


def main():
    symbol = "EURUSD"
    timeframe = "M15"
    days = 180

    print(f"\nLoading {timeframe} data for {symbol}...")
    df = load_data(symbol, timeframe, days)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    validate_data_for_backtest(df)

    # Split train/test
    train_df, test_df = split_data_for_backtest(df, 0.8)
    print(f"Train/Test split: {len(train_df)} / {len(test_df)} bars")

    # Get the test period dates
    test_start = test_df['time'].min()
    test_end = test_df['time'].max()

    # Calculate H1 data period (exactly 45 days before test start)
    h1_start = test_start - pd.Timedelta(days=45)

    print(f"\nLoading H1 data for SR levels...")
    print(f"Period: {h1_start} to {test_end}")

    for attempt in range(3):
        df_h1 = load_data(
            symbol=symbol,
            timeframe="H1",
            start_date=h1_start - pd.Timedelta(days=attempt*15),  # Extend period on each retry
            end_date=test_end
        )

        if df_h1 is not None and not df_h1.empty:
            h1_completeness = len(df_h1) / ((test_end - h1_start).days * 24 * 5/7)
            if h1_completeness >= 0.90:
                break

        logging.warning(f"H1 data attempt {attempt+1} completeness: {h1_completeness:.1%}")

    if df_h1.empty:
        print("Failed to load H1 data")
        return

    validate_data_for_backtest(df_h1)

    # Only proceed if we have enough quality data
    h1_completeness = len(df_h1) / ((test_end - h1_start).days * 24 * 5/7)
    if h1_completeness < 0.9:  # Require at least 90% data completeness
        print(f"Insufficient H1 data completeness: {h1_completeness:.1%}")
        return

    strategy = SR_Bounce_Strategy()

    if hasattr(strategy, "update_weekly_levels"):
        strategy.update_weekly_levels(df_h1, weeks=2, weekly_buffer=0.00075)
        print("Weekly levels found:", strategy.valid_levels)
        print(f"Number of valid levels found: {len(strategy.valid_levels)}")
        print(f"Valid levels: {strategy.valid_levels}")

    bounce_details = check_training_bounces(strategy, train_df)

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

    print("\n--- MONTE CARLO (1000 shuffles) ---")

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"BACKTEST_REPORT_{symbol}_{now_str}.md")

    monthly_data = {}
    monthly_levels = []
    weekly_levels = []

    with ReportWriter(report_file) as writer:
        writer.write_data_overview(test_df)
        writer.write_trades_section(trades)
        writer.write_stats_section(stats, final_balance)
        writer.write_monthly_breakdown(monthly_data)
        writer.write_sr_levels(monthly_levels, weekly_levels)

    print(f"\nDetailed report written to: {report_file}")


if __name__ == "__main__":
    # Normal backtest run (unchanged):
    main()

    # Optional: M15 -> H1 resample
    fetch_and_resample_m15_to_h1()

    # [ADDED LINES] Uncomment below if you want to fetch ~2 years of data:
    fetch_long_term_data()
