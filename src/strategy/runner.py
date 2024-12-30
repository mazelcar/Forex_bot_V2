import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import pytz

# Import your local modules
from src.strategy.data_storage import save_data_to_csv, load_data_from_csv
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


def load_data(symbol="EURUSD", timeframe="H1", days=None, start_date=None, end_date=None, max_retries=3) -> pd.DataFrame:
    """
    Enhanced data loader with Step 1 date fix + Step 2 CSV storage.
    Fetches from broker only if no local CSV or CSV is empty.
    """
    # -------------------------------------------------
    # STEP 2: Check local CSV first, to avoid re-fetching
    # -------------------------------------------------
    csv_filename = f"{symbol}_{timeframe}_data.csv"
    if os.path.exists(csv_filename):
        print(f"Found local CSV: {csv_filename}, skipping broker fetch...")
        df_local = load_data_from_csv(csv_filename)
        if not df_local.empty:
            print(f"Loaded {len(df_local)} bars from local CSV.")
            return df_local
        else:
            print("Local CSV is empty; proceeding with broker fetch...")

    mt5 = MT5Handler(debug=True)
    print(f"Attempting to load {symbol} {timeframe} data...")

    # ----- STEP 1: FIX DATE RANGE (already implemented) -----
    if days and not start_date and not end_date:
        end_date = datetime(2023, 1, 1, tzinfo=pytz.UTC)  # Fixed date instead of datetime.now()
        start_date = end_date - timedelta(days=days)
        print(f"Date range: {start_date} to {end_date}")
    # --------------------------------------------------------

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")
            df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)

            if df is None:
                print(f"MT5 returned None on attempt {attempt + 1}")
                continue

            if df.empty:
                print(f"MT5 returned empty DataFrame on attempt {attempt + 1}")
                continue

            print(f"Retrieved {len(df)} bars")

            # --------------------------------------
            # STEP 2: Save data to CSV after fetch
            # --------------------------------------
            save_data_to_csv(df, csv_filename)

            return df

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if start_date:
                start_date -= timedelta(days=5)
                print(f"Retrying with new start date: {start_date}")

    print("Failed to load data after all attempts")
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
        ~df['is_extreme_spread'] &
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

    df_h1_resampled = df_m15.resample("1h").agg({
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
        return pd.DataFrame()
    df_h1_resampled = resample_m15_to_h1(df_m15)
    try:
        validate_data_for_backtest(df_h1_resampled, timeframe="H1")
    except:
        logger.warning("Resampled H1 data is also incomplete or invalid.")
    return df_h1_resampled


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    """
    Enhanced backtest engine that:
    - Tracks floating PnL
    - Monitors FTMO daily and max drawdown limits
    - Forces position closure when limits are breached
    - Returns final trades & balance
    """
    if df.empty:
        logger.warning("DataFrame empty in run_backtest. Returning no trades.")
        return {"Trades": [], "final_balance": initial_balance}

    trades: list["SR_Bounce_Strategy.Trade"] = []
    balance = initial_balance
    active_trade: Optional["SR_Bounce_Strategy.Trade"] = None
    daily_high_balance = initial_balance
    current_day = None

    logger.debug("Starting backtest run with FTMO compliance...")

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]

        if len(current_segment) < 5:
            continue

        current_bar = current_segment.iloc[-1]
        bar_date = pd.to_datetime(current_bar['time']).date()

        # Reset daily tracking on new day
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New trading day: {bar_date}, Balance reset: {balance}")

        # Update daily high water mark
        daily_high_balance = max(daily_high_balance, balance)

        # Calculate floating PnL if trade is active
        if active_trade:
            current_price = float(current_bar['close'])
            if active_trade.type == "BUY":
                floating_pnl = (current_price - active_trade.entry_price) * 10000.0 * active_trade.size
            else:  # SELL
                floating_pnl = (active_trade.entry_price - current_price) * 10000.0 * active_trade.size

            # Check FTMO daily drawdown limit including floating PnL
            total_daily_drawdown = (balance + floating_pnl - daily_high_balance) / initial_balance
            if total_daily_drawdown < -0.05:  # 5% daily drawdown limit
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning(f"Force closed trade due to daily drawdown limit at {current_bar['time']}")
                active_trade = None
                continue

            # Check max drawdown limit (10%)
            total_drawdown = (balance + floating_pnl - initial_balance) / initial_balance
            if total_drawdown < -0.10:
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning(f"Force closed trade due to max drawdown limit at {current_bar['time']}")
                active_trade = None
                continue

            # Check regular exit conditions if no force close
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

        # Try to open new trade if none active
        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                trades.append(active_trade)
                logger.debug(f"New trade opened: {new_trade.type} at {new_trade.entry_price}")

    # Force close any remaining trade at the end
    if active_trade:
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])
        if active_trade.type == "BUY":
            pnl = (last_close - active_trade.entry_price) * 10000.0 * active_trade.size
        else:
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

def run_backtest_step5(strategy: SR_Bounce_Strategy, symbol_data_dict: Dict[str, pd.DataFrame], initial_balance=10000.0) -> Dict:
    """
    STEP 5: Advanced Trade Management with multi-symbol parallel processing,
    combined FTMO risk tracking, cross-pair exposure management,
    and dynamic position sizing based on correlation.

    Args:
        strategy (SR_Bounce_Strategy): Our trading strategy instance.
        symbol_data_dict (Dict[str, pd.DataFrame]): A dictionary of symbols -> validated price data.
        initial_balance (float): Starting account balance for backtest.
    Returns:
        Dict with all trades from all symbols and final combined balance.
    """

    if not symbol_data_dict:
        logger.warning("No symbol data provided to run_backtest_step5. Returning empty result.")
        return {"Trades": [], "final_balance": initial_balance}

    # 1) Merge all symbols' data into a single DataFrame with a 'symbol' column
    #    so we can iterate chronologically across all pairs in parallel.
    merged_frames = []
    for sym, df in symbol_data_dict.items():
        temp = df.copy()
        temp["symbol"] = sym
        merged_frames.append(temp)
    all_data = pd.concat(merged_frames, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Global account equity
    balance = initial_balance

    # Track open trades per symbol: { "EURUSD": Trade or None, "GBPUSD": Trade or None, ... }
    active_trades: Dict[str, Optional[strategy.Trade]] = {sym: None for sym in symbol_data_dict.keys()}

    # Master list of closed trades from all symbols
    closed_trades = []

    # For FTMO tracking across all pairs
    daily_high_balance = balance
    current_day = None

    logger.debug("Starting multi-symbol backtest run with advanced trade management...")

    # 2) Main loop: process each bar in chronological order
    for i in range(len(all_data)):
        row = all_data.iloc[i]
        symbol = row["symbol"]
        # Current bar subset for this symbol only
        # (We slice up to i+1 but only where symbol matches)
        symbol_slice = all_data.iloc[: i + 1]
        symbol_slice = symbol_slice[symbol_slice["symbol"] == symbol]

        if len(symbol_slice) < 5:
            continue

        current_bar = symbol_slice.iloc[-1]
        bar_date = pd.to_datetime(current_bar["time"]).date()

        # --- Daily resets for FTMO checks (combined) ---
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New day: {bar_date} | Reset daily high balance: {balance:.2f}")

        # Update daily high watermark
        daily_high_balance = max(daily_high_balance, balance)

        # --- Check if trade is active for this symbol ---
        if active_trades[symbol]:
            # Evaluate floating PnL
            trade = active_trades[symbol]
            current_price = float(current_bar["close"])

            if trade.type == "BUY":
                floating_pnl = (current_price - trade.entry_price) * 10000.0 * trade.size
            else:
                floating_pnl = (trade.entry_price - current_price) * 10000.0 * trade.size

            # Combined daily drawdown check
            total_daily_drawdown = (balance + floating_pnl - daily_high_balance) / initial_balance
            if total_daily_drawdown < -strategy.daily_drawdown_limit:
                # Force close
                balance += floating_pnl
                trade.close_i = current_bar.name
                trade.close_time = current_bar["time"]
                trade.close_price = current_price
                trade.pnl = floating_pnl
                closed_trades.append(trade)
                logger.warning(f"[{symbol}] Force-closed trade due to daily drawdown at {current_bar['time']}")
                active_trades[symbol] = None
                continue

            # Combined max drawdown check
            total_drawdown = (balance + floating_pnl - initial_balance) / initial_balance
            if total_drawdown < -strategy.max_drawdown_limit:
                balance += floating_pnl
                trade.close_i = current_bar.name
                trade.close_time = current_bar["time"]
                trade.close_price = current_price
                trade.pnl = floating_pnl
                closed_trades.append(trade)
                logger.warning(f"[{symbol}] Force-closed trade due to max drawdown at {current_bar['time']}")
                active_trades[symbol] = None
                continue

            # Normal exit conditions
            should_close, fill_price, pnl = strategy.exit_trade(symbol_slice, trade)
            if should_close:
                balance += pnl
                trade.close_i = current_bar.name
                trade.close_time = current_bar["time"]
                trade.close_price = fill_price
                trade.pnl = pnl
                closed_trades.append(trade)
                logger.debug(f"[{symbol}] Trade closed: PnL={pnl:.2f}")
                active_trades[symbol] = None

        # --- Try to open new trade if none active for this symbol ---
        if active_trades[symbol] is None:
            new_trade = strategy.open_trade(symbol_slice, balance, i, symbol=symbol)
            if new_trade:
                # Check cross-pair exposure & correlation before finalizing
                can_open, reason = strategy.validate_cross_pair_exposure(new_trade, active_trades, balance)
                if not can_open:
                    logger.debug(f"[{symbol}] Cross-pair check failed: {reason}")
                    continue

                active_trades[symbol] = new_trade
                logger.debug(f"[{symbol}] New trade opened: {new_trade.type} at {new_trade.entry_price:.5f}")

    # 3) At end: close any remaining trades
    for symbol, trade in active_trades.items():
        if trade is not None:
            # Close at last known price for that symbol
            sym_df = all_data[all_data["symbol"] == symbol].iloc[-1]
            last_close = float(sym_df["close"])

            if trade.type == "BUY":
                pnl = (last_close - trade.entry_price) * 10000.0 * trade.size
            else:
                pnl = (trade.entry_price - last_close) * 10000.0 * trade.size

            balance += pnl
            trade.close_i = sym_df.name
            trade.close_time = sym_df["time"]
            trade.close_price = last_close
            trade.pnl = pnl
            closed_trades.append(trade)
            logger.debug(f"[{symbol}] Final trade closed at end: PnL={pnl:.2f}")

    return {
        "Trades": [t.to_dict() for t in closed_trades],
        "final_balance": balance
    }



# -------------------------------------------------------------
#  STEP 3: Multi-Symbol Data Management (Modified main)
# -------------------------------------------------------------
def main():
    print("\nStarting backtest with enhanced data validation...")

    # We'll load multiple symbols in parallel
    symbols = ["EURUSD", "GBPUSD"]
    timeframe = "M15"
    days = 180

    # Dictionary to hold validated data for each symbol
    symbol_data_dict = {}

    # Load and validate data for each symbol
    for symbol in symbols:
        print(f"\nLoading {timeframe} data for {symbol}...")
        df = load_data(symbol, timeframe, days=days)

        if df.empty:
            print(f"ERROR: No valid data loaded for {symbol}. Skipping.")
            continue

        if not validate_data_for_backtest(df, timeframe):
            print(f"ERROR: Validation failed for {symbol}. Skipping.")
            continue

        symbol_data_dict[symbol] = df

    # If we have at least 2 symbols, we'll do a multi-symbol backtest using STEP 5
    if len(symbol_data_dict) >= 2:
        # Quick correlation check for demonstration
        df_eu = symbol_data_dict["EURUSD"].copy()
        df_gb = symbol_data_dict["GBPUSD"].copy()
        df_eu.rename(columns={"close": "close_eu"}, inplace=True)
        df_gb.rename(columns={"close": "close_gb"}, inplace=True)
        merged = pd.merge(
            df_eu[["time", "close_eu"]],
            df_gb[["time", "close_gb"]],
            on="time",
            how="inner",
        ).sort_values("time").reset_index(drop=True)
        corr = merged["close_eu"].corr(merged["close_gb"])
        print(f"\nCorrelation between EURUSD and GBPUSD (close prices): {corr:.4f}")

        # Instantiate the strategy
        strategy = SR_Bounce_Strategy(config_file=None)

        # Optional: If you want to fetch H1 data for each symbol for weekly S/R:
        for sym in symbol_data_dict:
            df_sym = symbol_data_dict[sym]
            test_start = pd.to_datetime(df_sym['time'].iloc[-1]) - timedelta(days=45)
            test_end = pd.to_datetime(df_sym['time'].iloc[-1])
            df_h1 = check_h1_data_or_resample(sym, test_start, test_end)
            if not df_h1.empty:
                strategy.update_weekly_levels(df_h1, symbol=sym, weeks=2, weekly_buffer=0.00075)

        # Now run the multi-symbol Step 5 backtest
        results = run_backtest_step5(strategy, symbol_data_dict, initial_balance=10000.0)
        trades = results["Trades"]
        final_balance = results["final_balance"]

        # Analyze trades
        stats = analyze_trades(trades, 10000.0)
        print("\n--- MULTI-SYMBOL BACKTEST (Step 5) COMPLETE ---")
        print(f"Symbols used: {list(symbol_data_dict.keys())}")
        print(f"Total Trades: {stats['count']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${stats['total_pnl']:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")

        # STEP 6: Generate comprehensive report
        from src.strategy.report_writer import ReportWriter

        # Prepare correlation data as dictionary, e.g. from strategy.symbol_correlations or manual
        correlation_data = {
            "EURUSD": {"GBPUSD": corr},
            "GBPUSD": {"EURUSD": corr}
        }

        # FTMO data (example placeholders)
        ftmo_data = {
            "daily_drawdown_limit": 0.05,
            "max_drawdown_limit": 0.10,
            "profit_target": 0.10,
            "current_daily_dd": 0.02,   # example
            "current_total_dd": 0.03   # example
        }

        # Dummy placeholders for monthly_data, monthly_levels, weekly_levels if you want them
        monthly_data = {}
        monthly_levels = []
        weekly_levels = []

        report_path = "step6_report.md"
        with ReportWriter(report_path) as rw:
            rw.generate_full_report(
                df_test=symbol_data_dict["EURUSD"],  # or any main symbol for an overview
                trades=trades,
                stats=stats,
                mc_results={},             # if you have MonteCarlo results pass here
                final_balance=final_balance,
                monthly_data=monthly_data,
                monthly_levels=monthly_levels,
                weekly_levels=weekly_levels,
                correlation_data=correlation_data,
                ftmo_data=ftmo_data
            )
        print(f"\nStep 6 report generated: {report_path}")

    else:
        # Fallback to single-symbol approach
        if len(symbol_data_dict) == 0:
            print("No valid symbols loaded. Exiting main().")
            return

        default_symbol = next(iter(symbol_data_dict.keys()))
        df = symbol_data_dict[default_symbol]

        # Split data
        train_df, test_df = split_data_for_backtest(df, 0.8)
        print(f"\nSingle-Symbol Approach => {default_symbol}")
        print(f"Train/Test split: {len(train_df)} / {len(test_df)}")

        # Optional: fetch H1 for weekly S/R
        test_start = pd.to_datetime(test_df['time'].min())
        test_end = pd.to_datetime(test_df['time'].max())
        h1_start = test_start - timedelta(days=45)
        df_h1 = check_h1_data_or_resample(default_symbol, h1_start, test_end, threshold=0.90)

        strategy = SR_Bounce_Strategy(config_file=None)
        if not df_h1.empty:
            strategy.update_weekly_levels(df_h1, symbol=default_symbol, weeks=2, weekly_buffer=0.00075)

        # Basic bounces (training)
        bounce_count = 0
        for i in range(len(train_df)):
            current_segment = train_df.iloc[: i + 1]
            sig = strategy.generate_signals(current_segment)
            if sig["type"] != "NONE":
                bounce_count += 1
        print(f"Training-set bounces detected: {bounce_count}")

        # Single-pair backtest
        single_result = run_backtest(strategy, test_df, initial_balance=10000.0)
        sp_trades = single_result["Trades"]
        sp_final_balance = single_result["final_balance"]

        sp_stats = analyze_trades(sp_trades, 10000.0)
        print("\n--- SINGLE-SYMBOL BACKTEST COMPLETE ---")
        print(f"Total Trades: {sp_stats['count']}")
        print(f"Win Rate: {sp_stats['win_rate']:.2f}%")
        print(f"Profit Factor: {sp_stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${sp_stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${sp_stats['total_pnl']:.2f}")
        print(f"Final Balance: ${sp_final_balance:.2f}")



if __name__ == "__main__":
    main()
