# --------------------------------------------------------------
# runner.py
# --------------------------------------------------------------
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

# Local modules
from src.strategy.data_storage import save_data_to_csv, load_data_from_csv
from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.data_validator import DataValidator
from src.strategy.report_writer import ReportWriter, analyze_trades


def get_logger(name="runner", logfile="runner_debug.log") -> logging.Logger:
    """Global logger for runner.py."""
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


def load_data(
    symbol="EURUSD",
    timeframe="H1",
    days=None,
    start_date=None,
    end_date=None,
    max_retries=3,
) -> pd.DataFrame:
    """
    Load data from CSV if available; otherwise fetch missing from broker.
    Combines partial coverage logic, resaves merged data to CSV.
    """
    csv_filename = f"{symbol}_{timeframe}_data.csv"
    df_local = pd.DataFrame()

    # Load local CSV if exists
    if os.path.exists(csv_filename):
        df_local = load_data_from_csv(csv_filename)
        if not df_local.empty:
            df_local["time"] = pd.to_datetime(df_local["time"], utc=True)

    # If days specified without explicit range, choose date range
    if days and not start_date and not end_date:
        end_date = datetime.now(pytz.UTC)
        # Shift end date to a weekday near market close (avoid future data)
        while end_date.weekday() >= 5:  # Saturday=5, Sunday=6
            end_date -= timedelta(days=1)
        if end_date.hour < 21:
            end_date -= timedelta(days=1)
        end_date = end_date.replace(hour=21, minute=45, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        print(f"Date range: {start_date} to {end_date}")

    # If local data covers requested range, slice & return
    if not df_local.empty and start_date and end_date:
        local_min = df_local['time'].min()
        local_max = df_local['time'].max()
        if local_min <= start_date and local_max >= end_date:
            df_requested = df_local[(df_local['time'] >= start_date) & (df_local['time'] <= end_date)]
            if not df_requested.empty:
                print(f"Local CSV covers {symbol} {timeframe} from {start_date} to {end_date}, "
                      f"returning {len(df_requested)} bars.")
                return df_requested
            else:
                print("Local CSV has no bars in the sub-range, fetching from broker...")

        # Partial coverage
        else:
            missing_start = None
            missing_end = None

            if local_min > start_date:
                missing_start = start_date
                missing_end = local_min - timedelta(minutes=1)

            if local_max < end_date:
                if missing_start is None:
                    missing_start = local_max + timedelta(minutes=1)
                missing_end = end_date

            if missing_start and missing_end:
                print(f"Partial coverage. Fetching missing portion: {missing_start} to {missing_end}")
                mt5 = MT5Handler(debug=True)
                df_missing = pd.DataFrame()

                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1} of {max_retries} for missing portion")
                        df_partial = mt5.get_historical_data(
                            symbol, timeframe, missing_start, missing_end
                        )
                        if df_partial is not None and not df_partial.empty:
                            df_partial["time"] = pd.to_datetime(df_partial["time"], utc=True)
                            df_missing = pd.concat([df_missing, df_partial], ignore_index=True)
                            print(f"Fetched {len(df_partial)} bars for the missing portion.")
                        else:
                            print("Broker returned empty or None data for missing portion.")
                        break
                    except Exception as e:
                        print(f"Error fetching missing portion on attempt {attempt + 1}: {str(e)}")
                        missing_start -= timedelta(days=5)

                if not df_missing.empty:
                    df_merged = pd.concat([df_local, df_missing], ignore_index=True)
                    df_merged["time"] = pd.to_datetime(df_merged["time"], utc=True)
                    df_merged.drop_duplicates(subset=["time"], keep="last", inplace=True)
                    df_merged.sort_values("time", inplace=True)
                    df_local = df_merged.reset_index(drop=True)

                    save_data_to_csv(df_local, csv_filename)
                    df_requested = df_local[
                        (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                    ]
                    print(f"Returning merged data slice with {len(df_requested)} bars.")
                    return df_requested
                else:
                    print("Failed to fetch any missing data, returning local CSV subset.")
                    return df_local[
                        (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                    ]
            else:
                df_requested = df_local[
                    (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                ]
                print(f"No broker fetch needed. Returning {len(df_requested)} bars from local CSV.")
                return df_requested

    elif not df_local.empty and not start_date and not end_date:
        print(f"Found local CSV: {csv_filename}, no date range requested, returning entire file.")
        return df_local

    # Fallback: fetch from broker if no local or partial coverage
    print(f"Fetching {symbol} {timeframe} from broker, no local coverage or partial coverage.")
    mt5 = MT5Handler(debug=True)

    for attempt in range(max_retries):
        try:
            print(f"Broker fetch attempt {attempt + 1} of {max_retries}")
            df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
            if df is None or df.empty:
                print(f"Broker returned no data on attempt {attempt + 1}")
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            print(f"Retrieved {len(df)} bars from broker.")
            save_data_to_csv(df, csv_filename)
            return df
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if start_date:
                start_date -= timedelta(days=5)
                print(f"Retrying with new start date: {start_date}")

    print("Failed to load data after all attempts.")
    return pd.DataFrame()


def validate_data_for_backtest(df: pd.DataFrame, timeframe: str = "M15") -> bool:
    """Validate data quality before using in a backtest."""
    if df.empty:
        print("ERROR: No data loaded.")
        return False

    current_time = datetime.now(pytz.UTC)
    df_time_max = pd.to_datetime(df['time'].max())
    if not df_time_max.tzinfo:
        df_time_max = pytz.UTC.localize(df_time_max)
    if df_time_max > current_time:
        print(f"ERROR: Data has future dates! {df_time_max}")
        return False

    df['time'] = pd.to_datetime(df['time'])
    if not hasattr(df['time'].dt, 'tz'):
        df['time'] = df['time'].dt.tz_localize(pytz.UTC)

    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False

    df['spread'] = df['high'] - df['low']
    median_spread = df['spread'].median()
    spread_std = df['spread'].std()
    df['is_extreme_spread'] = df['spread'] > (median_spread + 5 * spread_std)

    invalid_prices = df[
        ~df['is_extreme_spread'] &
        ((df['high'] < df['low']) |
         (df['close'] > df['high']) |
         (df['close'] < df['low']) |
         (df['open'] > df['high']) |
         (df['open'] < df['low']))
    ]
    if not invalid_prices.empty:
        print("ERROR: Found truly invalid price data:")
        print(invalid_prices[['time', 'open', 'high', 'low', 'close', 'spread']].head())
        return False

    if (df[['open', 'high', 'low', 'close']] == 0).any().any():
        print("ERROR: Zero prices detected in data.")
        return False

    date_min = df["time"].min()
    date_max = df["time"].max()
    date_range = date_max - date_min
    total_days = date_range.days

    print(f"\nData Range Analysis:\nStart: {date_min}\nEnd: {date_max}\nTotal days: {total_days}")

    bars_per_day = {"M15": 96, "M5": 288, "H1": 24}.get(timeframe, 24)
    expected_bars = total_days * bars_per_day * (5 / 7)
    actual_bars = len(df)
    completeness = (actual_bars / (expected_bars if expected_bars else 1)) * 100

    print(f"\nBar Count Analysis:\nExpected bars: {expected_bars:.0f}\n"
          f"Actual bars: {actual_bars}\nData completeness: {completeness:.1f}%")

    df_sorted = df.sort_values("time").reset_index(drop=True)
    time_diffs = df_sorted["time"].diff()

    if timeframe.startswith("M"):
        freq_minutes = int(timeframe[1:])
        expected_diff = pd.Timedelta(minutes=freq_minutes)
    elif timeframe.startswith("H"):
        freq_hours = int(timeframe[1:])
        expected_diff = pd.Timedelta(hours=freq_hours)
    else:
        expected_diff = pd.Timedelta(minutes=15)

    weekend_gaps = time_diffs[
        (df_sorted['time'].dt.dayofweek == 0) & (time_diffs > pd.Timedelta(days=1))
    ]
    unexpected_gaps = time_diffs[
        (time_diffs > expected_diff * 1.5) &
        ~((df_sorted['time'].dt.dayofweek == 0) & (time_diffs > pd.Timedelta(days=1)))
    ]

    print(f"\nGap Analysis:\nWeekend gaps: {len(weekend_gaps)}\nUnexpected gaps: {len(unexpected_gaps)}")
    if len(unexpected_gaps) > 0:
        print("Largest unexpected gaps:")
        largest_gaps = unexpected_gaps.nlargest(3)
        for idx in largest_gaps.index:
            if idx > 0:
                gap_start = df_sorted.loc[idx - 1, 'time']
                print(f"Gap of {time_diffs[idx]} at {gap_start}")

    if completeness < 90:
        print("ERROR: Data completeness below 90%")
        return False
    if len(unexpected_gaps) > total_days * 0.1:
        print("ERROR: Too many unexpected gaps in data")
        return False

    print("\nValidation passed. Data is suitable for backtest.")
    return True


def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def resample_m15_to_h1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Simple aggregator from M15 to H1."""
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
    Fetch H1 data. If completeness < threshold, fallback to M15 and resample.
    """
    df_h1 = load_data(symbol=symbol, timeframe="H1", start_date=h1_start, end_date=h1_end)
    if df_h1.empty:
        logger.warning("No H1 data returned, trying M15 fallback.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    try:
        validate_data_for_backtest(df_h1, timeframe="H1")
    except ValueError as e:
        logger.warning(f"Validation error on H1: {str(e)}. Falling back to M15.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    df_h1_min = pd.to_datetime(df_h1["time"].min())
    df_h1_max = pd.to_datetime(df_h1["time"].max())
    day_span = (df_h1_max - df_h1_min).days
    expected_bars = day_span * 24 * (5/7)
    actual_bars = len(df_h1)
    completeness = actual_bars / (expected_bars if expected_bars > 0 else 1e-9)

    if completeness < threshold:
        logger.warning(f"H1 data completeness {completeness:.1%} < {threshold:.1%}, using M15 fallback.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    return df_h1


def fallback_resample_from_m15(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    logger.info("Attempting fallback: fetch M15 data, then resample to H1.")
    df_m15 = load_data(symbol=symbol, timeframe="M15", start_date=start, end_date=end)
    if df_m15.empty:
        logger.error("M15 fallback data also empty.")
        return pd.DataFrame()

    df_h1_resampled = resample_m15_to_h1(df_m15)
    try:
        validate_data_for_backtest(df_h1_resampled, timeframe="H1")
    except:
        logger.warning("Resampled H1 data incomplete or invalid.")
    return df_h1_resampled


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    """
    Single-symbol backtest with FTMO-like daily drawdown and max drawdown checks.
    """
    if df.empty:
        logger.warning("Empty DataFrame in run_backtest, returning no trades.")
        return {"Trades": [], "final_balance": initial_balance}

    trades: List[SR_Bounce_Strategy.Trade] = []
    balance = initial_balance
    active_trade: Optional[SR_Bounce_Strategy.Trade] = None
    daily_high_balance = initial_balance
    current_day = None

    logger.debug("Starting single-symbol backtest with FTMO rules...")

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]
        if len(current_segment) < 5:
            continue

        current_bar = current_segment.iloc[-1]
        bar_date = pd.to_datetime(current_bar['time']).date()

        # Daily reset
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New day: {bar_date} - resetting daily high to {balance:.2f}")

        daily_high_balance = max(daily_high_balance, balance)

        # If trade is active, check floating PnL + exits
        if active_trade:
            current_price = float(current_bar["close"])
            if active_trade.type == "BUY":
                floating_pnl = (current_price - active_trade.entry_price) * 10000.0 * active_trade.size
            else:
                floating_pnl = (active_trade.entry_price - current_price) * 10000.0 * active_trade.size

            # Check daily drawdown
            total_daily_drawdown = (balance + floating_pnl - daily_high_balance) / initial_balance
            if total_daily_drawdown < -0.05:
                # Force close
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning("Force-closed trade due to daily drawdown limit.")
                active_trade = None
                continue

            # Check max drawdown
            total_drawdown = (balance + floating_pnl - initial_balance) / initial_balance
            if total_drawdown < -0.10:
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning("Force-closed trade due to max drawdown limit.")
                active_trade = None
                continue

            # Normal exit
            should_close, fill_price, pnl = strategy.exit_trade(current_segment, active_trade)
            if should_close:
                balance += pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar["time"]
                active_trade.close_price = fill_price
                active_trade.pnl = pnl
                trades.append(active_trade)
                logger.debug(f"Closed trade with PnL={pnl:.2f}")
                active_trade = None

        # Attempt new trade if none active
        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                logger.debug(f"Opened new trade: {new_trade.type} at {new_trade.entry_price:.5f}")

    # Force close any remaining trade at end
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
        trades.append(active_trade)
        logger.debug(f"Final forced-close trade PnL={pnl:.2f}")
        active_trade = None

    return {
        "Trades": [t.to_dict() for t in trades],
        "final_balance": balance,
    }


def run_backtest_step5(
    strategy: SR_Bounce_Strategy,
    symbol_data_dict: Dict[str, pd.DataFrame],
    initial_balance=10000.0
) -> Dict:
    """
    Multi-symbol backtest with advanced FTMO + cross-pair correlation checks.
    """
    logger.debug("Starting multi-symbol backtest step5...")

    if not symbol_data_dict:
        logger.warning("No symbol data in run_backtest_step5, returning empty.")
        return {"Trades": [], "final_balance": initial_balance}

    # Merge all symbols into a single DataFrame
    merged_frames = []
    for sym, df_sym in symbol_data_dict.items():
        temp = df_sym.copy()
        temp["symbol"] = sym
        merged_frames.append(temp)

    all_data = pd.concat(merged_frames, ignore_index=True).sort_values("time").reset_index(drop=True)

    balance = initial_balance
    daily_high_balance = balance
    current_day = None
    active_trades: Dict[str, Optional[SR_Bounce_Strategy.Trade]] = {
        s: None for s in symbol_data_dict.keys()
    }
    closed_trades: List[SR_Bounce_Strategy.Trade] = []

    for i in range(len(all_data)):
        row = all_data.iloc[i]
        symbol = row["symbol"]

        # Filter out all bars for this symbol up to current index
        symbol_slice = all_data.iloc[: i + 1]
        symbol_slice = symbol_slice[symbol_slice["symbol"] == symbol]
        if len(symbol_slice) < 5:
            continue

        current_bar = symbol_slice.iloc[-1]
        bar_date = pd.to_datetime(current_bar["time"]).date()

        # Check daily reset
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New day: {bar_date} -> daily high {balance:.2f}")

        daily_high_balance = max(daily_high_balance, balance)

        # 1) If a trade is active for this symbol, check for exit + drawdown
        if active_trades[symbol]:
            trade = active_trades[symbol]
            exited, fill_price, pnl = strategy.exit_trade(symbol_slice, trade, symbol)
            if exited:
                balance += pnl
                trade.close_i = current_bar.name
                trade.close_time = current_bar["time"]
                trade.close_price = fill_price
                trade.pnl = pnl
                closed_trades.append(trade)
                logger.debug(f"[{symbol}] Trade closed with PnL={pnl:.2f}")
                active_trades[symbol] = None
            else:
                # If not exited, compute floating PnL
                current_price = float(current_bar["close"])
                if trade.type == "BUY":
                    floating_pnl = (current_price - trade.entry_price) * 10000.0 * trade.size
                else:
                    floating_pnl = (trade.entry_price - current_price) * 10000.0 * trade.size

                # Daily drawdown
                daily_dd_ratio = (balance + floating_pnl - daily_high_balance) / initial_balance
                total_dd_ratio = (balance + floating_pnl - initial_balance) / initial_balance

                # Force close if daily or max drawdown is exceeded
                if daily_dd_ratio < -strategy.daily_drawdown_limit:
                    logger.warning(f"[{symbol}] Force-closed trade (daily drawdown).")
                    balance += floating_pnl
                    trade.close_i = current_bar.name
                    trade.close_time = current_bar["time"]
                    trade.close_price = current_price
                    trade.pnl = floating_pnl
                    trade.exit_reason = "Daily drawdown forced close"
                    closed_trades.append(trade)
                    active_trades[symbol] = None
                elif total_dd_ratio < -strategy.max_drawdown_limit:
                    logger.warning(f"[{symbol}] Force-closed trade (max drawdown).")
                    balance += floating_pnl
                    trade.close_i = current_bar.name
                    trade.close_time = current_bar["time"]
                    trade.close_price = current_price
                    trade.pnl = floating_pnl
                    trade.exit_reason = "Max drawdown forced close"
                    closed_trades.append(trade)
                    active_trades[symbol] = None

        # 2) Attempt to open a new trade if none active for this symbol
        if active_trades[symbol] is None:
            # Enforce the global 'max_positions' limit across all symbols
            currently_open_positions = sum(t is not None for t in active_trades.values())
            if currently_open_positions >= strategy.max_positions:
                # Skip opening new trade since we hit the limit
                continue

            new_trade = strategy.open_trade(symbol_slice, balance, i, symbol=symbol)
            if new_trade:
                # Cross-pair exposure checks
                can_open, reason = strategy.validate_cross_pair_exposure(
                    new_trade, active_trades, balance
                )
                if can_open:
                    active_trades[symbol] = new_trade
                    logger.debug(f"[{symbol}] Opened trade: {new_trade.type} at {new_trade.entry_price:.5f}")
                else:
                    logger.debug(f"[{symbol}] Cross-pair or exposure check failed: {reason}")

    # End of data: force close any leftover trades
    for sym, trade in active_trades.items():
        if trade is not None:
            sym_df = all_data[all_data["symbol"] == sym].iloc[-1]
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
            trade.exit_reason = "Forced close end of data"
            closed_trades.append(trade)
            logger.debug(f"[{sym}] Final forced-close trade PnL={pnl:.2f}")

    return {
        "Trades": [t.to_dict() for t in closed_trades],
        "final_balance": balance,
    }


def main():
    """Main function orchestrating the backtest process."""
    print("Starting backtest with simplified code...\n")

    # ----------------------------------------------------------------
    # We keep the same config, but reduce 'days' if needed
    # ----------------------------------------------------------------
    backtest_config = {
        "symbols": ["EURUSD", "GBPUSD"],
        "timeframe": "M15",
        "days": 365,  # or reduce to see faster tests
        "sr_lookback_days": 90,  # a bit less to speed up S/R detection
        "initial_balance": 10000.0,
        "report_path": "comprehensive_backtest_report.md",
        "ftmo_limits": {
            "daily_drawdown_limit": 0.05,
            "max_drawdown_limit": 0.10,
            "profit_target": 0.10,
            "current_daily_dd": 0.02,
            "current_total_dd": 0.03
        },
    }

    symbol_data_dict = {}
    for symbol in backtest_config["symbols"]:
        print(f"\nLoading {backtest_config['timeframe']} data for {symbol} ...")
        df = load_data(
            symbol=symbol,
            timeframe=backtest_config["timeframe"],
            days=backtest_config["days"],
        )
        if df.empty:
            print(f"ERROR: No data loaded for {symbol}. Skipping.")
            continue
        if not validate_data_for_backtest(df, backtest_config["timeframe"]):
            print(f"ERROR: Validation failed for {symbol}. Skipping.")
            continue
        symbol_data_dict[symbol] = df

    if not symbol_data_dict:
        print("No valid symbols, exiting.")
        return

    # Single vs Multi
    if len(symbol_data_dict) > 1:
        print("Detected multiple symbols, proceeding with multi-symbol Step 5 backtest.")
        strategy = SR_Bounce_Strategy()
        # Update correlation for multi-symbol (example with EURUSD/GBPUSD)
        if "EURUSD" in symbol_data_dict and "GBPUSD" in symbol_data_dict:
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
            strategy.symbol_correlations["EURUSD"]["GBPUSD"] = corr
            strategy.symbol_correlations["GBPUSD"] = {"EURUSD": corr}
            print(f"Correlation (EURUSD/GBPUSD): {corr:.4f}")

        # Optional: fetch H1 data for S/R
        for sym, df_sym in symbol_data_dict.items():
            test_start = pd.to_datetime(df_sym["time"].iloc[-1]) - timedelta(
                days=backtest_config["sr_lookback_days"]
            )
            test_end = pd.to_datetime(df_sym["time"].iloc[-1])
            df_h1 = check_h1_data_or_resample(sym, test_start, test_end)
            if not df_h1.empty:
                strategy.update_weekly_levels(df_h1, symbol=sym, weeks=2, weekly_buffer=0.00075)

        results = run_backtest_step5(strategy, symbol_data_dict, backtest_config["initial_balance"])
        trades = results["Trades"]
        final_balance = results["final_balance"]
        stats = analyze_trades(trades, backtest_config["initial_balance"])

        print("\n--- MULTI-SYMBOL BACKTEST COMPLETE ---")
        print(f"Symbols: {list(symbol_data_dict.keys())}")
        print(f"Total Trades: {stats['count']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${stats['total_pnl']:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")

        minimal_strategy_config = {
            "params": {"min_touches": 3, "risk_reward": 2.0},
            "pair_settings": {
                "EURUSD": {"tolerance": 0.0005, "min_volume_threshold": 500, "min_bounce_volume": 400},
                "GBPUSD": {"tolerance": 0.0007, "min_volume_threshold": 600, "min_bounce_volume": 500},
            },
            "ftmo_limits": backtest_config["ftmo_limits"],
        }
        correlation_data = {"EURUSD": {"GBPUSD": corr}, "GBPUSD": {"EURUSD": corr}}
        ftmo_data = backtest_config["ftmo_limits"]
        monthly_data = {}
        monthly_levels = []
        weekly_levels = []

        with ReportWriter(backtest_config["report_path"]) as rw:
            rw.generate_full_report(
                strategy_config=minimal_strategy_config,
                df_test=symbol_data_dict["EURUSD"],  # example
                trades=trades,
                stats=stats,
                final_balance=final_balance,
                monthly_data=monthly_data,
                monthly_levels=monthly_levels,
                weekly_levels=weekly_levels,
                correlation_data=correlation_data,
                ftmo_data=ftmo_data,
            )
        print(f"\nReport generated: {backtest_config['report_path']}\n")

    else:
        # Single-Symbol
        default_symbol = next(iter(symbol_data_dict.keys()))
        df = symbol_data_dict[default_symbol]
        print(f"Single-symbol approach => {default_symbol}")

        train_df, test_df = split_data_for_backtest(df, 0.8)
        print(f"Train/Test split: {len(train_df)} / {len(test_df)}")

        test_start = pd.to_datetime(test_df['time'].min())
        test_end = pd.to_datetime(test_df['time'].max())
        h1_start = test_start - timedelta(days=backtest_config["sr_lookback_days"])
        df_h1 = check_h1_data_or_resample(default_symbol, h1_start, test_end, threshold=0.90)

        strategy = SR_Bounce_Strategy()
        if not df_h1.empty:
            strategy.update_weekly_levels(df_h1, symbol=default_symbol, weeks=2, weekly_buffer=0.00075)

        # Simple bounce detection on train set
        bounce_count = 0
        for i in range(len(train_df)):
            seg = train_df.iloc[: i + 1]
            sig = strategy.generate_signals(seg, symbol=default_symbol)
            if sig["type"] != "NONE":
                bounce_count += 1
        print(f"Training-set bounces detected: {bounce_count}")

        single_result = run_backtest(strategy, test_df, backtest_config["initial_balance"])
        sp_trades = single_result["Trades"]
        sp_final_balance = single_result["final_balance"]

        sp_stats = analyze_trades(sp_trades, backtest_config["initial_balance"])
        print("\n--- SINGLE-SYMBOL BACKTEST COMPLETE ---")
        print(f"Total Trades: {sp_stats['count']}")
        print(f"Win Rate: {sp_stats['win_rate']:.2f}%")
        print(f"Profit Factor: {sp_stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${sp_stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${sp_stats['total_pnl']:.2f}")
        print(f"Final Balance: ${sp_final_balance:.2f}")

        # (Optional) Generate report
        # with ReportWriter(backtest_config["report_path"]) as rw:
        #     ...
        # print(f"Single-symbol report generated: {backtest_config['report_path']}")


if __name__ == "__main__":
    main()
