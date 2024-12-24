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
def load_data(symbol="EURUSD", timeframe="H1", days=365) -> pd.DataFrame:
    mt5 = MT5Handler(debug=True)
    end_date = datetime.now()
    logging.debug(f"End Date: {end_date}")
    start_date = end_date - timedelta(days=days)
    logging.debug(f"Start Date: {start_date}")
    df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        return pd.DataFrame()
    return df.sort_values("time").reset_index(drop=True)


def validate_data_for_backtest(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No data loaded.")

    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data not valid: missing columns {missing_cols}")
    else:
        print("Validation passed: required columns found, no empty DataFrame.")


def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    # If no data, bail out
    if df.empty:
        return {"Trades": [], "final_balance": initial_balance}

    trades: list["SR_Bounce_Strategy.Trade"] = []
    logging.debug("Before trades debug line")
    logging.debug(f"trades: {trades}")
    logging.debug("After trades debug line")

    balance = initial_balance
    active_trade: Optional["SR_Bounce_Strategy.Trade"] = None

    for i in range(len(df)):
        # 1) current_segment is the data up to the i-th bar
        current_segment = df.iloc[: i + 1]

        # 2) Basic skip if too few bars to form signals
        if len(current_segment) < 5:
            continue

        # 3) If we have no open trade, attempt to open one
        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            logging.debug("Before open trade debug line")
            logging.debug(f"trades: {new_trade}")
            logging.debug("After open trades debug line")

            if new_trade:
                active_trade = new_trade
                trades.append(active_trade)
        else:
            # 4) If we do have an active trade, check for exit
            should_close, fill_price, pnl = strategy.exit_trade(current_segment, active_trade)
            if should_close:
                balance += pnl
                # Optionally update trade fields
                last_bar = current_segment.iloc[-1]
                active_trade.close_i = last_bar.name
                active_trade.close_time = last_bar["time"]
                active_trade.close_price = fill_price
                active_trade.pnl = pnl
                # Mark trade closed
                active_trade = None

    # 5) If still open at the end, close at final bar
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
        active_trade = None

    # Return final results
    return {
        "Trades": [t.to_dict() for t in trades],
        "final_balance": balance
    }


def check_training_bounces(strategy: SR_Bounce_Strategy, df_train: pd.DataFrame):
    """
    Print out any signals the strategy would generate on the training set,
    but do not actually open or manage trades. This is purely diagnostic.
    """
    print("\n--- Checking for bounces in the TRAINING SET ---")
    bounce_count = 0

    for i in range(len(df_train)):
        current_segment = df_train.iloc[: i + 1]
        sig = strategy.generate_signals(current_segment)
        if sig["type"] != "NONE":
            bar_time = current_segment.iloc[-1]["time"]
            print(f"Bounce found at index={i}, time={bar_time}, signal={sig}")
            bounce_count += 1

    print(f"Total bounces detected in training set: {bounce_count}")


##################################################
# 3) MAIN
##################################################
def main():
    symbol = "EURUSD"
    timeframe = "M15"
    days = 180

    # 1) Load data
    df = load_data(symbol, timeframe, days)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    # 2) Validate
    validate_data_for_backtest(df)

    # 4) Split train/test
    train_df, test_df = split_data_for_backtest(df, 0.8)
    print(f"Train size={len(train_df)} | Test size={len(test_df)}")

    # 5) Create strategy
    strategy = SR_Bounce_Strategy()

    df_h1 = load_data(symbol, "H1", 45)  # 45 days, or however many you prefer
    validate_data_for_backtest(df_h1)

    # 7) (Optional) If strategy has a method that populates self.valid_levels:
    if hasattr(strategy, "update_weekly_levels"):
        strategy.update_weekly_levels(df_h1, weeks=2, weekly_buffer=0.00075)
        print("Weekly levels found:", strategy.valid_levels)
        print(f"Number of valid levels found: {len(strategy.valid_levels)}")
        print(f"Valid levels: {strategy.valid_levels}")

    # 8) Diagnostic: Check signals in the training set
    check_training_bounces(strategy, train_df)

    # 9) run_backtest on test data
    backtest_result = run_backtest(strategy, test_df, initial_balance=10000.0)
    trades = backtest_result["Trades"]
    final_balance = backtest_result["final_balance"]

    # 10) analyze trades
    stats = analyze_trades(trades, 10000.0)

    # 11) Print summary
    print("\n--- BACKTEST COMPLETE (Console Summary) ---")
    print(f"Total Trades: {stats['count']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")

    print("\n--- MONTE CARLO (1000 shuffles) ---")

    # 12) Optionally write a markdown report
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"BACKTEST_REPORT_{symbol}_{now_str}.md")

    monthly_data = {}
    monthly_levels = []
    weekly_levels = []

    # If you have a real ReportWriter
    with ReportWriter(report_file) as writer:
        writer.write_data_overview(test_df)
        writer.write_trades_section(trades)
        writer.write_stats_section(stats, final_balance)
        writer.write_monthly_breakdown(monthly_data)
        writer.write_sr_levels(monthly_levels, weekly_levels)

    print(f"\nDetailed report written to: {report_file}")


if __name__ == "__main__":
    main()
