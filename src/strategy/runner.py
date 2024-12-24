# file: src/strategy/runner.py

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy  # MUST have open_trade(...) & exit_trade(...)
from src.strategy.report_writer import ReportWriter
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy


##################################################
# 1) DATA LOADING & VALIDATION
##################################################
def load_data(symbol="EURUSD", timeframe="H1", days=365) -> pd.DataFrame:
    """
    Fetch historical data from MT5Handler, sort by time,
    return a DataFrame with columns: [time, open, high, low, close, tick_volume].
    """
    mt5 = MT5Handler(debug=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        return pd.DataFrame()
    return df.sort_values("time").reset_index(drop=True)


def validate_data_for_backtest(df: pd.DataFrame) -> None:
    """
    Raise ValueError if df is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("No data loaded.")

    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data not valid: missing columns {missing_cols}")
    else:
        print("Validation passed: required columns found, no empty DataFrame.")


def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into train and test sets by the given ratio.
    """
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


##################################################
# 2) BACKTEST LOGIC
##################################################
def run_backtest(
    strategy: SR_Bounce_Strategy,
    df: pd.DataFrame,
    initial_balance=10000.0
) -> Dict:
    """
    A step-by-step backtest loop:
      - For each bar, call `open_trade(...)` if no trade is active
      - If a trade is active, call `exit_trade(...)`
      - If exit triggered, we close the trade & update balance
      - If still active at the end, close at final bar
    Returns a dict with "Trades" list (serialized trade objects) and "final_balance".
    """

    # If no data, bail out
    if df.empty:
        return {"Trades": [], "final_balance": initial_balance}

    trades: list["SR_Bounce_Strategy.Trade"] = []
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


def analyze_trades(trades: List[Dict], initial_balance: float) -> Dict:
    """
    Compute basic performance metrics:
      - total trades, win_rate, profit_factor, max_drawdown, total_pnl
    """
    import numpy as np

    if not trades:
        return {
            "count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_win = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in losses))

    count = len(trades)
    win_rate = (len(wins) / count) * 100.0 if count else 0.0
    profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
    total_pnl = sum(t["pnl"] for t in trades)

    # Max drawdown
    running = initial_balance
    peak = running
    dd = 0.0
    for t in trades:
        running += t["pnl"]
        if running > peak:
            peak = running
        drawdown = peak - running
        if drawdown > dd:
            dd = drawdown

    return {
        "count": count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": dd,
        "total_pnl": total_pnl
    }


def run_monte_carlo(trades: List[Dict], initial_balance: float, iterations: int = 1000) -> Dict:
    """
    Shuffle trade outcomes many times, track final equity distribution.
    """
    import numpy as np

    if not trades:
        return {
            "avg_final": initial_balance,
            "worst_case": initial_balance,
            "best_case": initial_balance
        }

    results = []
    for _ in range(iterations):
        # Shuffle trades
        shuffled = trades.copy()
        random.shuffle(shuffled)
        eq = initial_balance
        peak = eq
        dd = 0.0
        for t in shuffled:
            eq += t["pnl"]
            if eq > peak:
                peak = eq
            ddown = peak - eq
            if ddown > dd:
                dd = ddown
        results.append(eq)

    return {
        "avg_final": float(np.mean(results)),
        "worst_case": float(np.min(results)),
        "best_case": float(np.max(results))
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
    """
    Main entry point for backtesting.
    - Load data
    - Validate
    - Optionally build volume validator
    - Split train/test
    - Create SR_Bounce_Strategy
    - Possibly do "analyze_higher_timeframe_levels" or other steps
    - Check signals in training
    - run_backtest on test
    - analyze results
    - write a report
    """

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
    strategy = SR_Bounce_Strategy(
        config_file="config/my_bounce_config.json",
        news_file="config/market_news.json"
    )

    # 7) (Optional) If strategy has a method that populates self.valid_levels:
    if hasattr(strategy, "analyze_higher_timeframe_levels"):
        strategy.analyze_higher_timeframe_levels(train_df)
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
    mc_results = run_monte_carlo(trades, 10000.0)

    # 11) Print summary
    print("\n--- BACKTEST COMPLETE (Console Summary) ---")
    print(f"Total Trades: {stats['count']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")

    print("\n--- MONTE CARLO (1000 shuffles) ---")
    print(f"Average Final: {mc_results['avg_final']:.2f}")
    print(f"Worst Case: {mc_results['worst_case']:.2f}")
    print(f"Best Case: {mc_results['best_case']:.2f}")

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
        writer.write_monte_carlo_section(mc_results)
        writer.write_monthly_breakdown(monthly_data)
        writer.write_sr_levels(monthly_levels, weekly_levels)

    print(f"\nDetailed report written to: {report_file}")


if __name__ == "__main__":
    main()
