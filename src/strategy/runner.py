import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.data_validator import DataValidator
from src.strategy.report_writer import ReportWriter

# Data Loading
def load_data(symbol="EURUSD", timeframe="H1", days=365) -> pd.DataFrame:
    mt5 = MT5Handler(debug=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        return pd.DataFrame()
    return df.sort_values("time").reset_index(drop=True)

# Data splitting
def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test

# Trade simulation backtesting
class Trade:
    def __init__(self, open_i: int, open_time: str, type: str, entry_price: float,
                 sl: float, tp: float, size: float):
        self.open_i = open_i
        self.open_time = open_time
        self.type = type
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.size = size
        self.close_i = None
        self.close_time = None
        self.close_price = None
        self.pnl = 0.0
        self.entry_volume = 0.0
        self.prev_3_avg_volume = 0.0
        self.hour_avg_volume = 0.0
        # New fields for S/R analysis
        self.level = 0.0
        self.distance_to_level = 0.0
        self.level_type = ""  # "Support" or "Resistance"

    def to_dict(self) -> Dict:
        return {
            "open_i": self.open_i,
            "open_time": self.open_time,
            "type": self.type,
            "entry_price": self.entry_price,
            "sl": self.sl,
            "tp": self.tp,
            "size": self.size,
            "close_i": self.close_i,
            "close_time": self.close_time,
            "close_price": self.close_price,
            "pnl": self.pnl,
            "entry_volume": self.entry_volume,
            "prev_3_avg_volume": self.prev_3_avg_volume,
            "hour_avg_volume": self.hour_avg_volume,
            "level": self.level,
            "distance_to_level": self.distance_to_level,
            "level_type": self.level_type
        }

def check_trade_exit(
    strategy: SR_Bounce_Strategy,
    df_segment: pd.DataFrame,
    trade: Trade
) -> Tuple[bool, float, float]:

    position_dict = {
        "type": trade.type,          # 'BUY' or 'SELL'
        "stop_loss": trade.sl,       # or "sl"
        "take_profit": trade.tp,     # or "tp"
        "size": trade.size           # optional if your exit logic depends on it
    }

    should_close, reason = strategy.trade_manager.check_exit_conditions(df_segment, position_dict)
    if not should_close:
        # No exit condition is met, so do nothing.
        return (False, 0.0, 0.0)

    last_bar = df_segment.iloc[-1]
    hi = float(last_bar["high"])
    lo = float(last_bar["low"])

    close_price = float(last_bar["close"])  # Default to bar’s close if not SL/TP
    pnl = 0.0

    if trade.type == "BUY":
        # If the bar’s low is below SL, we fill at SL first.
        if lo <= trade.sl:
            close_price = trade.sl
            pnl = (close_price - trade.entry_price) * 10000.0 * trade.size
        # Else if the bar’s high is above TP, fill at TP.
        elif hi >= trade.tp:
            close_price = trade.tp
            pnl = (close_price - trade.entry_price) * 10000.0 * trade.size
        else:
            # If no exact intrabar hit, close on bar’s close (or however you wish).
            pnl = (close_price - trade.entry_price) * 10000.0 * trade.size

    else:  # SELL
        # If the bar’s high is above SL, fill at SL.
        if hi >= trade.sl:
            close_price = trade.sl
            pnl = (trade.entry_price - close_price) * 10000.0 * trade.size
        # Else if the bar’s low is below TP, fill at TP.
        elif lo <= trade.tp:
            close_price = trade.tp
            pnl = (trade.entry_price - close_price) * 10000.0 * trade.size
        else:
            pnl = (trade.entry_price - close_price) * 10000.0 * trade.size

    # 5) Return the same tuple that runner.py expects.
    return (True, close_price, pnl)

def close_trade(trade: Trade, bar: pd.Series, close_price: float, pnl: float) -> None:
    """Update trade with closing details"""
    trade.close_i = bar.name
    trade.close_time = bar["time"]
    trade.close_price = close_price
    trade.pnl = pnl

def open_trade(strategy: 'SR_Bounce_Strategy', current_segment: pd.DataFrame,
               balance: float, i: int) -> Optional[Trade]:
    """Try to open new trade based on strategy signals"""
    sig = strategy.generate_signals(current_segment)
    if not strategy.validate_signal(sig, current_segment):
        return None

    last_bar = current_segment.iloc[-1]
    entry_price = float(last_bar["close"])
    stop_loss = strategy.calculate_stop_loss(sig, current_segment)
    dist = abs(entry_price - stop_loss)
    position_size = strategy.calculate_position_size(balance, dist)
    take_profit = strategy.calculate_take_profit(entry_price, stop_loss)

    trade = Trade(
        open_i=i,
        open_time=last_bar["time"],
        type=sig["type"],
        entry_price=entry_price,
        sl=stop_loss,
        tp=take_profit,
        size=position_size
    )

    # Add volume data
    trade.entry_volume = float(last_bar["tick_volume"])
    trade.prev_3_avg_volume = float(current_segment["tick_volume"].tail(4).head(3).mean())
    trade.hour_avg_volume = float(strategy.volume_validator.avg_volume_by_hour.get(pd.to_datetime(last_bar["time"]).hour, 0))

    # Add S/R level data
    trade.level = sig.get("level", 0.0)
    if trade.level > 0:  # Only calculate if we have a valid level
        trade.distance_to_level = abs(entry_price - trade.level)
        trade.level_type = "Resistance" if sig["type"] == "SELL" else "Support"

    return trade

def run_backtest(strategy: 'SR_Bounce_Strategy', df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    """Main backtest function that coordinates the overall process"""
    if df.empty:
        return {"Trades": [], "final_balance": initial_balance}

    trades: List[Trade] = []
    balance = initial_balance
    active_trade: Optional[Trade] = None

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]
        if len(current_segment) < 5:
            continue

        if active_trade is None:
            new_trade = open_trade(strategy, current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                trades.append(active_trade)
        else:
            # We have an active trade, let's ask the strategy if we should close
            position_dict = {
                "type": active_trade.type,
                "stop_loss": active_trade.sl,
                "take_profit": active_trade.tp,
                # If your exit logic also needs the level, volume, RSI, etc., pass it here
                "level": 0.0,  # or however you store it in Trade
            }

            should_close, reason = strategy.check_exit(current_segment, position_dict)
            if should_close:
                # Decide on the close price. Maybe you use the bar’s close:
                last_bar = current_segment.iloc[-1]
                close_price = float(last_bar["close"])

                # Calculate PnL the same way you did before
                if active_trade.type == "BUY":
                    pnl = (close_price - active_trade.entry_price) * 10000.0 * active_trade.size
                else:
                    pnl = (active_trade.entry_price - close_price) * 10000.0 * active_trade.size

                close_trade(active_trade, last_bar, close_price, pnl)
                balance += pnl
                active_trade = None

    # Close any remaining open trade at the last price
    if active_trade:
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])

        if active_trade.type == "BUY":
            pnl = (last_close - active_trade.entry_price) * 10000.0 * active_trade.size
        else:
            pnl = (active_trade.entry_price - last_close) * 10000.0 * active_trade.size

        close_trade(active_trade, last_bar, last_close, pnl)
        balance += pnl

    return {
        "Trades": [trade.to_dict() for trade in trades],
        "final_balance": balance
    }


def analyze_trades(trades: List[Dict], initial_balance: float) -> Dict:
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
    win_rate = (len(wins)/count)*100.0 if count else 0.0
    profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
    total_pnl = sum(t["pnl"] for t in trades)

    # max drawdown
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
    if not trades:
        return {
            "avg_final": initial_balance,
            "worst_case": initial_balance,
            "best_case": initial_balance
        }

    results = []
    for _ in range(iterations):
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

def check_training_bounces(strategy, df_train: pd.DataFrame):
    """
    Runs the signal-generation logic on the training set
    to see if any bounce signals appear there.
    We won't actually 'open' trades, just log the signals.
    """
    print("\n--- Checking for bounces in the TRAINING SET ---")
    bounce_count = 0

    for i in range(len(df_train)):
        # We mimic the backtest approach: pass all bars up to index i.
        current_segment = df_train.iloc[: i + 1]

        # Generate potential signal for the current bar.
        sig = strategy.generate_signals(current_segment)

        # If we get a real signal (i.e. type != NONE), log it.
        if sig["type"] != "NONE":
            bar_time = current_segment.iloc[-1]["time"]
            print(f"Bounce found at index={i}, time={bar_time}, signal={sig}")
            bounce_count += 1

    print(f"Total bounces detected in training set: {bounce_count}")

def main():
    symbol = "EURUSD"
    timeframe = "M15"
    days = 180

    df = load_data(symbol, timeframe=timeframe, days=days)
    if df.empty:
        print("No data loaded.")
        return

    # 1) Validate basic columns
    validator = DataValidator(["time","open","high","low","close","tick_volume"])
    ok, msg = validator.validate_data(df)
    if not ok:
        print(f"Data not valid: {msg}")
        return

    # 2) BUILD AND PREP VOLUME VALIDATOR
    from src.strategy.volume_analysis import VolumeValidator
    volume_validator = VolumeValidator(
        expansion_factor=1.2,   # require 20% higher volume than recent avg
        lookback_bars=3,       # compare to the last 3 bars
        time_adjustment=True    # normalize by hour-of-day
    )
    volume_validator.update_avg_volume_by_hour(df)

    # 3) Now proceed with train/test split
    train_df, test_df = split_data_for_backtest(df, 0.8)
    print(f"Train size={len(train_df)}  Test size={len(test_df)}")

    # 4) Create the strategy
    strategy = SR_Bounce_Strategy(
        config_file="config/my_bounce_config.json",
        news_file="config/market_news.json"
    )

    # 5) Pass the volume_validator to strategy so it can reference the new method
    strategy.volume_validator = volume_validator

    strategy.analyze_higher_timeframe_levels(train_df)
    print(f"Number of valid levels found: {len(strategy.valid_levels)}")
    print(f"Valid levels: {strategy.valid_levels}")

    # Call our debug function to see if the strategy generates any signals in training
    check_training_bounces(strategy, train_df)

    result = run_backtest(strategy, test_df, initial_balance=10000.0)
    trades = result["Trades"]
    final_balance = result["final_balance"]

    stats = analyze_trades(trades, 10000.0)
    mc = run_monte_carlo(trades, 10000.0)

    # Print short console summary
    print("\n--- BACKTEST COMPLETE (Console Summary) ---")
    print(f"Total Trades: {stats['count']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")

    print("\n--- MONTE CARLO (1000 shuffles) ---")
    print(f"Average Final: {mc['avg_final']:.2f}")
    print(f"Worst Case: {mc['worst_case']:.2f}")
    print(f"Best Case: {mc['best_case']:.2f}")

    # Prepare report filename
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"BACKTEST_REPORT_{symbol}_{now_str}.md")

    # -- Here's the integration with ReportWriter:
    # If you have extra data like monthly_data, monthly_levels, weekly_levels,
    # gather or compute them here. If you don’t have them, just pass empty structures.

    monthly_data = {}       # or however you compute monthly stats
    monthly_levels = []     # or however you gather monthly S/R
    weekly_levels = []      # or weekly S/R

    with ReportWriter(report_file) as writer:
        # Option 1: Call the step-by-step writing methods individually:
        writer.write_data_overview(test_df)
        writer.write_trades_section(trades)
        writer.write_stats_section(stats, final_balance)
        writer.write_monte_carlo_section(mc)
        writer.write_monthly_breakdown(monthly_data)
        writer.write_sr_levels(monthly_levels, weekly_levels)

        # Option 2: Or call the built-in "generate_full_report" if you want a single call:
        # writer.generate_full_report(
        #     df_test=test_df,
        #     trades=trades,
        #     stats=stats,
        #     mc_results=mc,
        #     final_balance=final_balance,
        #     monthly_data=monthly_data,
        #     monthly_levels=monthly_levels,
        #     weekly_levels=weekly_levels
        # )

    print(f"\nDetailed report written to: {report_file}")


if __name__ == "__main__":
    main()
