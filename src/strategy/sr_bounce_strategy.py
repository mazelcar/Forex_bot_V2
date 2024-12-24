import logging
import json
from typing import List, Optional, Dict, Any

import pandas as pd

# We keep is_volume_sufficient if you still want that function from volume_analysis:
from src.strategy.signal_generator import SignalGenerator
from src.strategy.trade_manager import TradeManager
from typing import List

def identify_sr_weekly(
    df_h1: pd.DataFrame,
    weeks: int = 2,
    chunk_size: int = 24,
    weekly_buffer: float = 0.0003
) -> List[float]:
    """
    Gathers multiple S/R lines from H1 data spanning the last `weeks` weeks.
    1) Filter to the last X weeks of data.
    2) Chunk into windows of `chunk_size` bars (e.g. 24 bars ~ 1 day).
    3) Record max/min from each chunk as potential S/R.
    4) (Optional) merge nearby lines within `pip_threshold`.
    5) Return sorted list of lines (lowest -> highest).
    """

    if df_h1.empty:
        return []

    # 1) Filter to last `weeks` weeks
    last_time = pd.to_datetime(df_h1["time"].max())
    cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
    recent_df = df_h1[df_h1["time"] >= cutoff_time]
    if recent_df.empty:
        return []

    # 2) Gather potential levels by chunking
    potential_levels = []
    end_index = len(recent_df) - chunk_size
    if end_index < 0:
        # If we don't even have chunk_size bars, just take one max/min
        hi = recent_df["high"].max()
        lo = recent_df["low"].min()
        return [lo, hi]

    for i in range(end_index + 1):
        window = recent_df.iloc[i : i + chunk_size]
        hi = window["high"].max()
        lo = window["low"].min()
        potential_levels.append(hi)
        potential_levels.append(lo)

    # 3) Remove duplicates & sort
    potential_levels = sorted(set(potential_levels))

    # 4) (Optional) Merge lines that are very close (within `pip_threshold`)
    merged_levels = []
    for lvl in potential_levels:
        if not merged_levels:
            merged_levels.append(lvl)
            continue
        prev = merged_levels[-1]
        # If they're within pip_threshold, replace the last with midpoint
        if abs(lvl - prev) <= weekly_buffer:
            midpoint = (lvl + prev) / 2.0
            merged_levels[-1] = midpoint
        else:
            merged_levels.append(lvl)

    return merged_levels


class SR_Bounce_Strategy:

    def __init__(
        self,
        config_file: Optional[str] = None,
        logger: logging.Logger = None,
        news_file: str = "config/market_news.json"
    ):
        # 1) Default params (can be overwritten if config_file is provided)
        self.params = {
            "min_touches": 8,
            "min_volume_threshold": 1500,
            "margin_pips": 0.0030,
            "risk_reward": 2.0,
            "lookforward_minutes": 30,
        }
        if config_file:
            self._load_config(config_file)

        # 2) Setup logger
        self.logger = logger or self._create_default_logger()

        # 4) Data structures for levels, etc.
        self.valid_levels = []
        self.avg_atr = 0.0005  # Only if you use ATR; otherwise ignore.

        # 5) Initialize SignalGenerator (notice we do NOT pass any volume_validator here)
        self.signal_generator = SignalGenerator(
            valid_levels=self.valid_levels,
            params=self.params,
            log_file="results/signals_debug.log"
        )

        # 6) Initialize TradeManager
        self.trade_manager = TradeManager(risk_reward=1.2)

    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            self.params.update(user_cfg)
        except Exception as e:
            print(f"[WARNING] Unable to load {config_file}: {e}")

    def _create_default_logger(self) -> logging.Logger:
        logger = logging.getLogger("SR_Bounce_Strategy")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger


    def update_weekly_levels(self, df_h1, weeks: int = 2, weekly_buffer: float = 0.00075):
        w_levels = identify_sr_weekly(df_h1, weeks=weeks, weekly_buffer=weekly_buffer)
        """
        Identify weekly S/R from the last 'weeks' of H1 data, apply a small buffer,
        and merge into self.valid_levels.
        """
        w_levels = identify_sr_weekly(df_h1, weeks=weeks, weekly_buffer=weekly_buffer)
        # w_levels is like [weekly_support, weekly_resistance]

        if not w_levels:
            self.logger.warning("No weekly levels found. (Empty data?)")
            return

        # Merge them into self.valid_levels. For simplicity, just extend and sort:
        self.valid_levels.extend(w_levels)
        self.valid_levels = sorted(set(self.valid_levels))  # remove duplicates, sort
        self.logger.info(f"Weekly levels merged: {w_levels}  Now total: {len(self.valid_levels)}")

    def generate_signals(self, df_segment):
        return self.signal_generator.generate_signal(df_segment)

    def validate_signal(self, signal, df_segment):
        return self.signal_generator.validate_signal(signal, df_segment)

    # -------------------------------------------------------
    # Trade Management
    # -------------------------------------------------------
    def calculate_stop_loss(self, signal, df_segment) -> float:
        return self.trade_manager.calculate_stop_loss(signal, df_segment)

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        return self.trade_manager.calculate_position_size(account_balance, stop_distance)

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        return self.trade_manager.calculate_take_profit(entry_price, sl)

    def check_exit(self, df_segment, position):
        return self.trade_manager.check_exit_conditions(df_segment, position)

    def open_trade(strategy: "SR_Bounce_Strategy", current_segment, balance: float, i: int):
        signal = strategy.generate_signals(current_segment)
        if signal["type"] == "NONE":
            return None

        # Possibly validate volume, session, momentum, etc.
        last_bar = current_segment.iloc[-1]
        if last_bar["tick_volume"] < strategy.params["min_volume_threshold"]:
            return None  # no trade if volume is too low

        # 1) Define an entry price (e.g., last bar's close).
        #    If you prefer using the bar's open or the S/R level, adjust here.
        entry_price = float(last_bar["close"])

        # 2) Calculate stop loss using your TradeManager
        stop_loss = strategy.calculate_stop_loss(signal, current_segment)

        # 3) Calculate the distance between entry & stop
        stop_distance = abs(entry_price - stop_loss)

        # 4) Calculate the position size
        size = strategy.calculate_position_size(balance, stop_distance)

        # 5) Calculate take profit
        take_profit = strategy.calculate_take_profit(entry_price, stop_loss)

        # 6) Now build the trade object
        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=size
        )
        return new_trade


    # -------------------------------------------------------
    # For completeness, your Trade inner-class or open_trade, etc.
    # -------------------------------------------------------
    class Trade:
        def __init__(
            self,
            open_i: int,
            open_time: str,
            type: str,
            entry_price: float,
            sl: float,
            tp: float,
            size: float
        ):
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

            self.level = 0.0
            self.distance_to_level = 0.0
            self.level_type = ""

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
                "level_type": self.level_type,
            }




    def exit_trade(self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade"):
        """
        1) Check if SL or TP is triggered via TradeManager
        2) If so, compute fill_price & PnL
        3) Return (should_close, fill_price, pnl)
        """
        # 1) Build a position dict that TradeManager expects
        position_dict = {
            "type": trade.type,
            "stop_loss": trade.sl,
            "take_profit": trade.tp
            # you could add "level" or other fields if needed
        }

        # 2) Check conditions
        should_close, reason = self.trade_manager.check_exit_conditions(df_segment, position_dict)

        if should_close:
            # 3) Determine fill_price based on reason
            # If the reason is "Stop loss hit" or "Take profit hit,"
            #   we assume the fill price = SL or TP.
            #   Otherwise, you could default to the last bar's close.
            if reason == "Stop loss hit":
                fill_price = trade.sl
            elif reason == "Take profit hit":
                fill_price = trade.tp
            else:
                # fallback: last bar's close
                last_bar = df_segment.iloc[-1]
                fill_price = float(last_bar["close"])

            # 4) Compute PnL based on BUY/SELL
            if trade.type == "BUY":
                pnl = (fill_price - trade.entry_price) * 10000.0 * trade.size
            else:  # SELL
                pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size

            return (True, fill_price, pnl)

        # If no exit condition met, do nothing
        return (False, 0.0, 0.0)

    def close_trade(trade: "SR_Bounce_Strategy.Trade", bar, close_price: float, pnl: float) -> None:
        # ...
        pass

