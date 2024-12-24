import logging
import json
from typing import List, Optional, Dict, Any

import pandas as pd

# We keep is_volume_sufficient if you still want that function from volume_analysis:
from src.strategy.volume_analysis import is_volume_sufficient

from src.strategy.news_validator import NewsValidator
from src.strategy.signal_generator import SignalGenerator
from src.strategy.trade_manager import TradeManager
from src.strategy.level_analysis import identify_yearly_extremes, merge_levels_to_zones


# Functions for charting trading zones for S/R zones
import pandas as pd
from typing import List

def identify_sr_yearly(df_daily: pd.DataFrame, buffer_pips: float = 0.003) -> List[float]:
        if df_daily.empty:
            return []

        # The absolute extremes from your daily data
        yearly_high = df_daily["high"].max()
        yearly_low = df_daily["low"].min()

        # Subtract buffer from the top, add buffer to the bottom
        # so you avoid placing lines exactly on the absolute extremes
        adjusted_high = yearly_high - buffer_pips
        adjusted_low = yearly_low + buffer_pips

        # If the buffer is too large (makes them cross over), just return the raw extremes
        if adjusted_low >= adjusted_high:
            # fallback: if the difference between high and low is smaller than 2 * buffer
            return [yearly_low, yearly_high]

        return [adjusted_low, adjusted_high]

def identify_sr_monthly(
    df_h1: pd.DataFrame,
    months: int = 2,
    monthly_buffer: float = 0.0015
) -> List[float]:
    """
    Identifies a broad monthly support & resistance from H1 data
    over the last X 'months' (~30.44 days each).

    Returns [monthly_support, monthly_resistance].
    """
    if df_h1.empty:
        return []

    # 1) Figure out the cutoff date/time (X months back from the last bar).
    last_time = pd.to_datetime(df_h1["time"].max())
    approx_days = int(30.44 * months)
    cutoff_time = last_time - pd.Timedelta(days=approx_days)

    # 2) Filter the dataset
    recent_df = df_h1[df_h1["time"] >= cutoff_time]
    if recent_df.empty:
        return []

    # 3) Highest & lowest in this range
    highest = recent_df["high"].max()
    lowest  = recent_df["low"].min()

    # 4) Apply monthly buffer: push the top down & bottom up by monthly_buffer
    monthly_res  = highest - monthly_buffer
    monthly_supp = lowest  + monthly_buffer

    return [float(monthly_supp), float(monthly_res)]

def identify_sr_weekly(df_h1: pd.DataFrame, weeks: int = 2, weekly_buffer: float = 0.00075) -> list[float]:
    if df_h1.empty:
        return []

    # 1) Find the most recent timestamp in df_h1
    last_time = pd.to_datetime(df_h1["time"].max())

    # 2) Convert weeks -> days, e.g. 2 weeks -> 14 days
    total_days = weeks * 7
    cutoff_time = last_time - pd.Timedelta(days=total_days)

    # 3) Filter rows to only those after cutoff_time
    recent_df = df_h1[df_h1["time"] >= cutoff_time.strftime("%Y-%m-%d %H:%M:%S")]
    if recent_df.empty:
        return []

    # 4) Highest high and lowest low in the recent subset
    highest = float(recent_df["high"].max())
    lowest  = float(recent_df["low"].min())

    # 5) Subtract a small buffer from the highest, add to the lowest
    weekly_res = highest - weekly_buffer
    weekly_supp = lowest + weekly_buffer

    # Return in ascending order: support first, then resistance
    return [weekly_supp, weekly_res]


class SR_Bounce_Strategy:
    """
    Shows an advanced S/R bounce approach with:
      - 2-bounce volume logic
      - (Optional) momentum filter (RSI + ADX)
      - (Optional) news avoidance
      - dynamic exit conditions
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        logger: logging.Logger = None,
        news_file: str = "config/market_news.json"
    ):
        # 1) Default params (can be overwritten if config_file is provided)
        self.params = {
            "min_touches": 8,
            "min_volume_threshold": 380000.0,
            "margin_pips": 0.0030,
            "session_filter": True,
            "session_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15],
            "risk_reward": 2.0,
            "lookforward_minutes": 30,
            "use_momentum_filter": True
        }
        if config_file:
            self._load_config(config_file)

        # 2) Setup logger
        self.logger = logger or self._create_default_logger()

        # 3) News Validator (if you still want it)
        self.news_validator = NewsValidator(
            news_file=news_file,
            lookforward_minutes=self.params["lookforward_minutes"]
        )

        # 4) Data structures for levels, etc.
        self.valid_levels = []
        self.avg_atr = 0.0005  # Only if you use ATR; otherwise ignore.

        # 5) Initialize SignalGenerator (notice we do NOT pass any volume_validator here)
        self.signal_generator = SignalGenerator(
            volume_validator=None,  # or remove param entirely if your SignalGenerator can handle that
            news_validator=self.news_validator,
            valid_levels=self.valid_levels,
            params=self.params,
            log_file="results/signals_debug.log"
        )

        # 6) Initialize TradeManager
        self.trade_manager = TradeManager(self.params["risk_reward"])

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

    # -------------------------------------------------------
    # Example pass-through to your signal generator
    # -------------------------------------------------------
    def update_weekly_levels(self, df_h1, weeks: int = 2, weekly_buffer: float = 0.00075):
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

    # Example exit_trade or open_trade if you still need them:
    def exit_trade(self, df_segment, trade: "SR_Bounce_Strategy.Trade"):
        # ...
        return (False, 0.0, 0.0)

    def close_trade(trade: "SR_Bounce_Strategy.Trade", bar, close_price: float, pnl: float) -> None:
        # ...
        pass

    def open_trade(strategy: "SR_Bounce_Strategy", current_segment, balance: float, i: int):
        # ...
        pass
