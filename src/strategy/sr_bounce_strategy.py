import logging
from typing import Tuple
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.strategy.ftmo_risk_manager import FTMORiskManager

import pandas as pd

from src.strategy.signal_generator import SignalGenerator
from src.strategy.trade_manager import TradeManager


def get_strategy_logger(name="SR_Bounce_Strategy"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class SR_Bounce_Strategy:
    def __init__(
        self,
        config_file: Optional[str] = None,
        logger: logging.Logger = None,
        news_file: str = "config/market_news.json"
    ):
        # Default params
        self.params = {
            "min_touches": 8,
            "min_volume_threshold": 1500,
            "margin_pips": 0.0030,
            "risk_reward": 2.0,          # Use 2.0 consistently
            "lookforward_minutes": 30,
        }

        # Load config if specified
        if config_file:
            self._load_config(config_file)

        # Setup logger
        self.logger = logger or get_strategy_logger()

        # Data structures
        self.valid_levels = []
        self.avg_atr = 0.0005  # If used for something later

        # Initialize SignalGenerator
        self.signal_generator = SignalGenerator(
            valid_levels=self.valid_levels,
            params=self.params,
            log_file="results/signals_debug.log"
        )

        # Initialize TradeManager with same risk_reward
        self.trade_manager = TradeManager(risk_reward=self.params["risk_reward"])

        self.risk_manager = FTMORiskManager()
        self.daily_pnl = 0.0


    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            self.params.update(user_cfg)
        except Exception as e:
            print(f"[WARNING] Unable to load {config_file}: {e}")


    def identify_sr_weekly(
        self,
        df_h1: pd.DataFrame,
        weeks: int = 12,
        chunk_size: int = 24,
        weekly_buffer: float = 0.0003
    ) -> List[float]:
        """
        Identify significant S/R levels from H1 data over the last `weeks` weeks.
        """
        try:
            if df_h1.empty:
                self.logger.error("Empty dataframe in identify_sr_weekly.")
                return []

            # Filter to last `weeks` weeks
            last_time = pd.to_datetime(df_h1["time"].max())
            cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
            recent_df = df_h1[df_h1["time"] >= cutoff_time].copy()
            recent_df.sort_values("time", inplace=True)

            self.logger.info(f"Analyzing data from {recent_df['time'].min()} to {recent_df['time'].max()}")

            if recent_df.empty:
                self.logger.warning("No data after filtering for recent weeks.")
                return []

            # Calculate average volume for significance
            avg_volume = recent_df['tick_volume'].mean()
            volume_threshold = avg_volume * 1.5

            potential_levels = []
            # Slide window of chunk_size bars
            for i in range(0, len(recent_df), chunk_size):
                window = recent_df.iloc[i:i + chunk_size]
                if len(window) < chunk_size / 2:  # skip small windows
                    continue

                # High & Low
                high = float(window['high'].max())
                low = float(window['low'].min())

                high_volume = float(window.loc[window['high'] == high, 'tick_volume'].max())
                low_volume = float(window.loc[window['low'] == low, 'tick_volume'].max())

                if high_volume > volume_threshold:
                    potential_levels.append(high)
                    self.logger.debug(f"High level found at {high:.5f} with volume {high_volume}")

                if low_volume > volume_threshold:
                    potential_levels.append(low)
                    self.logger.debug(f"Low level found at {low:.5f} with volume {low_volume}")

            # Sort & merge nearby
            potential_levels = sorted(set(potential_levels))
            merged_levels = []
            for lvl in potential_levels:
                if not merged_levels or abs(lvl - merged_levels[-1]) > weekly_buffer:
                    merged_levels.append(lvl)
                else:
                    merged_levels[-1] = (merged_levels[-1] + lvl) / 2.0

            self.logger.info(f"Identified {len(merged_levels)} valid S/R levels")
            return merged_levels

        except Exception as e:
            self.logger.error(f"Error in identify_sr_weekly: {str(e)}")
            return []


    def update_weekly_levels(self, df_h1, weeks: int = 2, weekly_buffer: float = 0.00075):
        """
        Update the strategy's valid levels using weekly S/R from identify_sr_weekly.
        """
        try:
            w_levels = self.identify_sr_weekly(
                df_h1,
                weeks=weeks,
                weekly_buffer=weekly_buffer
            )
            if not w_levels:
                self.logger.warning("No weekly levels found.")
                return

            self.valid_levels = w_levels
            self.logger.info(f"Updated valid levels. Total: {len(self.valid_levels)}")

            # Update signal generator's levels
            self.signal_generator.valid_levels = self.valid_levels

        except Exception as e:
            self.logger.error(f"Error updating weekly levels: {str(e)}")


    def generate_signals(self, df_segment):
        return self.signal_generator.generate_signal(df_segment)

    def validate_signal(self, signal, df_segment):
        return self.signal_generator.validate_signal(signal, df_segment)


    # --- Trade Management Wrappers ---
    def calculate_stop_loss(self, signal, df_segment) -> float:
        return self.trade_manager.calculate_stop_loss(signal, df_segment)

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        return self.trade_manager.calculate_position_size(account_balance, stop_distance)

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        return self.trade_manager.calculate_take_profit(entry_price, sl)

    def check_exit(self, df_segment, position):
        return self.trade_manager.check_exit_conditions(df_segment, position)


    def open_trade(self, current_segment, balance: float, i: int) -> Optional["SR_Bounce_Strategy.Trade"]:
        """
        Enhanced trade opening with FTMO safety checks
        """
        # Get current market conditions
        last_bar = current_segment.iloc[-1]
        current_time = pd.to_datetime(last_bar['time'])
        bar_range = float(last_bar['high']) - float(last_bar['low'])
        current_spread = bar_range * 0.1

        # FTMO validation
        can_trade, reason = self.risk_manager.can_open_trade(
            current_time=current_time,
            spread=current_spread,
            daily_pnl=self.daily_pnl
        )

        if not can_trade:
            self.logger.debug(f"FTMO check failed: {reason}")
            return None

        # Continue with regular strategy
        signal = self.generate_signals(current_segment)
        if signal["type"] == "NONE":
            return None

        # Volume check
        if last_bar["tick_volume"] < self.params["min_volume_threshold"]:
            return None  # skip if volume too low

        entry_price = float(last_bar["close"])
        stop_loss = self.calculate_stop_loss(signal, current_segment)

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.00001:
            return None

        size = self.calculate_position_size(balance, stop_distance)
        take_profit = self.calculate_take_profit(entry_price, stop_loss)

        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=size
        )

        # Additional fields
        new_trade.level = signal.get('level', 0.0)
        new_trade.level_type = "Support" if signal["type"] == "BUY" else "Resistance"
        new_trade.distance_to_level = abs(entry_price - signal.get('level', entry_price))
        new_trade.entry_volume = float(last_bar['tick_volume'])
        new_trade.prev_3_avg_volume = float(current_segment['tick_volume'].tail(3).mean())
        new_trade.hour_avg_volume = float(current_segment['tick_volume'].tail(4).mean())

        # Update FTMO tracking
        self.risk_manager.update_trade_history({
            'time': new_trade.open_time,
            'type': new_trade.type,
            'size': new_trade.size
        })

        self.logger.debug(f"Opening trade: {signal['type']} at {entry_price:.5f}, level={new_trade.level}")
        return new_trade

    def exit_trade(self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade") -> Tuple[bool, float, float]:
        """
        Enhanced exit with FTMO tracking
        """
        position_dict = {
            "type": trade.type,
            "stop_loss": trade.sl,
            "take_profit": trade.tp
        }
        should_close, reason = self.trade_manager.check_exit_conditions(df_segment, position_dict)

        if should_close:
            last_bar = df_segment.iloc[-1]
            if reason == "Stop loss hit":
                fill_price = trade.sl
            elif reason == "Take profit hit":
                fill_price = trade.tp
            else:
                fill_price = float(last_bar["close"])

            if trade.type == "BUY":
                pnl = (fill_price - trade.entry_price) * 10000.0 * trade.size
            else:  # SELL
                pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size

            # Update daily PnL tracking for FTMO
            self.daily_pnl += pnl

            return True, fill_price, pnl

        return False, 0.0, 0.0


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
