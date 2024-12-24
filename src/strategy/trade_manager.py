from typing import Any, Dict, Tuple
import pandas as pd
from src.strategy.price_analysis import detect_price_hovering


class TradeManager:
    def __init__(self, risk_reward: float):
        self.risk_reward = risk_reward

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        last_bar = df_segment.iloc[-1]
        low_ = float(last_bar['low'])
        high_ = float(last_bar['high'])
        pip_buffer = 8 / 10000.0

        if signal["type"] == "BUY":
            return low_ - pip_buffer
        else:
            return high_ + pip_buffer

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        risk = account_balance * 0.01
        pip_value = 1.0
        stop_pips = stop_distance * 10000.0
        if stop_pips <= 0:
            return 0.01
        lots = risk / (stop_pips * pip_value)
        return round(lots, 2)

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        dist = abs(entry_price - sl)
        if entry_price > sl:
            return entry_price + dist * self.risk_reward
        else:
            return entry_price - dist * self.risk_reward

    def check_exit_conditions(
        self,
        df_segment: pd.DataFrame,
        position: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Checks if exit conditions are met for the given position.

        Returns:
            (should_close, reason):
                should_close (bool): True if we want to exit now
                reason (str): Explanation of which condition triggered the exit
        """
        last_bar = df_segment.iloc[-1]
        current_price = float(last_bar["close"])

        # Example: Basic stop loss & take profit checks
        if position["type"] == "BUY":
            if current_price <= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price >= position["take_profit"]:
                return True, "Take profit hit"
        else:  # SELL position
            if current_price >= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price <= position["take_profit"]:
                return True, "Take profit hit"

        # Example: Additional conditions
        # if some_rsi_condition:
        #     return True, "RSI exit"
        #
        # if detect_price_hovering(df_segment, position["level"]):
        #     return True, "Price hovering exit"

        # 2) Fallback:
        return False, "No exit condition met"

