import logging
from typing import Any, Dict, Tuple
import pandas as pd

def get_trade_manager_logger(name="TradeManager"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class TradeManager:
    def __init__(self, risk_reward: float):
        self.risk_reward = risk_reward
        self.logger = get_trade_manager_logger()

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        """Calculate stop loss based on signal type and recent price action."""
        if df_segment.empty:
            return 0.0

        last_bar = df_segment.iloc[-1]
        close_price = float(last_bar['close'])
        low = float(last_bar['low'])
        high = float(last_bar['high'])

        pip_buffer = 0.0008

        if signal["type"] == "BUY":
            stop_loss = low - pip_buffer
            self.logger.debug(f"Buy signal => SL set below last low: {stop_loss:.5f}")
        else:  # SELL
            stop_loss = high + pip_buffer
            self.logger.debug(f"Sell signal => SL set above last high: {stop_loss:.5f}")

        return stop_loss

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """
        Basic 1% risk model:
        - risk_amount = 1% of balance
        - position_size = risk_amount / (stop_pips * pip_value)
        """
        if account_balance <= 0:
            self.logger.error(f"Invalid account_balance: {account_balance}")
            return 0.01
        if stop_distance <= 0:
            self.logger.error(f"Invalid stop_distance: {stop_distance}")
            return 0.01

        risk_amount = account_balance * 0.01
        stop_pips = stop_distance * 10000.0
        pip_value = 10.0  # For EURUSD in 1 standard lot

        position_size = risk_amount / (stop_pips * pip_value)
        position_size = round(position_size, 2)

        if position_size < 0.01:
            position_size = 0.01  # ensure min

        self.logger.info(f"Position size calculated: {position_size} lots (Balance={account_balance}, Stop={stop_distance:.5f})")
        return position_size

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        """
        If risk_reward=2.0, the distance from entry to SL is multiplied by 2
        for the TP distance.
        """
        dist = abs(entry_price - sl)
        if entry_price > sl:
            return entry_price + (dist * self.risk_reward)
        else:
            return entry_price - (dist * self.risk_reward)

    def check_exit_conditions(self, df_segment: pd.DataFrame, position: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if SL or TP is touched by the last bar's close.
        Return (should_close, reason).
        """
        if df_segment.empty:
            return False, "No data"

        last_bar = df_segment.iloc[-1]
        current_price = float(last_bar["close"])
        pos_type = position.get("type", "BUY")

        if pos_type == "BUY":
            if current_price <= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price >= position["take_profit"]:
                return True, "Take profit hit"
        else:  # SELL
            if current_price >= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price <= position["take_profit"]:
                return True, "Take profit hit"

        return False, "No exit condition met"
