import logging
from typing import Any, Dict, Tuple
import pandas as pd
from src.strategy.price_analysis import detect_price_hovering


# trade_manager.py
from typing import Any, Dict, Tuple
import pandas as pd
import logging

class TradeManager:
    def __init__(self, risk_reward: float):
        self.risk_reward = risk_reward
        self.logger = logging.getLogger("TradeManager")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        """Calculate stop loss based on signal type and recent price action."""
        try:
            last_bar = df_segment.iloc[-1]
            low = float(last_bar['low'])
            high = float(last_bar['high'])

            # Default pip buffer (8 pips)
            pip_buffer = 0.0008

            # Calculate stop loss based on signal type
            if signal["type"] == "BUY":
                # For buys, stop goes below recent low
                stop_loss = low - pip_buffer
                self.logger.debug(f"Buy signal stop loss: {stop_loss:.5f}")
            else:  # SELL
                # For sells, stop goes above recent high
                stop_loss = high + pip_buffer
                self.logger.debug(f"Sell signal stop loss: {stop_loss:.5f}")

            return stop_loss

        except Exception as e:
            self.logger.error(f"Stop loss calculation error: {str(e)}")
            # Return a default stop loss in case of error
            return last_bar['close'] * 0.99 if signal["type"] == "BUY" else last_bar['close'] * 1.01

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            # Validate inputs
            if account_balance <= 0:
                self.logger.error(f"Invalid account balance: {account_balance}")
                return 0.01

            if stop_distance <= 0:
                self.logger.error(f"Invalid stop distance: {stop_distance}")
                return 0.01

            # Risk calculation (1% risk per trade)
            risk_amount = account_balance * 0.01

            # Convert stop distance to pips
            stop_pips = stop_distance * 10000.0

            # Calculate pip value (EURUSD standard lot = $10 per pip)
            pip_value = 10.0

            # Calculate position size in lots
            position_size = risk_amount / (stop_pips * pip_value)

            # Round to 2 decimal places and ensure minimum size
            position_size = max(round(position_size, 2), 0.01)

            self.logger.info(f"Position size calculated: {position_size} lots")
            self.logger.debug(f"Details: Balance=${account_balance}, Risk=${risk_amount}, Stop={stop_pips}pips")

            return position_size

        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.01

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        """Calculate take profit based on risk/reward ratio."""
        dist = abs(entry_price - sl)
        if entry_price > sl:  # Long position
            return entry_price + (dist * self.risk_reward)
        else:  # Short position
            return entry_price - (dist * self.risk_reward)

    def check_exit_conditions(
        self,
        df_segment: pd.DataFrame,
        position: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if any exit conditions are met."""
        try:
            last_bar = df_segment.iloc[-1]
            current_price = float(last_bar["close"])

            # Basic stop loss & take profit checks
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

            return False, "No exit condition met"

        except Exception as e:
            self.logger.error(f"Exit condition check error: {str(e)}")
            return False, "Error checking exit conditions"