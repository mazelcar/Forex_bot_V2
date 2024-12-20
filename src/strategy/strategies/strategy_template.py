# src/strategy/strategies/strategy_template.py
from typing import Any, Dict

class StrategyTemplate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate_market_state(self, market_data: Any) -> bool:
        if market_data is None:
            return False
        if hasattr(market_data, 'empty') and market_data.empty:
            return False
        return True

    def calculate_position(self, account_balance: float, risk_per_trade: float = 0.01) -> float:
        return 0.1

    def generate_signals(self, market_data: Any, account_info: Dict[str, Any]) -> Dict[str, Any]:
        if self.validate_market_state(market_data):
            position_size = self.calculate_position(account_info.get("balance", 10000))
            return {"type": "BUY", "size": position_size}
        return {"type": "NONE"}
