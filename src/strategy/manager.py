# src/strategy/manager.py
from typing import Any, Dict
from src.strategy.strategies.strategy_template import StrategyTemplate

class StrategyManager:
    def __init__(self):
        self._strategies: Dict[str, StrategyTemplate] = {}

    def register_strategy(self, name: str, strategy: StrategyTemplate) -> None:
        self._strategies[name] = strategy

    def get_strategy(self, name: str) -> StrategyTemplate:
        return self._strategies.get(name, None)

    def execute_strategy(self, name: str, market_data: Any, account_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the strategy logic (validate market, generate signals).
        """
        strategy = self.get_strategy(name)
        if strategy is None:
            return {"type": "NONE"}  # No strategy found
        # Directly call generate_signals for simplicity
        return strategy.generate_signals(market_data, account_info)
