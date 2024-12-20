class MARSIVolumeStrategy(StrategyTemplate):
    """Specific implementation of MA/RSI/Volume strategy.
    All strategy-specific logic goes here."""

    def __init__(self):
        super().__init__()
        # Strategy-specific indicators
        self.indicators = IndicatorHandler()

    def analyze_conditions(self, data) -> dict:
        # Strategy-specific market analysis
        ma_cross = self.indicators.check_ma_cross(data)
        rsi_value = self.indicators.calculate_rsi(data)
        volume_ok = self.indicators.check_volume(data)
        return {
            "ma_signal": ma_cross,
            "rsi_valid": rsi_value,
            "volume_confirmed": volume_ok
        }

    def execute_strategy(self, data) -> dict:
        # 1. First validate market state
        if not self.validate_market_state(data):
            return {"signal": "NONE"}

        # 2. Analyze conditions
        conditions = self.analyze_conditions(data)

        # 3. Generate signal based on strategy rules
        signal = self._generate_signal(conditions)

        # 4. Calculate position size if signal exists
        if signal["type"] != "NONE":
            signal["size"] = self.calculate_position(data, signal)

        return signal