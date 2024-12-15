"""Indicator Validation Module for Forex Trading Bot V2.

This module handles validation of technical indicators including:
1. EMA calculation validation
2. RSI calculation validation
3. Volume indicators validation

Author: mazelcar
Created: December 2024
"""

from typing import Dict
import pandas as pd

class IndicatorValidator:
    """Validates technical indicators calculation requirements."""

    def __init__(self, strategy_config: Dict):
        """Initialize with strategy configuration.

        Args:
            strategy_config: Strategy configuration containing indicator parameters
        """
        self.config = strategy_config

        # Extract indicator periods from config
        self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
        self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']
        self.rsi_period = self.config['indicators']['rsi']['period']
        self.volume_period = self.config['indicators']['volume']['period']

    def validate_indicator_warmup(self, data: pd.DataFrame) -> Dict:
        """
        Step 1.3: INDICATOR WARMUP VALIDATION
        Validates that there's enough data for each indicator's warmup period.

        Args:
            data: DataFrame containing the market data

        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'overall_pass': True,
            'checks': {},
            'error_messages': []
        }

        print("\n========== STEP 1.3: INDICATOR WARMUP VALIDATION ==========")

        # 1.3.1 EMA Warmup Validation
        ema_validation = self._validate_ema_warmup(data)
        validation_results['checks']['ema'] = ema_validation
        if not ema_validation['pass']:
            validation_results['overall_pass'] = False
            validation_results['error_messages'].extend(ema_validation['messages'])

        # 1.3.2 RSI Warmup Validation
        rsi_validation = self._validate_rsi_warmup(data)
        validation_results['checks']['rsi'] = rsi_validation
        if not rsi_validation['pass']:
            validation_results['overall_pass'] = False
            validation_results['error_messages'].extend(rsi_validation['messages'])

        # 1.3.3 Volume Indicator Warmup Validation
        volume_validation = self._validate_volume_warmup(data)
        validation_results['checks']['volume'] = volume_validation
        if not volume_validation['pass']:
            validation_results['overall_pass'] = False
            validation_results['error_messages'].extend(volume_validation['messages'])

        # Print validation summary
        print("\n========== VALIDATION SUMMARY ==========")
        print(f"Overall Status: {'[PASS]' if validation_results['overall_pass'] else '[FAIL]'}")
        if validation_results['error_messages']:
            print("\nError Messages:")
            for msg in validation_results['error_messages']:
                print(f"• {msg}")
        print("=======================================")

        return validation_results

    def _validate_ema_warmup(self, data: pd.DataFrame) -> Dict:
        """Validate EMA calculation requirements."""
        print("\n1.3.1 EMA Warmup Requirements:")

        # Calculate required bars for EMA
        required_bars = max(self.fast_ema_period, self.slow_ema_period) * 2
        available_bars = len(data)

        print(f"-> Fast EMA Period: {self.fast_ema_period} bars")
        print(f"-> Slow EMA Period: {self.slow_ema_period} bars")
        print(f"-> Required Warmup: {required_bars} bars")
        print(f"-> Available Bars: {available_bars}")

        is_valid = available_bars >= required_bars
        print(f"-> Status: {'[PASS]' if is_valid else '[FAIL]'} EMA Warmup Check")

        return {
            'pass': is_valid,
            'messages': [] if is_valid else [
                f"Insufficient data for EMA calculation. Need {required_bars}, have {available_bars}"
            ]
        }

    def _validate_rsi_warmup(self, data: pd.DataFrame) -> Dict:
        """Validate RSI calculation requirements."""
        print("\n1.3.2 RSI Warmup Requirements:")

        # Calculate required bars for RSI
        required_bars = self.rsi_period * 2  # Double period for proper calculation
        available_bars = len(data)

        print(f"-> RSI Period: {self.rsi_period} bars")
        print(f"-> Required Warmup: {required_bars} bars")
        print(f"-> Available Bars: {available_bars}")

        is_valid = available_bars >= required_bars
        print(f"-> Status: {'[PASS]' if is_valid else '[FAIL]'} RSI Warmup Check")

        return {
            'pass': is_valid,
            'messages': [] if is_valid else [
                f"Insufficient data for RSI calculation. Need {required_bars}, have {available_bars}"
            ]
        }

    def _validate_volume_warmup(self, data: pd.DataFrame) -> Dict:
        """Validate volume indicators calculation requirements."""
        print("\n1.3.3 Volume Indicator Warmup Requirements:")

        # Calculate required bars for volume indicators
        required_bars = self.volume_period * 2
        available_bars = len(data)

        print(f"-> Volume Period: {self.volume_period} bars")
        print(f"-> Required Warmup: {required_bars} bars")
        print(f"-> Available Bars: {available_bars}")

        # Check if volume data is present
        has_volume = any(col in data.columns for col in ['tick_volume', 'real_volume'])

        is_valid = available_bars >= required_bars and has_volume
        print(f"-> Status: {'[PASS]' if is_valid else '[FAIL]'} Volume Warmup Check")

        messages = []
        if not has_volume:
            messages.append("No volume data available")
        if available_bars < required_bars:
            messages.append(f"Insufficient data for volume calculation. Need {required_bars}, have {available_bars}")

        return {
            'pass': is_valid,
            'messages': messages
        }