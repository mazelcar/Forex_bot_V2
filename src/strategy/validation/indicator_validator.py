# src/strategy/validation/indicator_validator.py

from typing import Dict
import pandas as pd

class IndicatorValidator:
    def __init__(self, config: Dict):
        self.config = config

        # Example configuration expected:
        # self.config['indicators'] = {
        #   'ema': {'fast_period': 12, 'slow_period': 26},
        #   'rsi': {'period': 14}
        # }

    def validate_indicator_warmup(self, data: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        # Check for required data length for EMA
        fast_period = self.config.get('indicators', {}).get('ema', {}).get('fast_period', 12)
        slow_period = self.config.get('indicators', {}).get('ema', {}).get('slow_period', 26)
        rsi_period = self.config.get('indicators', {}).get('rsi', {}).get('period', 14)

        # The idea: we need at least max(slow_period, rsi_period, ...) * 2 bars to properly warm up indicators
        # You can adjust this logic as needed.
        min_required = max(slow_period, rsi_period) * 2

        actual_bars = len(data)
        if actual_bars < min_required:
            results['passed'] = False
            results['messages'].append(
                f"Not enough bars for indicator warmup. Required: {min_required}, Found: {actual_bars}"
            )

        return results

    def validate_no_lookahead(self, data: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        # Extract periods from config
        rsi_period = self.config.get('indicators', {}).get('rsi', {}).get('period', 14)

        # Check if RSI column exists
        if 'rsi' not in data.columns:
            # If RSI not computed or missing, skip this check
            # Ideally, we ensure RSI is always computed at this stage
            results['passed'] = False
            results['messages'].append("RSI column not found, cannot verify look-ahead bias.")
            return results

        # We'll do a simple incremental RSI check on the first rsi_period*2 bars
        # and ensure it matches the bulk calculation at that point.

        subset = data['close'].iloc[:rsi_period*2].copy()
        if len(subset) < rsi_period*2:
            # Not enough data to perform a meaningful test
            # You could pass or fail gracefully here, let's just pass
            return results

        # Incremental RSI calculation
        delta = subset.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))

        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

        rs = avg_gain / avg_loss
        incremental_rsi = 100 - (100 / (1 + rs))

        # The last RSI value in incremental_rsi should match the RSI calculated in the full dataset at the same point
        full_rsi_value = data['rsi'].iloc[rsi_period*2 - 1]
        incremental_rsi_value = incremental_rsi.iloc[-1]

        # If they differ significantly, it suggests a potential look-ahead (or calculation inconsistency)
        # Allow a tiny tolerance for floating-point differences
        if abs(full_rsi_value - incremental_rsi_value) > 1e-7:
            results['passed'] = False
            results['messages'].append(
                f"RSI incremental calculation differs from bulk calculation. Potential look-ahead bias.\n"
                f"Bulk RSI at bar {rsi_period*2-1}: {full_rsi_value}, Incremental RSI: {incremental_rsi_value}"
            )

        return results

    def calculate_and_verify_indicators(self, data: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        # Extract periods from config
        fast_period = self.config.get('indicators', {}).get('ema', {}).get('fast_period', 12)
        slow_period = self.config.get('indicators', {}).get('ema', {}).get('slow_period', 26)
        rsi_period = self.config.get('indicators', {}).get('rsi', {}).get('period', 14)

        # Check if 'close' column is present
        if 'close' not in data.columns:
            results['passed'] = False
            results['messages'].append("No 'close' column found for indicator calculations.")
            return results

        # Calculate EMAs
        data['ema_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()

        # Verify EMAs are computed without NaNs after warmup
        # Allow first few bars to be NaN due to warmup but ensure stable values after the largest period
        largest_period = max(fast_period, slow_period)
        if data['ema_fast'].iloc[largest_period:].isnull().any():
            results['passed'] = False
            results['messages'].append("EMA fast calculation contains NaNs after warmup period.")
        if data['ema_slow'].iloc[largest_period:].isnull().any():
            results['passed'] = False
            results['messages'].append("EMA slow calculation contains NaNs after warmup period.")

        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()


        # Start RSI calculation after rsi_period
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))

        print("Debug: Indicator calculations completed, verifying values...")

        # Verify RSI values after the warmup
        if data['rsi'].iloc[rsi_period:].isnull().any():
            results['passed'] = False
            results['messages'].append("RSI calculation contains NaNs after warmup period.")

        # Check if RSI stays within 0 to 100 range (normal RSI range)
        if not data['rsi'].iloc[rsi_period:].between(0, 100).all():
            results['passed'] = False
            results['messages'].append("RSI values fall outside the 0-100 range, indicating a calculation issue.")

        return results