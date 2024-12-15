"""Data Validation Module for Forex Trading Bot V2.

This module handles all data validation checks including:
1. Basic data structure validation
2. Test period coverage validation
3. Volume data quality checks
4. Weekend/Holiday handling

Author: mazelcar
Created: December 2024
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import json

class DataValidator:
    """Handles all data validation checks for the trading strategy."""

    def __init__(self, strategy_config: Dict):
        """Initialize with strategy configuration.

        Args:
            strategy_config: Strategy configuration containing periods and parameters
        """
        self.config = strategy_config

        # Extract required parameters from config
        self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
        self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']
        self.rsi_period = self.config['indicators']['rsi']['period']
        self.volume_period = self.config['indicators']['volume']['period']

    def validate_basic_data(self, data: pd.DataFrame) -> Dict:
        """
        Step 1.1: Basic Bars and Columns Validation
        Validates basic data structure, required bars, and columns.

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

        print("\n========== STEP 1: BASIC DATA VALIDATION ==========")

        # 1.1 Price Data Check
        min_required_bars = max(
            self.fast_ema_period * 2,  # For EMA calculation
            self.slow_ema_period * 2,  # For EMA calculation
            self.rsi_period * 2,       # For RSI calculation
            self.volume_period * 2,    # For volume analysis
            50                         # Minimum for trend analysis
        )

        has_enough_data = len(data) >= min_required_bars
        validation_results['checks']['enough_data'] = has_enough_data
        print(f"\n1.1.1 Price Data Check:")
        print(f"-> Required Bars: {min_required_bars}")
        print(f"-> Available Bars: {len(data)}")
        print(f"-> Status: {'[PASS]' if has_enough_data else '[FAIL]'} Price Data Check")

        if not has_enough_data:
            msg = f"Insufficient data: need {min_required_bars}, have {len(data)}"
            validation_results['error_messages'].append(msg)
            validation_results['overall_pass'] = False

        # 1.2 Required Columns Check
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        has_required_columns = len(missing_columns) == 0
        validation_results['checks']['required_columns'] = has_required_columns

        print(f"\n1.1.2 Required Columns Check:")
        print(f"-> Looking for: {', '.join(required_columns)}")
        print(f"-> Found Columns: {', '.join(data.columns)}")
        print(f"-> Status: {'[PASS]' if has_required_columns else '[FAIL]'} Required Columns Check")

        if not has_required_columns:
            msg = f"Missing required columns: {missing_columns}"
            validation_results['error_messages'].append(msg)
            validation_results['overall_pass'] = False

        # 1.3 Data Structure Validation
        validation_results.update(
            self._validate_data_structure(data)
        )

        # 1.4 Volume Data Check
        validation_results.update(
            self._validate_volume_data(data)
        )

        # Print overall validation summary
        self._print_validation_summary(validation_results)

        return validation_results

    def validate_data(self, data: pd.DataFrame, start_date: datetime = None,
                        end_date: datetime = None) -> Dict:
            """Main validation entry point.

            Args:
                data: DataFrame to validate
                start_date: Optional start date for period validation
                end_date: Optional end date for period validation

            Returns:
                Dict containing validation results
            """
            validation_results = {
                'basic_validation': self.validate_basic_data(data),
                'test_period': None
            }

            # Only perform test period validation if dates are provided
            if start_date and end_date:
                validation_results['test_period'] = self.validate_test_period(
                    data, start_date, end_date
                )

            return validation_results

    def validate_test_period(self, data: pd.DataFrame, start_date: datetime,
                           end_date: datetime) -> Dict:
        """
        Step 1.2: TEST PERIOD COVERAGE
        Validates the test period coverage, including date range and data consistency.

        Args:
            data: DataFrame containing the market data
            start_date: Requested start date for testing
            end_date: Requested end date for testing

        Returns:
            Dictionary containing validation results
        """
        print("\nEntering test period validation...")

        validation_results = {
            'overall_pass': True,
            'checks': {},
            'error_messages': []
        }

        try:
            print("\n========== STEP 1.2: TEST PERIOD COVERAGE ==========")

            # 1.2.1 Date Range Verification
            validation_results.update(
                self._validate_date_range(data, start_date, end_date)
            )

            # 1.2.2 Timestamp Gaps Check
            validation_results.update(
                self._validate_timestamp_continuity(data)
            )

            # 1.2.3 Data Frequency Consistency
            validation_results.update(
                self._validate_frequency_consistency(data)
            )

            # Print validation summary
            self._print_validation_summary(validation_results)

            return validation_results

        except Exception as e:
            print(f"\nError in test period validation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def _validate_data_structure(self, data: pd.DataFrame) -> Dict:
        """Validate data structure and types."""
        results = {
            'checks': {},
            'error_messages': []
        }

        print("\n1.1.3 Data Structure Validation:")

        # Check if input is DataFrame
        is_dataframe = isinstance(data, pd.DataFrame)
        results['checks']['is_dataframe'] = is_dataframe
        print(f"-> Is DataFrame: {is_dataframe}")

        if not is_dataframe:
            results['error_messages'].append("Input is not a pandas DataFrame")
            results['overall_pass'] = False
            return results

        # Define expected data types
        expected_types = {
            'time': 'datetime64[ns]',
            'open': ['float64', 'float32'],
            'high': ['float64', 'float32'],
            'low': ['float64', 'float32'],
            'close': ['float64', 'float32'],
            'tick_volume': ['int64', 'float64', 'float32', 'uint64']
        }

        # Check column data types
        results.update(self._check_column_types(data, expected_types))

        # Check for NULL values
        results.update(self._check_null_values(data))

        return results

    def _validate_volume_data(self, data: pd.DataFrame) -> Dict:
        """Validate volume data quality."""
        results = {
            'checks': {},
            'error_messages': []
        }

        volume_cols = ['tick_volume', 'real_volume']
        has_volume = any(col in data.columns for col in volume_cols)
        results['checks']['has_volume'] = has_volume

        print(f"\n1.1.4 Volume Data Check:")
        print(f"Looking for any of: {', '.join(volume_cols)}")

        if has_volume:
            found_vol = next(col for col in volume_cols if col in data.columns)
            valid_volume = not data[found_vol].isnull().any() and (data[found_vol] >= 0).all()
            print(f"Found {found_vol} data")
            print(f"Volume data validation: {'Valid' if valid_volume else 'Invalid'}")
            print("-> Status: [PASS] Volume Data Check")
        else:
            print("-> Status: [FAIL] No volume data found")
            msg = "No volume data available"
            results['error_messages'].append(msg)
            results['overall_pass'] = False

        return results

    def _validate_date_range(self, data: pd.DataFrame, start_date: datetime,
                           end_date: datetime) -> Dict:
        """Validate date range coverage."""
        results = {
            'checks': {},
            'error_messages': []
        }

        data_start = data['time'].min()
        data_end = data['time'].max()

        print("\n1.2.1 Date Range Check:")
        print(f"Requested Period: {start_date} to {end_date}")
        print(f"Available Period: {data_start} to {data_end}")

        min_required_bars = max(
            self.fast_ema_period * 2,
            self.slow_ema_period * 2,
            self.rsi_period * 2,
            self.volume_period * 2,
            50
        )

        has_enough_coverage = len(data) >= min_required_bars
        results['checks']['dates_match'] = has_enough_coverage
        print(f"Required bars: {min_required_bars}")
        print(f"Available bars: {len(data)}")
        print(f"{'[PASS]' if has_enough_coverage else '[FAIL]'}: Date Range Check")

        if not has_enough_coverage:
            msg = f"Insufficient data coverage. Need {min_required_bars} bars, have {len(data)}"
            results['error_messages'].append(msg)
            results['overall_pass'] = False

        return results

    def _validate_timestamp_continuity(self, data: pd.DataFrame) -> Dict:
        """Validate timestamp continuity and gaps."""
        results = {
            'checks': {},
            'error_messages': []
        }

        print("\n1.2.2 Timestamp Continuity Check:")
        timeframe_minutes = 5  # M5 timeframe
        expected_diff = pd.Timedelta(minutes=timeframe_minutes)
        time_diffs = data['time'].diff()

        gaps = self._find_unexpected_gaps(data, time_diffs, expected_diff)

        has_gaps = len(gaps) > 0
        results['checks']['no_gaps'] = not has_gaps

        print(f"Expected interval: {timeframe_minutes} minutes")
        if has_gaps:
            self._print_gaps(gaps)
            results['error_messages'].append(f"Found {len(gaps)} unexpected gaps in data")
            results['overall_pass'] = False
        else:
            print("No unexpected gaps found")

        print(f"{'[PASS]' if not has_gaps else '[FAIL]'}: Continuity Check")
        return results

    def _validate_frequency_consistency(self, data: pd.DataFrame) -> Dict:
        """Validate data frequency consistency."""
        results = {
            'checks': {},
            'error_messages': []
        }

        print("\n1.2.3 Frequency Consistency Check:")
        expected_diff = pd.Timedelta(minutes=5)
        time_diffs = data['time'].diff()
        valid_diffs = time_diffs[time_diffs <= expected_diff * 2]

        mean_diff = valid_diffs.mean()
        std_diff = valid_diffs.std()

        is_consistent = (std_diff <= pd.Timedelta(minutes=1))
        results['checks']['frequency_consistent'] = is_consistent

        print(f"Mean interval: {mean_diff}")
        print(f"Standard deviation: {std_diff}")
        print(f"{'[PASS]' if is_consistent else '[FAIL]'}: Frequency Consistency")

        if not is_consistent:
            msg = f"Inconsistent data frequency. Mean: {mean_diff}, Std: {std_diff}"
            results['error_messages'].append(msg)
            results['overall_pass'] = False

        return results

    def _check_column_types(self, data: pd.DataFrame, expected_types: Dict) -> Dict:
        """Check column data types."""
        results = {
            'checks': {'type_checks': {}},
            'error_messages': []
        }

        print("\n-> Column Types:")
        for col, expected in expected_types.items():
            if col not in data.columns:
                continue

            current_type = data[col].dtype.name
            if isinstance(expected, list):
                type_valid = current_type in expected
            else:
                type_valid = current_type == expected

            results['checks']['type_checks'][col] = type_valid
            print(f"   {col}: {current_type} {'[PASS]' if type_valid else '[FAIL]'}")

            if not type_valid:
                msg = f"Column {col} has incorrect type: {current_type}, expected {expected}"
                results['error_messages'].append(msg)
                results['overall_pass'] = False

        return results

    def _check_null_values(self, data: pd.DataFrame) -> Dict:
        """Check for NULL values in critical columns."""
        results = {
            'checks': {},
            'error_messages': []
        }

        critical_columns = ['time', 'open', 'high', 'low', 'close']
        null_checks = data[critical_columns].isnull().sum()
        has_nulls = null_checks.sum() > 0

        print("\n-> NULL Value Check:")
        if has_nulls:
            print("   Found NULL values in:")
            for col, count in null_checks[null_checks > 0].items():
                print(f"   {col}: {count} NULL values")
                msg = f"Found {count} NULL values in column {col}"
                results['error_messages'].append(msg)
                results['overall_pass'] = False
        else:
            print("   No NULL values found [PASS]")

        results['checks']['null_checks'] = not has_nulls
        return results

    def _find_unexpected_gaps(self, data: pd.DataFrame, time_diffs: pd.Series,
                            expected_diff: pd.Timedelta) -> List:
        """Find unexpected gaps in timestamp data."""
        gaps = []
        for i in range(1, len(data)):
            current_time = data['time'].iloc[i]
            prev_time = data['time'].iloc[i-1]

            # Skip weekend gaps
            if prev_time.weekday() == 4 and current_time.weekday() == 0:
                continue

            # Skip daily gaps (market close to open)
            if prev_time.hour == 23 and current_time.hour == 0:
                continue

            # Check for unexpected gaps
            if time_diffs.iloc[i] > expected_diff:
                gaps.append((prev_time, current_time))

        return gaps

    def _print_gaps(self, gaps: List) -> None:
        """Print gap information."""
        print("Found unexpected gaps:")
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            print(f"  Gap from {gap_start} to {gap_end} (Duration: {gap_duration})")

    def _print_validation_summary(self, results: Dict) -> None:
        """Print validation summary."""
        print("\n========== VALIDATION SUMMARY ==========")
        print(f"Overall Status: {'[PASS]' if results['overall_pass'] else '[FAIL]'}")
        if results['error_messages']:
            print("\nError Messages:")
            for msg in results['error_messages']:
                print(f"• {msg}")
        print("=======================================")