from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class DataValidator:
    """Validates market data for backtesting purposes (Basic Version for Day 1-2)."""

    DEFAULT_CONFIG = {
        'minimum_bars': 100,
        'required_columns': [
            'time', 'open', 'high', 'low', 'close',
            'tick_volume', 'spread', 'real_volume',
            'pip_value', 'point', 'symbol'
        ],
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataValidator with optional configuration.

        Args:
            config: Dictionary containing validation configuration parameters
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

    def validate_basic_data(self, data: pd.DataFrame) -> Dict:
        """
        Perform basic validation on market data:
        - Minimum bars check
        - Required columns presence
        - Basic time column checks

        Args:
            data: DataFrame containing market data

        Returns:
            Dict containing validation results and any error messages
        """
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        # Perform basic validations
        basic_validation = self.validate_basic_bars_and_columns(data)
        if not basic_validation['passed']:
            results['passed'] = False
            results['messages'].extend(basic_validation['messages'])
            results['details'].update(basic_validation.get('details', {}))
            # If this fails, no need to check further
            return results

        # Check the structure minimally: time column format and ordering
        structure_validation = self.validate_time_structure(data)
        if not structure_validation['passed']:
            results['passed'] = False
            results['messages'].extend(structure_validation['messages'])
            results['details'].update(structure_validation.get('details', {}))

        return results

    def validate_volume_quality(self, data: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        if 'tick_volume' not in data.columns:
            results['passed'] = False
            results['messages'].append("No 'tick_volume' column found for volume quality checks.")
            return results

        if (data['tick_volume'] == 0).all():
            results['passed'] = False
            results['messages'].append("All bars have zero tick_volume, suspicious volume data.")

        return results

    def validate_basic_bars_and_columns(self, data: pd.DataFrame) -> Dict:
        """
        Validate basic bar count and required columns.

        Args:
            data: DataFrame containing market data

        Returns:
            Dict containing validation results and any error messages
        """
        results = {
            'passed': True,
            'messages': [],
            'details': {
                'bar_count': len(data),
                'columns_present': list(data.columns),
            }
        }

        # Check minimum bars
        min_bars = self.config['minimum_bars']
        if len(data) < min_bars:
            results['passed'] = False
            results['messages'].append(
                f"Insufficient number of bars. Required: {min_bars}, Found: {len(data)}"
            )
            results['details']['minimum_required'] = min_bars

        # Check required columns
        required_cols = self.config['required_columns']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            results['passed'] = False
            results['messages'].append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )
            results['details']['missing_columns'] = missing_cols

        return results

    def validate_time_structure(self, data: pd.DataFrame) -> Dict:
        """
        Validate that the 'time' column is in datetime format and strictly increasing.

        Args:
            data: DataFrame containing market data

        Returns:
            Dict containing validation results and any error messages
        """
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        if 'time' not in data.columns:
            # This should have been caught above, but just in case
            results['passed'] = False
            results['messages'].append("'time' column is missing, cannot validate ordering.")
            return results

        if not pd.api.types.is_datetime64_any_dtype(data['time']):
            results['passed'] = False
            results['messages'].append("'time' column is not in datetime format")

        if not data['time'].is_monotonic_increasing:
            results['passed'] = False
            results['messages'].append("'time' column is not strictly increasing")

        return results

    def validate_test_period_coverage(self, data: pd.DataFrame, requested_start: datetime, requested_end: datetime, expected_freq='5T') -> Dict:
        """
        Validate if the data covers the requested test period adequately.
        Args:
            data: pd.DataFrame with 'time' column
            requested_start: datetime requested start
            requested_end: datetime requested end
            expected_freq: expected frequency, e.g. '5T' for 5 minutes

        Returns:
            dict with 'passed', 'messages', and optional 'details'
        """
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        if 'time' not in data.columns:
            results['passed'] = False
            results['messages'].append("Cannot verify period coverage without 'time' column.")
            return results

        actual_start = data['time'].min()
        actual_end = data['time'].max()

        # Check coverage
        if actual_start > requested_start:
            results['passed'] = False
            results['messages'].append(
                f"Data starts later than requested period. Requested start: {requested_start}, Actual start: {actual_start}"
            )

        if actual_end < requested_end:
            results['passed'] = False
            results['messages'].append(
                f"Data ends before requested period end. Requested end: {requested_end}, Actual end: {actual_end}"
            )

        # Optional: Check for large gaps
        # Estimate expected bars:
        expected_bars = int(((requested_end - requested_start).total_seconds() / 60) / 5)  # For M5 data
        actual_bars = len(data)

        # If actual bars are significantly less than expected (e.g., less than 70%?), warn or fail
        if actual_bars < expected_bars * 0.7:
            results['passed'] = False
            results['messages'].append(
                f"Significant data gap. Expected ~{expected_bars} bars, got {actual_bars} bars."
            )

        return results

    def validate_weekend_holiday_handling(self, data: pd.DataFrame, holidays: Dict[str, List[str]]) -> Dict:
        """
        Validate that there are no bars during weekends or known holidays.

        Args:
            data: DataFrame containing the market data with a 'time' column.
            holidays: A dictionary of holidays keyed by year, e.g.:
                      {
                        "2024": {
                            "Sydney": [
                               {"date": "2024-12-25", "name": "Christmas"},
                               {"date": "2024-12-26", "name": "Boxing Day"}
                            ],
                            "Tokyo": [...]
                        },
                        "2025": { ... }
                      }
                      For now, we just need to know if the bar date is a holiday for any major market session.
                      Adjust as per your actual holiday structure.

        Returns:
            dict with 'passed', 'messages', and optional 'details'.
        """
        results = {
            'passed': True,
            'messages': [],
            'details': {}
        }

        if 'time' not in data.columns:
            results['passed'] = False
            results['messages'].append("Cannot perform weekend/holiday check without 'time' column.")
            return results

        # Extract the day of week: Monday=0, Sunday=6
        # Weekend is Saturday (5) and Sunday (6)
        weekend_bars = data[data['time'].dt.dayofweek >= 5]
        if not weekend_bars.empty:
            results['passed'] = False
            count_weekend_bars = len(weekend_bars)
            results['messages'].append(f"Found {count_weekend_bars} bars on weekends, which is not expected.")

        # Check holidays:
        # Convert 'time' to date string in YYYY-MM-DD format and check if it’s in the holiday list
        # We'll just check if any date in data falls on a known holiday date from the holidays dict
        # We assume that if a date is listed in ANY session's holiday, it's a holiday.

        holiday_dates = set()
        for year, markets in holidays.items():
            for market, hol_list in markets.items():
                for hol in hol_list:
                    holiday_dates.add(hol["date"])  # e.g. "2024-12-25"

        # Create a column with just date (YYYY-MM-DD)
        data['date_str'] = data['time'].dt.strftime('%Y-%m-%d')
        holiday_bars = data[data['date_str'].isin(holiday_dates)]
        if not holiday_bars.empty:
            results['passed'] = False
            count_holiday_bars = len(holiday_bars)
            results['messages'].append(f"Found {count_holiday_bars} bars on holidays, which should be closed market days.")

        # Clean up the temporary column
        data.drop(columns=['date_str'], inplace=True)

        return results
