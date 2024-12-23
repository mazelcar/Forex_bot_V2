"""
data_validator.py

Provides a basic DataValidator class to ensure your DataFrame
has the expected columns and is not empty before continuing
with a backtest or other logic.

Usage:
    from data_validator import DataValidator

    validator = DataValidator(required_columns=['time', 'open', 'high', 'low', 'close'])

    df = load_some_data()
    is_valid, error_message = validator.validate_data(df)
    if not is_valid:
        print("Data not valid:", error_message)
        # Handle the error (e.g. return, raise an exception, etc.)
"""

import pandas as pd
from typing import List, Tuple

class DataValidator:
    def __init__(self, required_columns: List[str] = None, allow_empty: bool = False):
        """
        Args:
            required_columns (List[str]): A list of column names that must be present.
            allow_empty (bool): Whether an empty DataFrame is acceptable or not.
        """
        if required_columns is None:
            # Default required columns
            required_columns = ['time', 'open', 'high', 'low', 'close']
        self.required_columns = required_columns
        self.allow_empty = allow_empty

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates that the DataFrame:
         1) is not empty (unless allow_empty=True)
         2) has all required columns.

        Returns:
            (bool, str): (is_valid, error_message)
                         is_valid = True if DF is valid, False otherwise
                         error_message = reason why validation failed, if any
        """

        # Check if DataFrame is empty
        if not self.allow_empty and df.empty:
            return (False, "DataFrame is empty.")

        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            return (False, f"Missing required columns: {missing_cols}")

        # If all checks pass:
        return (True, "Data is valid.")
