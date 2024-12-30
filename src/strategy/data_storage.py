import os
import pandas as pd

def save_data_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Saves the provided DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The target CSV filename.
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a DataFrame, parsing dates if the file exists.
    Returns an empty DataFrame if the file does not exist or is invalid.

    Args:
        filename (str): The CSV filename to load from.

    Returns:
        pd.DataFrame: Loaded DataFrame (empty if not found or invalid).
    """
    if not os.path.exists(filename):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=['time'])
        # Ensure 'time' is UTC localized if needed:
        if 'time' in df.columns and not df['time'].dt.tz:
            df['time'] = df['time'].dt.tz_localize('UTC')
        return df
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()
