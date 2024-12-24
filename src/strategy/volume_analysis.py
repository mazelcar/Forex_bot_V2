import pandas as pd

def is_volume_sufficient(
    df: pd.DataFrame,
    current_index: int,
    lookback_bars: int = 5,
    min_ratio: float = 0.8
) -> bool:
    """
    Checks if the current bar's volume is at least `min_ratio` times
    the average volume of the previous `lookback_bars`.

    Args:
        df (pd.DataFrame): The DataFrame containing price/volume data.
        current_index (int): The index of the current bar.
        lookback_bars (int): How many bars to look back for the average volume.
        min_ratio (float): The fraction of average volume considered "sufficient" (default=0.8).

    Returns:
        bool: True if current volume >= min_ratio * average volume, False otherwise.
    """
    if current_index < 1:
        return False

    current_vol = df.iloc[current_index]['tick_volume']
    start_idx = max(current_index - lookback_bars, 0)
    recent_vol = df['tick_volume'].iloc[start_idx:current_index]

    if len(recent_vol) == 0:
        return False

    avg_vol = recent_vol.mean()
    return current_vol >= (min_ratio * avg_vol)
