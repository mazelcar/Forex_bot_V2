# src/strategy/support_resistance.py

import numpy as np
import pandas as pd
from typing import Tuple

def identify_support_resistance_zones(higher_tf_data: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify support and resistance levels from higher timeframe data (H1).
    Steps:
    - Find swing highs/lows: a swing high is where 'high' > highs of bars around it.
    - Similarly for swing lows.
    - Filter levels by requiring multiple touches (using median as a simple approach).
    - Optionally filter support levels that align with a simple EMA trend filter.

    Args:
        higher_tf_data (pd.DataFrame): A DataFrame containing at least 'time', 'high', and 'low' columns.
                                       Should be sorted by time in ascending order.

    Returns:
        (support_levels, resistance_levels): Two lists containing identified support and resistance levels.
    """
    # Ensure data sorted by time
    higher_tf_data = higher_tf_data.sort_values('time').reset_index(drop=True)

    # Compute a short EMA for trend confirmation (e.g., 20-period EMA)
    higher_tf_data['ema_short'] = higher_tf_data['close'].ewm(span=20, adjust=False).mean()

    swing_highs = []
    swing_lows = []
    lookback = 2  # how many bars before/after to confirm a swing

    # Identify swings by checking if current bar is higher/lower than bars around it
    for i in range(lookback, len(higher_tf_data)-lookback):
        h = higher_tf_data.loc[i, 'high']
        l = higher_tf_data.loc[i, 'low']
        # Check if it's a swing high
        if all(h > higher_tf_data.loc[i-j, 'high'] for j in range(1, lookback+1)) and \
           all(h > higher_tf_data.loc[i+j, 'high'] for j in range(1, lookback+1)):
            swing_highs.append(h)
        # Check if it's a swing low
        if all(l < higher_tf_data.loc[i-j, 'low'] for j in range(1, lookback+1)) and \
           all(l < higher_tf_data.loc[i+j, 'low'] for j in range(1, lookback+1)):
            swing_lows.append(l)

    # Create support/resistance from medians if multiple swing points found
    if len(swing_lows) > 2:
        support_levels = [np.median(swing_lows)]
    else:
        support_levels = [1.05]  # fallback dummy

    if len(swing_highs) > 2:
        resistance_levels = [np.median(swing_highs)]
    else:
        resistance_levels = [1.06]  # fallback dummy

    # Filter support levels to those below the EMA (assuming a bullish bias if price > EMA)
    # In this example, we keep only levels below the mean of the ema_short as "valid" supports.
    support_levels = [lvl for lvl in support_levels if lvl < higher_tf_data['ema_short'].mean()]

    # If no valid supports found after filtering, fallback
    if not support_levels:
        support_levels = [1.045]

    # If no valid resistances found, fallback
    if not resistance_levels:
        resistance_levels = [1.065]

    return support_levels, resistance_levels


def is_bounce_candle(row: pd.Series, level: float, direction: str = 'support') -> bool:
    """
    Check if there's a bullish bounce at support or a bearish bounce at resistance.

    For a bullish bounce at support:
    - Candle's low should be near the support level within a certain tolerance.
    - Candle should close higher than it opened (bullish).

    For a bearish bounce at resistance:
    - Candle's high should be near the resistance level within a certain tolerance.
    - Candle should close lower than it opened (bearish).

    Args:
        row (pd.Series): A DataFrame row representing a bar/candle with at least 'open', 'high', 'low', 'close' columns.
        level (float): The support or resistance level to check against.
        direction (str): 'support' for bullish bounce check, 'resistance' for bearish bounce check.

    Returns:
        bool: True if the candle suggests a bounce at the given level, False otherwise.
    """
    tolerance = 0.0005  # Example tolerance to consider a "touch" of the level

    if direction == 'support':
        near_support = abs(row['low'] - level) <= tolerance
        bullish_candle = row['close'] > row['open']
        return near_support and bullish_candle
    else:
        near_resistance = abs(row['high'] - level) <= tolerance
        bearish_candle = row['close'] < row['open']
        return near_resistance and bearish_candle
