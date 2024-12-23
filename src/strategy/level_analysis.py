from typing import List, Tuple
import pandas as pd


def identify_yearly_extremes(df: pd.DataFrame, buffer_pips: float = 0.0050) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    yearly_high = df['high'].max()
    yearly_low = df['low'].min()

    high_zone = (yearly_high - buffer_pips, yearly_high + buffer_pips)
    low_zone = (yearly_low - buffer_pips, yearly_low + buffer_pips)

    return high_zone, low_zone

def merge_levels_to_zones(levels: List[float], pip_threshold: float = 0.0015) -> List[Tuple[float, float]]:

    if not levels:
        return []

    sorted_levels = sorted(levels)
    zones = []
    current_zone = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if level - current_zone[-1] <= pip_threshold:
            current_zone.append(level)
        else:
            zones.append((min(current_zone), max(current_zone)))
            current_zone = [level]

    if current_zone:
        zones.append((min(current_zone), max(current_zone)))

    return zones

def calculate_cleanliness_score(df: pd.DataFrame, level: float, pip_tolerance: float = 0.0003) -> int:

    touches = 0
    fakeouts = 0
    in_fakeout = False

    for i in range(1, len(df) - 1):
        prev_bar = df.iloc[i-1]
        curr_bar = df.iloc[i]
        next_bar = df.iloc[i+1]

        # Check if price touched the level
        if (abs(curr_bar['high'] - level) <= pip_tolerance or
            abs(curr_bar['low'] - level) <= pip_tolerance):

            if not in_fakeout:
                touches += 1

            # Check for fakeout
            if ((curr_bar['close'] > level + pip_tolerance and next_bar['close'] < level - pip_tolerance) or
                (curr_bar['close'] < level - pip_tolerance and next_bar['close'] > level + pip_tolerance)):
                fakeouts += 1
                in_fakeout = True
            else:
                in_fakeout = False

    # Calculate score based on touches and fakeouts
    if touches >= 8 and fakeouts <= 1:
        return 5
    elif touches >= 5 and fakeouts <= 3:
        return 4
    elif touches >= 5 and fakeouts <= 5:
        return 3
    elif touches >= 3 or fakeouts > 5:
        return 2
    else:
        return 1