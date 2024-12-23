import pandas as pd


def detect_price_hovering(df: pd.DataFrame, level: float, pip_tolerance: float = 0.0003, bars: int = 3) -> bool:

    if len(df) < bars:
        return False

    recent_bars = df.tail(bars)
    distance_high = abs(recent_bars['high'] - level)
    distance_low = abs(recent_bars['low'] - level)

    return (
        (distance_high <= pip_tolerance).all() and
        (distance_low <= pip_tolerance).all()
    )