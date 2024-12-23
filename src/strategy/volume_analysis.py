import pandas as pd

def calculate_volume_ma(volume: pd.Series, period: int = 50) -> pd.Series:
    return volume.rolling(period).mean()

def is_volume_sufficient(current_vol: float, average_vol: float) -> bool:
    return current_vol >= 0.8 * average_vol

class VolumeValidator:
    """
    A more dynamic VolumeValidator that checks:
      1. Time-of-day (hourly) volume averages.
      2. Recent-bar volume expansion relative to prior bars.
    """

    def __init__(
        self,
        expansion_factor: float = 1.5,
        lookback_bars: int = 5,
        time_adjustment: bool = True
    ):
        """
        Args:
            expansion_factor: Multiplier for checking expansion vs. recent average.
                             (e.g., 1.0 -> volume must be >= prior avg to qualify)
            lookback_bars: How many recent bars to average for 'prior volume' reference.
            time_adjustment: Whether to normalize volume by the average volume for that hour of the day.
        """
        self.expansion_factor = expansion_factor
        self.lookback_bars = lookback_bars
        self.time_adjustment = time_adjustment
        self.avg_volume_by_hour = {}

    def update_avg_volume_by_hour(self, df: pd.DataFrame) -> None:
        """
        Pre-calculates the average volume for each hour of day (0-23) across the entire DataFrame.
        Call this once after loading your data, before calling is_volume_sufficient in a loop.
        """
        volume_sum_by_hour = {}
        count_by_hour = {}

        for i, row in df.iterrows():
            hr = pd.to_datetime(row['time']).hour
            volume_sum_by_hour[hr] = volume_sum_by_hour.get(hr, 0.0) + row['tick_volume']
            count_by_hour[hr] = count_by_hour.get(hr, 0) + 1

        for hr in volume_sum_by_hour:
            if count_by_hour[hr] > 0:
                volume_sum_by_hour[hr] /= float(count_by_hour[hr])
            else:
                volume_sum_by_hour[hr] = 0.0

        self.avg_volume_by_hour = volume_sum_by_hour

    def is_volume_sufficient(self, df: pd.DataFrame, current_index: int) -> bool:
        """
        Determines if volume at 'current_index' is 'sufficient' based on:
          - Time-of-day normalization (if enabled),
          - Recent-bar volume expansion.

        Returns:
            True if volume is sufficient, False otherwise.
        """
        # Must have at least 1 prior bar
        if current_index < 1:
            return False

        current_row = df.iloc[current_index]
        current_vol = current_row['tick_volume']
        current_time = pd.to_datetime(current_row['time'])
        hr = current_time.hour

        # --- 1) Time-of-Day Normalization ---
        # e.g., if hour's baseline is zero or missing, fallback to 1.0
        # so normalized_vol = (current_vol / baseline) indicates how big volume is relative to average for that hour
        if self.time_adjustment and hr in self.avg_volume_by_hour and self.avg_volume_by_hour[hr] != 0:
            normalized_vol = current_vol / self.avg_volume_by_hour[hr]
        else:
            normalized_vol = 1.0  # fallback if we can't do time-of-day normalization

        # --- 2) Recent-Bar Volume Expansion ---
        start_idx = max(current_index - self.lookback_bars, 0)
        recent_vol_series = df.iloc[start_idx:current_index]['tick_volume']
        if len(recent_vol_series) == 0:
            return False
        prior_avg_vol = recent_vol_series.mean()

        # Example condition:
        # Must have current_vol >= expansion_factor * prior_avg_vol AND
        # normalized_vol >= 1.0 (i.e. at least average for that time-of-day).
        # Adjust as you see fit.
        if current_vol >= self.expansion_factor * prior_avg_vol and normalized_vol >= 1.0:
            return True

        return False