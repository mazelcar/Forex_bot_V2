import pandas as pd
from datetime import datetime
import logging
from typing import Optional, Dict, Any

def is_volume_sufficient(
    df: pd.DataFrame,
    current_index: int,
    lookback_bars: int = 20,
    min_ratio: float = 0.5
) -> bool:
    """
    Checks if the current bar's volume is at least `min_ratio`
    times the average of the last `lookback_bars` volumes.
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


def get_signal_logger(name="SignalGenerator", log_file="signals_debug.log"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class SignalGenerator:
    """
    Checks for bounces at valid_levels with volume confirmation.
    """
    def __init__(
        self,
        valid_levels,
        params,
        log_file: str = "signals_debug.log"
    ):
        self.valid_levels = valid_levels
        self.params = params
        self.bounce_registry = {}

        # Setup logger
        self.logger = get_signal_logger(log_file=log_file)

        # Adding this line: Initialize statistics dictionary
        self.signal_stats = {
            "volume_filtered": 0,
            "first_bounce_recorded": 0,
            "second_bounce_low_volume": 0,
            "signals_generated": 0,
            "tolerance_misses": 0  # Adding this line: Track near misses of tolerance
        }

    def generate_signal(self, df_segment: pd.DataFrame) -> Dict[str, Any]:
        last_idx = len(df_segment) - 1
        if last_idx < 0:
            return self._create_no_signal("Segment has no rows")

        last_bar = df_segment.iloc[last_idx]

        # Volume check with stats tracking
        if not is_volume_sufficient(df_segment, last_idx):
            # Adding this line: Track volume filtered signals
            self.signal_stats["volume_filtered"] += 1
            return self._create_no_signal("Volume too low vs. recent average")

        # Check if bar is bullish or bearish
        close_ = float(last_bar['close'])
        open_ = float(last_bar['open'])
        high_ = float(last_bar['high'])
        low_ = float(last_bar['low'])
        bullish = close_ > open_
        bearish = close_ < open_

        # Tolerance for "touch"
        tol = 0.0005

        for lvl in self.valid_levels:
            near_support = bullish and (abs(low_ - lvl) <= tol)
            near_resistance = bearish and (abs(high_ - lvl) <= tol)

            # Adding this block: Track near misses
            if bullish and not near_support and abs(low_ - lvl) <= tol * 2:
                self.signal_stats["tolerance_misses"] += 1
            if bearish and not near_resistance and abs(high_ - lvl) <= tol * 2:
                self.signal_stats["tolerance_misses"] += 1

            if near_support or near_resistance:
                # We have potential bounce
                self.logger.debug(f"Potential bounce at level={lvl}, barTime={last_bar['time']}, volume={last_bar['tick_volume']}")
                signal = self._process_bounce(lvl, float(last_bar['tick_volume']), last_bar['time'], near_support)
                if signal and signal["type"] != "NONE":
                    # Adding this line: Track successful signals
                    self.signal_stats["signals_generated"] += 1
                    return signal

        return self._create_no_signal("No bounce off valid levels")

    def _process_bounce(self, level, volume, time, is_support) -> Optional[Dict[str, Any]]:
        """
        1. If first bounce at this level, record it
        2. If second bounce, check:
            - Time since last bounce (cooldown)
            - Volume requirements
            Then generate BUY/SELL if conditions met
        """
        if level not in self.bounce_registry:
            # Mark first bounce
            self.bounce_registry[level] = {
                "first_bounce_volume": volume,
                "timestamp": time,
                "last_trade_time": None
            }
            # Adding this line: Track first bounces
            self.signal_stats["first_bounce_recorded"] += 1
            reason = f"[1st bounce] volume={volume} at lvl={level}"
            self.logger.debug(reason)
            return self._create_no_signal(reason)

        # Check cooldown period if we already traded this level
        if self.bounce_registry[level].get("last_trade_time"):
            last_trade = pd.to_datetime(self.bounce_registry[level]["last_trade_time"])
            current_time = pd.to_datetime(time)
            cooldown_period = pd.Timedelta(hours=4)  # 4-hour cooldown

            if current_time - last_trade < cooldown_period:
                self.logger.debug(f"Level {level} in cooldown. Time since last trade: {current_time - last_trade}")
                return self._create_no_signal(f"Level {level} in cooldown period")

        # If we have an existing bounce record
        first_vol = self.bounce_registry[level]["first_bounce_volume"]

        if volume < first_vol * 0.7:  # Allow 70% of first bounce volume
            # Adding this line: Track second bounces with insufficient volume
            self.signal_stats["second_bounce_low_volume"] += 1
            self.logger.debug(f"Second bounce volume {volume} < first {first_vol}*0.7, skipping signal")
            return None

        if is_support:
            bounce_type = "BUY"
            reason = f"Bouncing support {level} with second-bounce vol {volume} >= {first_vol}*0.7"
        else:
            bounce_type = "SELL"
            reason = f"Bouncing resistance {level} with second-bounce vol {volume} >= {first_vol}*0.7"

        # Update last trade time for this level
        self.bounce_registry[level]["last_trade_time"] = time

        self.logger.debug(f"Signal generated: {bounce_type} @ level={level}, volume={volume}")
        return {
            "type": bounce_type,
            "strength": 0.8,
            "reasons": [reason],
            "level": level
        }


    def validate_signal(self, signal, df_segment) -> bool:
        # Placeholder for extra validations if needed
        return signal["type"] != "NONE"


    def _create_no_signal(self, reason: str) -> Dict[str, Any]:
        self.logger.debug(f"No signal: {reason}")
        return {
            "type": "NONE",
            "strength": 0.0,
            "reasons": [reason],
            "level": None
        }
