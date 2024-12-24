import pandas as pd
from datetime import datetime
import logging
from typing import Optional, Dict, Any

def is_volume_sufficient(
    df: pd.DataFrame,
    current_index: int,
    lookback_bars: int = 5,
    min_ratio: float = 0.8
) -> bool:

    if current_index < 1:
        return False

    current_vol = df.iloc[current_index]['tick_volume']
    start_idx = max(current_index - lookback_bars, 0)
    recent_vol = df['tick_volume'].iloc[start_idx:current_index]

    if len(recent_vol) == 0:
        return False

    avg_vol = recent_vol.mean()
    return current_vol >= (min_ratio * avg_vol)



class SignalGenerator:
    def __init__(
        self,
        valid_levels,
        params,
        log_file: str = "signals_debug.log"
    ):
        self.valid_levels = valid_levels
        self.params = params
        self.bounce_registry = {}

        # --- Setup Logger ---
        self.logger = logging.getLogger("SignalGenerator")
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info("SignalGenerator initialized with log_file=%s", log_file)

    def generate_signal(self, df_segment: pd.DataFrame) -> Dict[str, Any]:
        # 2) Identify last bar in df_segment
        last_idx = len(df_segment) - 1
        if last_idx < 0:
            return self._create_no_signal("df_segment has no rows")

        last_bar = df_segment.iloc[last_idx]
        current_time = last_bar['time']

        # 3) Volume check
        if not is_volume_sufficient(df_segment, last_idx):
            return self._create_no_signal("Volume too low...")

        # 4) If volume is OK, proceed:
        bar_volume = last_bar.get('tick_volume', 0.0)
        signal_dict = self._check_levels(last_bar, current_time, bar_volume)
        return signal_dict

    def _check_levels(self, last_bar, current_time, bar_volume) -> Dict[str, Any]:
        close_ = float(last_bar['close'])
        open_ = float(last_bar['open'])
        high_ = float(last_bar['high'])
        low_ = float(last_bar['low'])
        bullish = close_ > open_
        bearish = close_ < open_
        tol = 0.0003

        for lvl in self.valid_levels:
            near_support = (abs(low_ - lvl) <= tol and bullish)
            near_resistance = (abs(high_ - lvl) <= tol and bearish)
            if not (near_support or near_resistance):
                continue

            # We have a potential bounce off this level
            self.logger.debug(
                f"Potential bounce at level={lvl}, barTime={current_time}, "
                f"volume={bar_volume}, bullish={bullish}, bearish={bearish}"
            )

            signal = self._process_bounce(lvl, bar_volume, current_time, near_support)
            if signal and signal["type"] != "NONE":
                return signal

        return self._create_no_signal("No bounce off any level")

    def _process_bounce(self, level, volume, time, is_support) -> Optional[Dict[str, Any]]:
        self.logger.debug(f"_process_bounce called with level={level}, volume={volume}, is_support={is_support}")

        if level not in self.bounce_registry:
            # Mark first bounce
            self.bounce_registry[level] = {
                "first_bounce_volume": volume,
                "timestamp": time
            }
            reason = f"[1st bounce] Mark volume={volume} at lvl={level}"
            self.logger.debug(reason)
            return self._create_no_signal(reason)

        # We have a registry for this level (means second bounce?)
        first_vol = self.bounce_registry[level]["first_bounce_volume"]
        if volume < first_vol:
            self.logger.debug(f"Second bounce volume {volume} < first bounce volume {first_vol}, skipping signal")
            return None

        if is_support:
            bounce_type = "BUY"
            reason = f"Bouncing support {level} with second bounce volume >= first ({volume} >= {first_vol})"
        else:
            bounce_type = "SELL"
            reason = f"Bouncing resistance {level} with second bounce volume >= first ({volume} >= {first_vol})"

        self.logger.debug(f"Signal generated: {bounce_type} @ level={level}, volume={volume}")
        return {
            "type": bounce_type,
            "strength": 0.8,
            "reasons": [reason],
            "level": level
        }

    def _create_no_signal(self, reason: str) -> Dict[str, Any]:
        self.logger.debug(f"No signal: {reason}")
        return {
            "type": "NONE",
            "strength": 0.0,
            "reasons": [reason],
            "level": None
        }

