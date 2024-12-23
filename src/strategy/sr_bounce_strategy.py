import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

from src.strategy.technical_indicators import calculate_atr
from src.strategy.price_analysis import detect_price_hovering
from src.strategy.level_analysis import (
    identify_yearly_extremes,
    merge_levels_to_zones,
    calculate_cleanliness_score
)

# Validation
from src.strategy.volume_analysis import (
    calculate_volume_ma,
    is_volume_sufficient,
    VolumeValidator
)
from src.strategy.news_validator import NewsValidator

# Signal and Trade Management
from src.strategy.signal_generator import SignalGenerator
from src.strategy.trade_manager import TradeManager

class SR_Bounce_Strategy:
    """
    Shows an advanced S/R bounce approach with:
      - 2-bounce volume logic
      - optional momentum filter (RSI + ADX)
      - news avoidance
      - dynamic exit conditions
    """

    def __init__(self, config_file: Optional[str] = None, logger: logging.Logger = None,
                 news_file: str = "config/market_news.json"):
        # 1. Load config first
        self.params = {
            "min_touches": 8,
            "min_volume_threshold": 380000.0,
            "margin_pips": 0.0030,
            "session_filter": True,
            "session_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15],
            "risk_reward": 2.0,
            "lookforward_minutes": 30,
            "use_momentum_filter": True
        }
        if config_file:
            self._load_config(config_file)

        # 2. Initialize logger
        self.logger = logger or self._create_default_logger()

        # 3. Initialize validators and core components
        #    Replace the old threshold-based volume validator with the new approach
        #    For example, requiring 1.2x expansion factor, 3-bar lookback, time-of-day normalization
        self.volume_validator = VolumeValidator(
            expansion_factor=1.2,
            lookback_bars=3,
            time_adjustment=True
        )

        # For news events
        self.news_validator = NewsValidator(
            news_file=news_file,
            lookforward_minutes=self.params["lookforward_minutes"]
        )

        self.valid_levels = []
        self.avg_atr = 0.0005

        # 4. Initialize signal generator and trade manager
        #    Now pass self.volume_validator into SignalGenerator
        self.signal_generator = SignalGenerator(
            self.volume_validator,
            self.news_validator,
            self.valid_levels,
            self.params,
            log_file="results/signals_debug.log"
        )
        self.trade_manager = TradeManager(self.params["risk_reward"])

    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            self.params.update(user_cfg)
        except Exception as e:
            print(f"[WARNING] Unable to load {config_file}: {e}")

    def _create_default_logger(self):
        logger = logging.getLogger("SR_Bounce_Strategy")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    def analyze_higher_timeframe_levels(self, df_htf: pd.DataFrame):
        print("Entering analyze_higher_timeframe_levels with df_htf of length:", len(df_htf))
        if df_htf.empty:
            self.logger.warning("[analyze_higher_timeframe_levels] Received empty df.")
            return

        # Get yearly extreme zones
        high_zone, low_zone = identify_yearly_extremes(df_htf, buffer_pips=self.params["margin_pips"])
        print(f"Identified extreme zones: high_zone={high_zone}, low_zone={low_zone}")

        # 1) Create potential zones from rolling 20-bar windows
        potential_levels = []
        for period in range(len(df_htf) - 20):
            window = df_htf.iloc[period : period + 20]
            high_price = window['high'].max()
            low_price  = window['low'].min()

            # (Optional) skip levels in extreme zones if desired
            # if (low_zone[0] <= high_price <= low_zone[1] or
            #     high_zone[0] <= high_price <= high_zone[1] or
            #     low_zone[0] <= low_price  <= low_zone[1]  or
            #     high_zone[0] <= low_price <= high_zone[1]):
            #     continue

            potential_levels.extend([high_price, low_price])

        # Remove duplicates and sort
        potential_levels = sorted(set(potential_levels))
        print(f"Potential levels (before merging): {potential_levels[:20]} ... (showing first 20 if large)")

        # 2) Merge close levels into zones
        level_zones = merge_levels_to_zones(potential_levels, pip_threshold=0.0003)
        print("Zones after merging:", level_zones[:10], "... (showing first 10 if large)")

        # 3) Validate each zone with cleanliness & volume
        valid_levels = []
        for zone_start, zone_end in level_zones:
            zone_mid = (zone_start + zone_end) / 2
            score = calculate_cleanliness_score(df_htf, zone_mid)

            print(f"Zone ({zone_start:.4f}-{zone_end:.4f}) mid={zone_mid:.4f}, cleanliness_score={score}")

            if score >= 4:  # Only keep levels with good cleanliness
                volume_ma = calculate_volume_ma(df_htf['tick_volume'])
                if is_volume_sufficient(df_htf['tick_volume'].iloc[-1], volume_ma.iloc[-1]):
                    valid_levels.append(zone_mid)

        # 4) Assign final valid_levels, update signal generator
        self.valid_levels = valid_levels
        self.signal_generator.valid_levels = valid_levels

        # 5) Calculate average ATR for reference
        self.avg_atr = calculate_atr(df_htf, period=14)

        print("Final valid_levels:", self.valid_levels)
        self.logger.info(f"[SR_Bounce_Strategy] Valid levels loaded: {len(self.valid_levels)}")

    def generate_signals(self, df_segment: pd.DataFrame) -> Dict[str, Any]:
        return self.signal_generator.generate_signal(df_segment)

    def validate_signal(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> bool:
        return self.signal_generator.validate_signal(signal, df_segment)

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        return self.trade_manager.calculate_stop_loss(signal, df_segment)

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        return self.trade_manager.calculate_position_size(account_balance, stop_distance)

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        return self.trade_manager.calculate_take_profit(entry_price, sl)

    def check_exit(self, df_segment: pd.DataFrame, position: Dict[str, Any]) -> Tuple[bool, str]:
        return self.trade_manager.check_exit_conditions(df_segment, position)
