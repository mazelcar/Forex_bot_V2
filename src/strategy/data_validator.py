import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict
import pytz

class DataValidator:
    def __init__(self, log_file: str = "data_validation.log"):
        self.logger = self._setup_logger(log_file)

        # Market schedule constants
        self.market_open = {
            0: True,  # Monday
            1: True,  # Tuesday
            2: True,  # Wednesday
            3: True,  # Thursday
            4: True,  # Friday
            5: False, # Saturday
            6: False  # Sunday
        }

        # Expected gaps by timeframe
        self.TIMEFRAME_MINUTES = {
            'M5': 5,
            'M15': 15,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }

        # EURUSD specific thresholds
        self.MAX_NORMAL_SPREAD = 0.0003  # 3 pips
        self.MAX_NEWS_SPREAD = 0.0010    # 10 pips
        self.MIN_VOLUME_PERCENTILE = 5
        self.MAX_PRICE_CHANGE = 0.003    # 30 pips

        # Initialize quality metrics
        self.quality_metrics = self._init_metrics()

    def _init_metrics(self) -> Dict:
        return {
            'invalid_bars': 0,
            'true_gaps_detected': 0,
            'weekend_gaps': 0,
            'session_gaps': 0,
            'expected_gaps': 0,  # Add this for runner.py compatibility
            'low_volume_bars': 0,
            'high_spread_bars': 0,
            'suspicious_prices': 0,
            'quality_score': 0.0
        }

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DataValidator')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(fh)
        return logger

    def validate_and_clean_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        """Main validation function"""
        if df.empty:
            return df, self._init_metrics()

        # Reset metrics
        self.quality_metrics = self._init_metrics()

        try:
            df_clean = df.copy()

            # 1. Ensure consistent timezone handling
            df_clean['time'] = pd.to_datetime(df_clean['time'])
            if df_clean['time'].dt.tz is None:
                df_clean['time'] = df_clean['time'].dt.tz_localize(pytz.UTC)
            elif df_clean['time'].dt.tz != pytz.UTC:
                df_clean['time'] = df_clean['time'].dt.tz_convert(pytz.UTC)

            # 2. Sort and handle gaps
            df_clean = df_clean.sort_values('time').reset_index(drop=True)
            df_clean = self._handle_gaps(df_clean, timeframe)

            # 3. Clean invalid OHLC data
            df_clean = self._clean_invalid_prices(df_clean)

            # 4. Handle spreads and volume
            df_clean = self._validate_market_data(df_clean)

            # 5. Calculate final quality score
            self.quality_metrics['quality_score'] = self._calculate_quality_score(df_clean, df)

            return df_clean, self.quality_metrics

        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return df, self.quality_metrics

    def _handle_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Process and validate time gaps"""
        expected_minutes = self.TIMEFRAME_MINUTES.get(timeframe, 15)
        expected_gap = pd.Timedelta(minutes=expected_minutes)

        time_diffs = df['time'].diff()

        for i, diff in time_diffs[1:].items():
            if pd.isna(diff):
                continue

            current_time = df['time'].iloc[i]
            prev_time = df['time'].iloc[i-1]

            # Skip if gap is normal
            if diff <= expected_gap * 1.5:
                continue

            # Check weekend gap
            if prev_time.weekday() == 4 and current_time.weekday() == 0:
                gap_hours = diff.total_seconds() / 3600
                if 48 <= gap_hours <= 72:  # Normal weekend gap
                    self.quality_metrics['weekend_gaps'] += 1
                    continue

            # Check if outside trading hours
            if not self.market_open[current_time.weekday()]:
                self.quality_metrics['session_gaps'] += 1
                continue

            # True gap during trading hours
            self.quality_metrics['true_gaps_detected'] += 1
            self.logger.warning(f"Unexpected {diff} gap at {prev_time}")

        return df

    def _clean_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data"""
        # Check OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )

        # Check price changes
        df['price_change'] = abs(df['close'].pct_change())
        suspicious_mask = df['price_change'] > self.MAX_PRICE_CHANGE

        # Update metrics
        self.quality_metrics['invalid_bars'] = invalid_mask.sum()
        self.quality_metrics['suspicious_prices'] = suspicious_mask.sum()

        # Remove invalid bars
        df_clean = df[~(invalid_mask | suspicious_mask)].copy()
        df_clean.drop('price_change', axis=1, inplace=True)

        return df_clean

    def _validate_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate spreads and volumes"""
        # Calculate spreads
        df['spread'] = df['high'] - df['low']

        # Set spread thresholds
        df['max_spread'] = self.MAX_NORMAL_SPREAD
        news_hours = [8, 12, 14]  # Common news hours (UTC)
        df.loc[df['time'].dt.hour.isin(news_hours), 'max_spread'] = self.MAX_NEWS_SPREAD

        # Volume threshold
        min_volume = df['tick_volume'].quantile(self.MIN_VOLUME_PERCENTILE / 100)

        # Create masks
        spread_mask = df['spread'] > df['max_spread']
        volume_mask = df['tick_volume'] < min_volume

        # Update metrics
        self.quality_metrics['high_spread_bars'] = spread_mask.sum()
        self.quality_metrics['low_volume_bars'] = volume_mask.sum()

        # Clean data
        df_clean = df[~(spread_mask | volume_mask)].copy()
        df_clean.drop(['spread', 'max_spread'], axis=1, inplace=True)

        return df_clean

    def _calculate_quality_score(self, clean_df: pd.DataFrame, original_df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        if len(original_df) == 0:
            return 0.0

        # Calculate component scores
        completeness = len(clean_df) / len(original_df)
        gap_score = max(0, 1 - (self.quality_metrics['true_gaps_detected'] / len(original_df)))
        price_score = max(0, 1 - (self.quality_metrics['suspicious_prices'] / len(original_df)))

        # Weighted final score
        quality_score = (
            completeness * 0.4 +
            gap_score * 0.3 +
            price_score * 0.3
        ) * 100

        return round(quality_score, 2)

    def is_data_valid_for_trading(self, quality_metrics: Dict) -> Tuple[bool, str]:
        """Check if data meets minimum quality standards"""
        MIN_QUALITY_SCORE = 85
        MAX_TRUE_GAPS = 3

        if quality_metrics['quality_score'] < MIN_QUALITY_SCORE:
            return False, f"Quality score {quality_metrics['quality_score']}% below minimum {MIN_QUALITY_SCORE}%"

        if quality_metrics['true_gaps_detected'] > MAX_TRUE_GAPS:
            return False, f"Too many unexpected gaps: {quality_metrics['true_gaps_detected']}"

        if quality_metrics['suspicious_prices'] > len(quality_metrics) * 0.01:
            return False, f"Too many suspicious prices: {quality_metrics['suspicious_prices']}"

        return True, "Data passed quality checks"