"""MA RSI Volume Strategy Implementation for Forex Trading Bot V2.

This module implements a trading strategy that combines:
1. Dynamic EMA crossovers
2. RSI confirmation with adaptive levels
3. Volume analysis with smart thresholds
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


from src.strategy.validation.indicator_validator import IndicatorValidator
from .base import Strategy
from .validation.data_validator import DataValidator
from src.strategy.optimization.performance_optimizer import PerformanceOptimizer

import logging
logger = logging.getLogger('audit')


class MA_RSI_Volume_Strategy(Strategy):
    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
        self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']
        self.rsi_period = self.config['indicators']['rsi']['period']
        self.volume_period = self.config['indicators']['volume']['period']

        self.rsi_center = self.config['indicators']['rsi']['dynamic_levels']['center_line']['base']
        self.rsi_ob = self.config['indicators']['rsi']['dynamic_levels']['extreme_levels']['base_overbought']
        self.rsi_os = self.config['indicators']['rsi']['dynamic_levels']['extreme_levels']['base_oversold']

        self.market_phase = "unknown"
        self.last_signals: Dict[str, Dict] = {}
        self.risk_percent = 0.01  # Risk 1% per trade
        self.default_pip_value_per_lot = 10.0  # For EURUSD
        self.point_value = 0.0001  # EURUSD point

        # Initialize DataValidator
        self.data_validator = DataValidator(self.config)
        self.indicator_validator = IndicatorValidator(self.config)

        # Add performance optimizer
        self.performance_optimizer = PerformanceOptimizer(config_file)

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        return self._calculate_atr(data, period=14)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            # Ensure we have enough data
            if len(data) < period + 1:
                return 0.0

            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            # Calculate the three differences
            tr1 = high[1:] - low[1:]  # Current high - current low
            tr2 = abs(high[1:] - close[:-1])  # Current high - previous close
            tr3 = abs(low[1:] - close[:-1])  # Current low - previous close

            # Stack the differences and find the maximum
            true_ranges = np.vstack([tr1, tr2, tr3])
            true_range = np.max(true_ranges, axis=0)

            # Calculate ATR
            atr = np.mean(true_ranges[-period:])
            return float(atr) if not np.isnan(atr) else 0.0

        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            return 0.0

    def _adjust_parameters(self) -> None:
        """Adjust strategy parameters based on current market conditions."""
        try:
            # Use class-level market condition that was already analyzed
            market_condition = self.current_market_condition
            volatility = self.current_volatility

            # Adjust periods based on both volatility and market condition
            if market_condition.get('phase') == 'trending':
                self.fast_ema_period = min(
                    self.config['indicators']['moving_averages']['fast_ma']['dynamic_adjustment']['volatility_based']['high_volatility'],
                    5  # Minimum period
                )
                self.rsi_period = max(self.rsi_period, 14)
            else:  # ranging market
                self.fast_ema_period = max(
                    self.config['indicators']['moving_averages']['fast_ma']['period'],
                    8  # Default period
                )
                self.rsi_period = min(self.rsi_period, 10)

            # Adjust based on volatility thresholds from config
            if volatility > self.config['market_context']['volatility_measurement']['thresholds']['high']:
                self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['dynamic_adjustment']['volatility_based']['high_volatility']
                self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['dynamic_adjustment']['volatility_based']['high_volatility']
            elif volatility < self.config['market_context']['volatility_measurement']['thresholds']['low']:
                self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['dynamic_adjustment']['volatility_based']['low_volatility']
                self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['dynamic_adjustment']['volatility_based']['low_volatility']

        except Exception as e:
            print(f"Error in _adjust_parameters: {str(e)}")
            # Revert to default periods if there's an error
            self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
            self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        # Run validations through validators
        data_validation = self.data_validator.validate_basic_data(data)
        if not data_validation['overall_pass']:
            return {'type': 'NONE', 'strength': 0}

        indicator_validation = self.indicator_validator.validate_indicator_warmup(data)
        if not indicator_validation['overall_pass']:
            return {'type': 'NONE', 'strength': 0}

        try:
            min_periods = max(self.fast_ema_period, self.slow_ema_period, self.rsi_period, 20)

            if len(data) < min_periods:
                logger.info(f"Insufficient data: {len(data)} < {min_periods}")
                return {'type': 'NONE', 'strength': 0}

            # Calculate indicators
            fast_ema = data['close'].ewm(span=self.fast_ema_period, adjust=False).mean()
            slow_ema = data['close'].ewm(span=self.slow_ema_period, adjust=False).mean()
            rsi = self._calculate_rsi(data['close'])

            # Market conditions
            trend = self._analyze_market_trend(data)
            volume = self._analyze_volume_conditions(data)
            volume_ratio = volume['volume_ratio']
            current_rsi = rsi.iloc[-1]
            crossover = self._detect_crossover(
                fast_ema=fast_ema,
                slow_ema=slow_ema,
                volume_ratio=volume_ratio,
                current_rsi=current_rsi
            )

            current_rsi = rsi.iloc[-1]
            current_close = data['close'].iloc[-1]

            logger.info("\nDetailed Market Analysis:")
            logger.info(f"Current Price: {current_close:.5f}")
            logger.info(f"RSI: {current_rsi:.2f}")
            logger.info(f"Trend: {trend['direction']} ({trend['strength']})")
            logger.info(f"Volume Ratio: {volume['volume_ratio']:.2f}")
            logger.info(f"Signal Type: {crossover['type']}")

            signal = {'type': 'NONE', 'strength': 0}

            if crossover['type'] == 'BULLISH':
                logger.info("\nChecking Buy Conditions:")
                logger.info(f"RSI < 80: {current_rsi < 80} ({current_rsi:.2f})")
                logger.info(f"RSI > 20: {current_rsi > 20} ({current_rsi:.2f})")
                logger.info(f"Volume > 0.5: {volume['volume_ratio'] > 0.5} ({volume['volume_ratio']:.2f})")
                logger.info(f"Valid Trend (UP): {trend['direction'] == 'UP'} ({trend['direction']})")
                logger.info(f"Trend Strength: {trend['strength']}")

                if (current_rsi > 20 and
                    current_rsi < 80 and
                    volume['volume_ratio'] > 0.2 and
                    trend['direction'] in ['UP', 'DOWN', 'MIXED']):

                    # Debug logs before assigning signal
                    logger.info("\n[DEBUG] Bullish conditions met:")
                    logger.info(f"  RSI condition: {20 < current_rsi < 80}")
                    logger.info(f"  Volume condition: {volume['volume_ratio'] > 0.5}")
                    logger.info(f"  Trend condition: {trend['direction']} in ['UP', 'MIXED']")

                    signal = {
                        'type': 'BUY',
                        'strength': crossover['strength'],
                        'entry_price': current_close
                    }
                    logger.info("\nBUY Signal Generated!")
                    logger.info(f"Entry Price: {current_close:.5f}")
                    logger.info(f"Signal Strength: {signal['strength']:.2f}")

            elif crossover['type'] == 'BEARISH':
                logger.info("\nChecking Sell Conditions:")
                logger.info(f"RSI > 20: {current_rsi > 20} ({current_rsi:.2f})")
                logger.info(f"RSI < 80: {current_rsi < 80} ({current_rsi:.2f})")
                logger.info(f"Volume > 0.5: {volume['volume_ratio'] > 0.5} ({volume['volume_ratio']:.2f})")
                logger.info(f"Valid Trend (DOWN): {trend['direction'] in ['DOWN', 'MIXED']} ({trend['direction']})")
                logger.info(f"Trend Strength: {trend['strength']}")

                if (current_rsi > 20 and
                    current_rsi < 80 and
                    volume['volume_ratio'] > 0.5 and
                    trend['direction'] in ['DOWN', 'MIXED'] and
                    abs(crossover['spread']) > 0.5):

                    # Debug logs before assigning signal
                    logger.info("\n[DEBUG] Bearish conditions met:")
                    logger.info(f"  RSI condition: {20 < current_rsi < 80}")
                    logger.info(f"  Volume condition: {volume['volume_ratio'] > 0.5}")
                    logger.info(f"  Trend condition: {trend['direction']} in ['DOWN', 'MIXED']")

                    signal = {
                        'type': 'SELL',
                        'strength': crossover['strength'],
                        'entry_price': current_close
                    }
                    logger.info("\nSELL Signal Generated!")
                    logger.info(f"Entry Price: {current_close:.5f}")
                    logger.info(f"Signal Strength: {signal['strength']:.2f}")

            # Additional detailed logging before returning the final signal
            logger.info("\n=== Detailed Signal Decision Logging ===")
            logger.info(f"Detected crossover type: {crossover['type']}")
            logger.info(f"Signal candidate: {signal['type']} with strength {signal['strength']:.2f}")
            logger.info(f"  EMA Conditions: Fast EMA={fast_ema.iloc[-1]:.5f}, Slow EMA={slow_ema.iloc[-1]:.5f}")
            logger.info(f"  RSI={current_rsi:.2f}, RSI OB={self.rsi_ob}, RSI OS={self.rsi_os}")
            logger.info(f"  Volume Ratio={volume['volume_ratio']:.2f}, Above Avg={volume['above_average']}, High Volume={volume['high_volume']}")
            logger.info(f"  Trend Direction={trend['direction']}, Trend Strength={trend['strength']}")
            logger.info(f"  Final decision: Returning {signal}")

            return signal

        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'type': 'NONE', 'strength': 0}




    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


    def _analyze_market_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze the market trend using multiple timeframes."""
        try:
            # Use shorter periods for trend analysis
            ema10 = data['close'].ewm(span=10, adjust=False).mean()
            ema20 = data['close'].ewm(span=20, adjust=False).mean()
            ema50 = data['close'].ewm(span=50, adjust=False).mean()

            current_price = data['close'].iloc[-1]

            # Calculate price position relative to EMAs
            price_above_10 = current_price > ema10.iloc[-1]
            price_above_20 = current_price > ema20.iloc[-1]
            price_above_50 = current_price > ema50.iloc[-1]

            # Calculate EMA alignment
            ema_10_above_20 = ema10.iloc[-1] > ema20.iloc[-1]
            ema_20_above_50 = ema20.iloc[-1] > ema50.iloc[-1]

            # Determine trend direction with adjusted conditions
            if price_above_10 and price_above_20 and price_above_50 and ema_10_above_20 and ema_20_above_50:
                direction = 'UP'
                strength = 'STRONG'
            elif not price_above_10 and not price_above_20 and not price_above_50 and not ema_10_above_20 and not ema_20_above_50:
                direction = 'DOWN'
                strength = 'STRONG'
            elif (price_above_10 and price_above_20) or (ema_10_above_20 and price_above_20):
                direction = 'UP'
                strength = 'MODERATE'
            elif (not price_above_10 and not price_above_20) or (not ema_10_above_20 and not price_above_20):
                direction = 'DOWN'
                strength = 'MODERATE'
            else:
                direction = 'MIXED'
                strength = 'WEAK'

            return {
                'direction': direction,
                'strength': strength,
                'price_above_10': price_above_10,
                'price_above_20': price_above_20,
                'price_above_50': price_above_50,
                'ema_aligned': ema_10_above_20 and ema_20_above_50
            }

        except Exception as e:
            print(f"Error analyzing market trend: {str(e)}")
            return {
                'direction': 'MIXED',
                'strength': 'WEAK',
                'price_above_10': False,
                'price_above_20': False,
                'price_above_50': False,
                'ema_aligned': False
            }

    def _analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        try:
            atr = self._calculate_atr(data, period=14)
            ema_200 = data['close'].ewm(span=200, adjust=False).mean()
            if len(ema_200) >= 2:
                current_price = data['close'].iloc[-1]
                last_ema = ema_200.iloc[-1]
                if last_ema != 0:
                    trend_strength = (current_price - last_ema) / last_ema
                else:
                    trend_strength = 0
            else:
                trend_strength = 0

            if 'tick_volume' in data.columns:
                volume = data['tick_volume']
            elif 'real_volume' in data.columns:
                volume = data['real_volume']
            else:
                volume = pd.Series([0])

            volume_sma = volume.rolling(window=20).mean()
            if len(volume_sma) > 0 and volume_sma.iloc[-1] != 0:
                volume_strength = (volume.iloc[-1] / volume_sma.iloc[-1]) - 1
            else:
                volume_strength = 0

            if trend_strength > 0.01:
                phase = "uptrend"
            elif trend_strength < -0.01:
                phase = "downtrend"
            else:
                phase = "ranging"

            return {
                'phase': phase,
                'volatility': atr,
                'trend_strength': trend_strength,
                'volume_strength': volume_strength
            }
        except:
            return {
                'phase': 'unknown',
                'volatility': 0,
                'trend_strength': 0,
                'volume_strength': 0
            }

    def _analyze_volume_conditions(self, data: pd.DataFrame) -> Dict:
        """Analyze volume with more lenient conditions."""
        try:
            if 'tick_volume' in data.columns:
                volume = data['tick_volume']
            elif 'real_volume' in data.columns:
                volume = data['real_volume']
            else:
                return {'above_average': True, 'high_volume': False, 'volume_ratio': 1.0}

            # Calculate shorter-term volume averages
            volume_sma = volume.rolling(window=min(self.volume_period, 10)).mean()
            volume_std = volume.rolling(window=min(self.volume_period, 10)).std()

            if len(volume_sma) == 0 or volume_sma.iloc[-1] == 0:
                return {'above_average': True, 'high_volume': False, 'volume_ratio': 1.0}

            current_volume = volume.iloc[-1]
            sma = volume_sma.iloc[-1]

            # More lenient volume ratio calculation
            volume_ratio = current_volume / sma if sma > 0 else 1.0

            return {
                'above_average': volume_ratio > 0.6,  # More lenient threshold
                'high_volume': volume_ratio > 1.0,
                'volume_ratio': volume_ratio
            }
        except Exception as e:
            print(f"Error in volume analysis: {e}")
            return {'above_average': True, 'high_volume': False, 'volume_ratio': 1.0}

    def _detect_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series, volume_ratio: float = 0.0, current_rsi: float = 50.0) -> Dict:

        try:
            # Get recent values
            fast_values = fast_ema.iloc[-5:].values
            slow_values = slow_ema.iloc[-5:].values

            # Step 1: Initial Position Check
            is_above = fast_values[-1] > slow_values[-1]
            logger.info("\nStep 1: Position Check")
            logger.info(f"Fast above Slow: {is_above}")
            logger.info(f"Fast EMA: {fast_values[-1]:.5f}")
            logger.info(f"Slow EMA: {slow_values[-1]:.5f}")

            # Step 2: Movement Analysis
            fast_diff = (fast_values[-1] - fast_values[-2]) * 10000
            slow_diff = (slow_values[-1] - slow_values[-2]) * 10000
            ema_spread = (fast_values[-1] - slow_values[-1]) * 10000

            logger.info("\nStep 2: Movement Analysis (in pips)")
            logger.info(f"Fast Movement: {fast_diff:.1f} (threshold: 0.03)")
            logger.info(f"Slow Movement: {slow_diff:.1f}")
            logger.info(f"EMA Spread: {ema_spread:.1f} (threshold: 0.2)")

            # Step 3: Trend Analysis
            fast_trend = (fast_values[-1] - fast_values[0]) * 10000
            slow_trend = (slow_values[-1] - slow_values[0]) * 10000

            logger.info("\nStep 3: Trend Analysis (in pips)")
            logger.info(f"Fast 5-bar Trend: {fast_trend:.1f}")
            logger.info(f"Slow 5-bar Trend: {slow_trend:.1f}")

            # Step 4: Bullish Conditions Check
            bullish_conditions = {
                'is_above': is_above,
                'fast_movement': fast_diff > 0.02,
                'ema_spread': ema_spread > 0.1
            }

            logger.info("\nStep 4: Bullish Conditions Check")
            for condition, result in bullish_conditions.items():
                logger.info(f"{condition}: {'PASS' if result else 'FAIL'}")

            # Step 5: Bearish Conditions Check
            bearish_conditions = {
                'is_below': not is_above,
                'fast_movement': abs(fast_diff) > 0.02,
                'fast_trend': fast_trend < -0.2,
                'ema_spread': ema_spread < -0.3
            }

            logger.info("\nStep 5: Bearish Conditions Check")
            for condition, result in bearish_conditions.items():
                logger.info(f"{condition}: {'PASS' if result else 'FAIL'}")

            # Final Signal Decision
            if all(bullish_conditions.values()):
                trend_strength = min((fast_trend / 3.0), 1.0)
                logger.info(f"\nStep 6: BULLISH Signal Generated (Strength: {trend_strength:.2f})")
                return {
                    'type': 'BULLISH',
                    'strength': trend_strength,
                    'momentum': 'Up',
                    'spread': ema_spread
                }
            elif all(bearish_conditions.values()):
                trend_strength = min((abs(fast_trend) / 3.0), 1.0)
                logger.info(f"\nStep 6: BEARISH Signal Generated (Strength: {trend_strength:.2f})")
                return {
                    'type': 'BEARISH',
                    'strength': trend_strength,
                    'momentum': 'Down',
                    'spread': ema_spread
                }

            logger.info("\nStep 6: No Signal Generated - Conditions not met")
            logger.info("\nDetailed Crossover Analysis:")
            logger.info(f"Price movement in pips: {fast_diff:.1f}")
            logger.info(f"EMA separation: {ema_spread:.1f} pips")
            logger.info(f"Current volume ratio: {volume_ratio:.2f}")
            logger.info(f"RSI value: {current_rsi:.2f}")
            logger.info(f"Market phase: {self.current_market_condition.get('phase')}")

            return {
                'type': 'NONE',
                'strength': 0,
                'momentum': 'Flat',
                'spread': ema_spread
            }

        except Exception as e:
            logger.error(f"Error in signal detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'type': 'NONE', 'strength': 0, 'momentum': 'Error'}
    def _calculate_signal_strength(self, crossover: Dict, rsi: float,
                                   volume: Dict, fast_ema: float, slow_ema: float) -> float:
        strength = crossover['strength']

        if crossover['type'] == 'BULLISH':
            rsi_factor = (self.rsi_ob - rsi) / (self.rsi_ob - self.rsi_os)
        else:
            rsi_factor = (rsi - self.rsi_os) / (self.rsi_ob - self.rsi_os)

        volume_factor = volume['volume_ratio']

        total_strength = (
            strength * self.config['signal_strength']['calculation']['weights']['ema']['base'] +
            rsi_factor * self.config['signal_strength']['calculation']['weights']['rsi']['base'] +
            volume_factor * self.config['signal_strength']['calculation']['weights']['volume']['base']
        )

        return min(max(total_strength, 0), 1)

    def calculate_position_size(self, account_info: Dict, data: pd.DataFrame) -> float:
        """Calculate conservative position size with dynamic risk adjustment."""
        try:
            balance = account_info.get('balance', 10000)

            # Reduce risk for trending conditions
            if self.current_market_condition.get('phase') == 'trending':
                self.risk_percent = 0.005  # 0.5% risk in trends
            else:
                self.risk_percent = 0.0025  # 0.25% risk in ranging

            risk_amount = balance * self.risk_percent

            # Use ATR for pip distance calculation
            atr_pips = (self._calculate_atr(data) / self.point_value)
            pip_distance = max(atr_pips * 1.5, 25.0)  # At least 25 pips or 1.5 * ATR

            lot_size = risk_amount / (self.default_pip_value_per_lot * pip_distance)
            return max(min(lot_size, 0.5), 0.01)  # Cap at 0.5 lots, minimum 0.01

        except Exception as e:
            print(f"Error in position sizing: {e}")
            return 0.01  # Default to minimum size on error

    def calculate_stop_loss(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate dynamic stop loss based on ATR and market conditions."""
        try:
            atr = self._calculate_atr(data)
            current_price = data['close'].iloc[-1]

            # Use larger stops in trending markets
            atr_multiplier = 2.0 if self.current_market_condition.get('phase') == 'trending' else 1.5

            if signal['type'] == 'BUY':
                return current_price - (atr * atr_multiplier)
            else:
                return current_price + (atr * atr_multiplier)

        except Exception as e:
            print(f"Error calculating stop loss: {e}")
            return None

    def _calculate_volatility_adjustment(self) -> float:
        """Calculate volatility-based adjustment factor."""
        if not hasattr(self, 'current_volatility') or self.current_volatility == 0:
            return 0.0

        base_volatility = self.config['market_context']['volatility_measurement']['thresholds']['normal'][0]
        current_vol = self.current_volatility

        # Calculate adjustment factor (0.0 to 1.0)
        adjustment = (current_vol - base_volatility) / base_volatility
        return max(min(adjustment, 1.0), 0.0)

    def calculate_take_profit(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate dynamic take profit with market-based adjustments."""
        try:
            atr = self._calculate_atr(data)
            current_price = data['close'].iloc[-1]

            # Use larger targets in trending markets
            atr_multiplier = 4.0 if self.current_market_condition.get('phase') == 'trending' else 3.0

            if signal['type'] == 'BUY':
                return current_price + (atr * atr_multiplier)
            else:
                return current_price - (atr * atr_multiplier)

        except Exception as e:
            print(f"Error calculating take profit: {e}")
            return None

    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        # Identify key support/resistance levels
        pivots = self._calculate_pivot_points(data)

        # Analyze price action patterns
        patterns = self._identify_patterns(data)

        # Check if price is near significant levels
        near_level = self._check_price_levels(data, pivots)

        return {
            'pivots': pivots,
            'patterns': patterns,
            'near_level': near_level
        }
