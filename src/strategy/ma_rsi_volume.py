"""MA RSI Volume Strategy Implementation for Forex Trading Bot V2.

This module implements a trading strategy that combines:
1. Dynamic EMA crossovers
2. RSI confirmation with adaptive levels
3. Volume analysis with smart thresholds

The strategy adapts its parameters based on market conditions and includes:
- Volatility-based parameter adjustment
- Multiple timeframe analysis
- Advanced risk management

Author: mazelcar
Created: December 2024
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .base import Strategy

class MA_RSI_Volume_Strategy(Strategy):
    """Trading strategy using Moving Averages, RSI, and Volume analysis."""

    def __init__(self, config_file: str):
        """Initialize the strategy.

        Args:
            config_file: Path to strategy configuration JSON
        """
        super().__init__(config_file)

        # Initialize indicator parameters
        self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
        self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']
        self.rsi_period = self.config['indicators']['rsi']['period']
        self.volume_period = self.config['indicators']['volume']['period']

        # Initialize adaptive thresholds
        self.rsi_center = self.config['indicators']['rsi']['dynamic_levels']['center_line']['base']
        self.rsi_ob = self.config['indicators']['rsi']['dynamic_levels']['extreme_levels']['base_overbought']
        self.rsi_os = self.config['indicators']['rsi']['dynamic_levels']['extreme_levels']['base_oversold']

        # Track current market phase
        self.market_phase = "unknown"
        self.last_signals: Dict[str, Dict] = {}

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility using ATR.

        Args:
            data: Market data for volatility calculation

        Returns:
            Current ATR value
        """
        return self._calculate_atr(data, period=14)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range.

        Args:
            data: Market data
            period: ATR period

        Returns:
            ATR value
        """
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1]

    def _adjust_parameters(self) -> None:
        """Adjust strategy parameters based on market conditions."""
        volatility = self.current_volatility

        # Adjust MA periods based on volatility
        if volatility > self.config['market_context']['volatility_measurement']['thresholds']['high']:
            self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['dynamic_adjustment']['volatility_based']['high_volatility']
            self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['dynamic_adjustment']['volatility_based']['high_volatility']
        elif volatility < self.config['market_context']['volatility_measurement']['thresholds']['low']:
            self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['dynamic_adjustment']['volatility_based']['low_volatility']
            self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['dynamic_adjustment']['volatility_based']['low_volatility']
        else:
            self.fast_ema_period = self.config['indicators']['moving_averages']['fast_ma']['period']
            self.slow_ema_period = self.config['indicators']['moving_averages']['slow_ma']['period']

        # Adjust RSI levels based on market phase
        if self.market_phase == "trending":
            self.rsi_center += self.config['indicators']['rsi']['dynamic_levels']['center_line']['market_condition_modifiers']['trend_strength'][1]
        else:
            self.rsi_center = self.config['indicators']['rsi']['dynamic_levels']['center_line']['base']

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on strategy rules.

        Args:
            data: Market data for signal generation

        Returns:
            Dictionary containing signal information
        """
        try:
            # Calculate indicators
            fast_ema = data['close'].ewm(span=self.fast_ema_period, adjust=False).mean()
            slow_ema = data['close'].ewm(span=self.slow_ema_period, adjust=False).mean()
            rsi = self._calculate_rsi(data['close'])
            volume_signal = self._analyze_volume_conditions(data)

            # Detect crossovers
            crossover = self._detect_crossover(fast_ema, slow_ema)

            # Generate signal
            signal = self._combine_signals(crossover, rsi.iloc[-1], volume_signal)

            # Calculate signal strength
            signal['strength'] = self._calculate_signal_strength(
                crossover,
                rsi.iloc[-1],
                volume_signal,
                fast_ema.iloc[-1],
                slow_ema.iloc[-1]
            )

            self.last_signals[data['symbol'].iloc[-1]] = signal
            return signal

        except Exception as e:
            print(f"Error generating signals: {e}")
            return {'type': 'NONE', 'strength': 0}

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator.

        Args:
            prices: Price series

        Returns:
            RSI values
        """
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        """Analyze current market conditions.

        Args:
            data: Market data for analysis

        Returns:
            Dictionary containing market condition assessment
        """
        try:
            # Calculate ATR for volatility
            atr = self._calculate_atr(data, period=14)

            # Determine trend strength using EMA
            ema_200 = data['close'].ewm(span=200, adjust=False).mean()
            if len(ema_200) >= 2:  # Make sure we have enough data
                current_price = data['close'].iloc[-1]
                last_ema = ema_200.iloc[-1]
                if last_ema != 0:  # Prevent division by zero
                    trend_strength = (current_price - last_ema) / last_ema
                else:
                    trend_strength = 0
            else:
                trend_strength = 0

            # Analyze volume using tick_volume
            if 'tick_volume' in data.columns:
                volume = data['tick_volume']
                volume_sma = volume.rolling(window=20).mean()
                if len(volume_sma) > 0 and volume_sma.iloc[-1] != 0:
                    volume_strength = (volume.iloc[-1] / volume_sma.iloc[-1]) - 1
                else:
                    volume_strength = 0
            else:
                volume_strength = 0

            # Determine market phase
            if trend_strength > 0.02:
                phase = "uptrend"
            elif trend_strength < -0.02:
                phase = "downtrend"
            else:
                phase = "ranging"

            return {
                'phase': phase,
                'volatility': atr,
                'trend_strength': trend_strength,
                'volume_strength': volume_strength
            }

        except Exception as e:
            print(f"Error in market condition analysis: {e}")
            return {
                'phase': 'unknown',
                'volatility': 0,
                'trend_strength': 0,
                'volume_strength': 0
            }

    def _analyze_volume_conditions(self, data: pd.DataFrame) -> Dict:
        """Analyze volume conditions.

        Args:
            data: Market data

        Returns:
            Volume analysis results
        """
        try:
            # First try tick_volume, then real_volume
            if 'tick_volume' in data.columns:
                volume = data['tick_volume']
            elif 'real_volume' in data.columns:
                volume = data['real_volume']
            else:
                # If no volume data available, return default values
                return {
                    'above_average': True,
                    'high_volume': False,
                    'volume_ratio': 1.0
                }

            volume_sma = volume.rolling(window=self.volume_period).mean()
            volume_std = volume.rolling(window=self.volume_period).std()

            current_volume = volume.iloc[-1]
            sma = volume_sma.iloc[-1]

            if sma == 0:
                return {
                    'above_average': True,
                    'high_volume': False,
                    'volume_ratio': 1.0
                }

            upper_threshold = sma + (volume_std.iloc[-1] *
                self.config['indicators']['volume']['dynamic_thresholds']['threshold_multipliers']['high'])

            return {
                'above_average': current_volume > sma,
                'high_volume': current_volume > upper_threshold,
                'volume_ratio': current_volume / sma
            }

        except Exception as e:
            print(f"Volume analysis error: {e}")
            # Return default values if anything fails
            return {
                'above_average': True,
                'high_volume': False,
                'volume_ratio': 1.0
            }

    def _detect_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> Dict:
        """Detect EMA crossovers.

        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series

        Returns:
            Crossover signal information
        """
        current_diff = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        previous_diff = fast_ema.iloc[-2] - slow_ema.iloc[-2]

        if current_diff > 0 and previous_diff < 0:
            return {'type': 'BULLISH', 'strength': abs(current_diff)}
        elif current_diff < 0 and previous_diff > 0:
            return {'type': 'BEARISH', 'strength': abs(current_diff)}
        else:
            return {'type': 'NONE', 'strength': 0}

    def _combine_signals(self, crossover: Dict, rsi: float, volume: Dict) -> Dict:
        """Combine different signal components.

        Args:
            crossover: EMA crossover signal
            rsi: Current RSI value
            volume: Volume analysis results

        Returns:
            Combined trading signal
        """
        signal = {'type': 'NONE', 'strength': 0}

        if crossover['type'] == 'BULLISH':
            if rsi < self.rsi_ob and volume['above_average']:
                signal = {'type': 'BUY', 'strength': crossover['strength']}
        elif crossover['type'] == 'BEARISH':
            if rsi > self.rsi_os and volume['above_average']:
                signal = {'type': 'SELL', 'strength': crossover['strength']}

        return signal

    def _calculate_signal_strength(self, crossover: Dict, rsi: float,
                                 volume: Dict, fast_ema: float, slow_ema: float) -> float:
        """Calculate overall signal strength.

        Args:
            crossover: Crossover signal
            rsi: RSI value
            volume: Volume analysis
            fast_ema: Current fast EMA
            slow_ema: Current slow EMA

        Returns:
            Signal strength value
        """
        # Base strength from crossover
        strength = crossover['strength']

        # Adjust based on RSI
        if crossover['type'] == 'BULLISH':
            rsi_factor = (self.rsi_ob - rsi) / (self.rsi_ob - self.rsi_os)
        else:
            rsi_factor = (rsi - self.rsi_os) / (self.rsi_ob - self.rsi_os)

        # Adjust based on volume
        volume_factor = volume['volume_ratio']

        # Combine factors
        total_strength = (
            strength * self.config['signal_strength']['calculation']['weights']['ema']['base'] +
            rsi_factor * self.config['signal_strength']['calculation']['weights']['rsi']['base'] +
            volume_factor * self.config['signal_strength']['calculation']['weights']['volume']['base']
        )

        return min(max(total_strength, 0), 1)  # Normalize between 0 and 1

    def calculate_position_size(self, account_info: Dict) -> float:
        """Calculate position size based on risk parameters.

        Args:
            account_info: Current account information

        Returns:
            Position size in lots
        """
        # Implement position sizing based on:
        # 1. Account balance
        # 2. Risk per trade
        # 3. Current market volatility
        return 0.01  # Placeholder - implement actual calculation

    def calculate_stop_loss(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate stop loss price for a trade.

        Args:
            data: Market data
            signal: Signal information

        Returns:
            Stop loss price
        """
        # Implement stop loss calculation based on:
        # 1. ATR
        # 2. Recent swing levels
        # 3. Risk parameters
        return None  # Placeholder - implement actual calculation

    def calculate_take_profit(self, data: pd.DataFrame, signal: Dict) -> Optional[float]:
        """Calculate take profit price for a trade.

        Args:
            data: Market data
            signal: Signal information

        Returns:
            Take profit price
        """
        # Implement take profit calculation based on:
        # 1. Risk:Reward ratio
        # 2. Recent market structure
        # 3. Volatility
        return None  # Placeholder - implement actual calculation