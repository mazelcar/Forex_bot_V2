import logging
from typing import Tuple
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

# from src.strategy.signal_generator import SignalGenerator




def get_strategy_logger(name="SR_Bounce_Strategy", debug=False):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO if not debug else logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if not debug else logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class SR_Bounce_Strategy:


    def __init__(
        self,
        config_file: Optional[str] = None,
        logger: logging.Logger = None,
        news_file: str = "config/market_news.json"
    ):
        # Default params
        self.params = {
            "min_touches": 8,
            "min_volume_threshold": 1200,
            "margin_pips": 0.0030,
            "risk_reward": 3.0,
            "lookforward_minutes": 30,
        }

        # Load config if specified
        if config_file:
            self._load_config(config_file)

        # Setup logger
        self.logger = logger or get_strategy_logger()

        # Multi-symbol data structures
        self.symbol_data = {}  # Store data per symbol
        self.symbol_levels = {}  # Store S/R levels per symbol
        self.symbol_bounce_registry = {}  # Store bounces per symbol

        # Correlation tracking
        self.symbol_correlations = {
            "EURUSD": {
                "GBPUSD": 0.0,
                "USDJPY": 0.0
            }
        }

        # FTMO multi-pair limits
        self.ftmo_limits = {
            "daily_loss_per_pair": 5000,  # $5000 max loss per pair
            "total_exposure": 25000,      # $25000 max total exposure
            "correlation_limit": 0.75,    # Max correlation between pairs
            "max_correlated_positions": 2  # Max number of correlated pairs
        }

        # Default symbol
        self.default_symbol = "EURUSD"

        # Initialize data structures for default symbol
        self.symbol_data[self.default_symbol] = []
        self.symbol_levels[self.default_symbol] = []
        self.symbol_bounce_registry[self.default_symbol] = {}

        # Existing FTMO Parameters - now per symbol
        self.initial_balance = 100000.0
        self.current_balance = self.initial_balance
        self.daily_high_balance = self.initial_balance
        self.daily_trades = {}  # Now stores trades per symbol

        # FTMO Limits
        self.daily_drawdown_limit = 0.05  # 5% daily
        self.max_drawdown_limit = 0.10    # 10% total
        self.profit_target = 0.10         # 10% profit target

        # Trading rules
        self.max_positions = 3
        self.max_daily_trades = 8
        self.max_spread = 0.002  # Maximum 2 pip spread

        self.last_reset = datetime.now().date()

        # Initialize signal generator with symbol awareness
        self.signal_stats = {
            "volume_filtered": 0,
            "first_bounce_recorded": 0,
            "second_bounce_low_volume": 0,
            "signals_generated": 0,
            "tolerance_misses": 0
        }

        # Initialize internal SignalGenerator
        self.signal_generator = self.SignalGenerator(
            valid_levels=self.symbol_levels[self.default_symbol],
            params=self.params,
            logger=self.logger,
            debug=False
        )


    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            self.params.update(user_cfg)
        except Exception as e:
            print(f"[WARNING] Unable to load {config_file}: {e}")

    def _validate_ftmo_rules(self, current_time: datetime, spread: float) -> Tuple[bool, str]:
        """Internal FTMO validation logic"""
        trade_date = pd.to_datetime(current_time).date()

        # Reset counters if needed
        if trade_date != self.last_reset:
            self.daily_trades = []
            self.last_reset = trade_date
            self.daily_pnl = 0.0

        # Check daily trade count
        daily_trades_count = len([t for t in self.daily_trades
                                if pd.to_datetime(t['time']).date() == trade_date])

        if daily_trades_count >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({daily_trades_count}/{self.max_daily_trades})"

        if abs(min(0, self.daily_pnl)) >= self.initial_balance * self.daily_drawdown_limit:
            return False, f"Daily drawdown limit reached"

        if spread > self.max_spread:
            return False, f"Spread too high: {spread:.5f}"

        return True, "Trade validated"


    def identify_sr_weekly(
        self,
        df_h1: pd.DataFrame,
        weeks: int = 12,
        chunk_size: int = 24,
        weekly_buffer: float = 0.0003
    ) -> List[float]:
        """
        Identify significant S/R levels from H1 data over the last `weeks` weeks.
        """
        try:
            if df_h1.empty:
                self.logger.error("Empty dataframe in identify_sr_weekly.")
                return []

            # Filter to last `weeks` weeks
            last_time = pd.to_datetime(df_h1["time"].max())
            cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
            recent_df = df_h1[df_h1["time"] >= cutoff_time].copy()
            recent_df.sort_values("time", inplace=True)

            self.logger.info(f"Analyzing data from {recent_df['time'].min()} to {recent_df['time'].max()}")

            if recent_df.empty:
                self.logger.warning("No data after filtering for recent weeks.")
                return []

            # Calculate average volume for significance
            avg_volume = recent_df['tick_volume'].mean()
            volume_threshold = avg_volume * 1.5

            potential_levels = []
            # Slide window of chunk_size bars
            for i in range(0, len(recent_df), chunk_size):
                window = recent_df.iloc[i:i + chunk_size]
                if len(window) < chunk_size / 2:  # skip small windows
                    continue

                # High & Low
                high = float(window['high'].max())
                low = float(window['low'].min())

                high_volume = float(window.loc[window['high'] == high, 'tick_volume'].max())
                low_volume = float(window.loc[window['low'] == low, 'tick_volume'].max())

                if high_volume > volume_threshold:
                    potential_levels.append(high)
                    self.logger.debug(f"High level found at {high:.5f} with volume {high_volume}")

                if low_volume > volume_threshold:
                    potential_levels.append(low)
                    self.logger.debug(f"Low level found at {low:.5f} with volume {low_volume}")

            # Sort & merge nearby
            potential_levels = sorted(set(potential_levels))
            merged_levels = []
            for lvl in potential_levels:
                if not merged_levels or abs(lvl - merged_levels[-1]) > weekly_buffer:
                    merged_levels.append(lvl)
                else:
                    merged_levels[-1] = (merged_levels[-1] + lvl) / 2.0

            self.logger.info(f"Identified {len(merged_levels)} valid S/R levels")
            return merged_levels

        except Exception as e:
            self.logger.error(f"Error in identify_sr_weekly: {str(e)}")
            return []


    def update_weekly_levels(self, df_h1, weeks: int = 3, weekly_buffer: float = 0.00060):
        """
        Update the strategy's valid levels using weekly S/R from identify_sr_weekly.
        """
        try:
            w_levels = self.identify_sr_weekly(
                df_h1,
                weeks=weeks,
                weekly_buffer=weekly_buffer
            )
            if not w_levels:
                self.logger.warning("No weekly levels found.")
                return

            self.valid_levels = w_levels
            self.logger.info(f"Updated valid levels. Total: {len(self.valid_levels)}")

            # Update signal generator's levels
            self.signal_generator.valid_levels = self.valid_levels

        except Exception as e:
            self.logger.error(f"Error updating weekly levels: {str(e)}")


    def generate_signals(self, df_segment):
        return self.signal_generator.generate_signal(df_segment)


    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        """Calculate stop loss based on signal type and recent price action."""
        if df_segment.empty:
            return 0.0

        last_bar = df_segment.iloc[-1]
        close_price = float(last_bar['close'])
        low = float(last_bar['low'])
        high = float(last_bar['high'])

        pip_buffer = 0.0008

        if signal["type"] == "BUY":
            stop_loss = low - pip_buffer
            self.logger.debug(f"Buy signal => SL set below last low: {stop_loss:.5f}")
        else:  # SELL
            stop_loss = high + pip_buffer
            self.logger.debug(f"Sell signal => SL set above last high: {stop_loss:.5f}")

        return stop_loss

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """
        Basic 1% risk model:
        - risk_amount = 1% of balance
        - position_size = risk_amount / (stop_pips * pip_value)
        """
        if account_balance <= 0:
            self.logger.error(f"Invalid account_balance: {account_balance}")
            return 0.01
        if stop_distance <= 0:
            self.logger.error(f"Invalid stop_distance: {stop_distance}")
            return 0.01

        risk_amount = account_balance * 0.01
        stop_pips = stop_distance * 10000.0
        pip_value = 10.0  # For EURUSD in 1 standard lot

        position_size = risk_amount / (stop_pips * pip_value)
        position_size = round(position_size, 2)

        if position_size < 0.01:
            position_size = 0.01  # ensure min

        self.logger.info(f"Position size calculated: {position_size} lots (Balance={account_balance}, Stop={stop_distance:.5f})")
        return position_size

    def calculate_take_profit(self, entry_price: float, sl: float) -> float:
        """
        If risk_reward=2.0, the distance from entry to SL is multiplied by 2
        for the TP distance.
        """
        dist = abs(entry_price - sl)
        if entry_price > sl:
            return entry_price + (dist * self.params["risk_reward"])
        else:
            return entry_price - (dist * self.params["risk_reward"])

    def check_exit_conditions(self, df_segment: pd.DataFrame, position: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if SL or TP is touched by the last bar's close.
        Return (should_close, reason).
        """
        if df_segment.empty:
            return False, "No data"

        last_bar = df_segment.iloc[-1]
        current_price = float(last_bar["close"])
        pos_type = position.get("type", "BUY")

        if pos_type == "BUY":
            if current_price <= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price >= position["take_profit"]:
                return True, "Take profit hit"
        else:  # SELL
            if current_price >= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price <= position["take_profit"]:
                return True, "Take profit hit"

        return False, "No exit condition met"


    def open_trade(self, current_segment, balance: float, i: int) -> Optional["SR_Bounce_Strategy.Trade"]:
        """
        Enhanced trade opening with FTMO safety checks
        """
        # Get current market conditions
        last_bar = current_segment.iloc[-1]
        current_time = pd.to_datetime(last_bar['time'])
        bar_range = float(last_bar['high']) - float(last_bar['low'])
        current_spread = bar_range * 0.1

        # FTMO validation using internal rules
        can_trade, reason = self._validate_ftmo_rules(
            current_time=current_time,
            spread=current_spread
        )

        if not can_trade:
            self.logger.debug(f"FTMO check failed: {reason}")
            return None

        # Continue with regular strategy
        signal = self.generate_signals(current_segment)
        if signal["type"] == "NONE":
            return None

        # Volume check
        if last_bar["tick_volume"] < self.params["min_volume_threshold"]:
            return None  # skip if volume too low

        entry_price = float(last_bar["close"])
        stop_loss = self.calculate_stop_loss(signal, current_segment)

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.00001:
            return None

        size = self.calculate_position_size(balance, stop_distance)
        take_profit = self.calculate_take_profit(entry_price, stop_loss)

        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=size
        )

        # Additional fields
        new_trade.level = signal.get('level', 0.0)
        new_trade.level_type = "Support" if signal["type"] == "BUY" else "Resistance"
        new_trade.distance_to_level = abs(entry_price - signal.get('level', entry_price))
        new_trade.entry_volume = float(last_bar['tick_volume'])
        new_trade.prev_3_avg_volume = float(current_segment['tick_volume'].tail(3).mean())
        new_trade.hour_avg_volume = float(current_segment['tick_volume'].tail(4).mean())

        # Add trade to our own tracking
        self.daily_trades.append({
            'time': new_trade.open_time,
            'type': new_trade.type,
            'size': new_trade.size
        })

        self.logger.debug(f"Opening trade: {signal['type']} at {entry_price:.5f}, level={new_trade.level}")
        return new_trade

    def exit_trade(self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade") -> Tuple[bool, float, float]:
        """
        Enhanced exit with FTMO tracking
        """
        position_dict = {
            "type": trade.type,
            "stop_loss": trade.sl,
            "take_profit": trade.tp
        }
        should_close, reason = self.check_exit_conditions(df_segment, position_dict)

        if should_close:
            last_bar = df_segment.iloc[-1]
            if reason == "Stop loss hit":
                fill_price = trade.sl
            elif reason == "Take profit hit":
                fill_price = trade.tp
            else:
                fill_price = float(last_bar["close"])

            if trade.type == "BUY":
                pnl = (fill_price - trade.entry_price) * 10000.0 * trade.size
            else:  # SELL
                pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size

            # Update daily PnL tracking for FTMO
            self.daily_pnl += pnl

            return True, fill_price, pnl

        return False, 0.0, 0.0


    class Trade:
        def __init__(
            self,
            open_i: int,
            open_time: str,
            type: str,
            entry_price: float,
            sl: float,
            tp: float,
            size: float
        ):
            self.open_i = open_i
            self.open_time = open_time
            self.type = type
            self.entry_price = entry_price
            self.sl = sl
            self.tp = tp
            self.size = size

            self.close_i = None
            self.close_time = None
            self.close_price = None
            self.pnl = 0.0

            self.entry_volume = 0.0
            self.prev_3_avg_volume = 0.0
            self.hour_avg_volume = 0.0

            self.level = 0.0
            self.distance_to_level = 0.0
            self.level_type = ""

        def to_dict(self) -> Dict:
            return {
                "open_i": self.open_i,
                "open_time": self.open_time,
                "type": self.type,
                "entry_price": self.entry_price,
                "sl": self.sl,
                "tp": self.tp,
                "size": self.size,
                "close_i": self.close_i,
                "close_time": self.close_time,
                "close_price": self.close_price,
                "pnl": self.pnl,
                "entry_volume": self.entry_volume,
                "prev_3_avg_volume": self.prev_3_avg_volume,
                "hour_avg_volume": self.hour_avg_volume,
                "level": self.level,
                "distance_to_level": self.distance_to_level,
                "level_type": self.level_type,
            }

    class SignalGenerator:
        """
        Inner class for signal generation and bounce detection
        """
        def __init__(self, valid_levels, params, logger, debug=False):
            self.valid_levels = valid_levels
            self.params = params
            self.logger = get_strategy_logger("SignalGenerator", debug=debug)
            self.bounce_registry = {}
            self.signal_stats = {
                "volume_filtered": 0,
                "first_bounce_recorded": 0,
                "second_bounce_low_volume": 0,
                "signals_generated": 0,
                "tolerance_misses": 0
            }

        def generate_signal(self, df_segment: pd.DataFrame) -> Dict[str, Any]:
            last_idx = len(df_segment) - 1
            if last_idx < 0:
                return self._create_no_signal("Segment has no rows")

            last_bar = df_segment.iloc[last_idx]

            # Volume check with stats tracking
            if not self._is_volume_sufficient(df_segment, last_idx):
                self.signal_stats["volume_filtered"] += 1
                return self._create_no_signal("Volume too low vs. recent average")

            # Check if bar is bullish/bearish
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

                # Add distance check
                distance_pips = abs(close_ - lvl) * 10000
                if distance_pips > 15:  # More than 15 pips away
                    continue  # Skip this level if too far

                # Track near misses
                if bullish and not near_support and abs(low_ - lvl) <= tol * 2:
                    self.signal_stats["tolerance_misses"] += 1
                if bearish and not near_resistance and abs(high_ - lvl) <= tol * 2:
                    self.signal_stats["tolerance_misses"] += 1

                if near_support or near_resistance:
                    # We have potential bounce
                    self.logger.debug(f"Potential bounce at level={lvl}, barTime={last_bar['time']}, volume={last_bar['tick_volume']}")
                    signal = self._process_bounce(lvl, float(last_bar['tick_volume']), last_bar['time'], near_support)
                    if signal and signal["type"] != "NONE":
                        self.signal_stats["signals_generated"] += 1
                        return signal

            return self._create_no_signal("No bounce off valid levels")

        def _create_no_signal(self, reason: str) -> Dict[str, Any]:
            self.logger.debug(f"No signal: {reason}")
            return {
                "type": "NONE",
                "strength": 0.0,
                "reasons": [reason],
                "level": None
            }

        def _process_bounce(self, level, volume, time, is_support) -> Optional[Dict[str, Any]]:
            """
            Process potential bounce:
            1. If first bounce at level, record it
            2. If second bounce, validate and generate signal
            """
            if level not in self.bounce_registry:
                # Mark first bounce
                self.bounce_registry[level] = {
                    "first_bounce_volume": volume,
                    "timestamp": time,
                    "last_trade_time": None
                }
                self.signal_stats["first_bounce_recorded"] += 1
                self.logger.debug(f"[1st bounce] volume={volume} at lvl={level}")
                return self._create_no_signal(f"First bounce recorded at {level}")

            # Check cooldown if already traded
            if self.bounce_registry[level].get("last_trade_time"):
                last_trade = pd.to_datetime(self.bounce_registry[level]["last_trade_time"])
                current_time = pd.to_datetime(time)
                cooldown_period = pd.Timedelta(hours=2)

                if current_time - last_trade < cooldown_period:
                    return self._create_no_signal(f"Level {level} in cooldown")

            # Process second bounce
            first_vol = self.bounce_registry[level]["first_bounce_volume"]
            if volume < first_vol * 0.6:
                self.signal_stats["second_bounce_low_volume"] += 1
                return self._create_no_signal("Second bounce volume insufficient")

            bounce_type = "BUY" if is_support else "SELL"
            reason = f"Valid bounce at {'support' if is_support else 'resistance'} {level}"

            # Update last trade time
            self.bounce_registry[level]["last_trade_time"] = time
            self.signal_stats["signals_generated"] += 1

            return {
                "type": bounce_type,
                "strength": 0.8,
                "reasons": [reason],
                "level": level
            }

        def _is_volume_sufficient(
            self,
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