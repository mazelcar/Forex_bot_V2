# --------------------------------------------------------------
# sr_bounce_strategy.py
# --------------------------------------------------------------
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd


def get_strategy_logger(name="SR_Bounce_Strategy", debug=False) -> logging.Logger:
    """Create or retrieve the strategy logger, avoiding duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class SR_Bounce_Strategy:
    """
    Main strategy class responsible for:
      - FTMO-like rule checks
      - S/R level detection
      - Generating signals and trades

    Simplified version that:
      - Lowers volume thresholds to allow more trades
      - Reduces correlation constraints
      - Relaxes the second bounce requirement
      - Shortens the cooldown from 2 hours to 1 hour
      - Reduces min_touches in S/R identification
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Adjusted to lower 'risk_reward' from 3.0 to 2.0,
        lowered volume thresholds, relaxed bounce checks,
        and correlation checks so we can see more trades.
        """
        self.logger = logger or get_strategy_logger()

        # Per-symbol settings: we now allow lower volume thresholds
        self.pair_settings = {
            "EURUSD": {"min_volume_threshold": 500, "risk_reward": 2.0},  # Was 1200
            "GBPUSD": {"min_volume_threshold": 600, "risk_reward": 2.0},  # Was 1500
        }

        # Data storage
        self.symbol_data = {}
        self.symbol_levels = {}
        self.symbol_bounce_registry = {}

        # Correlation data (limit raised to 0.90 to allow more trades)
        self.symbol_correlations = {
            "EURUSD": {"GBPUSD": 0.0, "USDJPY": 0.0},
        }

        # FTMO-like limits
        self.ftmo_limits = {
            "daily_loss_per_pair": 5000,
            "total_exposure": 25000,
            "correlation_limit": 0.90,  # was 0.75
            "max_correlated_positions": 2,
        }

        # Default symbol
        self.default_symbol = "EURUSD"
        self.symbol_data[self.default_symbol] = pd.DataFrame()
        self.symbol_levels[self.default_symbol] = []
        self.symbol_bounce_registry[self.default_symbol] = {}

        # Account/trade-limiting parameters
        self.initial_balance = 100000.0
        self.current_balance = self.initial_balance
        self.daily_high_balance = self.initial_balance
        self.daily_drawdown_limit = 0.05
        self.max_drawdown_limit = 0.10
        self.profit_target = 0.10
        self.max_positions = 3
        self.max_daily_trades = 8
        self.max_spread = 0.002  # 20 pips
        self.last_reset = datetime.now().date()
        self.daily_trades = {}

        # Track some signal stats
        self.signal_stats = {
            "volume_filtered": 0,
            "first_bounce_recorded": 0,
            "second_bounce_low_volume": 0,
            "signals_generated": 0,
            "tolerance_misses": 0,
        }

        # Create a default SignalGenerator for self.default_symbol
        self.signal_generator = self.SignalGenerator(
            valid_levels=self.symbol_levels[self.default_symbol],
            logger=self.logger,
            debug=False,
            parent_strategy=self,
        )

    # -------------------------------------------------------------------------
    # FTMO Checks
    # -------------------------------------------------------------------------
    def _validate_ftmo_rules(
        self, current_time: datetime, spread: float, symbol: str = "EURUSD"
    ) -> Tuple[bool, str]:
        """
        Check multiple FTMO-like rules:
          1) Daily trade limit
          2) Total exposure
          3) Correlation limit
          4) Daily loss limit
          5) Spread limit
        """
        trade_date = current_time.date()

        # Reset daily counters if needed
        if trade_date != self.last_reset:
            self.daily_trades = {}
            self.last_reset = trade_date

        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []

        # 1) Check daily trade limit
        passed, reason = self._check_daily_trade_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 2) Check total exposure
        passed, reason = self._check_exposure_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 3) Check correlation
        passed, reason = self._check_correlation_limit(symbol)
        if not passed:
            return False, reason

        # 4) Check daily loss limit
        passed, reason = self._check_daily_loss_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 5) Check spread
        if spread > self.max_spread:
            return False, f"Spread too high for {symbol}: {spread:.5f}"

        return True, "Trade validated"

    def _check_daily_trade_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        daily_trades_count = len(
            [
                t
                for t in self.daily_trades.get(symbol, [])
                if pd.to_datetime(t["time"]).date() == trade_date
            ]
        )
        if daily_trades_count >= self.max_daily_trades:
            return (
                False,
                f"Daily trade limit reached for {symbol} ({daily_trades_count}/{self.max_daily_trades})",
            )
        return True, ""

    def _check_exposure_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        total_exposure = sum(
            abs(t.get("exposure", 0))
            for sym in self.daily_trades
            for t in self.daily_trades[sym]
            if pd.to_datetime(t["time"]).date() == trade_date
        )
        if total_exposure >= self.ftmo_limits["total_exposure"]:
            return (
                False,
                f"Total exposure limit reached ({total_exposure}/{self.ftmo_limits['total_exposure']})",
            )
        return True, ""

    def _check_correlation_limit(self, symbol: str) -> Tuple[bool, str]:
        active_pairs = [sym for sym in self.daily_trades if self.daily_trades[sym]]
        if symbol not in self.symbol_correlations:
            return True, ""  # No correlation data for this symbol
        for other_symbol in active_pairs:
            corr_val = abs(self.symbol_correlations[symbol].get(other_symbol, 0.0))
            if corr_val > self.ftmo_limits["correlation_limit"]:
                return (
                    False,
                    f"Correlation too high between {symbol} and {other_symbol}",
                )
        return True, ""

    def _check_daily_loss_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        symbol_daily_loss = abs(
            min(
                0,
                sum(
                    t.get("pnl", 0)
                    for t in self.daily_trades.get(symbol, [])
                    if pd.to_datetime(t["time"]).date() == trade_date
                ),
            )
        )
        if symbol_daily_loss >= self.ftmo_limits["daily_loss_per_pair"]:
            return False, f"Daily loss limit reached for {symbol}"
        return True, ""

    # -------------------------------------------------------------------------
    # S/R Identification
    # -------------------------------------------------------------------------
    def identify_sr_weekly(
        self,
        df_h1: pd.DataFrame,
        symbol: str = "EURUSD",
        weeks: int = 12,
        chunk_size: int = 24,
        weekly_buffer: float = 0.0003,
    ) -> List[float]:
        """
        Identify significant S/R levels from H1 data over the last `weeks` weeks,
        grouped into chunks of `chunk_size` bars, with a small merging buffer.
        Lower min_touches from ~7-8 to ~3 for more lenient detection.
        """
        try:
            if df_h1.empty:
                self.logger.error(f"Empty dataframe in identify_sr_weekly for {symbol}")
                return []

            recent_df = self._filter_recent_weeks(df_h1, weeks)
            if recent_df.empty:
                self.logger.warning(f"No data after filtering for {weeks} weeks: {symbol}")
                return []

            volume_threshold = self._compute_volume_threshold(recent_df)
            potential_levels = self._collect_potential_levels(
                recent_df, chunk_size, volume_threshold, symbol
            )
            merged_levels = self._merge_close_levels(potential_levels, weekly_buffer)

            self.symbol_levels[symbol] = merged_levels
            self.logger.info(f"Identified {len(merged_levels)} valid S/R levels for {symbol}")
            return merged_levels

        except Exception as e:
            self.logger.error(f"Error in identify_sr_weekly for {symbol}: {str(e)}")
            return []

    def _filter_recent_weeks(self, df: pd.DataFrame, weeks: int) -> pd.DataFrame:
        last_time = pd.to_datetime(df["time"].max())
        cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
        recent_df = df[df["time"] >= cutoff_time].copy()
        recent_df.sort_values("time", inplace=True)
        return recent_df

    def _compute_volume_threshold(self, recent_df: pd.DataFrame) -> float:
        avg_volume = recent_df["tick_volume"].mean()
        return avg_volume * 1.5

    def _collect_potential_levels(
        self,
        recent_df: pd.DataFrame,
        chunk_size: int,
        volume_threshold: float,
        symbol: str,
    ) -> List[float]:
        potential_levels = []
        for i in range(0, len(recent_df), chunk_size):
            window = recent_df.iloc[i : i + chunk_size]
            if len(window) < chunk_size / 2:
                continue
            high = float(window["high"].max())
            low = float(window["low"].min())
            high_volume = float(window.loc[window["high"] == high, "tick_volume"].max())
            low_volume = float(window.loc[window["low"] == low, "tick_volume"].max())

            # Check volumes relative to threshold
            if high_volume > volume_threshold:
                potential_levels.append(high)
                self.logger.debug(f"{symbol} High level found {high:.5f} vol {high_volume}")
            if low_volume > volume_threshold:
                potential_levels.append(low)
                self.logger.debug(f"{symbol} Low level found {low:.5f} vol {low_volume}")

        potential_levels = sorted(set(potential_levels))
        return potential_levels

    def _merge_close_levels(
        self, potential_levels: List[float], buffer_val: float
    ) -> List[float]:
        merged = []
        for lvl in potential_levels:
            if not merged or abs(lvl - merged[-1]) > buffer_val:
                merged.append(lvl)
            else:
                # Merge close levels into their midpoint
                merged[-1] = (merged[-1] + lvl) / 2.0
        return merged

    def update_weekly_levels(
        self, df_h1: pd.DataFrame, symbol: str = "EURUSD", weeks: int = 3, weekly_buffer: float = 0.00060
    ):
        """Update or create weekly S/R levels for the given symbol, from H1 data."""
        try:
            w_levels = self.identify_sr_weekly(
                df_h1, symbol=symbol, weeks=weeks, weekly_buffer=weekly_buffer
            )
            if not w_levels:
                self.logger.warning(f"No weekly levels found for {symbol}")
                return

            w_levels = [float(level) for level in w_levels]
            self.symbol_levels[symbol] = w_levels
            self.logger.info(f"Updated valid levels for {symbol}. Total: {len(w_levels)}")

            # Attach or update a signal generator for this symbol
            if symbol == self.default_symbol:
                self.signal_generator.valid_levels = w_levels
            else:
                signal_gen_attr = f"signal_generator_{symbol}"
                if not hasattr(self, signal_gen_attr):
                    setattr(
                        self,
                        signal_gen_attr,
                        self.SignalGenerator(
                            valid_levels=w_levels,
                            logger=self.logger,
                            debug=False,
                            parent_strategy=self,
                        ),
                    )
                else:
                    getattr(self, signal_gen_attr).valid_levels = w_levels

        except Exception as e:
            self.logger.error(f"Error updating weekly levels for {symbol}: {str(e)}")

    # -------------------------------------------------------------------------
    # Signal and Trade Management
    # -------------------------------------------------------------------------
    def generate_signals(self, df_segment: pd.DataFrame, symbol="EURUSD"):
        """Generate signals by delegating to the correct SignalGenerator for the symbol."""
        if symbol == self.default_symbol:
            return self.signal_generator.generate_signal(df_segment, symbol)
        signal_gen = getattr(self, f"signal_generator_{symbol}", None)
        if signal_gen is None:
            self.logger.warning(f"No signal generator for {symbol}, creating one.")
            signal_gen = self.SignalGenerator(
                valid_levels=self.symbol_levels.get(symbol, []),
                logger=self.logger,
                debug=False,
                parent_strategy=self,
            )
            setattr(self, f"signal_generator_{symbol}", signal_gen)
        return signal_gen.generate_signal(df_segment, symbol)

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        """
        Slightly widened SL. 0.0012 pips buffer to reduce quick wicks.
        """
        if df_segment.empty:
            return 0.0
        last_bar = df_segment.iloc[-1]
        low = float(last_bar["low"])
        high = float(last_bar["high"])

        pip_buffer = 0.0012

        if signal["type"] == "BUY":
            return low - pip_buffer
        else:
            return high + pip_buffer

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """
        1% risk model, fallback to min/max lots.
        """
        try:
            risk_amount = account_balance * 0.01
            stop_pips = stop_distance * 10000
            if stop_pips == 0:
                return 0.0
            position_size = risk_amount / (stop_pips * 10)
            position_size = min(position_size, 5.0)
            position_size = max(position_size, 0.01)
            return round(position_size, 2)
        except Exception as e:
            self.logger.error(f"Position sizing error: {str(e)}")
            return 0.01

    def calculate_take_profit(self, entry_price: float, sl: float, symbol: str) -> float:
        """Simple R:R based TP using pair_settings' risk_reward."""
        dist = abs(entry_price - sl)
        rr = self.pair_settings[symbol]["risk_reward"]
        if entry_price > sl:
            return entry_price + (dist * rr)
        else:
            return entry_price - (dist * rr)

    def check_exit_conditions(
        self, df_segment: pd.DataFrame, position: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Checks if the position hits SL or TP on the last bar of df_segment."""
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
        else:
            if current_price >= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price <= position["take_profit"]:
                return True, "Take profit hit"
        return False, "No exit condition met"

    def open_trade(
        self, current_segment: pd.DataFrame, balance: float, i: int, symbol: str = "EURUSD"
    ) -> Optional["SR_Bounce_Strategy.Trade"]:
        """
        Open trade if all FTMO checks pass, volume is enough,
        and a valid signal is generated.

        Added improvements:
          - Only allow trades between 07:00 and 17:00 UTC (time filter)
          - Require bar range >= 0.0005 to skip tiny bars (range filter)
        """

        if current_segment.empty:
            return None

        last_bar = current_segment.iloc[-1]
        current_time = pd.to_datetime(last_bar["time"])

        # -----------------------
        # 1) Time Window Filter
        # -----------------------
        bar_hour = current_time.hour
        if bar_hour < 7 or bar_hour > 17:
            self.logger.debug(f"[{symbol}] Skipping trade, out-of-hour range: {bar_hour}")
            return None

        # -----------------------
        # 2) Bar Range Filter
        # -----------------------
        bar_range = float(last_bar["high"]) - float(last_bar["low"])
        if bar_range < 0.0005:
            self.logger.debug(f"[{symbol}] Skipping trade, bar range too small: {bar_range:.5f}")
            return None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (Below is the same logic as before)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        current_spread = bar_range * 0.1
        can_trade, reason = self._validate_ftmo_rules(current_time, current_spread, symbol)
        if not can_trade:
            self.logger.debug(f"[{symbol}] FTMO check failed: {reason}")
            return None

        signal = self.generate_signals(current_segment, symbol=symbol)
        if signal["type"] == "NONE":
            return None

        if float(last_bar["tick_volume"]) < self.pair_settings[symbol]["min_volume_threshold"]:
            return None

        entry_price = float(last_bar["close"])
        stop_loss = self.calculate_stop_loss(signal, current_segment)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.00001:
            return None

        base_size = self.calculate_position_size(balance, stop_distance)
        take_profit = self.calculate_take_profit(entry_price, stop_loss, symbol)

        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            symbol=symbol,
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=base_size,
        )
        new_trade.level = signal.get("level", 0.0)
        new_trade.level_type = "Support" if signal["type"] == "BUY" else "Resistance"
        new_trade.distance_to_level = abs(entry_price - signal.get("level", entry_price))
        new_trade.entry_volume = float(last_bar["tick_volume"])
        new_trade.prev_3_avg_volume = float(current_segment["tick_volume"].tail(3).mean())
        new_trade.hour_avg_volume = float(current_segment["tick_volume"].tail(4).mean())

        if "reasons" in signal:
            new_trade.entry_reason = " + ".join(signal["reasons"])
        else:
            new_trade.entry_reason = "No specific reason"

        # Log trade to daily trades
        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []
        self.daily_trades[symbol].append(
            {
                "time": new_trade.open_time,
                "type": new_trade.type,
                "size": new_trade.size,
                "exposure": new_trade.size * 10000.0,
            }
        )
        return new_trade

    def exit_trade(
        self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade", symbol: str = "EURUSD"
    ) -> Tuple[bool, float, float]:
        """
        Checks if the trade hits SL or TP intrabar (based on the bar's high & low).
        If intrabar hit occurs, calculates fill_price accordingly.
        Otherwise, returns no exit.
        """

        if df_segment.empty:
            return False, 0.0, 0.0

        last_bar = df_segment.iloc[-1]
        bar_open = float(last_bar["open"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        bar_close = float(last_bar["close"])

        stop_loss = trade.sl
        take_profit = trade.tp

        # We assume the bar moves from OPEN -> HIGH/LOW -> CLOSE or OPEN -> LOW/HIGH -> CLOSE.
        if trade.type == "BUY":
            # If bar_low <= SL and bar_high >= TP, check which is closer to open
            if bar_low <= stop_loss and bar_high >= take_profit:
                dist_to_sl = abs(bar_open - stop_loss)
                dist_to_tp = abs(bar_open - take_profit)
                if dist_to_sl < dist_to_tp:
                    fill_price = stop_loss
                    reason = "Stop loss hit intrabar"
                else:
                    fill_price = take_profit
                    reason = "Take profit hit intrabar"
            elif bar_low <= stop_loss:
                fill_price = stop_loss
                reason = "Stop loss hit intrabar"
            elif bar_high >= take_profit:
                fill_price = take_profit
                reason = "Take profit hit intrabar"
            else:
                return False, 0.0, 0.0

            pnl = (fill_price - trade.entry_price) * 10000.0 * trade.size
            trade.exit_reason = reason

        else:
            # SELL trade
            if bar_high >= stop_loss and bar_low <= take_profit:
                dist_to_sl = abs(bar_open - stop_loss)
                dist_to_tp = abs(bar_open - take_profit)
                if dist_to_sl < dist_to_tp:
                    fill_price = stop_loss
                    reason = "Stop loss hit intrabar"
                else:
                    fill_price = take_profit
                    reason = "Take profit hit intrabar"
            elif bar_high >= stop_loss:
                fill_price = stop_loss
                reason = "Stop loss hit intrabar"
            elif bar_low <= take_profit:
                fill_price = take_profit
                reason = "Take profit hit intrabar"
            else:
                return False, 0.0, 0.0

            pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size
            trade.exit_reason = reason

        return True, fill_price, pnl

    def validate_cross_pair_exposure(
        self,
        new_trade: "SR_Bounce_Strategy.Trade",
        active_trades: Dict[str, Optional["SR_Bounce_Strategy.Trade"]],
        current_balance: float,
    ) -> Tuple[bool, str]:
        """
        Validate that adding new_trade won't exceed correlation or total lot constraints.
        """
        HIGH_CORR_THRESHOLD = 0.95
        MEDIUM_CORR_THRESHOLD = 0.70
        self.logger.info(f"[{new_trade.symbol}] Starting cross-pair validation. Size: {new_trade.size}")

        # Check total open lots
        total_open_lots = sum(t.size for t in active_trades.values() if t is not None)
        if total_open_lots + new_trade.size > 10.0:
            return (
                False,
                f"Total open lots would exceed limit: {total_open_lots + new_trade.size:.2f}",
            )

        # Check correlation adjustments
        new_sym = new_trade.symbol
        for sym, open_trade in active_trades.items():
            if open_trade is None or sym == new_sym:
                continue

            corr = abs(self.symbol_correlations.get(new_sym, {}).get(sym, 0.0))
            if corr > HIGH_CORR_THRESHOLD:
                return (
                    False,
                    f"Correlation {corr:.2f} with {sym} > {HIGH_CORR_THRESHOLD} => blocking trade.",
                )
            elif corr >= MEDIUM_CORR_THRESHOLD:
                old_size = new_trade.size
                new_trade.size = round(new_trade.size * 0.20, 2)
                if new_trade.size < 0.01:
                    return False, "Partial correlation reduction made size < 0.01 => skip trade."
                self.logger.info(
                    f"Reducing trade size from {old_size:.2f} to {new_trade.size:.2f}"
                    f" due to correlation {corr:.2f} with {sym}."
                )
        return True, "OK"

    # -------------------------------------------------------------------------
    # Inner Classes
    # -------------------------------------------------------------------------
    class Trade:
        """Tracks relevant data for a single trade lifecycle."""

        def __init__(
            self,
            open_i: int,
            open_time: str,
            symbol: str,
            type: str,
            entry_price: float,
            sl: float,
            tp: float,
            size: float,
        ):
            self.open_i = open_i
            self.open_time = open_time
            self.symbol = symbol
            self.type = type
            self.entry_price = entry_price
            self.sl = sl
            self.tp = tp
            self.size = size

            self.close_i: Optional[int] = None
            self.close_time: Optional[str] = None
            self.close_price: Optional[float] = None
            self.pnl: float = 0.0

            self.entry_volume: float = 0.0
            self.prev_3_avg_volume: float = 0.0
            self.hour_avg_volume: float = 0.0

            self.level: float = 0.0
            self.distance_to_level: float = 0.0
            self.level_type: str = ""
            self.entry_reason: str = ""
            self.exit_reason: str = ""
            self.level_source: str = ""
            self.level_touches: int = 0
            self.indicator_snapshot: dict = {}

        def pips(self) -> float:
            """Number of pips gained/lost so far."""
            if self.close_price is None:
                return 0.0
            raw_diff = (
                self.close_price - self.entry_price
                if self.type == "BUY"
                else self.entry_price - self.close_price
            )
            return raw_diff * 10000.0

        def profit(self) -> float:
            """Monetary profit of the trade."""
            return self.pips() * self.size

        def holding_time(self) -> pd.Timedelta:
            """Time in the trade."""
            if not self.close_time:
                return pd.Timedelta(0, unit="seconds")
            open_t = pd.to_datetime(self.open_time)
            close_t = pd.to_datetime(self.close_time)
            return close_t - open_t

        def to_dict(self) -> dict:
            """Return dictionary representation for reporting/logging."""
            return {
                "open_i": self.open_i,
                "open_time": self.open_time,
                "symbol": self.symbol,
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
                "entry_reason": self.entry_reason,
                "exit_reason": self.exit_reason,
                "level_source": self.level_source,
                "level_touches": self.level_touches,
                "indicator_snapshot": self.indicator_snapshot,
            }

    class SignalGenerator:
        """
        Simple bounce-based signal generator. Checks volume thresholds, correlation,
        and adjacency to known S/R levels.

        Adjusted logic:
          - Second bounce volume threshold lowered to 50% (was 80%)
          - Reduced bounce cooldown from 2 hours to 1 hour
          - Lowered min_touches to ~3 in the docstring
        """

        def __init__(
            self,
            valid_levels: List[float],
            logger: logging.Logger,
            debug: bool = False,
            parent_strategy: Optional["SR_Bounce_Strategy"] = None,
        ):
            self.valid_levels = valid_levels
            self.logger = logger
            self.debug = debug
            self.parent_strategy = parent_strategy
            self.bounce_registry: Dict[str, Dict] = {}
            self.signal_stats = {
                "volume_filtered": 0,
                "first_bounce_recorded": 0,
                "second_bounce_low_volume": 0,
                "signals_generated": 0,
                "tolerance_misses": 0,
            }
            # Symbol-specific config
            self.pair_settings = {
                "EURUSD": {
                    "min_touches": 3,
                    "min_volume_threshold": 500,
                    "margin_pips": 0.0030,
                    "tolerance": 0.0005,
                    "min_bounce_volume": 400,  # Was 1000
                },
                "GBPUSD": {
                    "min_touches": 3,
                    "min_volume_threshold": 600,
                    "margin_pips": 0.0035,
                    "tolerance": 0.0007,
                    "min_bounce_volume": 500,  # Was 1200
                },
            }
            # 1-hour cooldown instead of 2 hours
            self.bounce_cooldown = pd.Timedelta(hours=1)

        def generate_signal(self, df_segment: pd.DataFrame, symbol: str) -> Dict[str, Any]:
            """Generate a simple BUY/SELL signal if last bar is near an S/R level with enough volume."""
            last_idx = len(df_segment) - 1
            if last_idx < 0:
                return self._create_no_signal("Segment has no rows")

            settings = self.pair_settings.get(symbol, self.pair_settings["EURUSD"])
            correlation_threshold = 0.95
            if self.parent_strategy:
                correlations = self.parent_strategy.symbol_correlations.get(symbol, {})
            else:
                correlations = {}

            # Quick correlation block
            for other_symbol, corr_val in correlations.items():
                if abs(corr_val) > correlation_threshold:
                    reason = f"Correlation {corr_val:.2f} with {other_symbol} exceeds {correlation_threshold}"
                    return self._create_no_signal(reason)

            last_bar = df_segment.iloc[last_idx]
            last_bar_volume = float(last_bar["tick_volume"])

            # Volume check vs. threshold
            if last_bar_volume < settings["min_volume_threshold"]:
                self.signal_stats["volume_filtered"] += 1
                return self._create_no_signal("Volume too low vs. threshold")

            close_ = float(last_bar["close"])
            open_ = float(last_bar["open"])
            high_ = float(last_bar["high"])
            low_ = float(last_bar["low"])

            bullish = close_ > open_
            bearish = close_ < open_
            tol = settings["tolerance"]

            # Look for near support/resistance
            for lvl in self.valid_levels:
                near_support = bullish and (abs(low_ - lvl) <= tol)
                near_resistance = bearish and (abs(high_ - lvl) <= tol)
                distance_pips = abs(close_ - lvl) * 10000

                # skip if level is more than 15 pips away from close
                if distance_pips > 15:
                    continue

                if near_support or near_resistance:
                    self.logger.debug(
                        f"{symbol} potential bounce at level={lvl}, time={last_bar['time']}, vol={last_bar_volume}"
                    )
                    signal = self._process_bounce(
                        lvl, last_bar_volume, last_bar["time"], is_support=near_support, symbol=symbol
                    )
                    if signal and signal["type"] != "NONE":
                        self.signal_stats["signals_generated"] += 1
                        return signal

            # No near bounce identified
            return self._create_no_signal("No bounce off valid levels")

        def _process_bounce(
            self, level: float, volume: float, time_val: Any, is_support: bool, symbol: str
        ) -> Optional[Dict[str, Any]]:
            """
            Handle the first bounce registration and second bounce signal creation.
            Lowered required second bounce volume to 50% of first bounce.
            1-hour cooldown between trades on the same level.
            """
            settings = self.pair_settings.get(symbol, self.pair_settings["EURUSD"])
            if symbol not in self.bounce_registry:
                self.bounce_registry[symbol] = {}
            level_key = str(level)

            # If no bounce record, record the first bounce
            if level_key not in self.bounce_registry[symbol]:
                self.bounce_registry[symbol][level_key] = {
                    "first_bounce_volume": volume,
                    "timestamp": time_val,
                    "last_trade_time": None,
                }
                self.signal_stats["first_bounce_recorded"] += 1
                return self._create_no_signal(f"First bounce recorded for {symbol} at {level}")

            # If there's a recent bounce, ensure cooldown
            if self.bounce_registry[symbol][level_key].get("last_trade_time"):
                last_trade = pd.to_datetime(self.bounce_registry[symbol][level_key]["last_trade_time"])
                current_time = pd.to_datetime(time_val)
                if current_time - last_trade < self.bounce_cooldown:
                    return self._create_no_signal(f"Level {level} in cooldown for {symbol}")

            first_vol = self.bounce_registry[symbol][level_key]["first_bounce_volume"]
            min_vol_threshold = settings["min_bounce_volume"]

            # If second bounce has insufficient volume (<50% of first bounce) or below min_bounce_volume
            if volume < min_vol_threshold or volume < (first_vol * 0.50):
                self.signal_stats["second_bounce_low_volume"] += 1
                return self._create_no_signal("Second bounce volume insufficient")

            bounce_type = "BUY" if is_support else "SELL"
            reason = f"Valid bounce at {'support' if is_support else 'resistance'} {level} for {symbol}"
            self.bounce_registry[symbol][level_key]["last_trade_time"] = time_val
            self.signal_stats["signals_generated"] += 1
            return {
                "type": bounce_type,
                "strength": 0.8,
                "reasons": [reason],
                "level": level,
            }

        def _create_no_signal(self, reason: str) -> Dict[str, Any]:
            """Return a dict representing no-signal."""
            self.logger.debug(f"No signal: {reason}")
            return {"type": "NONE", "strength": 0.0, "reasons": [reason], "level": None}

