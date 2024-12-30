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
        """
        Full constructor that loads configs, logger, multi-symbol structures,
        and attempts to load market news events from JSON.
        """
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

        # News file management
        self.news_file = news_file
        self._load_news_file()  # <-- NEW CALL

        # Multi-symbol data structures
        self.symbol_data = {}
        self.symbol_levels = {}
        self.symbol_bounce_registry = {}

        # Correlation tracking
        self.symbol_correlations = {
            "EURUSD": {
                "GBPUSD": 0.0,
                "USDJPY": 0.0
            }
        }

        # FTMO multi-pair limits
        self.ftmo_limits = {
            "daily_loss_per_pair": 5000,
            "total_exposure": 25000,
            "correlation_limit": 0.75,
            "max_correlated_positions": 2
        }

        # Default symbol
        self.default_symbol = "EURUSD"
        self.symbol_data[self.default_symbol] = []
        self.symbol_levels[self.default_symbol] = []
        self.symbol_bounce_registry[self.default_symbol] = {}

        # Existing FTMO Parameters
        self.initial_balance = 100000.0
        self.current_balance = self.initial_balance
        self.daily_high_balance = self.initial_balance
        self.daily_trades = {}
        self.daily_drawdown_limit = 0.05
        self.max_drawdown_limit = 0.10
        self.profit_target = 0.10
        self.max_positions = 3
        self.max_daily_trades = 8
        self.max_spread = 0.002
        self.last_reset = datetime.now().date()

        # Initialize signal stats
        self.signal_stats = {
            "volume_filtered": 0,
            "first_bounce_recorded": 0,
            "second_bounce_low_volume": 0,
            "signals_generated": 0,
            "tolerance_misses": 0
        }

        # Initialize internal SignalGenerator
        from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
        self.signal_generator = self.SignalGenerator(
            valid_levels=self.symbol_levels[self.default_symbol],
            params=self.params,
            logger=self.logger,
            debug=False,
            parent_strategy=self
        )

    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            self.params.update(user_cfg)
        except Exception as e:
            print(f"[WARNING] Unable to load {config_file}: {e}")

    def _load_news_file(self):
        """
        Loads news events from the JSON file specified by self.news_file.
        On error, logs and sets self.news_events = [] so we have no further crash.
        """
        try:
            with open(self.news_file, "r", encoding="utf-8") as f:
                self.news_events = json.load(f)
            self.logger.info(f"Loaded {len(self.news_events)} news events from {self.news_file}")
        except Exception as e:
            self.logger.error(f"Error loading {self.news_file}: {str(e)}")
            self.news_events = []

    def _validate_ftmo_rules(self, current_time: datetime, spread: float, symbol: str = "EURUSD") -> Tuple[bool, str]:
        """Enhanced FTMO validation logic with multi-pair support"""
        trade_date = pd.to_datetime(current_time).date()

        # Reset counters if needed
        if trade_date != self.last_reset:
            self.daily_trades = {}  # Reset as dictionary for all symbols
            self.last_reset = trade_date
            self.daily_pnl = 0.0

        # Initialize symbol tracking if needed
        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []

        # Check daily trade count for this symbol
        daily_trades_count = len([t for t in self.daily_trades.get(symbol, [])
                                if pd.to_datetime(t['time']).date() == trade_date])

        if daily_trades_count >= self.max_daily_trades:
            return False, f"Daily trade limit reached for {symbol} ({daily_trades_count}/{self.max_daily_trades})"

        # Calculate total exposure across all symbols
        total_exposure = sum(
            abs(t.get('exposure', 0))
            for sym in self.daily_trades
            for t in self.daily_trades[sym]
            if pd.to_datetime(t['time']).date() == trade_date
        )

        if total_exposure >= self.ftmo_limits["total_exposure"]:
            return False, f"Total exposure limit reached ({total_exposure}/{self.ftmo_limits['total_exposure']})"

        # Check correlation limits
        active_pairs = [sym for sym in self.daily_trades if self.daily_trades[sym]]
        if symbol in self.symbol_correlations:
            for other_symbol in active_pairs:
                if other_symbol in self.symbol_correlations[symbol]:
                    correlation = abs(self.symbol_correlations[symbol][other_symbol])
                    if correlation > self.ftmo_limits["correlation_limit"]:
                        return False, f"Correlation too high between {symbol} and {other_symbol}"

        # Check daily loss limit per symbol
        symbol_daily_loss = abs(min(0, sum(
            t.get('pnl', 0) for t in self.daily_trades.get(symbol, [])
            if pd.to_datetime(t['time']).date() == trade_date
        )))

        if symbol_daily_loss >= self.ftmo_limits["daily_loss_per_pair"]:
            return False, f"Daily loss limit reached for {symbol}"

        if spread > self.max_spread:
            return False, f"Spread too high for {symbol}: {spread:.5f}"

        return True, "Trade validated"


    def identify_sr_weekly(
        self,
        df_h1: pd.DataFrame,
        symbol: str = "EURUSD",
        weeks: int = 12,
        chunk_size: int = 24,
        weekly_buffer: float = 0.0003
    ) -> List[float]:
        """
        Identify significant S/R levels from H1 data over the last `weeks` weeks.
        Now symbol-aware and includes correlation checks.
        """
        try:
            if df_h1.empty:
                self.logger.error(f"Empty dataframe in identify_sr_weekly for {symbol}")
                return []

            # Filter to last `weeks` weeks
            last_time = pd.to_datetime(df_h1["time"].max())
            cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
            recent_df = df_h1[df_h1["time"] >= cutoff_time].copy()
            recent_df.sort_values("time", inplace=True)

            self.logger.info(f"Analyzing {symbol} data from {recent_df['time'].min()} to {recent_df['time'].max()}")

            if recent_df.empty:
                self.logger.warning(f"No data after filtering for recent weeks for {symbol}")
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

                # High & Low with volume validation
                high = float(window['high'].max())
                low = float(window['low'].min())

                high_volume = float(window.loc[window['high'] == high, 'tick_volume'].max())
                low_volume = float(window.loc[window['low'] == low, 'tick_volume'].max())

                # Add levels if volume significant
                if high_volume > volume_threshold:
                    potential_levels.append(high)
                    self.logger.debug(f"{symbol} High level found at {high:.5f} with volume {high_volume}")

                if low_volume > volume_threshold:
                    potential_levels.append(low)
                    self.logger.debug(f"{symbol} Low level found at {low:.5f} with volume {low_volume}")

            # Sort & merge nearby levels
            potential_levels = sorted(set(potential_levels))
            merged_levels = []
            for lvl in potential_levels:
                if not merged_levels or abs(lvl - merged_levels[-1]) > weekly_buffer:
                    merged_levels.append(lvl)
                else:
                    merged_levels[-1] = (merged_levels[-1] + lvl) / 2.0

            # Store in symbol_levels dictionary
            self.symbol_levels[symbol] = merged_levels

            self.logger.info(f"Identified {len(merged_levels)} valid S/R levels for {symbol}")
            return merged_levels

        except Exception as e:
            self.logger.error(f"Error in identify_sr_weekly for {symbol}: {str(e)}")
            return []


    def update_weekly_levels(self, df_h1, symbol: str = "EURUSD", weeks: int = 3, weekly_buffer: float = 0.00060):
        """
        Update the strategy's valid levels using weekly S/R from identify_sr_weekly.
        Now symbol-aware.
        """
        try:
            w_levels = self.identify_sr_weekly(
                df_h1,
                symbol=symbol,
                weeks=weeks,
                weekly_buffer=weekly_buffer
            )
            if not w_levels:
                self.logger.warning(f"No weekly levels found for {symbol}")
                return

            # Update the symbol-specific levels
            self.symbol_levels[symbol] = w_levels

            # For backward compatibility with existing code
            if symbol == self.default_symbol:
                self.valid_levels = w_levels

            self.logger.info(f"Updated valid levels for {symbol}. Total: {len(w_levels)}")

            # Update signal generator's levels for the current symbol
            if symbol == self.default_symbol:
                self.signal_generator.valid_levels = w_levels

        except Exception as e:
            self.logger.error(f"Error updating weekly levels for {symbol}: {str(e)}")


    def generate_signals(self, df_segment, symbol="EURUSD"):
        """
        Modified to pass the symbol to the signal generator.
        """
        return self.signal_generator.generate_signal(df_segment, symbol)


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
        Fixed position size of 0.5 lots regardless of any market conditions.
        Maintains strict FTMO-compliant risk management.
        """
        if account_balance <= 0:
            self.logger.error(f"Invalid account_balance: {account_balance}")
            return 0.01

        return 0.5  # Fixed 0.5 lots for every trade

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


    def open_trade(self, current_segment: pd.DataFrame, balance: float, i: int, symbol: str = "EURUSD") -> Optional["SR_Bounce_Strategy.Trade"]:
        """
        Enhanced trade opening that respects FTMO checks and applies correlation-based
        exposure tiers (block/partial/normal) before finalizing the trade.
        """
        if current_segment.empty:
            return None

        last_bar = current_segment.iloc[-1]
        current_time = pd.to_datetime(last_bar['time'])
        bar_range = float(last_bar['high']) - float(last_bar['low'])
        # Approximate spread for demonstration
        current_spread = bar_range * 0.1

        # 1) Check basic FTMO rules (daily limits, spread, etc.)
        can_trade, reason = self._validate_ftmo_rules(
            current_time=current_time,
            spread=current_spread,
            symbol=symbol
        )
        if not can_trade:
            self.logger.debug(f"[{symbol}] FTMO check failed: {reason}")
            return None

        # 2) Generate signal from strategy logic
        signal = self.generate_signals(current_segment, symbol=symbol)
        if signal["type"] == "NONE":
            return None

        # 3) Volume threshold check
        if float(last_bar["tick_volume"]) < self.params["min_volume_threshold"]:
            return None

        # 4) Basic open price & stop distance
        entry_price = float(last_bar["close"])
        stop_loss = self.calculate_stop_loss(signal, current_segment)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.00001:
            return None

        # 5) Initial position sizing (before correlation adjustments)
        base_size = self.calculate_position_size(balance, stop_distance)

        # 6) Calculate take-profit
        take_profit = self.calculate_take_profit(entry_price, stop_loss)

        # 7) Create a provisional new Trade object
        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            symbol=symbol,             # <--- pass symbol here
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=base_size
        )

        # Additional fields
        new_trade.level = signal.get('level', 0.0)
        new_trade.level_type = "Support" if signal["type"] == "BUY" else "Resistance"
        new_trade.distance_to_level = abs(entry_price - signal.get('level', entry_price))
        new_trade.entry_volume = float(last_bar['tick_volume'])
        new_trade.prev_3_avg_volume = float(current_segment['tick_volume'].tail(3).mean())
        new_trade.hour_avg_volume = float(current_segment['tick_volume'].tail(4).mean())

        # 8) We'll record this provisional trade into the daily_trades dict, but final approval
        #    depends on cross-pair correlation checks:
        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []
        self.daily_trades[symbol].append({
            'time': new_trade.open_time,
            'type': new_trade.type,
            'size': new_trade.size,
            'exposure': new_trade.size * 10000.0  # naive exposure estimate
        })

        # 9) Return the provisional trade so that the backtest logic can call
        #    validate_cross_pair_exposure(new_trade, active_trades, balance).
        #    If that validation fails, the trade won't be opened. If partial, it adjusts new_trade.size.
        return new_trade


    def exit_trade(self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade", symbol: str = "EURUSD") -> Tuple[bool, float, float]:
        """
        Enhanced exit with multi-symbol awareness.
        Returns (should_close, fill_price, pnl).
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
            else:
                pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size

            # Update daily PnL
            self.daily_pnl += pnl
            return True, fill_price, pnl

        return False, 0.0, 0.0

    def validate_cross_pair_exposure(
        self,
        new_trade: "SR_Bounce_Strategy.Trade",
        active_trades: Dict[str, Optional["SR_Bounce_Strategy.Trade"]],
        current_balance: float
    ) -> Tuple[bool, str]:
        """
        Enhanced correlation-based cross-pair exposure management.
        - correlation > 0.85: Block new trades on the second pair
        - correlation 0.70 - 0.85: Partial exposure; the new trade's size is reduced
        - correlation < 0.70: Normal rules apply
        """
        # Primary (HIGH) and secondary (MEDIUM) thresholds
        HIGH_CORR_THRESHOLD = 0.85
        MEDIUM_CORR_THRESHOLD = 0.70

        # 1) Calculate total open lots across all symbols
        total_open_lots = 0.0
        for sym, trade in active_trades.items():
            if trade is not None:
                total_open_lots += trade.size

        # 2) Check if adding new_trade would exceed our total lot cap (example 10 lots)
        if total_open_lots + new_trade.size > 10.0:
            return (False, f"Total open lots would exceed limit: {total_open_lots + new_trade.size:.2f}")

        # 3) Correlation-based logic
        #    Scan all active trades. If any correlation is above thresholds, apply the rules
        new_sym = new_trade.symbol

        for sym, open_trade in active_trades.items():
            if open_trade is None:
                continue
            if sym == new_sym:
                continue  # Skip same symbol

            # Look up correlation from self.symbol_correlations
            # Use abs in case negative correlation is also critical
            corr = abs(self.symbol_correlations.get(new_sym, {}).get(sym, 0.0))

            if corr > HIGH_CORR_THRESHOLD:
                # Block if correlation > 0.85
                return (False, f"Correlation {corr:.2f} with {sym} > {HIGH_CORR_THRESHOLD} => blocking new trade.")
            elif corr >= MEDIUM_CORR_THRESHOLD:
                # Partial exposure scenario if correlation in [0.70, 0.85)
                # We'll reduce the new trade size to 20% of the originally computed size.
                adjusted_size = round(new_trade.size * 0.20, 2)
                if adjusted_size < 0.01:
                    return (False, f"Partial correlation reduction made size < 0.01 lots => skip trade.")
                new_trade.size = adjusted_size

        # If we made it here, correlation checks are OK
        return (True, "OK")





    class Trade:
        def __init__(
            self,
            open_i: int,
            open_time: str,
            symbol: str,              # <--- NEW: explicitly store symbol
            type: str,
            entry_price: float,
            sl: float,
            tp: float,
            size: float
        ):
            self.open_i = open_i
            self.open_time = open_time
            self.symbol = symbol      # <--- store the symbol in the Trade object
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

        def to_dict(self) -> dict:
            return {
                "open_i": self.open_i,
                "open_time": self.open_time,
                "symbol": self.symbol,               # <--- included in to_dict
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
        Inner class for signal generation and bounce detection, now with:
         - Symbol-aware bounce_registry
         - Correlation-based signal filtering
         - Simple conflict check
         - Minimal volume comparison across pairs
        """

        def __init__(self, valid_levels, params, logger, debug=False, parent_strategy=None):
            self.valid_levels = valid_levels
            self.params = params
            self.logger = logger
            self.debug = debug

            # Link back to the parent strategy for correlation data
            self.parent_strategy = parent_strategy

            # Make the bounce registry symbol-aware:
            self.bounce_registry = {}

            # Moved signal_stats here for clarity
            self.signal_stats = {
                "volume_filtered": 0,
                "first_bounce_recorded": 0,
                "second_bounce_low_volume": 0,
                "signals_generated": 0,
                "tolerance_misses": 0
            }

        def generate_signal(self, df_segment: pd.DataFrame, symbol: str) -> Dict[str, Any]:
            """
            Main signal generation routine with correlation-based filtering and
            volume comparison across pairs.
            """
            last_idx = len(df_segment) - 1
            if last_idx < 0:
                return self._create_no_signal("Segment has no rows")

            # 1) Check correlation filter
            #    If correlation with any other symbol is above our threshold, skip signals.
            #    (This is a simplistic approach for demonstration.)
            correlation_threshold = 0.85
            correlations = self.parent_strategy.symbol_correlations.get(symbol, {})
            for other_symbol, corr_val in correlations.items():
                if abs(corr_val) > correlation_threshold:
                    # We do a simple skip if correlation is too high
                    reason = f"Correlation {corr_val:.2f} with {other_symbol} exceeds {correlation_threshold}"
                    return self._create_no_signal(reason)

            # 2) Basic volume comparison across pairs (optional demonstration)
            #    If we find another symbol with extremely higher volume in the last bar,
            #    we skip signals for this symbol. Just a simple demonstration approach.
            last_bar_volume = float(df_segment.iloc[last_idx]['tick_volume'])
            for other_symbol, corr_val in correlations.items():
                if abs(corr_val) > correlation_threshold:
                    continue  # Already skipped above if correlation too high
                # Attempt to get the other symbol's last bar volume
                if other_symbol in self.parent_strategy.symbol_data:
                    other_df = self.parent_strategy.symbol_data[other_symbol]
                    if len(other_df) > 0:
                        # We compare average volume of the last few bars as a naive approach
                        other_avg_vol = other_df['tick_volume'].tail(5).mean()
                        current_avg_vol = df_segment['tick_volume'].tail(5).mean()

                        # If other symbol is showing double the volume, skip
                        if other_avg_vol > (2 * current_avg_vol):
                            reason = (
                                f"Volume on {other_symbol} is significantly higher than on {symbol}. "
                                f"({other_avg_vol:.1f} vs {current_avg_vol:.1f})"
                            )
                            return self._create_no_signal(reason)

            # 3) Normal signal logic
            last_bar = df_segment.iloc[last_idx]

            # Check if volume is sufficient
            if not self._is_volume_sufficient(df_segment, last_idx):
                self.signal_stats["volume_filtered"] += 1
                return self._create_no_signal("Volume too low vs. recent average")

            # Evaluate bullish or bearish bar
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

                distance_pips = abs(close_ - lvl) * 10000
                if distance_pips > 15:
                    # More than 15 pips away from the level, skip
                    continue

                # Track near misses
                if bullish and not near_support and abs(low_ - lvl) <= tol * 2:
                    self.signal_stats["tolerance_misses"] += 1
                if bearish and not near_resistance and abs(high_ - lvl) <= tol * 2:
                    self.signal_stats["tolerance_misses"] += 1

                if near_support or near_resistance:
                    # Potential bounce
                    self.logger.debug(
                        f"{symbol} potential bounce at level={lvl}, "
                        f"time={last_bar['time']}, volume={last_bar['tick_volume']}"
                    )
                    signal = self._process_bounce(lvl, float(last_bar['tick_volume']), last_bar['time'],
                                                  is_support=near_support, symbol=symbol)
                    if signal and signal["type"] != "NONE":
                        self.signal_stats["signals_generated"] += 1
                        return signal

            return self._create_no_signal("No bounce off valid levels")

        def _process_bounce(self, level, volume, time, is_support, symbol) -> Optional[Dict[str, Any]]:
            """
            Make bounce registry symbol-aware: bounce_registry[symbol][level].
            Then implement the standard "first bounce / second bounce" logic.
            """
            # Ensure symbol key in bounce_registry
            if symbol not in self.bounce_registry:
                self.bounce_registry[symbol] = {}

            # If first bounce at level, record it
            if level not in self.bounce_registry[symbol]:
                self.bounce_registry[symbol][level] = {
                    "first_bounce_volume": volume,
                    "timestamp": time,
                    "last_trade_time": None
                }
                self.signal_stats["first_bounce_recorded"] += 1
                self.logger.debug(f"[1st bounce] {symbol} volume={volume} at lvl={level}")
                return self._create_no_signal(f"First bounce recorded for {symbol} at {level}")

            # Check cooldown if already traded
            if self.bounce_registry[symbol][level].get("last_trade_time"):
                import pandas as pd
                last_trade = pd.to_datetime(self.bounce_registry[symbol][level]["last_trade_time"])
                current_time = pd.to_datetime(time)
                cooldown_period = pd.Timedelta(hours=2)
                if current_time - last_trade < cooldown_period:
                    return self._create_no_signal(f"Level {level} in cooldown for {symbol}")

            # Process second bounce
            first_vol = self.bounce_registry[symbol][level]["first_bounce_volume"]
            if volume < first_vol * 0.6:
                self.signal_stats["second_bounce_low_volume"] += 1
                return self._create_no_signal("Second bounce volume insufficient")

            bounce_type = "BUY" if is_support else "SELL"
            reason = f"Valid bounce at {'support' if is_support else 'resistance'} {level} for {symbol}"

            # Update last trade time
            self.bounce_registry[symbol]["last_trade_time"] = time
            self.signal_stats["signals_generated"] += 1

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