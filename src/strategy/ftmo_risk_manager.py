from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

class FTMORiskManager:
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_high_balance = initial_balance
        self.daily_trades: List[Dict] = []

        # FTMO Limits
        self.daily_drawdown_limit = 0.05  # 5% daily
        self.max_drawdown_limit = 0.10    # 10% total
        self.profit_target = 0.10         # 10% profit target

        # Trading rules
        self.max_positions = 3
        self.max_daily_trades = 8
        self.max_spread = 0.002  # Maximum 3 pip spread

        self.last_reset = datetime.now().date()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('FTMORiskManager')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler('ftmo_risk.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(fh)
        return logger

    def can_open_trade(self, current_time: datetime, spread: float, daily_pnl: float) -> Tuple[bool, str]:
        """Enhanced trade validation with proper date handling"""
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

        # Other FTMO checks remain the same
        if abs(min(0, daily_pnl)) >= self.initial_balance * self.daily_drawdown_limit:
            return False, f"Daily drawdown limit reached"

        if spread > self.max_spread:
            return False, f"Spread too high: {spread:.5f}"

        return True, "Trade validated"

    def _is_valid_trading_time(self, current_time: datetime) -> bool:
        """Check if current time is within valid trading sessions"""
        hour = current_time.hour
        minute = current_time.minute

        # No trading in first 15 minutes of any session
        if minute < 15 and hour in [self.session_rules['london_open'],
                                  self.session_rules['ny_open']]:
            return False

        # Check if within London or NY session
        is_london = (self.session_rules['london_open'] <= hour <
                    self.session_rules['london_close'])
        is_ny = (self.session_rules['ny_open'] <= hour <
                 self.session_rules['ny_close'])

        return is_london or is_ny

    def update_trade_history(self, trade: Dict):
        """Track trade for daily limits with proper date handling"""
        current_date = pd.to_datetime(trade['time']).date()

        # Reset if new day
        if current_date != self.last_reset:
            self.daily_trades = []
            self.last_reset = current_date
            self.daily_pnl = 0.0
            self.logger.info(f"Daily stats reset for {current_date}")

        # Add trade to daily tracking
        self.daily_trades.append(trade)