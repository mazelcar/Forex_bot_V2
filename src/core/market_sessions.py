# src/core/market_sessions.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class MarketSessionManager:
    """Manages market sessions, holidays, and news events."""

    def __init__(self, logger=None):
        """Initialize market session manager."""
        self.logger = logger or self._get_default_logger()
        self.config_path = Path(__file__).parent.parent.parent / "config"
        self.sessions = self._load_json("market_session.json")
        self.holidays = self._load_json("market_holidays.json")
        self.news = self._load_json("market_news.json")

    def _get_default_logger(self):
        import logging
        logger = logging.getLogger('MarketSessionManager')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        return logger

    def _load_json(self, filename: str) -> Dict:
        try:
            with open(self.config_path / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {}

    def is_holiday(self, market: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        year = datetime.now().year
        try:
            market_holidays = self.holidays.get(str(year), {}).get(market, [])
            is_holiday = any(holiday["date"] == today for holiday in market_holidays)
            self.logger.debug(f"is_holiday({market}): Today={today}, Holiday={is_holiday}")
            return is_holiday
        except Exception as e:
            self.logger.error(f"Error checking holidays for {market}: {e}")
            return False

    def check_sessions(self) -> Dict[str, bool]:
        """Check which markets are currently open based on local time."""
        try:
            current_time_local = datetime.now()
            current_day = current_time_local.strftime("%A")
            current_hour = current_time_local.hour

            self.logger.debug(f"Current Local Time: {current_time_local}")
            self.logger.debug(f"Current Day: {current_day}")
            self.logger.debug(f"Current Hour: {current_hour}")

            weekday_num = current_time_local.weekday()  # Monday=0, Sunday=6
            is_weekend = weekday_num >= 5
            self.logger.debug(f"Is Weekend? {is_weekend}")

            sessions_data = self.sessions.get("sessions", {})
            self.logger.debug(f"Sessions Found: {list(sessions_data.keys())}")

            market_status = {}
            for market, info in sessions_data.items():
                # Skip non-dict entries like _description
                if not isinstance(info, dict):
                    self.logger.debug(f"Skipping {market} because info is not a dict.")
                    continue

                self.logger.debug(f"Checking session: {market}")
                self.logger.debug(f"{market} Market Days: {info.get('days', [])}")

                holiday_check = self.is_holiday(market)
                self.logger.debug(f"{market} Holiday Check: {holiday_check}")
                if holiday_check:
                    self.logger.debug(f"{market}: Closed due to holiday.")
                    market_status[market] = False
                    continue

                if current_day not in info.get("days", []):
                    self.logger.debug(f"{market}: {current_day} not in trading days.")
                    market_status[market] = False
                    continue

                if is_weekend:
                    self.logger.debug(f"{market}: Closed due to weekend.")
                    market_status[market] = False
                    continue

                try:
                    open_time_str = info["open"]
                    close_time_str = info["close"]
                    open_hour = int(open_time_str.split(":")[0])
                    close_hour = int(close_time_str.split(":")[0])

                    self.logger.debug(f"{market} Session Times: Open={open_time_str}, Close={close_time_str}")
                    self.logger.debug(f"Comparing {current_hour} with open={open_hour}, close={close_hour}")

                    if open_hour > close_hour:
                        is_open = (current_hour >= open_hour) or (current_hour < close_hour)
                        self.logger.debug(f"{market}: crosses midnight -> is_open={is_open}")
                    else:
                        is_open = (open_hour <= current_hour < close_hour)
                        self.logger.debug(f"{market}: normal hours -> is_open={is_open}")

                    market_status[market] = is_open

                except Exception as parse_e:
                    self.logger.error(f"Error parsing times for {market}: {parse_e}")
                    market_status[market] = False

            self.logger.debug(f"Final Market Status: {market_status}")
            return market_status

        except Exception as e:
            self.logger.error(f"Error checking sessions: {e}")
            return {market: False for market in self.sessions.get("sessions", {})}


    def get_status(self) -> Dict:
        self.logger.debug("Calling check_sessions()...")
        current_status = self.check_sessions()
        self.logger.debug(f"check_sessions() returned: {current_status}")
        overall_status = 'OPEN' if any(current_status.values()) else 'CLOSED'
        self.logger.debug(f"Overall market status: {overall_status}")
        return {
            'status': current_status,
            'overall_status': overall_status
        }
