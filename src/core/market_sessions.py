"""Market Sessions Module for Forex Trading Bot V2.

This module handles market sessions, holidays, and news events for the trading system.
It reads from configuration files and provides real-time market session information.

Author: mazelcar
Created: December 2024
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class MarketSessionManager:
    """Manages market sessions, holidays, and news events."""

    def __init__(self):
        """Initialize market session manager."""
        self.config_path = Path(__file__).parent.parent.parent / "config"
        self.sessions = self._load_json("market_session.json")
        self.holidays = self._load_json("market_holidays.json")
        self.news = self._load_json("market_news.json")

    def _load_json(self, filename: str) -> Dict:
        """Load and parse JSON configuration file."""
        try:
            with open(self.config_path / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}

    def is_holiday(self, market: str) -> bool:
        """Check if today is a holiday for the given market."""
        today = datetime.now().strftime("%Y-%m-%d")
        year = datetime.now().year

        try:
            market_holidays = self.holidays.get(str(year), {}).get(market, [])
            return any(holiday["date"] == today for holiday in market_holidays)
        except Exception as e:
            print(f"Error checking holidays: {e}")
            return False

    def check_sessions(self) -> Dict[str, bool]:
        """Check which markets are currently open."""
        try:
            current_time = datetime.now()
            current_day = current_time.strftime("%A")

            market_status = {}
            for market, info in self.sessions.get("sessions", {}).items():
                # Check if current day is a trading day
                if current_day not in info.get("days", []):
                    market_status[market] = False
                    continue

                # Check if holiday
                if self.is_holiday(market):
                    market_status[market] = False
                    continue

                # Get session times
                open_hour = int(info["open"].split(":")[0])
                close_hour = int(info["close"].split(":")[0])
                current_hour = current_time.hour

                # Handle overnight sessions
                if open_hour > close_hour:
                    is_open = current_hour >= open_hour or current_hour < close_hour
                else:
                    is_open = open_hour <= current_hour < close_hour

                market_status[market] = is_open

            return market_status

        except Exception as e:
            print(f"Error checking sessions: {e}")
            return {market: False for market in self.sessions.get("sessions", {})}

    def get_status(self) -> Dict:
        """Get complete market status including holidays and news."""
        current_status = self.check_sessions()

        return {
            'status': current_status,
            'overall_status': 'OPEN' if any(current_status.values()) else 'CLOSED'
        }