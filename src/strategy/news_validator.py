# src/strategy/news_validator.py

import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os

class NewsValidator:
    """
    Checks if upcoming high-impact news is within a certain window.
    Expects a JSON file with an array of events like:
      [
        {
          "time": "2024-12-13T14:30:00",
          "name": "FOMC Statement"
        },
        ...
      ]
    """
    def __init__(self, news_file: str, lookforward_minutes: int = 30):
        self.lookforward_minutes = lookforward_minutes
        self.news_events = []
        self.load_news_events(news_file)

    def load_news_events(self, path: str):
        if not os.path.exists(path):
            print(f"[NewsValidator] No file at {path}, skipping news load.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Expect "high_impact" list or something similar
                if isinstance(data, dict) and "high_impact" in data:
                    data = data["high_impact"]
                self.news_events = []
                for event in data:
                    # Must parse the date/time if it’s in separate fields or single ISO
                    evt_time_str = event.get("time")
                    if evt_time_str and "-" in evt_time_str:
                        # if "time" is like "2024-12-13T14:30:00"
                        evt_time = datetime.fromisoformat(evt_time_str)
                    else:
                        # fallback if "date" + "time" are separate
                        # (not 100% sure how your JSON looks, adapt as needed)
                        evt_date = event.get("date", "")
                        evt_t = event.get("time", "")
                        dt_str = f"{evt_date} {evt_t}"
                        evt_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                    self.news_events.append({
                        "time": evt_time,
                        "name": event.get("event") or event.get("name", "UnnamedEvent")
                    })
        except Exception as e:
            print(f"[NewsValidator] Error loading {path}: {e}")

    def is_news_window(self, current_time: datetime) -> Tuple[bool, str]:
        """
        Returns (True, eventName) if a news event is within the next lookforward_minutes,
        else (False, "").
        """
        for evt in self.news_events:
            delta = (evt["time"] - current_time).total_seconds() / 60.0
            if 0 <= delta <= self.lookforward_minutes:
                return (True, evt["name"])
        return (False, "")
