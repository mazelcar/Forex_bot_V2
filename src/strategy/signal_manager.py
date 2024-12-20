# src/strategy/signal_manager.py
from datetime import datetime, timedelta

class SignalManager:
    def __init__(self):
        self.current_signal = None  # e.g., {"type": "BUY", "strength": 0.5, "reasons": ["Low volume"]}
        self.history = []  # list of signals with timestamps

    def record_signal(self, signal_type, strength, reasons):
        # signal_type could be 'BUY', 'SELL', or 'NONE'
        # reasons is a list of strings explaining conditions
        self.current_signal = {
            "time": datetime.now(),
            "type": signal_type,
            "strength": strength,
            "reasons": reasons
        }
        self.history.append(self.current_signal)

    def get_current_signal(self):
        # Returns current signal dict or None
        return self.current_signal

    def get_signals(self, last_hours=24):
        cutoff = datetime.now() - timedelta(hours=last_hours)
        return [s for s in self.history if s['time'] >= cutoff]

    def clear_current_signal(self):
        self.current_signal = None

    def write_historical_signals_to_file(self, last_hours=8, filename="historical_signals.txt"):
        """
        Writes the last `last_hours` hours of signals to a fixed file, overwriting it each time.
        This keeps the file small and up-to-date with only recent signals.

        Args:
            last_hours (int): How many hours of signals to include.
            filename (str): The name of the file to write to.
        """
        signals = self.get_signals(last_hours=last_hours)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Historical Signals (Last {} Hours) ===\n".format(last_hours))
            if signals:
                for s in signals:
                    # s is expected to have keys: time, type, strength, reasons
                    signal_time_str = s['time'].strftime('%Y-%m-%d %H:%M:%S')
                    f.write("Time: {}, Type: {}, Strength: {:.2f}\n".format(
                        signal_time_str, s['type'], s['strength']))
                    f.write("Reasons:\n")
                    for r in s['reasons']:
                        f.write(" - {}\n".format(r))
            else:
                f.write("No signals recorded in the last {} hours.\n".format(last_hours))
            f.write("=== End of Historical Signals ===\n")
