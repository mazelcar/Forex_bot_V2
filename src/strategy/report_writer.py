from datetime import datetime
from typing import Dict, List

import pandas as pd

def analyze_trades(trades: List[Dict], initial_balance: float) -> Dict:
        """
        Compute basic performance metrics:
        - total trades, win_rate, profit_factor, max_drawdown, total_pnl
        """
        import numpy as np

        if not trades:
            return {
                "count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0
            }

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        total_win = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))

        count = len(trades)
        win_rate = (len(wins) / count) * 100.0 if count else 0.0
        profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
        total_pnl = sum(t["pnl"] for t in trades)

        # Max drawdown
        running = initial_balance
        peak = running
        dd = 0.0
        for t in trades:
            running += t["pnl"]
            if running > peak:
                peak = running
            drawdown = peak - running
            if drawdown > dd:
                dd = drawdown

        return {
            "count": count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": dd,
            "total_pnl": total_pnl
        }


class ReportWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_handler = None

    def __enter__(self):
        self.file_handler = open(self.filepath, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handler:
            self.file_handler.close()

    def write_data_overview(self, df_test: pd.DataFrame):
        self.file_handler.write("# Backtest Detailed Report\n\n")
        self.file_handler.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file_handler.write(f"## Data Overview\n")
        self.file_handler.write(f"- Total bars in Test Set: {len(df_test)}\n")
        self.file_handler.write(f"- Test Start: {df_test.iloc[0]['time'] if not df_test.empty else 'N/A'}\n")
        self.file_handler.write(f"- Test End: {df_test.iloc[-1]['time'] if not df_test.empty else 'N/A'}\n\n")

    def write_trades_section(self, trades: List[Dict]):
        self.file_handler.write(f"## Trades Executed\n\n")
        if not trades:
            self.file_handler.write("No trades were executed.\n\n")
            return

        self.file_handler.write("| # | Open Time | Type | Entry | Level | Level Type | Dist(pips) | Volume | 3-Bar Avg | Hour Avg | Close | PnL |\n")
        self.file_handler.write("|---|-----------|------|--------|--------|------------|------------|---------|------------|----------|-------|-----|\n")
        for i, t in enumerate(trades, start=1):
            dist_pips = t.get('distance_to_level', 0) * 10000  # Convert to pips
            self.file_handler.write(
                f"| {i} | {t['open_time']} | {t['type']} | "
                f"{t['entry_price']:.5f} | {t['level']:.5f} | {t['level_type']} | {dist_pips:.1f} | "
                f"{t['entry_volume']:.0f} | {t['prev_3_avg_volume']:.0f} | {t['hour_avg_volume']:.0f} | "
                f"{t['close_price']:.5f} | {t['pnl']:.2f} |\n"
            )

    def write_stats_section(self, stats: Dict, final_balance: float):
        self.file_handler.write("\n## Stats\n\n")
        self.file_handler.write(f"- Total Trades: {stats['count']}\n")
        self.file_handler.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
        self.file_handler.write(f"- Profit Factor: {stats['profit_factor']:.2f}\n")
        self.file_handler.write(f"- Max Drawdown: ${stats['max_drawdown']:.2f}\n")
        self.file_handler.write(f"- Total PnL: ${stats['total_pnl']:.2f}\n")
        self.file_handler.write(f"- Final Balance: ${final_balance:.2f}\n\n")

    def write_monthly_breakdown(self, monthly_data: Dict):
        self.file_handler.write("\n--- NARRATIVE MONTH/WEEK BREAKDOWN ---\n")
        for month, data in monthly_data.items():
            self.file_handler.write(f"\n=== {month} ===\n")
            self.file_handler.write(f" Time Range: {data['start']} -> {data['end']}\n")
            self.file_handler.write(f" Monthly O/H/L/C: {data['open']}/{data['high']}/{data['low']}/{data['close']}\n")
            for week in data['weeks']:
                self.file_handler.write(f"  Week {week['num']} ({week['start']} -> {week['end']}): \n")
                self.file_handler.write(f"    O/H/L/C = {week['open']}/{week['high']}/{week['low']}/{week['close']}\n")

    def write_sr_levels(self, monthly_levels: List[Dict], weekly_levels: List[Dict]):
        self.file_handler.write("\nMajor Monthly S/R Levels Detected:\n")
        for level in monthly_levels:
            self.file_handler.write(
                f" -> {level['price']} | First Detected: {level['first_date']} | "
                f"Touches: {level['touches']} | Last Touched: {level['last_date']} | "
                f"Trend@Formation: {level['trend']}\n"
            )

        self.file_handler.write("\nWeekly Sub-Levels Detected:\n")
        for level in weekly_levels:
            self.file_handler.write(
                f" -> {level['price']} | First Detected: {level['first_date']} | "
                f"Touches: {level['touches']} | Last Touched: {level['last_date']} | "
                f"Trend@Formation: {level['trend']}\n"
            )

    def generate_full_report(
        self,
        df_test: pd.DataFrame,
        trades: List[Dict],
        stats: Dict,
        mc_results: Dict,
        final_balance: float,
        monthly_data: Dict,
        monthly_levels: List[Dict],
        weekly_levels: List[Dict]
    ):
        self.write_data_overview(df_test)
        self.write_trades_section(trades)
        self.write_stats_section(stats, final_balance)
        self.write_monte_carlo_section(mc_results)
        self.write_monthly_breakdown(monthly_data)
        self.write_sr_levels(monthly_levels, weekly_levels)
        self.file_handler.write("\n---\n")
        self.file_handler.write("**End of Report**\n")

