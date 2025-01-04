import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def analyze_trades(trades: List[Dict], initial_balance: float) -> Dict:
    """
    Compute basic performance metrics:
    - total trades, win_rate, profit_factor, max_drawdown, total_pnl
    """
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

    def write_executive_summary(self, trades: List[Dict], stats: Dict, df_test: pd.DataFrame):
        """Writes executive summary with key metrics and patterns"""
        self.file_handler.write("# Executive Summary\n\n")

        # Performance Overview
        self.file_handler.write("## Performance Overview\n")
        self.file_handler.write(f"- Total Return: ${stats['total_pnl']:.2f}\n")
        self.file_handler.write(f"- Number of Trades: {stats['count']}\n")
        self.file_handler.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
        self.file_handler.write(f"- Profit Factor: {stats['profit_factor']:.2f}\n")
        self.file_handler.write(f"- Maximum Drawdown: ${stats['max_drawdown']:.2f}\n\n")

        # Key Findings
        self.file_handler.write("## Key Findings\n")

        # Analyze trading frequency
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty:
            df_trades['open_time'] = pd.to_datetime(df_trades['open_time'])
            df_trades['month'] = df_trades['open_time'].dt.strftime('%Y-%m')
            monthly_trades = df_trades.groupby('month').size()

            # Find months with no trades
            all_months = pd.date_range(df_test['time'].min(), df_test['time'].max(), freq='M')
            all_months = pd.Series(all_months.strftime('%Y-%m'))
            missing_months = all_months[~all_months.isin(monthly_trades.index)]

            if not missing_months.empty:
                self.file_handler.write("\n### Notable Periods Without Trading:\n")
                self.file_handler.write(f"- No trades executed in: {', '.join(missing_months)}\n")

            # Most active trading periods
            if not monthly_trades.empty:
                most_active = monthly_trades.idxmax()
                self.file_handler.write(f"\n- Most active trading month: {most_active} with {monthly_trades[most_active]} trades\n")

        # Analyze win streaks
        if trades:
            current_streak = 1
            max_streak = 1
            for i in range(1, len(trades)):
                if (trades[i]['pnl'] > 0 and trades[i-1]['pnl'] > 0) or \
                   (trades[i]['pnl'] < 0 and trades[i-1]['pnl'] < 0):
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1

            self.file_handler.write(f"- Longest streak of consecutive winning/losing trades: {max_streak}\n\n")

    # Continuing the ReportWriter class...

    def write_data_quality_analysis(self, df_test: pd.DataFrame):
        """Analyzes and writes data quality metrics"""
        self.file_handler.write("# Data Quality Analysis\n\n")

        # Check for gaps
        df_test['time'] = pd.to_datetime(df_test['time'])
        df_test = df_test.sort_values('time')
        time_diff = df_test['time'].diff()

        # Expected time difference (assuming 15-minute data)
        expected_diff = pd.Timedelta(minutes=15)
        gaps = time_diff[time_diff > expected_diff * 1.5]

        self.file_handler.write("## Data Coverage\n")
        self.file_handler.write(f"- Total number of bars: {len(df_test)}\n")
        self.file_handler.write(f"- Date range: {df_test['time'].min()} to {df_test['time'].max()}\n")

        if not gaps.empty:
            self.file_handler.write("\n## Data Gaps Detected\n")
            for idx in gaps.index:
                gap_start = df_test.loc[idx-1, 'time']
                gap_end = df_test.loc[idx, 'time']
                self.file_handler.write(f"- Gap from {gap_start} to {gap_end}\n")

        # Trading hours distribution
        df_test['hour'] = df_test['time'].dt.hour
        hour_dist = df_test.groupby('hour').size()

        self.file_handler.write("\n## Trading Hours Distribution\n")
        self.file_handler.write("Hour | Bar Count\n")
        self.file_handler.write("------|----------\n")
        for hour, count in hour_dist.items():
            self.file_handler.write(f"{hour:02d}:00 | {count}\n")
        self.file_handler.write("\n")

    def write_temporal_analysis(self, df_test: pd.DataFrame, trades: List[Dict]):
        """Analyzes and writes temporal patterns"""
        self.file_handler.write("# Temporal Analysis\n\n")

        # Monthly breakdown
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty:
            df_trades['open_time'] = pd.to_datetime(df_trades['open_time'])
            df_trades['month'] = df_trades['open_time'].dt.strftime('%Y-%m')
            monthly_stats = df_trades.groupby('month').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)

            self.file_handler.write("## Monthly Trading Activity\n")
            self.file_handler.write("Month | Trades | Total PnL | Avg PnL\n")
            self.file_handler.write("------|---------|-----------|----------\n")
            for month in monthly_stats.index:
                stats = monthly_stats.loc[month]
                self.file_handler.write(f"{month} | {stats[('pnl', 'count')]} | "
                                      f"${stats[('pnl', 'sum')]} | ${stats[('pnl', 'mean')]}\n")

        # Market volatility analysis
        df_test['volatility'] = (df_test['high'] - df_test['low']) * 10000  # Convert to pips
        df_test['month'] = df_test['time'].dt.strftime('%Y-%m')
        monthly_volatility = df_test.groupby('month')['volatility'].agg(['mean', 'max']).round(2)

        self.file_handler.write("\n## Monthly Market Volatility (in pips)\n")
        self.file_handler.write("Month | Average | Maximum\n")
        self.file_handler.write("------|----------|----------\n")
        for month in monthly_volatility.index:
            stats = monthly_volatility.loc[month]
            self.file_handler.write(f"{month} | {stats['mean']} | {stats['max']}\n")

        self.file_handler.write("\n")

    def write_trade_analysis(self, trades: List[Dict], df_test: pd.DataFrame):
        """Analyzes and writes detailed trade metrics"""
        self.file_handler.write("# Trade Analysis\n\n")

        # Convert trades to DataFrame for analysis
        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            self.file_handler.write("No trades to analyze.\n\n")
            return

        df_trades['open_time'] = pd.to_datetime(df_trades['open_time'])
        df_trades['close_time'] = pd.to_datetime(df_trades['close_time'])

        # Calculate holding times
        df_trades['holding_time'] = (df_trades['close_time'] - df_trades['open_time'])
        avg_holding = df_trades['holding_time'].mean()

        self.file_handler.write("## Timing Analysis\n")
        self.file_handler.write(f"- Average holding time: {avg_holding}\n")
        self.file_handler.write(f"- Shortest trade: {df_trades['holding_time'].min()}\n")
        self.file_handler.write(f"- Longest trade: {df_trades['holding_time'].max()}\n\n")

        # Win/Loss streaks
        df_trades['win'] = df_trades['pnl'] > 0
        streak_changes = df_trades['win'] != df_trades['win'].shift()
        streak_groups = streak_changes.cumsum()
        streaks = df_trades.groupby(streak_groups)['win'].agg(['first', 'size'])

        self.file_handler.write("## Win/Loss Streaks\n")
        self.file_handler.write("Length | Type | Start Date | End Date\n")
        self.file_handler.write("--------|------|------------|----------\n")

        for idx in streaks.index:
            is_win = streaks.loc[idx, 'first']
            length = streaks.loc[idx, 'size']
            streak_trades = df_trades[streak_groups == idx]
            start_date = streak_trades['open_time'].iloc[0].strftime('%Y-%m-%d')
            end_date = streak_trades['open_time'].iloc[-1].strftime('%Y-%m-%d')

            self.file_handler.write(f"{length} | {'Win' if is_win else 'Loss'} | "
                                  f"{start_date} | {end_date}\n")

        # Distance from S/R levels
        df_trades['dist_pips'] = df_trades['distance_to_level'] * 10000

        self.file_handler.write("\n## Distance from S/R Levels (pips)\n")
        self.file_handler.write(f"- Average: {df_trades['dist_pips'].mean():.1f}\n")
        self.file_handler.write(f"- Minimum: {df_trades['dist_pips'].min():.1f}\n")
        self.file_handler.write(f"- Maximum: {df_trades['dist_pips'].max():.1f}\n\n")

        # Volume analysis
        self.file_handler.write("## Volume Analysis\n")
        self.file_handler.write(f"- Average entry volume: {df_trades['entry_volume'].mean():.0f}\n")
        self.file_handler.write(f"- Average 3-bar volume: {df_trades['prev_3_avg_volume'].mean():.0f}\n")
        self.file_handler.write(f"- Average hourly volume: {df_trades['hour_avg_volume'].mean():.0f}\n\n")

    def write_data_overview(self, df_test: pd.DataFrame):
        """Basic data overview method"""
        self.file_handler.write("# Backtest Detailed Report\n\n")
        self.file_handler.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file_handler.write(f"## Data Overview\n")
        self.file_handler.write(f"- Total bars in Test Set: {len(df_test)}\n")
        self.file_handler.write(f"- Test Start: {df_test.iloc[0]['time'] if not df_test.empty else 'N/A'}\n")
        self.file_handler.write(f"- Test End: {df_test.iloc[-1]['time'] if not df_test.empty else 'N/A'}\n\n")

    def generate_full_report(
        self,
        df_test: pd.DataFrame,
        trades: List[Dict],
        stats: Dict,
        mc_results: Dict,
        final_balance: float,
        monthly_data: Dict,
        monthly_levels: List[Dict],
        weekly_levels: List[Dict],
        correlation_data: Dict[str, Dict[str, float]] = None,
        ftmo_data: Dict[str, float] = None
    ):
        """Generates comprehensive report with all sections"""
        self.write_data_overview(df_test)
        self.write_executive_summary(trades, stats, df_test)
        self.file_handler.write("\n---\n")

        self.write_data_quality_analysis(df_test)
        self.file_handler.write("\n---\n")

        self.write_temporal_analysis(df_test, trades)
        self.file_handler.write("\n---\n")

        self.write_trade_analysis(trades, df_test)
        self.file_handler.write("\n---\n")

        # Write original sections
        self.write_trades_section(trades)
        self.write_stats_section(stats, final_balance)
        self.write_multi_symbol_performance(trades)
        if correlation_data:
            self.write_correlation_report(correlation_data)
        if ftmo_data:
            self.write_ftmo_section(ftmo_data)
        self.write_monthly_breakdown(monthly_data)
        self.write_sr_levels(monthly_levels, weekly_levels)
        self.file_handler.write("\n**End of Report**\n")

    def write_stats_section(self, stats: Dict, final_balance: float):
        """Write basic statistics section"""
        self.file_handler.write("\n## Stats\n\n")
        self.file_handler.write(f"- Total Trades: {stats['count']}\n")
        self.file_handler.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
        self.file_handler.write(f"- Profit Factor: {stats['profit_factor']:.2f}\n")
        self.file_handler.write(f"- Max Drawdown: ${stats['max_drawdown']:.2f}\n")
        self.file_handler.write(f"- Total PnL: ${stats['total_pnl']:.2f}\n")
        self.file_handler.write(f"- Final Balance: ${final_balance:.2f}\n\n")

    def write_multi_symbol_performance(self, trades: List[Dict]):
        """Write per-symbol performance breakdown"""
        if not trades:
            self.file_handler.write("\n## Multi-Symbol Performance\nNo trades found.\n")
            return

        df = pd.DataFrame(trades)
        if 'symbol' not in df.columns:
            self.file_handler.write("\n## Multi-Symbol Performance\n(Symbol field not found)\n")
            return

        group = df.groupby('symbol')['pnl'].agg(['count', 'sum'])
        self.file_handler.write("\n## Multi-Symbol Performance\n\n")
        self.file_handler.write("| Symbol | Trades | Total PnL |\n")
        self.file_handler.write("|--------|--------|-----------|\n")
        for idx, row in group.iterrows():
            self.file_handler.write(f"| {idx} | {row['count']} | {row['sum']:.2f} |\n")
        self.file_handler.write("\n")

    def write_correlation_report(self, correlation_data: Dict[str, Dict[str, float]]):
        """Write correlation data among symbols"""
        if not correlation_data:
            self.file_handler.write("\n## Correlation Report\nNo correlation data provided.\n")
            return

        self.file_handler.write("\n## Correlation Report\n\n")
        for sym, corr_map in correlation_data.items():
            for other_sym, val in corr_map.items():
                self.file_handler.write(f"- Correlation {sym} vs {other_sym}: {val:.4f}\n")
        self.file_handler.write("\n")

    def write_ftmo_section(self, ftmo_data: Dict[str, float]):
        """Write FTMO compliance data"""
        if not ftmo_data:
            self.file_handler.write("\n## FTMO Compliance Report\nNo FTMO data provided.\n")
            return

        self.file_handler.write("\n## FTMO Compliance Report\n")
        self.file_handler.write(f"- Daily Drawdown Limit: {ftmo_data.get('daily_drawdown_limit', 'N/A')}\n")
        self.file_handler.write(f"- Max Drawdown Limit: {ftmo_data.get('max_drawdown_limit', 'N/A')}\n")
        self.file_handler.write(f"- Profit Target: {ftmo_data.get('profit_target', 'N/A')}\n")
        self.file_handler.write(f"- Current Daily DD: {ftmo_data.get('current_daily_dd', 'N/A')}\n")
        self.file_handler.write(f"- Current Total DD: {ftmo_data.get('current_total_dd', 'N/A')}\n\n")

    def write_monthly_breakdown(self, monthly_data: Dict):
        """Write monthly/weekly narrative breakdown"""
        if not monthly_data:
            return

        self.file_handler.write("\n--- NARRATIVE MONTH/WEEK BREAKDOWN ---\n")
        for month, data in monthly_data.items():
            self.file_handler.write(f"\n=== {month} ===\n")
            self.file_handler.write(f" Time Range: {data['start']} -> {data['end']}\n")
            self.file_handler.write(f" Monthly O/H/L/C: {data['open']}/{data['high']}/{data['low']}/{data['close']}\n")

    def write_sr_levels(self, monthly_levels: List[Dict], weekly_levels: List[Dict]):
        """Write support/resistance levels"""
        if monthly_levels:
            self.file_handler.write("\nMajor Monthly S/R Levels Detected:\n")
            for level in monthly_levels:
                self.file_handler.write(
                    f" -> {level['price']} | First: {level['first_date']} | "
                    f"Touches: {level['touches']} | Last: {level['last_date']} | "
                    f"Trend: {level['trend']}\n"
                )

        if weekly_levels:
            self.file_handler.write("\nWeekly Sub-Levels Detected:\n")
            for level in weekly_levels:
                self.file_handler.write(
                    f" -> {level['price']} | First: {level['first_date']} | "
                    f"Touches: {level['touches']} | Last: {level['last_date']} | "
                    f"Trend: {level['trend']}\n"
                )

    def write_trades_section(self, trades: List[Dict]):
        """
        Writes a list of trades in a tabular format.
        """
        self.file_handler.write("\n## Trade Details\n\n")

        if not trades:
            self.file_handler.write("No trades were executed.\n\n")
            return

        # Table headers
        self.file_handler.write(
            "| Open Time           | Symbol | Type | Entry Price | Stop Loss | Take Profit | Size | Close Time          | Close Price |   PnL    | Entry Reason                 | Exit Reason                  |\n"
        )
        self.file_handler.write(
            "|---------------------|--------|------|------------|----------|------------|------|----------------------|------------|----------|-----------------------------|-----------------------------|\n"
        )

        for trade in trades:
            self.file_handler.write(
                f"| {trade.get('open_time','')} "
                f"| {trade.get('symbol','')} "
                f"| {trade.get('type','')} "
                f"| {trade.get('entry_price',0.0):.5f} "
                f"| {trade.get('sl',0.0):.5f} "
                f"| {trade.get('tp',0.0):.5f} "
                f"| {trade.get('size',0.0):.2f} "
                f"| {trade.get('close_time','')} "
                f"| {trade.get('close_price',0.0):.5f} "
                f"| {trade.get('pnl',0.0):.2f} "
                f"| {trade.get('entry_reason','')} "
                f"| {trade.get('exit_reason','')} |\n"
            )

        self.file_handler.write("\n\n")
