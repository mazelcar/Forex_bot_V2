import logging
from typing import Any, Dict, List
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

    ########################################################################
    #  INSERT OR REPLACE THIS METHOD IN YOUR EXISTING ReportWriter CLASS
    ########################################################################
    # ====================== src/strategy/report_writer.py ======================


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
        strategy_config: Dict,  # REQUIRED to show "Strategy Overview"
        df_test: pd.DataFrame,
        trades: List[Dict],
        stats: Dict,
        final_balance: float,
        monthly_data: Dict,
        monthly_levels: List[Dict],
        weekly_levels: List[Dict],
        correlation_data: Dict[str, Dict[str, float]] = None,
        ftmo_data: Dict[str, float] = None,
        mc_results: Dict = None
    ):
        """
        Unified method that prints both:
        - The 'Strategy Overview' (from your old generate_comprehensive_report)
        - The 'full report' sections (Data Overview, Trade Analysis, etc.)

        HOW TO USE:
        1. Copy/paste this into your ReportWriter class in src/strategy/report_writer.py
        2. In runner.py, call it like this:
                rw.generate_full_report(
                    strategy_config=some_dict,
                    df_test=df,
                    trades=trades,
                    stats=stats,
                    final_balance=balance,
                    monthly_data=monthly_data,
                    monthly_levels=monthly_levels,
                    weekly_levels=weekly_levels,
                    correlation_data=correlation_data,
                    ftmo_data=ftmo_data
                )
        3. Run `python runner.py` to confirm the output.
        """

        # ---------------------------
        # 0) Basic Info & Header
        # ---------------------------
        self.file_handler.write("# Comprehensive Backtest Report\n\n")
        self.file_handler.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # -------------------------------------------------
        # 1) Strategy Overview (from old generate_comprehensive_report)
        # -------------------------------------------------
        if strategy_config and "params" in strategy_config and "pair_settings" in strategy_config:
            self.file_handler.write("## Strategy Overview\n\n")

            # Core Logic
            self.file_handler.write("### Core Logic\n")
            self.file_handler.write("- **Type:** Support/Resistance Bounce Strategy\n")
            self.file_handler.write("- **Timeframes:** Primary M15, H1 for level identification\n")
            self.file_handler.write(
                f"- **S/R Validation:** Minimum {strategy_config['params'].get('min_touches', '?')} "
                "touches required for level validation\n"
            )
            self.file_handler.write("- **Price Tolerance per Pair:**\n")
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                tol = settings.get('tolerance', 'n/a')
                self.file_handler.write(f"  * {pair}: {tol} tolerance\n")

            self.file_handler.write("- **Volume Requirements per Pair:**\n")
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                vol_req = settings.get('min_volume_threshold', 'n/a')
                self.file_handler.write(f"  * {pair}: Minimum {vol_req} threshold\n")
            self.file_handler.write("\n")

            # Entry Conditions
            self.file_handler.write("### Entry Conditions\n")
            self.file_handler.write(
                "1. Price must reach validated S/R level within tolerance bands\n"
                "2. First bounce requires minimum volume per pair:\n"
            )
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                bounce_vol = settings.get('min_bounce_volume', 'n/a')
                self.file_handler.write(f"   * {pair}: {bounce_vol} minimum\n")

            self.file_handler.write(
                "3. Second bounce volume must be >80% of first bounce\n"
                "4. 2-hour cooldown between trades on same level\n"
                "5. Cross-pair correlation checks must pass\n\n"
            )

            # Exit Conditions
            rr = strategy_config.get('params', {}).get('risk_reward', '?')
            self.file_handler.write("### Exit Conditions\n")
            self.file_handler.write(f"1. Take Profit: {rr}R from entry\n")
            self.file_handler.write("2. Stop Loss: Dynamic, based on recent price action\n")
            self.file_handler.write("3. Force exit triggers:\n")
            if 'ftmo_limits' in strategy_config:
                daily_loss = strategy_config['ftmo_limits'].get('daily_loss_per_pair', '?')
                total_exposure = strategy_config['ftmo_limits'].get('total_exposure', '?')
                self.file_handler.write(f"   * Daily drawdown reaches {daily_loss}\n")
                self.file_handler.write(f"   * Total exposure reaches {total_exposure}\n\n")

            # Risk Management
            self.file_handler.write("### Risk Management\n")
            self.file_handler.write("1. Position Sizing: 1% risk per trade\n")
            if 'ftmo_limits' in strategy_config:
                corr_lim = strategy_config['ftmo_limits'].get('correlation_limit', '?')
                max_corr_positions = strategy_config['ftmo_limits'].get('max_correlated_positions', '?')
                daily_loss_pp = strategy_config['ftmo_limits'].get('daily_loss_per_pair', '?')
                tot_exposure = strategy_config['ftmo_limits'].get('total_exposure', '?')
                self.file_handler.write(
                    f"2. Correlation Management:\n"
                    f"   * >{corr_lim}: Blocks new trades\n"
                    f"   * Maximum correlated positions: {max_corr_positions}\n"
                )
                self.file_handler.write(
                    f"3. FTMO Rules:\n"
                    f"   * {daily_loss_pp} daily loss limit per pair\n"
                    f"   * {tot_exposure} total exposure limit\n"
                    "   * Maximum 5 lots per position\n\n"
                )
        else:
            # If no strategy_config provided, just mention no overview
            self.file_handler.write("## Strategy Overview\n\n(No strategy_config provided, skipping details.)\n\n")

        # -------------------------------------------------
        # 2) Data Overview
        # -------------------------------------------------
        self.write_data_overview(df_test)

        # -------------------------------------------------
        # 3) Trade & Stats Analysis
        # -------------------------------------------------
        # 3.1 Data Quality
        self.file_handler.write("\n---\n")
        self.write_data_quality_analysis(df_test)

        # 3.2 Temporal & Market Volatility
        self.file_handler.write("\n---\n")
        self.write_temporal_analysis(df_test, trades)

        # 3.3 Trade Analysis
        self.file_handler.write("\n---\n")
        self.write_trade_analysis(trades, df_test)

        # Optionally, if you have MC results you want to show
        # (We won't do anything with mc_results by default,
        #  but you can add code here if needed.)

        # -------------------------------------------------
        # 4) Additional Sections
        # -------------------------------------------------
        self.file_handler.write("\n---\n")
        self.write_trades_section(trades)

        # Summaries
        self.file_handler.write("\n## Summary Stats\n\n")
        self.file_handler.write(f"- Total Trades: {stats['count']}\n")
        self.file_handler.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
        self.file_handler.write(f"- Profit Factor: {stats['profit_factor']:.2f}\n")
        self.file_handler.write(f"- Max Drawdown: ${stats['max_drawdown']:.2f}\n")
        self.file_handler.write(f"- Total PnL: ${stats['total_pnl']:.2f}\n")
        self.file_handler.write(f"- Final Balance: ${final_balance:.2f}\n\n")

        # If multi-symbol performance or correlation data
        self.write_multi_symbol_performance(trades)
        if correlation_data:
            self.write_correlation_report(correlation_data)
        if ftmo_data:
            self.write_ftmo_section(ftmo_data)

        # If you have monthly/weekly narratives or S/R info
        self.write_monthly_breakdown(monthly_data)
        self.write_sr_levels(monthly_levels, weekly_levels)

        self.file_handler.write("\n**End of Comprehensive Backtest Report**\n")


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