"""Performance Optimizer Module for Forex Trading Bot V2.

This module analyzes trading session performance to identify optimal trading windows.
Key functions:
1. Session performance analysis
2. Performance metrics calculation
3. Time window optimization
4. Session performance database

Author: mazelcar
Created: December 2024
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from venv import logger
import pandas as pd
import numpy as np
from pathlib import Path
import json

class PerformanceOptimizer:
    """Analyzes and optimizes trading performance across different time windows."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize performance optimizer.

        Args:
            config_file: Optional path to configuration file
        """
        self.config = self._load_config(config_file)
        self.session_metrics = {}
        self.performance_db = {}

    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration file if provided, else use defaults."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {
            'analysis_periods': ['1D', '1W', '1M'],
            'min_trades': 10,
            'session_windows': {
                'Sydney': ('21:00', '06:00'),
                'Tokyo': ('23:00', '08:00'),
                'London': ('03:00', '12:00'),
                'New York': ('08:00', '17:00')
            }
        }

    def _get_trading_session(self, hour: int) -> str:
        """Determine trading session based on hour."""
        if 21 <= hour or hour < 6:
            return 'Sydney'
        elif 23 <= hour or hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 17:
            return 'London'
        elif 13 <= hour < 22:
            return 'New York'
        return 'Off-Session'

    def _analyze_timeframe(self, trades: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze trading performance for a specific timeframe."""
        try:
            metrics = {}

            if timeframe == '1H':
                grouper = trades.groupby('hour')
            elif timeframe == '4H':
                trades['4h_block'] = trades['hour'] // 4
                grouper = trades.groupby('4h_block')
            else:  # Daily
                trades['date'] = trades['entry_time'].dt.date
                grouper = trades.groupby('date')

            for name, group in grouper:
                if len(group) >= self.config.get('min_trades', 10):
                    window_metrics = {
                        'total_trades': len(group),
                        'winning_trades': len(group[group['profit'] > 0]),
                        'total_profit': group['profit'].sum(),
                        'avg_profit': group['profit'].mean(),
                        'max_drawdown': self._calculate_drawdown(group),
                        'win_rate': len(group[group['profit'] > 0]) / len(group),
                        'profit_factor': self._calculate_profit_factor(group),
                        'sharpe_ratio': self._calculate_sharpe_ratio(group),
                        'risk_reward': self._calculate_risk_reward(group),
                        'trades': len(group)
                    }

                    # Add volatility metrics
                    if 'high' in group and 'low' in group:
                        window_metrics['avg_volatility'] = (
                            (group['high'] - group['low']).mean()
                        )

                    # Add volume metrics if available
                    if 'volume' in group:
                        window_metrics['avg_volume'] = group['volume'].mean()
                        window_metrics['volume_trend'] = (
                            group['volume'].pct_change().mean()
                        )

                    metrics[name] = window_metrics

            return metrics
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {str(e)}")
            return {}



    def analyze_time_windows(self, trades: pd.DataFrame, time_frames: Optional[List[str]] = None) -> Dict:
        """Analyze performance across different time windows with enhanced metrics.

        Args:
            trades: DataFrame containing trade history with columns:
                - entry_time: Trade entry timestamp
                - exit_time: Trade exit timestamp
                - profit: Trade profit/loss
                - type: Trade type (BUY/SELL)
            time_frames: Optional list of time frames to analyze (e.g., ['1H', '4H', 'D'])

        Returns:
            Dictionary containing comprehensive performance metrics by time window
        """
        try:
            if time_frames is None:
                time_frames = ['1H', '4H', 'D']

            print("\nAnalyzing Trading Windows...")
            print(f"Time frames to analyze: {time_frames}")
            print(f"Total trades: {len(trades)}")

            # Ensure datetime format
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
            trades['exit_time'] = pd.to_datetime(trades['exit_time'])

            # Add session information
            trades['hour'] = trades['entry_time'].dt.hour
            trades['session'] = trades['hour'].apply(self._get_trading_session)

            # Calculate base metrics
            trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
            trades['cumulative_profit'] = trades['profit'].cumsum()

            results = {}
            for tf in time_frames:
                print(f"\nAnalyzing {tf} time frame...")

                # Get metrics for this timeframe
                window_metrics = self._analyze_timeframe(trades, tf)

                # Add session analysis
                session_metrics = self._analyze_sessions(trades)

                # Combine metrics
                results[tf] = {
                    'window_metrics': window_metrics,
                    'session_metrics': session_metrics
                }

                # Rank and score windows
                ranked_windows = self._rank_time_windows(window_metrics)
                results[tf]['rankings'] = ranked_windows

                print(f"Analyzed {len(window_metrics)} windows in {tf} timeframe")
                self._print_top_windows(ranked_windows, tf)

                # Add detailed session analysis
                print("\nSession Performance Summary:")
                for session, metrics in session_metrics.items():
                    print(f"\n{session} Session:")
                    print(f"  Total Trades: {metrics['total_trades']}")
                    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
                    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
                    print(f"  Average Profit: ${metrics['avg_profit']:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error analyzing time windows: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _analyze_sessions(self, trades: pd.DataFrame) -> Dict:
        """Analyze performance by trading session."""
        try:
            session_metrics = {}
            for session in trades['session'].unique():
                session_trades = trades[trades['session'] == session]

                if len(session_trades) > 0:
                    metrics = {
                        'total_trades': len(session_trades),
                        'winning_trades': len(session_trades[session_trades['profit'] > 0]),
                        'total_profit': session_trades['profit'].sum(),
                        'avg_profit': session_trades['profit'].mean(),
                        'win_rate': len(session_trades[session_trades['profit'] > 0]) / len(session_trades),
                        'profit_factor': self._calculate_profit_factor(session_trades),
                        'avg_duration': session_trades['duration'].mean(),
                        'best_trade': session_trades['profit'].max(),
                        'worst_trade': session_trades['profit'].min(),
                        'avg_drawdown': self._calculate_drawdown(session_trades)
                    }

                    # Add time-based analysis
                    by_hour = session_trades.groupby(session_trades['entry_time'].dt.hour)
                    metrics['best_hour'] = by_hour['profit'].mean().idxmax()
                    metrics['worst_hour'] = by_hour['profit'].mean().idxmin()

                    session_metrics[session] = metrics

            return session_metrics
        except Exception as e:
            logger.error(f"Error in session analysis: {str(e)}")
            return {}

    def _calculate_window_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a time window.

        Args:
            trades: DataFrame containing trades for the time window

        Returns:
            Dictionary containing window performance metrics
        """
        total_trades = len(trades)
        winning_trades = len(trades[trades['profit'] > 0])

        metrics = {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_profit': trades['profit'].sum(),
            'avg_profit': trades['profit'].mean(),
            'profit_factor': self._calculate_profit_factor(trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades),
            'max_drawdown': self._calculate_drawdown(trades),
            'risk_reward': self._calculate_risk_reward(trades)
        }

        return metrics

    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor for a set of trades."""
        gross_profit = trades[trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades[trades['profit'] < 0]['profit'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_sharpe_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for a set of trades."""
        returns = trades['profit'].pct_change()
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0

    def _calculate_drawdown(self, trades: pd.DataFrame) -> float:
        """Calculate maximum drawdown for a set of trades."""
        cumulative = trades['profit'].cumsum()
        if len(cumulative) == 0:
            return 0
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max() if len(drawdown) > 0 else 0

    def _calculate_risk_reward(self, trades: pd.DataFrame) -> float:
        """Calculate risk-reward ratio for a set of trades."""
        avg_win = trades[trades['profit'] > 0]['profit'].mean()
        avg_loss = abs(trades[trades['profit'] < 0]['profit'].mean())
        return avg_win / avg_loss if avg_loss != 0 else 0

    def _rank_time_windows(self, metrics: Dict) -> Dict:
        """Rank time windows based on multiple performance metrics.

        Args:
            metrics: Dictionary containing metrics for each time window

        Returns:
            Dictionary containing ranked and scored time windows
        """
        if not metrics:
            return {}

        # Calculate composite score for each window
        scores = {}
        for window, window_metrics in metrics.items():
            # Weight different metrics for final score
            score = (
                window_metrics['win_rate'] * 0.3 +
                min(window_metrics['profit_factor'], 3) / 3 * 0.3 +
                min(window_metrics['sharpe_ratio'], 3) / 3 * 0.2 +
                (1 - window_metrics['max_drawdown']) * 0.2
            )
            scores[window] = {
                'score': score,
                'metrics': window_metrics
            }

        # Sort windows by score
        return dict(sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True))

    def _print_top_windows(self, ranked_windows: Dict, timeframe: str) -> None:
        """Print top performing time windows."""
        print(f"\nTop performing windows for {timeframe}:")
        for i, (window, data) in enumerate(ranked_windows.items(), 1):
            if i > 5:  # Show top 5
                break
            print(f"\n{i}. Window: {window}")
            print(f"   Score: {data['score']:.3f}")
            print(f"   Win Rate: {data['metrics']['win_rate']*100:.1f}%")
            print(f"   Profit Factor: {data['metrics']['profit_factor']:.2f}")
            print(f"   Sharpe Ratio: {data['metrics']['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {data['metrics']['max_drawdown']*100:.1f}%")

    def generate_session_report(self, ranked_windows: Dict) -> str:
        """Generate detailed session performance report.

        Args:
            ranked_windows: Dictionary containing ranked time window performance

        Returns:
            Formatted report string
        """
        report = ["=== Session Performance Report ===\n"]

        for timeframe, window_data in ranked_windows.items():
            report.append(f"\nTimeframe: {timeframe}")
            report.append("-" * 40)

            # Handle both window metrics and session metrics
            if 'window_metrics' in window_data:
                report.append("\nWindow Performance:")
                for window, metrics in window_data['window_metrics'].items():
                    report.append(f"\nWindow: {window}")
                    report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
                    report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                    report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                    report.append(f"Average Profit: ${metrics.get('avg_profit', 0):.2f}")

            if 'session_metrics' in window_data:
                report.append("\nSession Performance:")
                for session, metrics in window_data['session_metrics'].items():
                    report.append(f"\n{session} Session:")
                    report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
                    report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                    report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                    report.append(f"Average Profit: ${metrics.get('avg_profit', 0):.2f}")
                    if 'best_hour' in metrics:
                        report.append(f"Best Hour: {metrics['best_hour']:02d}:00")
                    if 'worst_hour' in metrics:
                        report.append(f"Worst Hour: {metrics['worst_hour']:02d}:00")

        return "\n".join(report)

    def save_analysis(self, ranked_windows: Dict, filepath: str) -> None:
        """Save performance analysis results to file.

        Args:
            ranked_windows: Dictionary containing performance analysis
            filepath: Path to save results
        """
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = {}
            for tf, windows in ranked_windows.items():
                serializable_results[tf] = {}
                for window, data in windows.items():
                    serializable_results[tf][str(window)] = {
                        'score': float(data['score']),
                        'metrics': {
                            k: float(v) if isinstance(v, (np.float32, np.float64))
                            else v for k, v in data['metrics'].items()
                        }
                    }

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)

            print(f"\nAnalysis results saved to: {filepath}")

        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")

    def load_analysis(self, filepath: str) -> Dict:
        """Load previous analysis results.

        Args:
            filepath: Path to analysis results file

        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading analysis results: {str(e)}")
            return {}