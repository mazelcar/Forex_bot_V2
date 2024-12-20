# src/strategy/analysis/performance_analyzer.py
import pandas as pd
from typing import List, Dict

class PerformanceAnalyzer:
    def __init__(self, trades: List[Dict]):
        self.trades = trades

    def compute_metrics(self) -> Dict:
        if not self.trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }

        df = pd.DataFrame(self.trades)
        profits = df['profit']

        wins = profits[profits > 0]
        losses = profits[profits < 0]

        win_rate = len(wins) / len(profits) * 100 if len(profits) > 0 else 0.0
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = losses.mean() if not losses.empty else 0.0

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0.0

        # Calculate equity curve and max drawdown:
        equity = profits.cumsum()
        peak = equity.cummax()
        drawdown = (equity - peak)
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        metrics = {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }
        return metrics
