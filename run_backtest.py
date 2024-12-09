"""Backtesting Example for MA RSI Volume Strategy.

This script demonstrates how to:
1. Initialize the strategy
2. Set up the backtester
3. Run a backtest
4. Analyze results
"""

from datetime import datetime, timedelta
from pathlib import Path
from src.strategy.ma_rsi_volume import MA_RSI_Volume_Strategy
from src.strategy.backtesting import Backtester

def main():
    # 1. Initialize your strategy
    strategy_config = str(Path("config/strategy.json"))
    strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)

    # 2. Create backtester instance
    backtester = Backtester(
        strategy=strategy,
        initial_balance=10000,  # $10,000 starting balance
        commission=2.0,         # $2 commission per trade
        spread=0.0001          # 1 pip spread (0.0001 for 4 digit pairs)
    )

    # 3. Set up backtest parameters
    symbol = "EURUSD"
    timeframe = "M5"  # 5-minute timeframe

    # Test last 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # 4. Run the backtest
    results = backtester.run(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    # 5. Print results
    print("\nBacktest Results:")
    print("-" * 40)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Profit: ${results['total_profit']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    # 6. Save detailed results to file
    results_path = Path("results") / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    backtester.save_results(str(results_path))

    # 7. Plot equity curve and drawdown
    backtester.plot_results()

if __name__ == "__main__":
    main()