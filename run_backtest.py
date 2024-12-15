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
    print("\n" + "="*50)
    print("FOREX TRADING BOT V2 - BACKTEST")
    print("="*50)

    # Initialize strategy
    strategy_config = str(Path("config/strategy.json"))
    strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
    print("\nStrategy initialized successfully")

    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_balance=10000,
        commission=0.07,
        spread=0.00002
    )
    print("Backtester created successfully")

    # Set test parameters
    symbol = "EURUSD"
    timeframe = "M5"
    end_date = datetime(2024, 12, 12, 23, 0)  # Thursday
    chunk_size = timedelta(hours=8)
    test_duration = timedelta(days=5)

    print("\nBacktest Configuration:")
    print("-" * 50)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"End Date: {end_date}")
    print(f"Chunk Size: {chunk_size.total_seconds()/3600:.1f} hours")
    print(f"Test Duration: {test_duration.days} days")
    print("-" * 50)

    # Start from most recent data chunk
    current_end = end_date
    all_results = []
    chunk_count = 0

    while test_duration.days > 0:
        current_start = current_end - chunk_size
        chunk_count += 1

        # Skip weekends
        if current_end.weekday() >= 5:  # Saturday = 5, Sunday = 6
            print(f"\nSkipping weekend: {current_end.strftime('%Y-%m-%d')}")
            current_end = current_end - timedelta(days=1)
            continue

        # Only test during market hours (approx 00:00 - 23:00 GMT)
        if current_end.hour < 23 and current_end.hour >= 0:
            print("\n" + "="*50)
            print(f"Processing Chunk {chunk_count}")
            print(f"Period: {current_start} to {current_end}")
            print("="*50)

            results = backtester.run(
                symbol=symbol,
                timeframe=timeframe,
                start_date=current_start,
                end_date=current_end
            )

            if results:  # Only store if we got data
                print(f"\nChunk {chunk_count} processed successfully")
                print(f"Trades in chunk: {results.get('total_trades', 0)}")
                all_results.append(results)
            else:
                print(f"\nChunk {chunk_count} processing failed")

        # Move to next chunk
        current_end = current_start
        test_duration -= chunk_size

    # Combine and print results
    print("\n" + "="*50)
    print("FINAL BACKTEST RESULTS")
    print("="*50)

    if all_results:
        final_results = combine_results(all_results)

        print("\nPerformance Metrics:")
        print("-" * 30)
        for key, value in final_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backtester.save_results(str(results_path))
        print(f"\nResults saved to: {results_path}")
    else:
        print("\nNo valid results generated")

    print("\nBacktest completed")
    print("="*50)

def combine_results(results_list):
    """Combine results from multiple chunks."""
    if not results_list:
        return {}

    combined = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'gross_profit': 0,
        'gross_loss': 0,
        'total_profit': 0,
    }

    for result in results_list:
        for key in combined.keys():
            combined[key] += result.get(key, 0)

    # Calculate final metrics
    if combined['total_trades'] > 0:
        combined['win_rate'] = combined['winning_trades'] / combined['total_trades']
        if combined['losing_trades'] > 0:
            combined['profit_factor'] = abs(combined['gross_profit'] / combined['gross_loss'])
        else:
            combined['profit_factor'] = float('inf') if combined['gross_profit'] > 0 else 0

    return combined

if __name__ == "__main__":
    main()