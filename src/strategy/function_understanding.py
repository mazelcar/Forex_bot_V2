from src.strategy.runner import load_data
from src.strategy.runner import split_data_for_backtest

def main():
    df_result = load_data(symbol="EURUSD", timeframe="H1", days=365)
    train_df, test_df = split_data_for_backtest(df_result, 0.8)
    train_df.to_csv("train_output.csv", index=False)
    test_df.to_csv("test_output.csv", index=False)
    print(f"Train set saved. Rows: {len(train_df)} | Test set saved. Rows: {len(test_df)}")

if __name__ == "__main__":
    main()

