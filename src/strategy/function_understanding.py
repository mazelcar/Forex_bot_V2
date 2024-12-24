import pandas as pd
from src.strategy.runner import load_data, validate_data_for_backtest
from src.strategy.sr_bounce_strategy import identify_sr_yearly, identify_sr_monthly
# NOTE: We'll define identify_sr_weekly in the same sr_bounce_strategy module or import it similarly:
from src.strategy.sr_bounce_strategy import identify_sr_weekly

def main():
    # 1) Load Daily data for Yearly
    df_daily = load_data(symbol="EURUSD", timeframe="D1", days=365)
    validate_data_for_backtest(df_daily)
    print("D1 range:", df_daily["time"].min(), "to", df_daily["time"].max())
    print("D1 rows:", len(df_daily))

    # 2) Load H1 data for Monthly & Weekly
    df_h1 = load_data(symbol="EURUSD", timeframe="H1", days=180)
    validate_data_for_backtest(df_h1)
    print("H1 range:", df_h1["time"].min(), "to", df_h1["time"].max())
    print("H1 rows:", len(df_h1))

    # 3) Identify Yearly, Monthly, and Weekly levels
    #    a) Yearly S/R from daily
    yearly_levels = identify_sr_yearly(df_daily, buffer_pips=0.003)
    # => [yearly_support, yearly_resistance]

    #    b) Monthly S/R from last 2 months of H1
    monthly_levels = identify_sr_monthly(df_h1, months=2, monthly_buffer=0.0015)
    # => [monthly_support, monthly_resistance]

    #    c) Weekly S/R from last 2 weeks of H1
    weekly_levels = identify_sr_weekly(df_h1, weeks=2, weekly_buffer=0.00075)
    # => [weekly_support, weekly_resistance]

    print("Yearly S/R:", yearly_levels)
    print("Monthly S/R:", monthly_levels)
    print("Weekly S/R:", weekly_levels)

    # 4) Create single row with labeled columns
    #    Ensure each list has exactly 2 entries
    if len(yearly_levels) == 2 and len(monthly_levels) == 2 and len(weekly_levels) == 2:
        df_out = pd.DataFrame({
            "yearly_support":    [yearly_levels[0]],
            "yearly_resistance": [yearly_levels[1]],
            "monthly_support":   [monthly_levels[0]],
            "monthly_resistance":[monthly_levels[1]],
            "weekly_support":    [weekly_levels[0]],
            "weekly_resistance": [weekly_levels[1]],
        })
        df_out.to_csv("sr_levels_output.csv", index=False)
        print("\nSaved yearly, monthly & weekly S/R to sr_levels_output.csv")
    else:
        print("\nWarning: Could not find valid levels. Check your data or logic.")

if __name__ == "__main__":
    main()
