# Comprehensive Backtest Report

**Generated on:** 2025-01-06 19:16:16

## Strategy Overview

### Core Logic
- **Type:** Support/Resistance Bounce Strategy
- **Timeframes:** Primary M15, H1 for level identification
- **S/R Validation:** Minimum 8 touches required for level validation
- **Price Tolerance per Pair:**
  * EURUSD: 0.0005 tolerance
  * GBPUSD: 0.0007 tolerance
- **Volume Requirements per Pair:**
  * EURUSD: Minimum 1200 threshold
  * GBPUSD: Minimum 1500 threshold

### Entry Conditions
1. Price must reach validated S/R level within tolerance bands
2. First bounce requires minimum volume per pair:
   * EURUSD: 1000 minimum
   * GBPUSD: 1200 minimum
3. Second bounce volume must be >80% of first bounce
4. 2-hour cooldown between trades on same level
5. Cross-pair correlation checks must pass

### Exit Conditions
1. Take Profit: 3.0R from entry
2. Stop Loss: Dynamic, based on recent price action
3. Force exit triggers:
   * Daily drawdown reaches ?
   * Total exposure reaches ?

### Risk Management
1. Position Sizing: 1% risk per trade
2. Correlation Management:
   * >?: Blocks new trades
   * Maximum correlated positions: ?
3. FTMO Rules:
   * ? daily loss limit per pair
   * ? total exposure limit
   * Maximum 5 lots per position

# Backtest Detailed Report

Generated on: 2025-01-06 19:16:16

## Data Overview
- Total bars in Test Set: 12037
- Test Start: 2024-07-10 21:45:00+00:00
- Test End: 2025-01-06 21:45:00+00:00


---
# Data Quality Analysis

## Data Coverage
- Total number of bars: 12037
- Date range: 2024-07-10 21:45:00+00:00 to 2025-01-06 21:45:00+00:00

## Data Gaps Detected
- Gap from 2024-07-12 23:45:00+00:00 to 2024-07-15 00:00:00+00:00
- Gap from 2024-07-19 23:45:00+00:00 to 2024-07-22 00:00:00+00:00
- Gap from 2024-07-26 23:45:00+00:00 to 2024-07-29 00:00:00+00:00
- Gap from 2024-07-29 08:00:00+00:00 to 2024-07-29 08:45:00+00:00
- Gap from 2024-08-02 23:45:00+00:00 to 2024-08-05 00:00:00+00:00
- Gap from 2024-08-09 23:45:00+00:00 to 2024-08-12 00:00:00+00:00
- Gap from 2024-08-12 08:00:00+00:00 to 2024-08-12 10:00:00+00:00
- Gap from 2024-08-16 23:45:00+00:00 to 2024-08-19 00:00:00+00:00
- Gap from 2024-08-23 23:45:00+00:00 to 2024-08-26 00:00:00+00:00
- Gap from 2024-08-30 23:45:00+00:00 to 2024-09-02 00:00:00+00:00
- Gap from 2024-09-06 23:45:00+00:00 to 2024-09-09 00:00:00+00:00
- Gap from 2024-09-13 23:45:00+00:00 to 2024-09-16 00:00:00+00:00
- Gap from 2024-09-20 23:45:00+00:00 to 2024-09-23 00:00:00+00:00
- Gap from 2024-09-27 23:45:00+00:00 to 2024-09-30 00:00:00+00:00
- Gap from 2024-10-04 23:45:00+00:00 to 2024-10-07 00:00:00+00:00
- Gap from 2024-10-11 23:45:00+00:00 to 2024-10-14 00:00:00+00:00
- Gap from 2024-10-18 23:45:00+00:00 to 2024-10-21 00:00:00+00:00
- Gap from 2024-10-25 23:45:00+00:00 to 2024-10-28 00:00:00+00:00
- Gap from 2024-11-01 23:45:00+00:00 to 2024-11-04 00:00:00+00:00
- Gap from 2024-11-08 23:45:00+00:00 to 2024-11-11 00:00:00+00:00
- Gap from 2024-11-15 23:45:00+00:00 to 2024-11-18 00:00:00+00:00
- Gap from 2024-11-22 23:45:00+00:00 to 2024-11-25 00:00:00+00:00
- Gap from 2024-11-29 23:45:00+00:00 to 2024-12-02 00:00:00+00:00
- Gap from 2024-12-06 23:45:00+00:00 to 2024-12-09 00:00:00+00:00
- Gap from 2024-12-13 23:45:00+00:00 to 2024-12-16 00:00:00+00:00
- Gap from 2024-12-20 23:45:00+00:00 to 2024-12-23 00:00:00+00:00
- Gap from 2024-12-24 21:45:00+00:00 to 2024-12-26 00:00:00+00:00
- Gap from 2024-12-27 23:45:00+00:00 to 2024-12-30 00:00:00+00:00
- Gap from 2024-12-30 15:00:00+00:00 to 2024-12-31 00:00:00+00:00
- Gap from 2024-12-31 21:45:00+00:00 to 2025-01-02 00:00:00+00:00
- Gap from 2025-01-03 23:45:00+00:00 to 2025-01-06 00:00:00+00:00

## Trading Hours Distribution
Hour | Bar Count
------|----------
00:00 | 504
01:00 | 504
02:00 | 504
03:00 | 504
04:00 | 504
05:00 | 504
06:00 | 504
07:00 | 504
08:00 | 499
09:00 | 500
10:00 | 504
11:00 | 504
12:00 | 504
13:00 | 504
14:00 | 504
15:00 | 501
16:00 | 500
17:00 | 500
18:00 | 500
19:00 | 500
20:00 | 500
21:00 | 501
22:00 | 492
23:00 | 492


---
# Temporal Analysis

## Monthly Trading Activity
Month | Trades | Total PnL | Avg PnL
------|---------|-----------|----------
2024-11 | 6.0 | $60.63 | $10.1
2024-12 | 6.0 | $19.91 | $3.32
2025-01 | 1.0 | $14.47 | $14.47

## Monthly Market Volatility (in pips)
Month | Average | Maximum
------|----------|----------
2024-07 | 4.19 | 52.6
2024-08 | 5.21 | 57.0
2024-09 | 5.46 | 75.3
2024-10 | 4.81 | 65.2
2024-11 | 7.82 | 112.7
2024-12 | 6.73 | 75.2
2025-01 | 8.12 | 60.0


---
# Trade Analysis

## Timing Analysis
- Average holding time: 0 days 10:46:09.230769230
- Shortest trade: 0 days 00:15:00
- Longest trade: 2 days 09:00:00

## Win/Loss Streaks
Length | Type | Start Date | End Date
--------|------|------------|----------
1 | Loss | 2024-11-21 | 2024-11-21
2 | Win | 2024-11-22 | 2024-11-26
2 | Loss | 2024-11-26 | 2024-11-26
1 | Win | 2024-11-27 | 2024-11-27
2 | Loss | 2024-12-19 | 2024-12-19
1 | Win | 2024-12-20 | 2024-12-20
2 | Loss | 2024-12-20 | 2024-12-30
2 | Win | 2024-12-31 | 2025-01-06

## Distance from S/R Levels (pips)
- Average: 8.7
- Minimum: 0.2
- Maximum: 13.1

## Volume Analysis
- Average entry volume: 1715
- Average 3-bar volume: 1626
- Average hourly volume: 1632


---

## Trade Details

| Open Time           | Symbol | Type | Entry Price | Stop Loss | Take Profit | Size | Close Time          | Close Price |   PnL    | Entry Reason                 | Exit Reason                  |
|---------------------|--------|------|------------|----------|------------|------|----------------------|------------|----------|-----------------------------|-----------------------------|
| 2024-11-21 19:00:00+00:00 | GBPUSD | SELL | 1.25793 | 1.25939 | 1.25355 | 0.68 | 2024-11-21 19:30:00+00:00 | 1.25939 | -9.93 | Valid bounce at resistance 1.25924 for GBPUSD | Stop loss hit |
| 2024-11-22 15:00:00+00:00 | GBPUSD | BUY | 1.25183 | 1.25005 | 1.25717 | 0.56 | 2024-11-25 00:00:00+00:00 | 1.25717 | 29.90 | Valid bounce at support 1.25065 for GBPUSD | Take profit hit |
| 2024-11-26 03:00:00+00:00 | GBPUSD | BUY | 1.25194 | 1.24988 | 1.25812 | 0.49 | 2024-11-26 12:30:00+00:00 | 1.25812 | 30.28 | Valid bounce at support 1.25065 for GBPUSD | Take profit hit |
| 2024-11-26 14:15:00+00:00 | GBPUSD | BUY | 1.25998 | 1.25781 | 1.26649 | 0.46 | 2024-11-26 16:00:00+00:00 | 1.25781 | -9.98 | Valid bounce at support 1.25924 for GBPUSD | Stop loss hit |
| 2024-11-26 17:30:00+00:00 | GBPUSD | BUY | 1.25610 | 1.25467 | 1.26039 | 0.70 | 2024-11-26 18:30:00+00:00 | 1.25467 | -10.01 | Valid bounce at support 1.25512 for GBPUSD | Stop loss hit |
| 2024-11-27 10:00:00+00:00 | GBPUSD | BUY | 1.26030 | 1.25810 | 1.26690 | 0.46 | 2024-11-27 17:45:00+00:00 | 1.26690 | 30.36 | Valid bounce at support 1.25924 for GBPUSD | Take profit hit |
| 2024-12-19 16:00:00+00:00 | GBPUSD | BUY | 1.25926 | 1.25776 | 1.26376 | 0.67 | 2024-12-19 17:00:00+00:00 | 1.25776 | -10.05 | Valid bounce at support 1.25924 for GBPUSD | Stop loss hit |
| 2024-12-19 18:00:00+00:00 | GBPUSD | SELL | 1.25413 | 1.25596 | 1.24864 | 0.55 | 2024-12-19 18:15:00+00:00 | 1.25596 | -10.06 | Valid bounce at resistance 1.25512 for GBPUSD | Stop loss hit |
| 2024-12-20 16:30:00+00:00 | GBPUSD | BUY | 1.25493 | 1.25373 | 1.25853 | 0.84 | 2024-12-20 19:00:00+00:00 | 1.25853 | 30.24 | Valid bounce at support 1.25512 for GBPUSD | Take profit hit |
| 2024-12-20 19:30:00+00:00 | GBPUSD | BUY | 1.26055 | 1.25890 | 1.26550 | 0.61 | 2024-12-20 22:30:00+00:00 | 1.25890 | -10.06 | Valid bounce at support 1.25924 for GBPUSD | Stop loss hit |
| 2024-12-30 16:45:00+00:00 | GBPUSD | BUY | 1.25551 | 1.25382 | 1.26058 | 0.60 | 2024-12-30 17:15:00+00:00 | 1.25382 | -10.14 | Valid bounce at support 1.25512 for GBPUSD | Stop loss hit |
| 2024-12-31 17:30:00+00:00 | EURUSD | SELL | 1.03647 | 1.03843 | 1.03059 | 0.51 | 2025-01-02 17:00:00+00:00 | 1.03059 | 29.99 | Valid bounce at resistance 1.03717 for EURUSD | Take profit hit |
| 2025-01-06 14:00:00+00:00 | GBPUSD | SELL | 1.25400 | 1.25592 | 1.24824 | 0.53 | 2025-01-06 21:45:00+00:00 | 1.25127 | 14.47 | Valid bounce at resistance 1.25512 for GBPUSD |  |



## Summary Stats

- Total Trades: 13
- Win Rate: 46.15%
- Profit Factor: 2.35
- Max Drawdown: $20.20
- Total PnL: $95.00
- Final Balance: $10095.00


## Multi-Symbol Performance

| Symbol | Trades | Total PnL |
|--------|--------|-----------|
| EURUSD | 1.0 | 29.99 |
| GBPUSD | 12.0 | 65.02 |


## Correlation Report

- Correlation EURUSD vs GBPUSD: 0.9081
- Correlation GBPUSD vs EURUSD: 0.9081


## FTMO Compliance Report
- Daily Drawdown Limit: 0.05
- Max Drawdown Limit: 0.1
- Profit Target: 0.1
- Current Daily DD: 0.02
- Current Total DD: 0.03


**End of Comprehensive Backtest Report**
