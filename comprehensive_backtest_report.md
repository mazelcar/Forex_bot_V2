# Comprehensive Backtest Report

**Generated on:** 2025-01-25 14:12:20

## Strategy Overview

### Core Logic
- **Type:** Support/Resistance Bounce Strategy
- **Timeframes:** Primary M15, H1 for level identification
- **S/R Validation:** Minimum 3 touches required for level validation
- **Price Tolerance per Pair:**
  * EURUSD: 0.0005 tolerance
  * GBPUSD: 0.0007 tolerance
- **Volume Requirements per Pair:**
  * EURUSD: Minimum 500 threshold
  * GBPUSD: Minimum 600 threshold

### Entry Conditions
1. Price must reach validated S/R level within tolerance bands
2. First bounce requires minimum volume per pair:
   * EURUSD: 400 minimum
   * GBPUSD: 500 minimum
3. Second bounce volume must be >80% of first bounce
4. 2-hour cooldown between trades on same level
5. Cross-pair correlation checks must pass

### Exit Conditions
1. Take Profit: 2.0R from entry
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

Generated on: 2025-01-25 14:12:20

## Data Overview
- Total bars in Test Set: 24835
- Test Start: 2024-01-24 21:45:00+00:00
- Test End: 2025-01-23 21:45:00+00:00


---
# Data Quality Analysis

## Data Coverage
- Total number of bars: 24835
- Date range: 2024-01-24 21:45:00+00:00 to 2025-01-23 21:45:00+00:00

## Data Gaps Detected
- Gap from 2024-01-26 23:45:00+00:00 to 2024-01-29 00:00:00+00:00
- Gap from 2024-02-02 23:45:00+00:00 to 2024-02-05 00:00:00+00:00
- Gap from 2024-02-09 23:45:00+00:00 to 2024-02-12 00:00:00+00:00
- Gap from 2024-02-16 23:45:00+00:00 to 2024-02-19 00:00:00+00:00
- Gap from 2024-02-23 23:45:00+00:00 to 2024-02-26 00:00:00+00:00
- Gap from 2024-03-01 23:45:00+00:00 to 2024-03-04 00:00:00+00:00
- Gap from 2024-03-08 23:45:00+00:00 to 2024-03-11 00:00:00+00:00
- Gap from 2024-03-15 23:45:00+00:00 to 2024-03-18 00:00:00+00:00
- Gap from 2024-03-22 23:45:00+00:00 to 2024-03-25 00:00:00+00:00
- Gap from 2024-03-29 22:45:00+00:00 to 2024-04-01 00:00:00+00:00
- Gap from 2024-04-05 23:45:00+00:00 to 2024-04-08 00:00:00+00:00
- Gap from 2024-04-12 23:45:00+00:00 to 2024-04-15 00:00:00+00:00
- Gap from 2024-04-19 23:45:00+00:00 to 2024-04-22 00:00:00+00:00
- Gap from 2024-04-26 23:45:00+00:00 to 2024-04-29 00:00:00+00:00
- Gap from 2024-05-03 23:45:00+00:00 to 2024-05-06 00:00:00+00:00
- Gap from 2024-05-10 23:45:00+00:00 to 2024-05-13 00:00:00+00:00
- Gap from 2024-05-17 23:45:00+00:00 to 2024-05-20 00:00:00+00:00
- Gap from 2024-05-24 23:45:00+00:00 to 2024-05-27 00:00:00+00:00
- Gap from 2024-05-31 23:45:00+00:00 to 2024-06-03 00:00:00+00:00
- Gap from 2024-06-07 23:45:00+00:00 to 2024-06-10 00:00:00+00:00
- Gap from 2024-06-14 23:45:00+00:00 to 2024-06-17 00:00:00+00:00
- Gap from 2024-06-21 23:45:00+00:00 to 2024-06-24 00:00:00+00:00
- Gap from 2024-06-28 23:45:00+00:00 to 2024-07-01 00:00:00+00:00
- Gap from 2024-07-04 00:15:00+00:00 to 2024-07-04 00:45:00+00:00
- Gap from 2024-07-05 23:45:00+00:00 to 2024-07-08 00:00:00+00:00
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
- Gap from 2024-12-31 21:45:00+00:00 to 2025-01-02 00:00:00+00:00
- Gap from 2025-01-03 23:45:00+00:00 to 2025-01-06 00:00:00+00:00
- Gap from 2025-01-10 23:45:00+00:00 to 2025-01-13 00:00:00+00:00
- Gap from 2025-01-17 23:45:00+00:00 to 2025-01-20 00:00:00+00:00

## Trading Hours Distribution
Hour | Bar Count
------|----------
00:00 | 1035
01:00 | 1036
02:00 | 1036
03:00 | 1036
04:00 | 1036
05:00 | 1036
06:00 | 1036
07:00 | 1036
08:00 | 1031
09:00 | 1032
10:00 | 1036
11:00 | 1036
12:00 | 1036
13:00 | 1036
14:00 | 1036
15:00 | 1036
16:00 | 1036
17:00 | 1036
18:00 | 1036
19:00 | 1036
20:00 | 1036
21:00 | 1037
22:00 | 1028
23:00 | 1024


---
# Temporal Analysis


## Monthly Market Volatility (in pips)
Month | Average | Maximum
------|----------|----------
2024-01 | 6.06 | 50.5
2024-02 | 4.87 | 87.4
2024-03 | 4.19 | 63.1
2024-04 | 5.13 | 83.6
2024-05 | 4.24 | 61.6
2024-06 | 4.66 | 66.8
2024-07 | 4.18 | 52.6
2024-08 | 5.21 | 57.0
2024-09 | 5.46 | 75.3
2024-10 | 4.81 | 65.2
2024-11 | 7.82 | 112.7
2024-12 | 6.78 | 75.2
2025-01 | 7.66 | 92.4


---
# Trade Analysis

No trades to analyze.


---

## Trade Details

No trades were executed.


## Summary Stats

- Total Trades: 0
- Win Rate: 0.00%
- Profit Factor: 0.00
- Max Drawdown: $0.00
- Total PnL: $0.00
- Final Balance: $10000.00


## Multi-Symbol Performance
No trades found.

## Correlation Report

- Correlation EURUSD vs GBPUSD: 0.8130
- Correlation GBPUSD vs EURUSD: 0.8130


## FTMO Compliance Report
- Daily Drawdown Limit: 0.05
- Max Drawdown Limit: 0.1
- Profit Target: 0.1
- Current Daily DD: 0.02
- Current Total DD: 0.03


**End of Comprehensive Backtest Report**
