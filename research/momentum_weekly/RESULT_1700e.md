# RESULT -- cell 1,700e: can we discover the bearish market and exit?

PREREG_1700e.md (FROZEN). Book = 1,700d's A1 (U2 large caps, 12-1 momentum, weekly, equal weight) plus top-10/top-50 robustness. 6 filters x 3 N = 18 cells x 3 windows = 54 rows in 1700e_cells.csv. Null = 300 draws/period (stated, matches 1700d); OFF periods draw 0 (cash) in both the real book and the null. Windows: whole 2016-01-01..2026-09-30, H1 ..2021-06-30, H2 2021-07-01..

## 18-cell table (filter x N): whole-window read + both-halves pass checks

| filter | N | ann_net | spy | excess_whole | excess_H1 | excess_H2 | alpha_t | maxDD_book | maxDD_spy | null_H1 | null_H2 | weeks_cash | switches | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 10 | +25.3% | +15.0% | +8.9% | +12.2% | +6.2% | 0.66 | -67.9% | -31.8% | 96.33 | 97.67 | +0.0% | 0 | fail |
| F0 | 20 | +24.8% | +15.0% | +8.5% | +7.2% | +9.6% | 0.60 | -63.0% | -31.8% | 92.33 | 100.00 | +0.0% | 0 | fail |
| F0 | 50 | +21.7% | +15.0% | +5.7% | +5.7% | +5.8% | 0.45 | -49.8% | -31.8% | 96.00 | 100.00 | +0.0% | 0 | fail |
| F1 | 10 | +16.4% | +15.0% | +1.1% | +2.5% | -0.0% | 0.75 | -67.6% | -31.8% | 92.33 | 92.33 | +15.4% | 26 | fail |
| F1 | 20 | +16.1% | +15.0% | +0.9% | -0.7% | +2.3% | 0.71 | -62.2% | -31.8% | 83.67 | 99.67 | +15.4% | 26 | fail |
| F1 | 50 | +15.1% | +15.0% | -0.0% | -0.1% | +0.1% | 0.67 | -49.3% | -31.8% | 97.67 | 100.00 | +15.4% | 26 | fail |
| F2 | 10 | +17.8% | +15.0% | +2.4% | +0.7% | +3.9% | 0.54 | -66.3% | -31.8% | 91.00 | 98.67 | +12.0% | 14 | fail |
| F2 | 20 | +17.6% | +15.0% | +2.2% | -3.1% | +6.9% | 0.46 | -60.3% | -31.8% | 72.33 | 100.00 | +12.0% | 14 | fail |
| F2 | 50 | +16.5% | +15.0% | +1.2% | -2.0% | +4.1% | 0.40 | -45.4% | -31.8% | 92.33 | 100.00 | +12.0% | 14 | fail |
| F3 | 10 | +7.7% | +15.0% | -6.4% | +3.6% | -14.1% | 0.08 | -67.2% | -31.8% | 97.67 | 45.67 | +15.2% | 20 | fail |
| F3 | 20 | +8.7% | +15.0% | -5.5% | -0.1% | -9.9% | -0.01 | -63.6% | -31.8% | 92.67 | 82.67 | +15.2% | 20 | fail |
| F3 | 50 | +9.7% | +15.0% | -4.7% | -0.1% | -8.4% | -0.08 | -52.4% | -31.8% | 99.33 | 99.33 | +15.2% | 20 | fail |
| F4 | 10 | +16.4% | +15.0% | +1.1% | +2.5% | -0.0% | 0.75 | -67.6% | -31.8% | 92.00 | 95.00 | +15.4% | 26 | fail |
| F4 | 20 | +17.7% | +15.0% | +2.3% | -0.2% | +4.5% | 0.73 | -62.2% | -31.8% | 81.33 | 100.00 | +13.8% | 18 | fail |
| F4 | 50 | +16.2% | +15.0% | +1.0% | -0.6% | +2.3% | 0.70 | -49.3% | -31.8% | 96.00 | 100.00 | +13.8% | 20 | fail |
| F5 | 10 | +16.9% | +15.0% | +1.6% | +8.2% | -3.6% | 0.76 | -66.6% | -31.8% | 96.67 | 90.67 | +14.8% | 14 | fail |
| F5 | 20 | +16.6% | +15.0% | +1.3% | +3.6% | -0.6% | 0.72 | -61.6% | -31.8% | 93.67 | 99.67 | +14.8% | 14 | fail |
| F5 | 50 | +15.8% | +15.0% | +0.6% | +3.5% | -1.7% | 0.72 | -49.9% | -31.8% | 99.33 | 99.67 | +14.8% | 14 | fail |

## Pass list (0/18): NONE -- no cell clears both-halves excess>0 + alpha t>=2.0 + maxDD<=1.25xSPY + null pct>=95 both halves.
**Bonferroni**: with 18 cells at a 5% per-half chance bar, ~0.05 cells are expected to clear BOTH halves by chance alone; 0 observed passes is within that noise floor.

## A1 (U2, top 20, 12-1, weekly) by year: book return per filter vs SPY

| Year | SPY | F0 | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|---|---|
| 2016 | +14.1% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| 2017 | +21.3% | +12.4% | +12.4% | +12.4% | +12.4% | +12.4% | +12.4% |
| 2018 | -4.3% | -5.5% | -11.2% | -5.3% | -6.2% | -4.6% | +0.4% |
| 2019 | +29.2% | +32.3% | +6.4% | +18.5% | +3.7% | +16.5% | +13.6% |
| 2020 | +19.3% | +68.0% | +58.4% | +19.9% | +58.4% | +37.8% | +58.4% |
| 2021 | +28.6% | -11.9% | -11.9% | -11.9% | -11.9% | -11.9% | -11.9% |
| 2022 | -18.0% | -15.7% | -17.8% | -9.6% | -22.9% | -17.8% | -15.9% |
| 2023 | +24.7% | +20.6% | +8.0% | +23.9% | +9.0% | +8.0% | +7.0% |
| 2024 | +27.9% | +68.9% | +68.9% | +68.9% | +68.9% | +68.9% | +68.9% |
| 2025 | +16.5% | +59.8% | +42.8% | +27.1% | -2.3% | +42.8% | +27.1% |
| 2026 | +12.8% | +49.8% | +33.9% | +49.8% | +6.1% | +49.8% | +27.5% |

## Sub-windows, A1: F0 vs each filter (book / SPY / excess / weeks-in-cash)

| Sub-window | Filter | Book | SPY | Excess | Weeks cash |
|---|---|---|---|---|---|
| 2020-02..04 | F0 | -11.5% | -12.7% | +1.2% | +0.0% |
| 2020-02..04 | F1 | -5.0% | -12.7% | +7.7% | +69.2% |
| 2020-02..04 | F2 | -30.9% | -12.7% | -18.2% | +46.2% |
| 2020-02..04 | F3 | -5.0% | -12.7% | +7.7% | +69.2% |
| 2020-02..04 | F4 | -17.4% | -12.7% | -4.7% | +61.5% |
| 2020-02..04 | F5 | -5.0% | -12.7% | +7.7% | +69.2% |
| 2021 | F0 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2021 | F1 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2021 | F2 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2021 | F3 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2021 | F4 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2021 | F5 | -11.9% | +28.6% | -40.5% | +0.0% |
| 2022 | F0 | -15.7% | -18.0% | +2.3% | +0.0% |
| 2022 | F1 | -17.8% | -18.0% | +0.1% | +80.8% |
| 2022 | F2 | -9.6% | -18.0% | +8.4% | +63.5% |
| 2022 | F3 | -22.9% | -18.0% | -4.9% | +67.3% |
| 2022 | F4 | -17.8% | -18.0% | +0.1% | +80.8% |
| 2022 | F5 | -15.9% | -18.0% | +2.1% | +75.0% |

## Methodology notes
- Universe/signal/cost/stats/null machinery copied from 1700d_grid.py (U2 ADV20>=$200M, price>=$10, >=273d history, 12-1 momentum t-252..t-21, 5bps/side + half spread-proxy cost, 300-draw null).
- F1/F2/F5 evaluated on prior_signal_date (prior close, no look-ahead); F3 evaluated at month-end close, held for the FOLLOWING calendar month (standard Faber/GTAA timing, not the same month).
- F4 (own-book 20% DD kill) is path-dependent per N: simulated in chronological order on that N's own net-of-cost equity curve; cleared only by SPY closing back above its 200-day SMA.
- OFF weeks = 0% return (cash); re-entry always buys the FRESH top-N ranking at the next ON week.
- Switch cost (full liquidation / full re-entry) charges each leg against its OWN portfolio size (new port for buys, prior port for sells) -- identical to 1700d's single-denom formula whenever portfolio size is constant (every F0 period), and the correct generalisation when it empties to cash.
- Null = 300 random same-N draws from the eligible pool on ON periods, 0 on OFF periods (cash, matched to the real book) -- isolates stock-selection skill from the shared regime timing.
- Early periods lacking SMA/return/month-end history default to ON (logged WARNING in 1700e.log), never OFF.
- No parameter outside this PREREG was tried after seeing numbers.
