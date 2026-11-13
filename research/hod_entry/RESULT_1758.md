# RESULT 1,758 - F3 leveraged-ETF rebalance, I2 calendar cuts (1758_score.py)
Data 2025-01..2026-05 (TRAIN 2025, VAL 2026). TEST dropped at load: 3,521 signals. VAL read once per read, after the TRAIN CSV. Count 1,757 to 1,758.

**F3: KILL.** SPY = paths.spy_close; coverage 344/344 signal days (TRAIN 242, VAL 102), 95 % of weekdays. Open = first bar, median 09:40 ET, so r is 09:40 to 15:00. Trade 15:00 to 15:55 closes with the sign of r, 2 bp round trip.
k = 1 %: VAL n 8 days, -7.9 bp, t -0.6 (up 3 days; down 5 days -9.2 bp). TRAIN n 28, +15.0 bp, t 1.2, halves +26.6 (n 19) / -9.7 (n 9): opposite. Comparator |r| < 1 %: VAL -1.4 bp, not below the trade-side. All three kill flags fire.
k = 0.5 %: VAL n 37, +1.0 bp, t 0.2. k = 1.5 %: TRAIN n 16 (+24 bp), VAL 0. The pre-picked k yields about 20 days a year, so the MDE is large.

**I2 calendar: noise.** Net R = B0 of 1735_per_signal.csv (12,010 arms). Nine levels in five cuts. Each level is an excess over the rest of the book, day-clustered t. No level has one sign on both TRAIN halves and VAL with |t| >= 2. Both VAL hits flip: opex Friday VAL -0.34 R (t -2.7) after +0.04/+0.07; Friday -0.19 R (t -2.1) after -0.08/+0.06. FOMC: -0.28/-0.24 R in TRAIN, +0.17 in VAL (t 1.3). The FOMC list is hard-coded in the script from memory of the Fed calendar (2016-2023 as asked, plus 2024-2026 because the data are 2025-26).

CHILDREN:
1. F3b: same 15:00 to 15:55 rule on a longer SPY history (2016-2023), where |r| >= 1 % has enough days.
2. F3c: signal on the overnight-plus-intraday return, not the raw intraday move.
3. I2b: opex Friday and FOMC as a risk cut on the HOD book, only if a second year of arms repeats the VAL sign.
