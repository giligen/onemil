# RESULT 1,771 — stop placement vs round numbers (FACT, no rule)

Cache (230 day files, 7,096 records): dict `symbol|exit_m|why|fill_min` -> `{measured, t0, bid_250, slip_bps, flag|reason}`.
NO tape window is stored, so read 2 is VOID (`1771_counterfactual.csv`); it needs a network re-fetch, outside
this cell's rules. Spread-tercile, minute and stop-distance cuts not run (no half_spread in the fills CSV).

Measured stops: TRAIN-H2 n 1,831 (128 days), VAL n 2,291 (102 days). All stops are above $5 (the $1-5 tier is empty).

## Read 1, primary: dollar grid, 0-2 c ABOVE a whole dollar vs > 10 c away (mean slip, bp)
| holdout | 0-2 c above (n) | > 10 c (n) | diff | day-t |
|---|---|---|---|---|
| TRAIN-H2 | 30.4 (90) | 36.6 (1,434) | -6.2 | -1.63 |
| VAL | 38.3 (118) | 35.1 (1,792) | +3.3 | +0.44 |

Opposite signs, VAL t 0.44: the hypothesis (just-above-round slips most) is not supported. Other dollar buckets
(TRAIN / VAL): 2-10 c above 36.2 / 25.3; 0-2 c below 28.3 / 54.3 (n 42 / 43); 2-10 c below 34.5 / 34.6.

Half-dollar, 0-2 c above vs > 10 c: TRAIN 38.0 vs 35.7 (t 0.60); VAL 40.6 vs 33.9 (+6.7, t 1.29). Dime grid: every
stop lies within 5 c of a dime, so the > 10 c bucket is empty (contrast undefined); all buckets 31.6-39.8 (flat).

Multiplicity: 3 grids x 5 buckets. The one nominal hint (half VAL, t 1.3) is uncorrected and insignificant.
Medians ~ 22 bp and p90 ~ 80 bp in every bucket.

Verdict: FACT. Measured stop slip does not depend on proximity to round numbers at any grid. No placement rule;
the "stop one tick below the round number" counterfactual remains untested.

CHILDREN:
1. 1,771b: store tape windows (prints + quotes around each stop minute) in a new cache, then run the paired
   counterfactual placement (needs a Databento GO).
2. Slip vs quote state at the stop instant (spread / bid depletion) using bid_250 and an entry half-spread join.
