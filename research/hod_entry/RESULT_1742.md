# RESULT 1,742 — retest-bid anatomy + V1 hold-gated bid + V2 retest-and-reclaim: FAIL (TEST sealed) — 2026-11-12
Programme count 1,742. Files: `1742_score.py`, `1742_features_b1.csv`, `1742_orders_b1.csv`, `1742_anatomy_b1.csv`, `1742_summary_b1.csv`, `1742_run.log`.
Arms = `1740_arms_b1.csv` (corrected true-SIP level, the RESULT_1740 answer; level = max h with m < entry_m-1). Scorer reproduces 1,740 M1 exactly
(strict TRAIN -0.088 / VAL -0.183; 10 bp -0.148 / -0.238). 904 arms have no C. No bars missing.

## Part 1 anatomy (hypotheses only; M1 strict fills, mean net R by quintile Q1..Q5, spread Q5-Q1, day-clustered t; rank = |VAL t| with TRAIN sign agreeing)
| feature | TRAIN Q1..Q5 | VAL spread (t) | TRAIN spread (t) | agree |
|---|---|---|---|---|
| retest_depth (3 bars from touch; ex-post) | +.27 +.08 -.10 -.17 -.52 | -0.70 (-10.6) | -0.79 (-4.8) | yes |
| half_spread | +.01 -.06 -.10 -.10 -.19 | -0.27 (-3.8) | -0.20 (-1.5) | yes |
| chase (B0 entry vs level) | flat | -0.08 (-1.2) | +0.13 (+0.8) | no |
| C minute-of-day / bars_since_open | Q5 (late) +.41 | -0.06 (-0.6) | +0.63 (+2.4) | no |
| prior-5-bar range | flat | -0.04 (-0.6) | +0.07 (+0.3) | no |
| hold_k | flat | -0.00 (-0.05) | -0.00 (-0.00) | yes (null) |
Same on B0 over all arms: hold_k Q5-Q1 +0.87/+0.93 (t 6.7/12.0; "never retests" = runner, mechanical), retest_depth -0.65/-0.54, half_spread -0.44/-0.57.
Hypotheses: (1) fills where the level barely pierced (depth ~0, shallow wick) earn +0.07..+0.27 but depth is only known AFTER the fill, so it is a
description of the loss, not a filter; a causal proxy would need to predict depth at the arm. (2) Cheap spread names lose less. (3) Hold length carries nothing for the M1 fill.

## Part 2 (strict, net R / day-t / n filled; orders placed)
| read | TRAIN H1 / H2 | TRAIN | VAL | ex-top5 T/V | stop share T/V | placebo T/V | fills/wk T/V |
|---|---|---|---|---|---|---|---|
| V1 k=5 (primary) | +.014 / -.025 | -0.006, t -0.07, 997 (1,599 orders) | -0.195, t -3.10, 650 (1,087) | -.11 / -.31 | .46 / .53 | 99 / 8 | 19 / 30 |
| V1 k=3 | +.058 / -.093 | -0.016, t -0.15, 1,526 | -0.201, t -3.46, 995 | -.12 / -.32 | .48 / .54 | 99 / 26 | 29 / 45 |
| V1 k=10 | -.122 / -.025 | -0.069, t -0.92, 465 | -0.182, t -2.18, 322 | -.18 / -.30 | .45 / .50 | 98 / 35 | 9 / 15 |
| V2 reclaim | -.011 / -.172 | -0.091, t -1.06, 4,979 | -0.209, t -5.37, 3,273 | -.20 / -.33 | .45 / .50 | 100 / 87 | 96 / 149 |
10 bp variant V1 k=5: -0.087 / -0.253 (k=3 -0.094/-0.248, k=10 -0.147/-0.257). Capped book (8/day, 4 conc.), V1 k=5: TRAIN n 758 -0.101 (t -1.9), VAL n 488 -0.226 (t -3.5); V2: TRAIN n 1,667 -0.231 (t -6.3), VAL n 742 -0.176 (t -3.2).
Filled vs unfilled B0 (adverse selection): V1 k=5 filled -0.24 / -0.45 vs unfilled -0.11 / -0.19 (TRAIN/VAL); V2 filled -0.05 / -0.18 vs unfilled -0.35 / -0.35 (the reclaim buys the better half of arms but not enough).

## Verdict: FAIL
Pass bar (V1 k=5 strict >= +0.10, t >= 2 on TRAIN and VAL, 10 bp >= +0.05, placebo >= 95 on VAL, ex-top5 >= 0) missed on every line: TRAIN -0.006, VAL -0.195, VAL placebo 8. TEST untouched.
Reads: 4 (k=3/5/10, V2) + 7 anatomy features; all negative on VAL, so no multiplicity rescue is needed.
Caveats: V1 order is dropped if any close in the hold window is >= 1.02 x level (my reading of the cancel rule); V2 placebo uses pseudo-levels scaled from random starts;
queue ignored (fills optimistic); stop includes the break bar's low (as 1,740); anatomy features hold_k / retest_depth are ex-post to the fill; VAL is a worse regime than TRAIN for every read.
Literal-level arms (`1740_arms.csv`) not run (1,740 showed the same sign).
