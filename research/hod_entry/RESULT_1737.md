# RESULT 1,737 — be the bid (PREREG_1737 + Amendment 1): **M1 PASSES the pre-committed bar (NOT yet independently rebuilt); M2 FAIL; M3 FAIL (post-hoc)**

Population 8,245 arms (2025-07..2026-05, TEST sealed). Script `1737_score.py`, orders `1737_orders.csv`, summary `1737_summary.csv`,
diag `1737_diag.txt`. Fact (a) confirmed: the bar at entry_m has h >= level in 87 % of arms (entry_m = break bar). Primary stop =
engine's own (consolidation_low at the last closed bar j-1; M2: at the arm bar a, causal); PREREG-literal bars j-4..j = M1L/M2L
sensitivities. Floor `min_r_pct` = 1.5 % (binds on 65 % of arms; median R 1.7 % of price). Missing bar -> next available row.
M2 arm walk: 3,828 arms get a resting window > 0 min (median 6 bars, p90 46); 1,692 never arm for that level, 2,725 have window 0.

| read | split | orders | fill share | fills/wk | mean net R | day-t | ex-top5 | stop | green wk | placebo pct (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | TRAIN | 3,425 | 45.7 % | 58 | +0.190 | 3.22 | +0.094 | 49 % | 74 % | 100 (-0.07) |
| M1 | VAL | 4,636 | 45.1 % | 95 | +0.186 | 3.73 | +0.090 | 49 % | 86 % | 100 (-0.08) |
| M2 | TRAIN | 1,598 | 93.2 % | 55 | -0.222 | -7.28 | -0.265 | 37 % | 7 % | 100 (-0.46) |
| M2 | VAL | 2,230 | 95.6 % | 97 | -0.292 | -14.6 | -0.339 | 41 % | 0 % | 100 (-0.49) |
| M3 | TRAIN | 386 | 100 % | 14 | -0.079 | -2.44 | -0.079 | 22 % | 37 % | 98 (-0.13) |
| M3 | VAL | 550 | 100 % | 25 | -0.033 | -1.09 | -0.033 | 20 % | 32 % | 100 (-0.09) |

M3 ETB-now: -0.071 / -0.021. M1L +0.182 / +0.178; M2L -0.68 / -0.82 (look-ahead stop, worse). Adverse selection (B0 of unfilled
minus filled result > 0.3 R): not flagged for M1/M2/M3 (M1 unfilled B0 -0.33/-0.38).

**M1 reading.** Every pass-bar item holds on TRAIN and VAL (mean >= +0.10, day-t >= 2, ex-top5 >= 0, placebo >= 95, >= 3 fills/wk,
green >= 55 %). Positive in all 11 months, ex-top-1 % +0.169, top-5 days 28 % of the sum. But: (1) weekly P10 -7.2 / -4.4 R and
max drawdown 57 / 40 R (cadence bar not met); (2) the resting-sell target fills on a touch and pays 0 half-spread — a 10 bp
through-trade rule on bid and target (`1737_summary_thru10bp.csv`) cuts M1 to +0.131 / +0.111 R, day-t 2.2 / 2.3, ex-top5 +0.03 / +0.01
(thin, still above the bar); (3) the universe (bf_zero/universe.csv superset) membership was NOT causality-traced here and B0
on the same arms is -0.10 / -0.16 R filled; (4) queue position ignored, fills are unconstrained by the live caps (2 concurrent / 6 per day);
(5) the PREREG's independent rebuild (REBUILD_1737) has NOT been run — the number must not reach the owner as final until it matches.
A pass = a paper instrument spec, never a live change. M3 is post-hoc (needs its own PREREG on a different regime); it fails anyway.
Cells counted: 1,737 plus sensitivities M1L / M2L / thru10bp (not new cells).
