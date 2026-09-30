# RESULT — cell 1,680: does the 50% partial at +1R buy consistency on ORB?

PREREG: research/orb_exit/PREREG_1680.md. R floor drops 1/601 fills (L1 EIDO_2026-06-09_195).

## Consistency bar clause table -- (d) 50% @ +1R vs base, L2 by year

| clause | 2025 base | 2025 (d) | 2025 pass | 2026 base | 2026 (d) | 2026 pass |
|---|---|---|---|---|---|---|
| mean_R_within_0.03 | 0.288 | 0.300 | PASS | 0.280 | 0.305 | PASS |
| green_week_share_ge_base_plus5pp | 0.548 | 0.703 | PASS | 0.655 | 0.710 | PASS |
| weekly_P10_ge_base_plus_0.5R | -1.132 | -1.300 | FAIL | -2.266 | -1.147 | PASS |
| weekly_sharpe_ge_base_x1.15 | 0.286 | 0.366 | PASS | 0.456 | 0.527 | PASS |
| gap_median_not_worse_1wk | 2.500 | 5.000 | FAIL | 4.000 | 3.500 | PASS |
| null_percentile_ge_95 | N/A | 48.500 | FAIL | N/A | 34.400 | FAIL |

**Overall verdict: FAIL** (all clauses must pass in BOTH years).

## Green-day share (day sum > 0 vs < 0, days with >= 1 fill)

| split | rule | n_days | green_share | daily P10 | worst day | MDD (R) | MDD ($) |
|---|---|---|---|---|---|---|---|
| L2_2025 | base | 127 | 0.449 | -1.038 | -3.071 | 11.45 | 4292 |
| L2_2025 | d | 127 | 0.528 | -1.047 | -3.070 | 7.80 | 2924 |
| L2_2025 | e | 127 | 0.480 | -2.043 | -3.141 | 19.12 | 7170 |
| L2_2026 | base | 132 | 0.462 | -1.396 | -4.598 | 8.81 | 3302 |
| L2_2026 | d | 132 | 0.515 | -1.380 | -7.155 | 9.19 | 3447 |
| L2_2026 | e | 132 | 0.424 | -2.051 | -7.155 | 15.50 | 5813 |
| L2_whole | base | 259 | 0.456 | -1.307 | -4.598 | 11.45 | 4292 |
| L2_whole | d | 259 | 0.521 | -1.102 | -7.155 | 9.19 | 3447 |
| L2_whole | e | 259 | 0.452 | -2.045 | -7.155 | 25.75 | 9656 |
| L1_whole | base | 53 | 0.396 | -2.308 | -3.326 | 28.40 | 9758 |
| L1_whole | d | 53 | 0.453 | -1.902 | -3.597 | 10.61 | 4165 |
| L1_whole | e | 53 | 0.377 | -2.730 | -3.597 | 26.86 | 9000 |

## Weekly reads (all splits, all rules)

| split | rule | n_wk | green_share | null_pctl | mean_R | SD | Sharpe | P10 | worst | strong_wk | gap_med | gap_p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L2_2025 | base | 53 | 0.548 | 8.9 | 1.152 | 4.023 | 0.286 | -1.132 | -5.521 | 5 | 2.5 | 14.2 |
| L2_2025 | d | 53 | 0.703 | 48.5 | 1.199 | 3.271 | 0.366 | -1.300 | -5.143 | 4 | 5.0 | 11.4 |
| L2_2025 | e | 53 | 0.475 | 10.1 | 1.081 | 5.180 | 0.209 | -2.254 | -7.383 | 6 | 4.0 | 11.8 |
| L2_2026 | base | 39 | 0.655 | 50.6 | 1.910 | 4.191 | 0.456 | -2.266 | -4.395 | 8 | 4.0 | 7.0 |
| L2_2026 | d | 39 | 0.710 | 34.4 | 2.077 | 3.941 | 0.527 | -1.147 | -2.400 | 7 | 3.5 | 7.5 |
| L2_2026 | e | 39 | 0.548 | 34.1 | 2.090 | 6.132 | 0.341 | -3.238 | -5.539 | 9 | 3.0 | 5.6 |
| L2_whole | base | 92 | 0.592 | 16.6 | 1.473 | 4.090 | 0.360 | -1.792 | -5.521 | 13 | 3.5 | 15.1 |
| L2_whole | d | 92 | 0.706 | 36.8 | 1.571 | 3.576 | 0.439 | -1.305 | -5.143 | 11 | 4.5 | 13.5 |
| L2_whole | e | 92 | 0.507 | 12.5 | 1.509 | 5.593 | 0.270 | -2.456 | -7.383 | 15 | 3.5 | 12.6 |
| L1_whole | base | 19 | 0.286 | 43.6 | -0.650 | 3.509 | -0.185 | -3.350 | -7.182 | 2 | 1 | 1 |
| L1_whole | d | 19 | 0.462 | 2.5 | 0.779 | 4.176 | 0.187 | -2.455 | -4.354 | 2 | 1 | 1 |
| L1_whole | e | 19 | 0.400 | 42.3 | 0.098 | 5.812 | 0.017 | -5.569 | -8.716 | 3 | 2.0 | 2.8 |

## Compounding read -- L2 whole (2025-01..2026-09), $65K start, weekly geometric compounding
Weekly $ = R_sum_week x fixed risk-per-fill ($375 current stage / $750 next rung); cap *= (1 + $/cap) each week.

| rule | risk/fill | n_wk | final capital | weekly geo growth |
|---|---|---|---|---|
| base | $375 | 92 | $115,823 | 0.630% |
| base | $750 | 92 | $166,645 | 1.029% |
| d | $375 | 92 | $119,203 | 0.661% |
| d | $750 | 92 | $173,406 | 1.072% |

## L1 (live) stability -- damaged-execution period, forward reference only

- base: n=122 fills, mean R=-0.101, weekly green share=0.286, n_weeks=19
- d: n=122 fills, mean R=0.121, weekly green share=0.462, n_weeks=19
- e: n=122 fills, mean R=0.015, weekly green share=0.400, n_weeks=19

Caveats: L1 spans 2026-05-19..2026-09-28 only (the damaged-execution period, small n);
strong-week gap (C1) requires >=2 weeks at R>=+5 -- most splits here never reach a single
+5R week at this book's per-fill scale, so gap_median/gap_p90/the clause built on them read N/A.
