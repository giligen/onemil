# RESULT 1,669 -- fast failures: anatomy, prediction at minute 0-1, value decomposition

PREREG: `research/hod_entry/PREREG_1669.md` (FROZEN 2026-09-29 19:45 UTC). Population n=300 (same 5,506-row 1.5%-floored primary book as cell 1,668). Exit-kind (unbounded walk) vs base_exit_type mismatches: 0.

## Part 1 -- anatomy of the actual exit (both halves)
| half | exit | n | <=1m | <=2m | <=5m | <=10m | <=30m | >30m | med min | MFE(R) | recross | clv0 | rng/ATR0 | clv+1 | ret+1(R) | volratio+1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | stop | 56 | 0.00 | 0.00 | 0.07 | 0.11 | 0.50 | 0.50 | 30.0 | 0.509 | 0.89 | 0.63 | 0.11 | 0.45 | -0.098 | 1.17 |
| TRAIN-H2 | target | 26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.88 | 83.0 | 2.140 | 1.00 | 0.70 | 0.11 | 0.70 | 0.039 | 1.15 |
| TRAIN-H2 | eod | 29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 354.0 | 1.185 | 1.00 | 0.65 | 0.15 | 0.51 | 0.022 | 1.04 |
| VAL | stop | 101 | 0.00 | 0.02 | 0.10 | 0.24 | 0.55 | 0.45 | 24.0 | 0.561 | 0.92 | 0.65 | 0.11 | 0.46 | -0.035 | 2.16 |
| VAL | target | 53 | 0.00 | 0.00 | 0.06 | 0.13 | 0.26 | 0.74 | 43.0 | 2.152 | 1.00 | 0.68 | 0.11 | 0.56 | 0.062 | 2.09 |
| VAL | eod | 35 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 343.0 | 1.321 | 1.00 | 0.63 | 0.12 | 0.57 | 0.098 | 1.00 |

## Candle story -- TA-Lib fire rate on bars fill-2..fill+1, top 12 by |stop_fast - stop_slow| gap (pooled)
| pattern | stop_fast rate (n) | stop_slow rate (n) | target rate (n) | eod rate (n) |
|---|---|---|---|---|
| CDLDOJI | 0.150 (14) | 0.541 (143) | 0.448 (79) | 0.441 (64) |
| CDLSHORTLINE | 0.200 (14) | 0.490 (143) | 0.391 (79) | 0.521 (64) |
| CDLLONGLEGGEDDOJI | 0.100 (14) | 0.349 (143) | 0.352 (79) | 0.290 (64) |
| CDLENGULFING | 0.525 (14) | 0.290 (143) | 0.190 (79) | 0.172 (64) |
| CDLCLOSINGMARUBOZU | 0.900 (14) | 0.672 (143) | 0.581 (79) | 0.736 (64) |
| CDLDOJISTAR | 0.000 (14) | 0.152 (143) | 0.115 (79) | 0.109 (64) |
| CDLSPINNINGTOP | 0.200 (14) | 0.337 (143) | 0.419 (79) | 0.361 (64) |
| CDLHANGINGMAN | 0.000 (14) | 0.136 (143) | 0.067 (79) | 0.132 (64) |
| CDLADVANCEBLOCK | 0.000 (14) | 0.132 (143) | 0.124 (79) | 0.209 (64) |
| CDLSHOOTINGSTAR | 0.000 (14) | 0.113 (143) | 0.038 (79) | 0.080 (64) |
| CDLDARKCLOUDCOVER | 0.125 (14) | 0.016 (143) | 0.000 (79) | 0.014 (64) |
| CDLHIKKAKE | 0.200 (14) | 0.294 (143) | 0.228 (79) | 0.155 (64) |

## Part 2 -- fast-failure classifier (out of sample)
| label | k | direction | n tr | n sc | AUC | placebo AUC | P/R@.5 | P/R@.6 | P/R@.7 | P/R@.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| FF2 | 0 | VAL->TRAIN-H2 (swap) | 189 | 111 | nan | nan | n/a | n/a | n/a | n/a |
| FF2 | 1 | VAL->TRAIN-H2 (swap) | 189 | 111 | nan | nan | n/a | n/a | n/a | n/a |
| FF5 | 0 | TRAIN->VAL | 111 | 189 | 0.274 | 0.463 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF5 | 0 | VAL->TRAIN-H2 (swap) | 189 | 111 | 0.512 | 0.572 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF5 | 1 | TRAIN->VAL | 111 | 189 | 0.431 | 0.549 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF5 | 1 | VAL->TRAIN-H2 (swap) | 189 | 111 | 0.565 | 0.315 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF10 | 0 | TRAIN->VAL | 111 | 189 | 0.514 | 0.503 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF10 | 0 | VAL->TRAIN-H2 (swap) | 189 | 111 | 0.633 | 0.568 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF10 | 1 | TRAIN->VAL | 111 | 189 | 0.515 | 0.561 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| FF10 | 1 | VAL->TRAIN-H2 (swap) | 189 | 111 | 0.632 | 0.452 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |

## Permutation importance (VAL-scored, TRAIN-H2->VAL, n_repeats=5, top 6)
| label | k | feature | importance mean | std |
|---|---|---|---|---|
| FF5 | 0 | minutes_since_open | 0.0063 | 0.0118 |
| FF5 | 0 | clv0 | 0.0011 | 0.0021 |
| FF5 | 0 | F14 | 0.0000 | 0.0000 |
| FF5 | 0 | volratio0 | 0.0000 | 0.0000 |
| FF5 | 0 | cdl_CDL3STARSINSOUTH_0 | 0.0000 | 0.0000 |
| FF5 | 0 | cdl_CDL3OUTSIDE_0 | 0.0000 | 0.0000 |
| FF5 | 1 | minutes_since_open | 0.0116 | 0.0040 |
| FF5 | 1 | cS7_mae_1 | 0.0116 | 0.0062 |
| FF5 | 1 | F12 | 0.0042 | 0.0040 |
| FF5 | 1 | F15 | 0.0042 | 0.0021 |
| FF5 | 1 | cS1_dist_level_1 | 0.0042 | 0.0021 |
| FF5 | 1 | cA1_progvol_1 | 0.0042 | 0.0021 |
| FF10 | 0 | minutes_since_open | 0.0116 | 0.0078 |
| FF10 | 0 | volratio0 | 0.0000 | 0.0000 |
| FF10 | 0 | cdl_CDL3WHITESOLDIERS_0 | 0.0000 | 0.0000 |
| FF10 | 0 | cdl_CDLBELTHOLD_0 | 0.0000 | 0.0000 |
| FF10 | 0 | cdl_CDL3STARSINSOUTH_0 | 0.0000 | 0.0000 |
| FF10 | 0 | cdl_CDL3OUTSIDE_0 | 0.0000 | 0.0000 |
| FF10 | 1 | F15 | 0.0063 | 0.0052 |
| FF10 | 1 | closeR0 | 0.0032 | 0.0026 |
| FF10 | 1 | F14 | 0.0011 | 0.0021 |
| FF10 | 1 | F11 | 0.0011 | 0.0021 |
| FF10 | 1 | talib_nbear_1 | 0.0011 | 0.0021 |
| FF10 | 1 | wick_last_1 | 0.0011 | 0.0021 |

## Part 3 -- value decomposition at tau=0.7 (full tau grid in 1669_reads.csv)
Formulas: saved (TP) = cut_gross - base_R; forgone (FP) = base_R - cut_gross; cost = CUT_BPS*next_open/R_unit (all fired). Identity: mean(dR|fired) = share_TP*saved - share_FP*forgone - cost. Break-even precision = (forgone+cost)/(saved+forgone+cost), PREREG formula, reported beside achieved precision.
| label | k | variant | direction | n fired | share cut | dR | day t | ex-top5% | achieved P | break-even P | saved | forgone | cost | identity residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF5 | 0 | C | TRAIN->VAL | 3 | 0.016 | -1.929 | -19.48 | -2.027 | 0.00 | nan | nan | 1.901 | 0.0278 | -2.22e-16 |
| FF5 | 0 | C' | TRAIN->VAL | 3 | 0.016 | -1.929 | -19.48 | -2.027 | 0.00 | nan | nan | 1.901 | 0.0278 | -2.22e-16 |
| FF5 | 0 | C | VAL->TRAIN-H2 (swap) | 1 | 0.009 | -1.197 | nan | -1.197 | 0.00 | nan | nan | 1.178 | 0.0187 | 0.00e+00 |
| FF5 | 1 | C | TRAIN->VAL | 1 | 0.005 | -2.114 | nan | -2.114 | 0.00 | nan | nan | 2.077 | 0.0369 | 0.00e+00 |
| FF5 | 1 | C' | TRAIN->VAL | 1 | 0.005 | -2.114 | nan | -2.114 | 0.00 | nan | nan | 2.077 | 0.0369 | 0.00e+00 |
| FF5 | 1 | C | VAL->TRAIN-H2 (swap) | 2 | 0.018 | -0.684 | -0.71 | -1.645 | 0.00 | nan | nan | 0.666 | 0.0181 | 1.11e-16 |
| FF10 | 0 | C | TRAIN->VAL | 6 | 0.032 | -0.519 | -0.79 | -0.851 | 0.00 | nan | nan | 0.487 | 0.0323 | 1.11e-16 |
| FF10 | 0 | C' | TRAIN->VAL | 6 | 0.032 | -0.519 | -0.79 | -0.851 | 0.00 | nan | nan | 0.487 | 0.0323 | 1.11e-16 |
| FF10 | 0 | C | VAL->TRAIN-H2 (swap) | 1 | 0.009 | -2.506 | nan | -2.506 | 0.00 | nan | nan | 2.467 | 0.0391 | 0.00e+00 |
| FF10 | 1 | C | TRAIN->VAL | 3 | 0.016 | -0.214 | -0.22 | -0.722 | 0.00 | nan | nan | 0.184 | 0.0301 | 2.78e-17 |
| FF10 | 1 | C' | TRAIN->VAL | 2 | 0.011 | -0.656 | -0.45 | -2.114 | 0.00 | nan | nan | 0.621 | 0.0346 | 0.00e+00 |
| FF10 | 1 | C | VAL->TRAIN-H2 (swap) | 3 | 0.027 | -0.236 | -0.33 | -0.641 | 0.00 | nan | nan | 0.208 | 0.0281 | 0.00e+00 |

**Pass bar (dR>=+0.05R, day t>=2.5, ex-top5%>0, BOTH out-of-sample scorings) across all 4 tau x 2 variants x 3 labels x 2 k = 96 cells: 0 pass -- none.**

## Verdicts
* Part 3 cells clearing the pass bar on both out-of-sample scorings: 0/96.
* cS5/S5 (SPY-relative) is void wherever bars_sip carries SPY for a single day only (same as cell 1,668); HistGradientBoostingClassifier handles the resulting NaNs natively, no imputation.

## Adequacy review
* minutes_to_exit is measured from the FILL BAR (bars["minarr"][i0]), matching Part 2's own label wording ("stop-out within k min of the fill bar"), not from the fractional fill instant -- a fill landing late in its own bar reads as up to ~1 minute faster here than a fill-instant clock would show.
* MFE and level re-cross include the exit bar's own high (intrabar order of high vs low is unknown, same convention as 1668's cS2_mfe); this is an optimistic (upper-bound) MFE, not a certified touch.
* Exit-kind from the unbounded walk matched base_exit_type on 100.0% of fills; mismatches (if any) are logged, not silently dropped.
* No Part 2/3 feature or label used a bar after its own decision instant (k=0 uses bars[0..i0] only; k=1 gates on walk_k's preempt=="" so a fill already resolved by bar fill+1 never enters the k=1 model).
* A null in Part 3 is a claim about these τ/k/label/variant cells specifically -- MDE at achieved n is reported per cell in 1669_reads.csv; break-even precision is compared to achieved precision, not asserted.
