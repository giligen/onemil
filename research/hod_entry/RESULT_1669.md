# RESULT 1,669 -- fast failures: anatomy, prediction at minute 0-1, value decomposition

PREREG: `research/hod_entry/PREREG_1669.md` (FROZEN 2026-09-29 19:45 UTC). Population n=5506 (same 5,506-row 1.5%-floored primary book as cell 1,668). Exit-kind (unbounded walk) vs base_exit_type mismatches: 13.

## Part 1 -- anatomy of the actual exit (both halves)
| half | exit | n | <=1m | <=2m | <=5m | <=10m | <=30m | >30m | med min | MFE(R) | recross | clv0 | rng/ATR0 | clv+1 | ret+1(R) | volratio+1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | stop | 1322 | 0.01 | 0.03 | 0.11 | 0.23 | 0.55 | 0.45 | 26.0 | 0.503 | 0.88 | 0.62 | 0.10 | 0.47 | -0.058 | 3.05 |
| TRAIN-H2 | target | 653 | 0.00 | 0.01 | 0.03 | 0.10 | 0.34 | 0.66 | 50.0 | 2.155 | 1.00 | 0.71 | 0.10 | 0.59 | 0.130 | 2.23 |
| TRAIN-H2 | eod | 373 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 343.0 | 1.198 | 1.00 | 0.63 | 0.12 | 0.52 | 0.004 | 1.15 |
| VAL | stop | 1781 | 0.01 | 0.02 | 0.12 | 0.24 | 0.54 | 0.46 | 26.0 | 0.488 | 0.87 | 0.62 | 0.10 | 0.48 | -0.048 | 2.30 |
| VAL | target | 848 | 0.00 | 0.01 | 0.03 | 0.08 | 0.33 | 0.67 | 50.0 | 2.142 | 1.00 | 0.68 | 0.10 | 0.58 | 0.138 | 1.76 |
| VAL | eod | 528 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 341.0 | 1.211 | 1.00 | 0.65 | 0.12 | 0.51 | 0.027 | 1.48 |

## Candle story -- TA-Lib fire rate on bars fill-2..fill+1, top 12 by |stop_fast - stop_slow| gap (pooled)
| pattern | stop_fast rate (n) | stop_slow rate (n) | target rate (n) | eod rate (n) |
|---|---|---|---|---|
| CDLENGULFING | 0.381 (357) | 0.271 (2746) | 0.245 (1501) | 0.225 (901) |
| CDLHIKKAKE | 0.319 (357) | 0.262 (2746) | 0.288 (1501) | 0.233 (901) |
| CDLADVANCEBLOCK | 0.044 (357) | 0.096 (2746) | 0.119 (1501) | 0.139 (901) |
| CDLDOJI | 0.433 (357) | 0.478 (2746) | 0.436 (1501) | 0.461 (901) |
| CDL3OUTSIDE | 0.152 (357) | 0.113 (2746) | 0.118 (1501) | 0.083 (901) |
| CDLSHORTLINE | 0.515 (357) | 0.484 (2746) | 0.452 (1501) | 0.455 (901) |
| CDLEVENINGSTAR | 0.048 (357) | 0.018 (2746) | 0.013 (1501) | 0.010 (901) |
| CDL3INSIDE | 0.064 (357) | 0.034 (2746) | 0.034 (1501) | 0.026 (901) |
| CDLGRAVESTONEDOJI | 0.098 (357) | 0.071 (2746) | 0.079 (1501) | 0.070 (901) |
| CDLINVERTEDHAMMER | 0.047 (357) | 0.023 (2746) | 0.018 (1501) | 0.019 (901) |
| CDLMARUBOZU | 0.472 (357) | 0.491 (2746) | 0.511 (1501) | 0.521 (901) |
| CDLDARKCLOUDCOVER | 0.035 (357) | 0.017 (2746) | 0.018 (1501) | 0.017 (901) |

## Part 2 -- fast-failure classifier (out of sample)
| label | k | direction | n tr | n sc | AUC | placebo AUC | P/R@.5 | P/R@.6 | P/R@.7 | P/R@.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| FF2 | 0 | TRAIN->VAL | 2348 | 3157 | 0.790 | 0.525 | 0.25/0.02 | 0.25/0.02 | 0.33/0.02 | 0.00/0.00 |
| FF2 | 0 | VAL->TRAIN-H2 (swap) | 3157 | 2348 | 0.778 | 0.599 | 0.80/0.09 | 0.80/0.09 | 0.67/0.04 | 0.67/0.04 |
| FF2 | 1 | TRAIN->VAL | 2330 | 3142 | 0.924 | 0.672 | 0.50/0.03 | 0.50/0.03 | 0.50/0.03 | 1.00/0.03 |
| FF2 | 1 | VAL->TRAIN-H2 (swap) | 3142 | 2330 | 0.937 | 0.588 | 0.00/0.00 | 0.00/0.00 | n/a | n/a |
| FF5 | 0 | TRAIN->VAL | 2348 | 3157 | 0.690 | 0.508 | 0.33/0.04 | 0.26/0.02 | 0.29/0.02 | 0.33/0.02 |
| FF5 | 0 | VAL->TRAIN-H2 (swap) | 3157 | 2348 | 0.679 | 0.463 | 0.43/0.06 | 0.50/0.06 | 0.62/0.06 | 0.78/0.05 |
| FF5 | 1 | TRAIN->VAL | 2330 | 3142 | 0.763 | 0.448 | 0.52/0.06 | 0.50/0.06 | 0.41/0.04 | 0.55/0.03 |
| FF5 | 1 | VAL->TRAIN-H2 (swap) | 3142 | 2330 | 0.810 | 0.487 | 0.41/0.16 | 0.43/0.12 | 0.50/0.11 | 0.62/0.10 |
| FF10 | 0 | TRAIN->VAL | 2348 | 3157 | 0.627 | 0.475 | 0.37/0.09 | 0.38/0.06 | 0.42/0.05 | 0.41/0.03 |
| FF10 | 0 | VAL->TRAIN-H2 (swap) | 3157 | 2348 | 0.653 | 0.491 | 0.37/0.09 | 0.43/0.06 | 0.46/0.04 | 0.71/0.03 |
| FF10 | 1 | TRAIN->VAL | 2330 | 3142 | 0.703 | 0.546 | 0.50/0.12 | 0.55/0.10 | 0.53/0.08 | 0.62/0.07 |
| FF10 | 1 | VAL->TRAIN-H2 (swap) | 3142 | 2330 | 0.719 | 0.510 | 0.48/0.19 | 0.48/0.15 | 0.49/0.11 | 0.50/0.08 |

## Permutation importance (VAL-scored, TRAIN-H2->VAL, n_repeats=5, top 6)
| label | k | feature | importance mean | std |
|---|---|---|---|---|
| FF2 | 0 | atr14_pct | 0.0014 | 0.0006 |
| FF2 | 0 | r_pct | 0.0005 | 0.0006 |
| FF2 | 0 | rangeATR0 | 0.0004 | 0.0006 |
| FF2 | 0 | volratio0 | 0.0002 | 0.0004 |
| FF2 | 0 | body0 | 0.0000 | 0.0004 |
| FF2 | 0 | cdl_CDL3OUTSIDE_0 | 0.0000 | 0.0000 |
| FF2 | 1 | body_last_1 | 0.0003 | 0.0002 |
| FF2 | 1 | r_pct | 0.0002 | 0.0003 |
| FF2 | 1 | closeR0 | 0.0001 | 0.0002 |
| FF2 | 1 | F12 | 0.0001 | 0.0003 |
| FF2 | 1 | rangeATR0 | 0.0001 | 0.0001 |
| FF2 | 1 | F15 | 0.0000 | 0.0000 |
| FF5 | 0 | r_pct | 0.0053 | 0.0013 |
| FF5 | 0 | atr14_pct | 0.0037 | 0.0010 |
| FF5 | 0 | minutes_since_open | 0.0030 | 0.0009 |
| FF5 | 0 | F13 | 0.0019 | 0.0007 |
| FF5 | 0 | closeR0 | 0.0018 | 0.0012 |
| FF5 | 0 | rangeATR0 | 0.0017 | 0.0017 |
| FF5 | 1 | r_pct | 0.0013 | 0.0006 |
| FF5 | 1 | cS1_dist_level_1 | 0.0013 | 0.0003 |
| FF5 | 1 | cS4_volratio_1 | 0.0011 | 0.0006 |
| FF5 | 1 | rangeATR0 | 0.0010 | 0.0003 |
| FF5 | 1 | minutes_since_open | 0.0008 | 0.0009 |
| FF5 | 1 | body0 | 0.0008 | 0.0007 |
| FF10 | 0 | r_pct | 0.0144 | 0.0018 |
| FF10 | 0 | minutes_since_open | 0.0113 | 0.0013 |
| FF10 | 0 | closeR0 | 0.0078 | 0.0015 |
| FF10 | 0 | rangeATR0 | 0.0063 | 0.0013 |
| FF10 | 0 | F15 | 0.0056 | 0.0019 |
| FF10 | 0 | atr14_pct | 0.0036 | 0.0022 |
| FF10 | 1 | cS7_mae_1 | 0.0055 | 0.0014 |
| FF10 | 1 | minutes_since_open | 0.0026 | 0.0014 |
| FF10 | 1 | cA1_progvol_1 | 0.0015 | 0.0015 |
| FF10 | 1 | clv_last_1 | 0.0013 | 0.0003 |
| FF10 | 1 | body_last_1 | 0.0012 | 0.0009 |
| FF10 | 1 | cdl_CDL3OUTSIDE_0 | 0.0008 | 0.0003 |

## Part 3 -- value decomposition at tau=0.7 (full tau grid in 1669_reads.csv)
Formulas: saved (TP) = cut_gross - base_R; forgone (FP) = base_R - cut_gross; cost = CUT_BPS*next_open/R_unit (all fired). Identity: mean(dR|fired) = share_TP*saved - share_FP*forgone - cost. Break-even precision = (forgone+cost)/(saved+forgone+cost), PREREG formula, reported beside achieved precision.
| label | k | variant | direction | n fired | share cut | dR | day t | ex-top5% | achieved P | break-even P | saved | forgone | cost | identity residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF2 | 0 | C | TRAIN->VAL | 3 | 0.001 | -1.283 | -1.76 | -1.988 | 0.33 | 0.93 | 0.158 | 1.967 | 0.0248 | 0.00e+00 |
| FF2 | 0 | C' | TRAIN->VAL | 1 | 0.000 | -1.668 | nan | -1.668 | 0.00 | nan | nan | 1.641 | 0.0269 | 0.00e+00 |
| FF2 | 0 | C | VAL->TRAIN-H2 (swap) | 3 | 0.001 | 0.045 | 0.14 | -0.180 | 0.67 | 0.77 | -0.152 | -0.523 | 0.0279 | 0.00e+00 |
| FF2 | 1 | C | TRAIN->VAL | 2 | 0.001 | 0.236 | 1.02 | 0.004 | 0.50 | 1.09 | 0.040 | -0.496 | 0.0321 | 2.78e-17 |
| FF5 | 0 | C | TRAIN->VAL | 14 | 0.004 | 0.030 | -0.01 | -0.060 | 0.29 | 0.05 | 0.152 | -0.019 | 0.0275 | 3.12e-17 |
| FF5 | 0 | C' | TRAIN->VAL | 1 | 0.000 | 1.195 | nan | 1.195 | 0.00 | nan | nan | -1.228 | 0.0331 | 0.00e+00 |
| FF5 | 0 | C | VAL->TRAIN-H2 (swap) | 13 | 0.006 | -0.283 | -0.81 | -0.374 | 0.62 | 0.79 | 0.320 | 1.171 | 0.0297 | -5.55e-17 |
| FF5 | 0 | C' | VAL->TRAIN-H2 (swap) | 1 | 0.000 | 0.647 | nan | 0.647 | 1.00 | nan | 0.674 | nan | 0.0274 | 0.00e+00 |
| FF5 | 1 | C | TRAIN->VAL | 17 | 0.005 | -0.364 | -0.99 | -0.431 | 0.41 | 0.70 | 0.379 | 0.841 | 0.0254 | 5.55e-17 |
| FF5 | 1 | C | VAL->TRAIN-H2 (swap) | 28 | 0.012 | 0.298 | 1.89 | 0.210 | 0.50 | -0.41 | 0.483 | -0.169 | 0.0278 | 0.00e+00 |
| FF10 | 0 | C | TRAIN->VAL | 48 | 0.015 | 0.246 | 1.05 | 0.157 | 0.42 | 0.09 | 0.716 | 0.040 | 0.0289 | 2.78e-17 |
| FF10 | 0 | C' | TRAIN->VAL | 20 | 0.006 | 0.471 | 1.88 | 0.401 | 0.25 | -0.37 | 1.056 | -0.318 | 0.0312 | -1.11e-16 |
| FF10 | 0 | C | VAL->TRAIN-H2 (swap) | 28 | 0.012 | -0.078 | -0.65 | -0.171 | 0.46 | 0.53 | 0.451 | 0.481 | 0.0301 | 1.39e-17 |
| FF10 | 0 | C' | VAL->TRAIN-H2 (swap) | 10 | 0.004 | -0.180 | -0.64 | -0.323 | 0.20 | 0.32 | 0.992 | 0.432 | 0.0325 | 5.55e-17 |
| FF10 | 1 | C | TRAIN->VAL | 64 | 0.020 | -0.001 | -0.29 | -0.066 | 0.53 | 0.51 | 0.438 | 0.436 | 0.0290 | -2.08e-17 |
| FF10 | 1 | C' | TRAIN->VAL | 5 | 0.002 | -0.239 | -0.32 | -0.558 | 0.00 | nan | nan | 0.206 | 0.0326 | -5.55e-17 |
| FF10 | 1 | C | VAL->TRAIN-H2 (swap) | 68 | 0.029 | -0.001 | -0.36 | -0.076 | 0.49 | 0.47 | 0.494 | 0.413 | 0.0277 | -1.10e-16 |
| FF10 | 1 | C' | VAL->TRAIN-H2 (swap) | 5 | 0.002 | -0.384 | -0.58 | -0.729 | 0.20 | 0.44 | 0.878 | 0.661 | 0.0305 | 0.00e+00 |

**Pass bar (dR>=+0.05R, day t>=2.5, ex-top5%>0, BOTH out-of-sample scorings) across all 4 tau x 2 variants x 3 labels x 2 k = 96 cells: 0 pass -- none.**

## Verdicts
* Part 3 cells clearing the pass bar on both out-of-sample scorings: 0/96.
* cS5/S5 (SPY-relative) is void wherever bars_sip carries SPY for a single day only (same as cell 1,668); HistGradientBoostingClassifier handles the resulting NaNs natively, no imputation.

## Adequacy review
* minutes_to_exit is measured from the FILL BAR (bars["minarr"][i0]), matching Part 2's own label wording ("stop-out within k min of the fill bar"), not from the fractional fill instant -- a fill landing late in its own bar reads as up to ~1 minute faster here than a fill-instant clock would show.
* MFE and level re-cross include the exit bar's own high (intrabar order of high vs low is unknown, same convention as 1668's cS2_mfe); this is an optimistic (upper-bound) MFE, not a certified touch.
* Exit-kind from the unbounded walk matched base_exit_type on 99.8% of fills; mismatches (if any) are logged, not silently dropped.
* No Part 2/3 feature or label used a bar after its own decision instant (k=0 uses bars[0..i0] only; k=1 gates on walk_k's preempt=="" so a fill already resolved by bar fill+1 never enters the k=1 model).
* A null in Part 3 is a claim about these τ/k/label/variant cells specifically -- MDE at achieved n is reported per cell in 1669_reads.csv; break-even precision is compared to achieved precision, not asserted.
