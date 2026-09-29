# RESULT 1,670 -- feature-timing map: where the information lives

PREREG: `research/hod_entry/PREREG_1670.md` (FROZEN, incl. amendments 1-2). Population n=5506 (1663 primary book, r_pct>=1.5%, 5,506 rows). SPY (family M) coverage by k: k0=100%, k1=100%, k2=100%, k5=100%, k10=100%, k30=100%, k60=100%.

## R1 -- AUC map (family x k), out-of-sample, label = stop-out after k

**TRAIN->VAL**

| k | A | P | V | M | S | ALL | n_open |
|---|---|---|---|---|---|---|---|
| 0 | 0.515 | 0.525 | 0.506 | 0.488 | 0.517 | 0.536 | 3157 |
| 1 | 0.511 | 0.565 | 0.562 | 0.486 | 0.544 | 0.573 | 3142 |
| 2 | 0.515 | 0.583 | 0.588 | 0.476 | 0.525 | 0.592 | 3100 |
| 5 | 0.507 | 0.620 | 0.615 | 0.519 | 0.537 | 0.626 | 2897 |
| 10 | 0.507 | 0.643 | 0.626 | 0.507 | 0.550 | 0.653 | 2605 |
| 30 | 0.505 | 0.667 | 0.651 | 0.497 | 0.558 | 0.692 | 1833 |
| 60 | 0.518 | 0.700 | 0.670 | 0.502 | 0.553 | 0.709 | 1308 |

**VAL->TRAIN-H2 (swap)**

| k | A | P | V | M | S | ALL | n_open |
|---|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.518 | 0.505 | 0.492 | 0.521 | 0.534 | 2349 |
| 1 | 0.514 | 0.573 | 0.565 | 0.484 | 0.550 | 0.575 | 2331 |
| 2 | 0.512 | 0.595 | 0.591 | 0.487 | 0.531 | 0.597 | 2292 |
| 5 | 0.500 | 0.633 | 0.622 | 0.473 | 0.553 | 0.641 | 2167 |
| 10 | 0.497 | 0.657 | 0.634 | 0.514 | 0.547 | 0.675 | 1942 |
| 30 | 0.500 | 0.668 | 0.657 | 0.505 | 0.552 | 0.698 | 1333 |
| 60 | 0.501 | 0.685 | 0.682 | 0.543 | 0.547 | 0.722 | 922 |

Placebo AUC (family=ALL, within-day label shuffle): k0/TRAIN->VAL=0.494; k0/VAL->TRAIN-H2 (swap)=0.491; k1/TRAIN->VAL=0.486; k1/VAL->TRAIN-H2 (swap)=0.521; k2/TRAIN->VAL=0.520; k2/VAL->TRAIN-H2 (swap)=0.494; k5/TRAIN->VAL=0.539; k5/VAL->TRAIN-H2 (swap)=0.486; k10/TRAIN->VAL=0.499; k10/VAL->TRAIN-H2 (swap)=0.494; k30/TRAIN->VAL=0.570; k30/VAL->TRAIN-H2 (swap)=0.524; k60/TRAIN->VAL=0.587; k60/VAL->TRAIN-H2 (swap)=0.588

## R2 -- money table (family=ALL, cut at fixed tau)

Best (k, tau, variant) by mean paired dR per direction (full 112-row table: `1670_reads.csv`, part=R2):

| direction | k | tau | variant | n_fired | mean_dR | iid_t | day_t | ex_top5_dR | mde | achieved_prec | breakeven_prec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN->VAL | 60 | 0.8 | remstop0.5 | 78 | 0.042 | 0.371 | -0.161 | -0.015 | 0.320 | 0.564 | 0.527 |
| VAL->TRAIN-H2 (swap) | 5 | 0.7 | remstop0.5 | 600 | 0.021 | 0.418 | 0.683 | -0.061 | 0.143 | 0.620 | 0.603 |

## R3 -- permutation importance, top 10, family=ALL, best-AUC k per scoring

Best-AUC k (TRAIN->VAL direction) = 60

| feature | importance_mean | importance_std |
|---|---|---|
| mtm_R_60 | 0.0731 | 0.0033 |
| progress_per_vol_60 | 0.0154 | 0.0042 |
| r_pct | 0.0040 | 0.0027 |
| F6 | 0.0034 | 0.0019 |
| talib_nbull_60 | 0.0034 | 0.0036 |
| mfe_R_60 | 0.0032 | 0.0011 |
| dvol_since_fill_adv20_60 | 0.0026 | 0.0020 |
| mae_R_60 | 0.0023 | 0.0023 |
| F15 | 0.0020 | 0.0018 |
| bars_since_higher_low_60 | 0.0017 | 0.0015 |

## R4 -- the ADD when P(target after k) >= tau

| variant | direction | k | tau | share_added | mean_dR | iid_t | day_t | ex_top5_dR | mde | add_own_R |
|---|---|---|---|---|---|---|---|---|---|---|
| A | TRAIN->VAL | 10 | 0.6 | 0.133 | 0.008 | 0.895 | -0.354 | -0.054 | 0.025 | 0.060 |
| A | VAL->TRAIN-H2 (swap) | 30 | 0.5 | 0.185 | 0.011 | 0.773 | -0.515 | -0.052 | 0.041 | 0.061 |
| A' | TRAIN->VAL | 10 | 0.6 | 0.133 | 0.008 | 1.145 | 0.322 | -0.046 | 0.019 | 0.057 |
| A' | VAL->TRAIN-H2 (swap) | 30 | 0.5 | 0.185 | 0.000 | 0.041 | -1.018 | -0.057 | 0.032 | 0.003 |

## R5 -- the ADD after +r R, whole-position stop locked (32 paired reads, part=R5 in `1670_reads.csv`)

| r | lock | target | model_gate | direction | share_reached | share_fired | mean_dR | iid_t | day_t | ex_top5_dR | mde | give_back | worst_day_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | breakeven | 2R | False | TRAIN->VAL | 0.454 | 0.454 | -0.021 | -1.345 | -1.417 | -0.081 | 0.043 | 0.381 | -0.634 |
| 1.0 | breakeven | 2R | False | VAL->TRAIN-H2 (swap) | 0.455 | 0.455 | -0.045 | -2.434 | -1.455 | -0.105 | 0.052 | 0.403 | -0.810 |
| 1.0 | breakeven | 2R | True | TRAIN->VAL | 0.454 | 0.097 | 0.007 | 1.010 | 1.187 | -0.040 | 0.020 | 0.346 | -0.300 |
| 1.0 | breakeven | 2R | True | VAL->TRAIN-H2 (swap) | 0.455 | 0.093 | -0.014 | -1.511 | -0.667 | -0.058 | 0.025 | 0.427 | -0.496 |
| 1.0 | breakeven | 3R | False | TRAIN->VAL | 0.454 | 0.454 | -0.003 | -0.109 | -1.098 | -0.164 | 0.073 | 0.477 | -1.093 |
| 1.0 | breakeven | 3R | False | VAL->TRAIN-H2 (swap) | 0.455 | 0.455 | -0.024 | -0.775 | 0.018 | -0.184 | 0.086 | 0.503 | -1.105 |
| 1.0 | breakeven | 3R | True | TRAIN->VAL | 0.454 | 0.097 | 0.026 | 1.901 | 1.095 | -0.095 | 0.038 | 0.484 | -0.590 |
| 1.0 | breakeven | 3R | True | VAL->TRAIN-H2 (swap) | 0.455 | 0.093 | -0.015 | -1.011 | -0.199 | -0.108 | 0.042 | 0.573 | -0.656 |
| 1.0 | plus0.5R | 2R | False | TRAIN->VAL | 0.454 | 0.454 | -0.022 | -1.512 | -1.023 | -0.083 | 0.042 | 0.556 | -0.626 |
| 1.0 | plus0.5R | 2R | False | VAL->TRAIN-H2 (swap) | 0.455 | 0.455 | -0.040 | -2.245 | -1.324 | -0.101 | 0.050 | 0.574 | -0.760 |
| 1.0 | plus0.5R | 2R | True | TRAIN->VAL | 0.454 | 0.097 | -0.008 | -1.051 | -0.376 | -0.055 | 0.020 | 0.536 | -0.296 |
| 1.0 | plus0.5R | 2R | True | VAL->TRAIN-H2 (swap) | 0.455 | 0.093 | -0.022 | -2.517 | -1.376 | -0.068 | 0.025 | 0.610 | -0.368 |
| 1.0 | plus0.5R | 3R | False | TRAIN->VAL | 0.454 | 0.454 | -0.026 | -1.155 | -1.197 | -0.183 | 0.063 | 0.672 | -0.634 |
| 1.0 | plus0.5R | 3R | False | VAL->TRAIN-H2 (swap) | 0.455 | 0.455 | -0.013 | -0.469 | -0.065 | -0.171 | 0.075 | 0.683 | -1.067 |
| 1.0 | plus0.5R | 3R | True | TRAIN->VAL | 0.454 | 0.097 | -0.001 | -0.117 | 0.195 | -0.098 | 0.032 | 0.683 | -0.408 |
| 1.0 | plus0.5R | 3R | True | VAL->TRAIN-H2 (swap) | 0.455 | 0.093 | -0.017 | -1.348 | -0.424 | -0.101 | 0.036 | 0.743 | -0.715 |
| 1.5 | breakeven | 2R | False | TRAIN->VAL | 0.323 | 0.323 | -0.023 | -2.262 | -1.767 | -0.055 | 0.028 | 0.178 | -0.448 |
| 1.5 | breakeven | 2R | False | VAL->TRAIN-H2 (swap) | 0.325 | 0.325 | -0.012 | -0.998 | -0.565 | -0.047 | 0.032 | 0.170 | -0.567 |
| 1.5 | breakeven | 2R | True | TRAIN->VAL | 0.323 | 0.128 | -0.013 | -1.960 | -1.615 | -0.038 | 0.019 | 0.181 | -0.322 |
| 1.5 | breakeven | 2R | True | VAL->TRAIN-H2 (swap) | 0.325 | 0.106 | -0.008 | -1.109 | -0.886 | -0.034 | 0.020 | 0.196 | -0.414 |
| 1.5 | breakeven | 3R | False | TRAIN->VAL | 0.323 | 0.323 | -0.017 | -0.728 | -1.381 | -0.151 | 0.065 | 0.312 | -1.161 |
| 1.5 | breakeven | 3R | False | VAL->TRAIN-H2 (swap) | 0.325 | 0.325 | 0.004 | 0.163 | 0.383 | -0.131 | 0.077 | 0.312 | -1.090 |
| 1.5 | breakeven | 3R | True | TRAIN->VAL | 0.323 | 0.128 | -0.004 | -0.258 | -1.091 | -0.130 | 0.044 | 0.342 | -0.693 |
| 1.5 | breakeven | 3R | True | VAL->TRAIN-H2 (swap) | 0.325 | 0.106 | -0.016 | -0.956 | -0.577 | -0.139 | 0.048 | 0.400 | -0.816 |
| 1.5 | plus0.5R | 2R | False | TRAIN->VAL | 0.323 | 0.323 | -0.027 | -2.722 | -2.159 | -0.059 | 0.028 | 0.270 | -0.517 |
| 1.5 | plus0.5R | 2R | False | VAL->TRAIN-H2 (swap) | 0.325 | 0.325 | -0.008 | -0.711 | 0.076 | -0.045 | 0.032 | 0.248 | -0.492 |
| 1.5 | plus0.5R | 2R | True | TRAIN->VAL | 0.323 | 0.128 | -0.023 | -3.348 | -2.719 | -0.049 | 0.019 | 0.295 | -0.349 |
| 1.5 | plus0.5R | 2R | True | VAL->TRAIN-H2 (swap) | 0.325 | 0.106 | -0.010 | -1.450 | -1.021 | -0.037 | 0.020 | 0.284 | -0.419 |
| 1.5 | plus0.5R | 3R | False | TRAIN->VAL | 0.323 | 0.323 | -0.027 | -1.317 | -1.592 | -0.160 | 0.058 | 0.452 | -0.804 |
| 1.5 | plus0.5R | 3R | False | VAL->TRAIN-H2 (swap) | 0.325 | 0.325 | 0.016 | 0.630 | 0.954 | -0.119 | 0.069 | 0.443 | -0.890 |
| 1.5 | plus0.5R | 3R | True | TRAIN->VAL | 0.323 | 0.128 | -0.023 | -1.646 | -1.674 | -0.143 | 0.039 | 0.505 | -0.512 |
| 1.5 | plus0.5R | 3R | True | VAL->TRAIN-H2 (swap) | 0.325 | 0.106 | -0.008 | -0.505 | 0.021 | -0.118 | 0.042 | 0.520 | -0.586 |

## Verdicts vs the pass bar

Pass bar (PREREG): paired dR>=+0.05R, t>=2.5, ex-top-5%>0 on BOTH scorings, placebo AUC<=0.55.

R2 (cut) reads clearing dR/day-t/ex-top5% individually (both scorings NOT yet cross-checked per-cell): 0/112. R4 (ADD): 0/112. R5 (ADD-after-+R): 0/32.

A pass on one scoring only is not a pass -- see `1670_reads.csv` for both-scoring cross-checks before any cell is proposed for paper.

## Adequacy

Not allowed items honored: tau/k not fit (grid only), no post-map feature/family changes, no exit-type feature in any family, pooled-only numbers avoided (both scorings shown throughout), training-half dR never reported as a result. R4 variant A' and R5 model-gate reuse the ALL-family T_k model at k*=nearest k<=minutes-elapsed, scored out of sample, per the PREREG's explicit rule.
