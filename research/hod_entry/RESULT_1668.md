# RESULT 1,668 -- post-entry failure detection

PREREG: `research/hod_entry/PREREG_1668.md` (FROZEN 2026-09-29 18:31 UTC) + amendments 1-3. Population n=5506 (5,506 x1.5%-floored primary book).

## Coverage
* k=3: computable 100.0% (5506/5506)
* k=5: computable 100.0% (5506/5506)
* k=10: computable 100.0% (5506/5506)
* k=15: computable 100.0% (5505/5506)
* level matched (causal join): 100.0%
* S5 (SPY) coverage at k=5: 8.0% -- bars_sip has SPY for a single day only; S5/cS5 are VOID by the rule's own escape clause, not reported further.

## Part A -- every rule x horizon, both halves (paired dR vs the base, whole computable book)
| Rule | k | n TR/VAL | fired% TR/VAL | dR TR/VAL | day-t TR/VAL | ex-top5% TR/VAL | fired-only dR TR/VAL |
|---|---|---|---|---|---|---|---|
| S1 | 3 | 2349/3157 | 0.42/0.44 | 0.021/-0.016 | 1.4/-0.1 | -0.033/-0.069 | 0.051/-0.036 |
| S2 | 3 | 2349/3157 | 0.18/0.18 | 0.010/0.009 | 1.0/1.6 | -0.035/-0.036 | 0.057/0.054 |
| S3 | 3 | 2349/3157 | 0.45/0.47 | 0.032/-0.016 | 1.8/-0.2 | -0.023/-0.071 | 0.072/-0.033 |
| S4 | 3 | 2349/3157 | 0.32/0.31 | 0.020/0.003 | 1.5/1.0 | -0.051/-0.066 | 0.064/0.011 |
| S5 | 3 | 96/138 | 0.00/0.00 | 0.000/0.000 | nan/nan | 0.000/0.000 | nan/nan |
| S6 | 3 | 2349/3157 | 0.00/0.00 | 0.001/0.001 | 1.6/2.0 | 0.000/0.000 | 0.314/0.330 |
| S7 | 3 | 2349/3157 | 0.18/0.17 | -0.010/-0.018 | -0.7/-1.0 | -0.056/-0.059 | -0.054/-0.106 |
| S8 | 3 | 2349/3157 | 0.09/0.08 | 0.010/0.004 | 1.3/1.1 | -0.029/-0.031 | 0.113/0.046 |
| S9 | 3 | 2349/3157 | 0.06/0.07 | 0.005/0.004 | 0.5/1.6 | -0.028/-0.036 | 0.088/0.062 |
| S1 | 5 | 2349/3157 | 0.40/0.42 | 0.015/-0.008 | 1.2/0.0 | -0.038/-0.062 | 0.038/-0.019 |
| S2 | 5 | 2349/3157 | 0.13/0.13 | 0.011/0.004 | 1.2/0.8 | -0.029/-0.035 | 0.085/0.035 |
| S3 | 5 | 2349/3157 | 0.42/0.44 | 0.015/-0.016 | 0.9/-0.5 | -0.040/-0.072 | 0.035/-0.036 |
| S4 | 5 | 2349/3157 | 0.30/0.29 | 0.008/-0.010 | 0.9/-0.2 | -0.063/-0.080 | 0.028/-0.034 |
| S5 | 5 | 182/260 | 0.00/0.00 | 0.000/0.000 | nan/nan | 0.000/0.000 | nan/nan |
| S6 | 5 | 2349/3157 | 0.00/0.01 | -0.001/0.002 | -0.9/1.5 | -0.002/-0.001 | -0.262/0.273 |
| S7 | 5 | 2349/3157 | 0.24/0.23 | 0.001/-0.022 | 0.2/-0.7 | -0.053/-0.072 | 0.003/-0.096 |
| S8 | 5 | 2349/3157 | 0.07/0.06 | 0.004/0.000 | 1.0/0.3 | -0.025/-0.026 | 0.064/0.001 |
| S9 | 5 | 2349/3157 | 0.04/0.05 | 0.008/0.004 | 2.0/1.5 | -0.013/-0.022 | 0.198/0.081 |
| S1 | 10 | 2349/3157 | 0.36/0.37 | 0.004/-0.014 | 1.2/-0.2 | -0.046/-0.066 | 0.011/-0.037 |
| S2 | 10 | 2349/3157 | 0.08/0.07 | 0.007/-0.002 | 1.0/-0.3 | -0.024/-0.029 | 0.085/-0.034 |
| S3 | 10 | 2349/3157 | 0.37/0.39 | 0.000/-0.014 | 0.7/-0.2 | -0.052/-0.067 | 0.000/-0.035 |
| S4 | 10 | 2349/3157 | 0.30/0.28 | 0.002/0.001 | 0.1/0.7 | -0.069/-0.072 | 0.007/0.005 |
| S5 | 10 | 407/552 | 0.00/0.00 | 0.000/0.000 | nan/nan | 0.000/0.000 | nan/nan |
| S6 | 10 | 2349/3157 | 0.01/0.01 | 0.002/-0.002 | 2.5/-0.5 | -0.000/-0.005 | 0.321/-0.186 |
| S7 | 10 | 2349/3157 | 0.30/0.30 | 0.002/-0.028 | 0.7/-0.9 | -0.057/-0.088 | 0.007/-0.094 |
| S8 | 10 | 2349/3157 | 0.04/0.04 | 0.004/-0.002 | 0.9/-0.1 | -0.013/-0.017 | 0.093/-0.051 |
| S9 | 10 | 2349/3157 | 0.03/0.03 | 0.003/-0.001 | 0.7/0.1 | -0.009/-0.014 | 0.130/-0.018 |
| S1 | 15 | 2348/3157 | 0.00/0.00 | -0.007/-0.019 | 0.3/-1.3 | -0.056/-0.068 | -0.024/-0.064 |
| S2 | 15 | 2348/3157 | 0.04/0.04 | 0.007/-0.007 | 2.3/-1.7 | -0.010/-0.021 | 0.164/-0.162 |
| S3 | 15 | 2348/3157 | 0.31/0.31 | -0.006/-0.018 | 0.4/-1.1 | -0.057/-0.069 | -0.020/-0.058 |
| S4 | 15 | 2348/3157 | 0.00/0.00 | 0.002/0.001 | 0.1/0.6 | -0.071/-0.075 | 0.006/0.005 |
| S5 | 15 | 619/843 | 0.00/0.00 | 0.000/0.000 | nan/nan | 0.000/0.000 | nan/nan |
| S6 | 15 | 2348/3157 | 0.00/0.00 | -0.001/-0.002 | 0.0/-0.8 | -0.005/-0.006 | -0.112/-0.114 |
| S7 | 15 | 2348/3157 | 0.30/0.30 | -0.005/-0.016 | 0.7/-0.5 | -0.068/-0.083 | -0.015/-0.053 |
| S8 | 15 | 2348/3157 | 0.02/0.02 | 0.006/-0.004 | 2.9/-0.8 | -0.004/-0.013 | 0.250/-0.181 |
| S9 | 15 | 2348/3157 | 0.00/0.00 | 0.004/0.001 | 0.9/0.3 | -0.006/-0.008 | 0.204/0.063 |

MDE at the median n (2349), fixed book SD: TRAIN-H2 0.078 R, VAL 0.077 R -- full per-cell MDE in `1668_reads.csv`.
**Part A pass bar (dR>=+0.05R, day t>=2.5, ex-top5%>0, BOTH halves): 0/36 rule x k cells pass -- none.**

## Part B -- trained classifier (out of sample)
| k | direction | n train | n score | tau | AUC | mean dR | iid t | day t | ex-top5% dR | share cut | train-half dR (in-sample, caveat) | placebo AUC | placebo dR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | TRAIN->VAL | 2167 | 2897 | 0.42 | 0.625 | -0.0538 | -2.95 | -0.86 | -0.1326 | 0.637 | 0.5069 | 0.530 | -0.0078 |
| 5 | VAL->TRAIN-H2 (swap) | 2897 | 2167 | 0.51 | 0.635 | -0.0123 | -0.65 | 0.07 | -0.0879 | 0.550 | 0.4988 | 0.498 | -0.0103 |
| 10 | TRAIN->VAL | 1942 | 2605 | 0.25 | 0.662 | -0.0503 | -2.56 | -0.11 | -0.1350 | 0.698 | 0.4744 | 0.549 | -0.0307 |
| 10 | VAL->TRAIN-H2 (swap) | 2605 | 1942 | 0.38 | 0.668 | -0.0142 | -0.69 | -0.07 | -0.0943 | 0.598 | 0.4626 | 0.545 | 0.0067 |

**Part B pass bar (both out-of-sample scorings, dR>=+0.05R, day t>=2.5, ex-top5%>0): fail.**

## Permutation importance (VAL-scored model, TRAIN-H2->VAL direction, n_repeats=5, top 8)
| k | feature | importance mean | importance std |
|---|---|---|---|
| 5 | cA1_progvol_5 | 0.0231 | 0.0053 |
| 5 | cS3_ret_5 | 0.0193 | 0.0041 |
| 5 | cS1_dist_level_5 | 0.0061 | 0.0042 |
| 5 | cS7_mae_5 | 0.0041 | 0.0039 |
| 5 | cdl_CDLDOJI_5 | 0.0026 | 0.0010 |
| 5 | cdl_CDLBELTHOLD_5 | 0.0012 | 0.0016 |
| 5 | cdl_CDL3OUTSIDE_5 | 0.0010 | 0.0005 |
| 5 | cdl_CDLENGULFING_5 | 0.0008 | 0.0006 |
| 10 | cS1_dist_level_10 | 0.0379 | 0.0037 |
| 10 | cS3_ret_10 | 0.0326 | 0.0030 |
| 10 | cS7_mae_10 | 0.0220 | 0.0055 |
| 10 | clv_last_10 | 0.0103 | 0.0043 |
| 10 | atr14_pct | 0.0096 | 0.0080 |
| 10 | cA1_progvol_10 | 0.0082 | 0.0034 |
| 10 | talib_nbull_10 | 0.0068 | 0.0029 |
| 10 | body_last_10 | 0.0058 | 0.0021 |

## Amendment 3 -- pattern fire rate on bars fill+1..fill+5, both halves (informational, not pass/fail)
Top 12 by |stop fire rate - target/EOD fire rate|, pooled rank (both halves shown):
| pattern | half | stop rate (n) | target/EOD rate (n) |
|---|---|---|---|
| CDLSHORTLINE | VAL | 0.671 (1545) | 0.610 (1352) |
| CDLDOJI | TRAIN-H2 | 0.648 (1161) | 0.591 (1006) |
| CDLSHORTLINE | TRAIN-H2 | 0.680 (1161) | 0.625 (1006) |
| CDLLONGLEGGEDDOJI | TRAIN-H2 | 0.440 (1161) | 0.390 (1006) |
| CDLADVANCEBLOCK | TRAIN-H2 | 0.047 (1161) | 0.084 (1006) |
| CDLHIKKAKE | TRAIN-H2 | 0.417 (1161) | 0.380 (1006) |
| CDLADVANCEBLOCK | VAL | 0.046 (1545) | 0.083 (1352) |
| CDLBELTHOLD | VAL | 0.682 (1545) | 0.717 (1352) |
| CDLHAMMER | TRAIN-H2 | 0.167 (1161) | 0.132 (1006) |
| CDLRICKSHAWMAN | VAL | 0.234 (1545) | 0.268 (1352) |
| CDLHARAMI | TRAIN-H2 | 0.434 (1161) | 0.402 (1006) |
| CDLHARAMICROSS | TRAIN-H2 | 0.189 (1161) | 0.157 (1006) |

## Verdicts
* Part A: 0/72 (9 rules x 4 k x 2 halves paired reads, 36 rule x k cells) clear the pass bar in both halves.
* Part B: no cell clears the out-of-sample pass bar on both scorings.
* S5 (market) is VOID throughout: bars_sip.db carries SPY for one distinct day only.

## Adequacy review
* Book SD (net_R) used for MDE: TRAIN-H2=1.341, VAL=1.331.
* This is a null-heavy design by construction: S1-S9 fire on a minority of still-open fills at each k, so most reads carry wide MDEs relative to the +0.05R bar; a non-pass here is a claim about a specific mechanically-defined cut rule, not about post-entry information in general.
* No read used a bar after its own cut minute; a stop/target/EOD hit at or before k always preempted the rule (mean dR forced to 0), matching the PREREG precedence rule.
* Base-exit-derived costs (entry/stop/target/EOD) were taken as-is from 1663_features.csv's net_R; only the early-cut leg (entry+cut bps) was computed here, on the same convention.
