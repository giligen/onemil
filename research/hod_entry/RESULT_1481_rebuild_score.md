Fills file: `research/hod_entry/rebuild_1481_fills.csv` -- 9911 rows (splits present: TRAIN, VAL only; TEST was not read, per the sealed-test guard).

1,486 join (day+symbol) to `model_1478_L3_predictions.csv`, column `hgb_prob_L3`: 9911/9911 matched (100.0%).

## Per cell x holdout

| cell | holdout | n_base | n_retest | fill_share | mean_R | t | ex_top5_R | fills/wk | med_r_pct_price | med_dip_bps | med_delay_min | dR_mean | dR_t | dR_ex5 | null_pctile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1481 | TRAIN-H2 | 4398 | 3957 | 90.0% | -0.113 | -2.88 | -0.224 | 50.7 | 1.5% | 53.6 | 0.9 | 0.207 | 23.74 | 0.132 | 100.0 |
| 1486 | TRAIN-H2 | 1466 | 1265 | 86.3% | 0.083 | 1.17 | -0.017 | 31.7 | 2.2% | 62.4 | 0.9 | 0.165 | 14.89 | 0.110 | 85.9 |
| 1481 | VAL | 5513 | 5016 | 91.0% | -0.089 | -2.21 | -0.199 | 50.5 | 1.5% | 58.0 | 0.8 | 0.223 | 35.52 | 0.152 | 100.0 |
| 1486 | VAL | 2116 | 1862 | 88.0% | -0.090 | -1.57 | -0.200 | 45.7 | 2.2% | 63.7 | 0.9 | 0.197 | 20.85 | 0.133 | 100.0 |

## Never-retest cohort (base fills with status == `no_retest`; the runners)

| holdout | n | mean base_net_R |
|---|---|---|
| TRAIN-H2 | 168 | 0.599 |
| VAL | 183 | 0.579 |

Excluded as indeterminate (`status == bar_tick_disagree`, not counted as retest or never-retest): TRAIN-H2 273, VAL 314.

## Pass bar verdicts (PREREG_1481.md, VAL-gated)

**Cell 1481: FAIL**
- FAIL: VAL mean_R -0.089 < +0.150
- FAIL: VAL t -2.21 < 2.5
- FAIL: VAL ex-top-5% -0.199 <= 0
- FAIL: TRAIN-H2 same-sign/t -0.113 (t=-2.88) fails same-sign t>=1

**Cell 1486: FAIL**
- FAIL: VAL mean_R -0.090 < +0.150
- FAIL: VAL t -1.57 < 2.5
- FAIL: VAL ex-top-5% -0.200 <= 0
- FAIL: TRAIN-H2 same-sign/t 0.083 (t=1.17) fails same-sign t>=1

## R-vs-spread rail (median R' must be >= 0.5% of price)

- no cell/holdout below the 0.5% rail

