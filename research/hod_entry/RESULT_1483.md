# RESULT 1,483-1,485 -- catalyst attribution and cash runway

Classification robustness (500-item seed-1483 sample, second independent prompt): agreement=0.810 (bar 0.85) -> catalyst cells VOID = True
News coverage (fills with a cached article stream): 0.100; XBRL runway coverage: 0.575

## Pass-bar table (cell x holdout)
| cell | holdout | n_kept | kept_R | t | ex_top5 | fills/wk | dropped_R | null_pctile | coverage | cache-only | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1483_CATALYST | TRAIN-H2 | 162 | -0.216 | -2.15 | -0.328 | 5.6 | -0.165 | 30.9 | 100.0% | 0.302 | None |
| 1483_CATALYST | VAL | 384 | -0.194 | -2.59 | -0.307 | 15.5 | -0.169 | 36.6 | 100.0% | 0.159 | False |
| 1484_RUNWAY | TRAIN-H2 | 1922 | -0.209 | -5.01 | -0.324 | 38.8 | -0.232 | 4.9 | 55.7% | 0.193 | None |
| 1484_RUNWAY | VAL | 2789 | -0.195 | -4.26 | -0.308 | 45.0 | -0.195 | 9.6 | 58.9% | 0.149 | False |
| 1485_JOINT | TRAIN-H2 | 128 | -0.126 | -1.26 | -0.228 | 4.6 | -0.219 | 61.6 | 55.7% | 0.344 | None |
| 1485_JOINT | VAL | 313 | -0.210 | -2.60 | -0.327 | 12.9 | -0.193 | 28.6 | 58.9% | 0.176 | False |

## Report-only: mean net R and big-day rate by news class x holdout
| cls                | holdout   |    n |   mean_net_R |   big_day_L1 |   big_day_L2 |
|:-------------------|:----------|-----:|-------------:|-------------:|-------------:|
| analyst_action     | TRAIN-H2  |   55 |   -0.0864198 |     0.472727 |     0.272727 |
| analyst_action     | VAL       |  147 |   -0.15336   |     0.408163 |     0.340136 |
| contract_product   | TRAIN-H2  |   20 |    0.316789  |     0.7      |     0.25     |
| contract_product   | VAL       |   59 |   -0.0524119 |     0.559322 |     0.457627 |
| earnings_guidance  | TRAIN-H2  |  117 |   -0.217359  |     0.65812  |     0.230769 |
| earnings_guidance  | VAL       |  288 |   -0.20449   |     0.600694 |     0.315972 |
| fda_clinical       | TRAIN-H2  |   21 |   -0.598647  |     0.52381  |     0.285714 |
| fda_clinical       | VAL       |   16 |   -0.0499792 |     0.6875   |     0.25     |
| financing_dilution | TRAIN-H2  |   17 |   -0.740466  |     0.588235 |     0.294118 |
| financing_dilution | VAL       |   23 |   -0.774197  |     0.608696 |     0.347826 |
| legal_regulatory   | TRAIN-H2  |    7 |    0.0103655 |     1        |     0.285714 |
| legal_regulatory   | VAL       |   15 |   -0.4006    |     0.466667 |     0.4      |
| ma_strategic       | TRAIN-H2  |   10 |    0.0565697 |     0.6      |     0.2      |
| ma_strategic       | VAL       |   23 |   -0.495338  |     0.434783 |     0.173913 |
| no_news            | TRAIN-H2  | 4087 |   -0.164488  |     0.565941 |     0.312209 |
| no_news            | VAL       | 4837 |   -0.163215  |     0.567707 |     0.344015 |
| sector_sympathy    | TRAIN-H2  |   64 |   -0.202247  |     0.609375 |     0.203125 |
| sector_sympathy    | VAL       |  105 |   -0.287009  |     0.447619 |     0.295238 |

## Report-only: mean net R and big-day rate by runway bucket x holdout
| bucket      | holdout   |    n |   mean_net_R |   big_day_L1 |
|:------------|:----------|-----:|-------------:|-------------:|
| 2-4         | TRAIN-H2  |  232 |    -0.281248 |     0.586207 |
| 2-4         | VAL       |  260 |    -0.165747 |     0.646154 |
| <2          | TRAIN-H2  |  297 |    -0.192804 |     0.555556 |
| <2          | VAL       |  200 |    -0.232508 |     0.55     |
| >=4         | TRAIN-H2  |  804 |    -0.198325 |     0.554726 |
| >=4         | VAL       |  988 |    -0.132178 |     0.587045 |
| NaN         | TRAIN-H2  | 1943 |    -0.110738 |     0.666495 |
| NaN         | VAL       | 2261 |    -0.136353 |     0.693056 |
| cf_positive | TRAIN-H2  | 1122 |    -0.210736 |     0.410873 |
| cf_positive | VAL       | 1804 |    -0.228195 |     0.374723 |

## Report-only: overnight close->next-open return by runway bucket
| bucket      |    n |   mean_overnight_ret |
|:------------|-----:|---------------------:|
| 2-4         |  492 |          0.000132698 |
| <2          |  497 |          0.00145199  |
| >=4         | 1792 |         -0.000404417 |
| NaN         | 4202 |          0.00275188  |
| cf_positive | 2926 |          0.00155433  |

## Judge (main session, 2026-09-26 18:35 UTC) — 1,483/1,485 VOID by construction, 1,484 FAIL/VOID

* News coverage, final (the re-fetch completed 17:53 after the disk incident): 1,673 of 9,911 fills (16.9 %) have at
  least one own-name Benzinga/Alpaca article between the previous close and the arm instant; 3,578 masked article rows
  in 18 batches. The PREREG's ≥ 70 % coverage rail cannot be met on this population with this source — 83 % of the
  fills have no own-name story to classify. The catalyst cell is VOID by construction, not a measured null; the A/B
  prompt agreement (0.81 < 0.85) on the partial run is moot. The masked corpus stays on disk (`news_1483/`) for any
  later catalyst study; the fact to carry: this population is mostly news-less by this source.
* Runway (1,484): XBRL coverage 57.5 % (rail 70 %); the buckets are flat within noise (< 2 quarters −0.19 / −0.23 R,
  ≥ 4 quarters −0.20 / −0.13 R, cash-flow positive −0.21 / −0.23 R), kept sets −0.21 / −0.20 R. No dilution-risk signal
  at the break. FAIL on the bar and below the coverage rail.
* Report-only classes on the partial run (n small): financing/dilution −0.74 / −0.77 R (as expected), every other
  class ≤ 0 on VAL. Nothing to pursue. Programme count for these cells: 1,485.
