# REBUILD_1552 -- full independent rebuild (all 7,754 CIKs)

Independent reimplementation from PREREG_1552.md + Amendment 1 prose only; raw submissions gzip cache parsed directly (not events_raw.csv). TEST split never computed. See rebuild_1552_full.py docstring for judgment calls.

## Class counts (Amendment 1 applied)

| class | raw candidates | after amendment (serial-issuer cap / 424B2 excl) |
|---|---|---|
| OFFERING | 44368 | 29431 |
| SHELF | 8477 | 8249 |
| REVERSE_SPLIT | 7727 | (unchanged) |
| AUDITOR | 1921 | (unchanged) |
| NON_RELIANCE | 613 | (unchanged) |
| LATE_FILING | 4668 | (unchanged) |
| OFFICER_EXIT | 40489 | (unchanged) |
| CONTRACT | 15075 | (unchanged) |
| ACTIVIST | 4096 | (unchanged) |
| BUYBACK_OR_INSIDER | NOT_COMPUTABLE | NOT_COMPUTABLE |

Report-only (not priced): 10-K=38080, BARE_8.01=0, EARNINGS_2.02=116014, REGFD_7.01=56019, 10-Q=112555

## Per cell x holdout (named leg per TRAIN, VAL never used to pick the leg)

| cell | class | named leg | split | n | ev/wk | mean bps | day-clust t | ex-top5% bps | winner-capped bps | share-dir | placebo margin bps | null pctile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1552 | OFFERING | E1 | TRAIN | 10451 | 50.66 | -0.9 | -0.15 | -57.7 | -3.3 | 0.49 | -4.3 |  |
| 1552 | OFFERING | E1 | VAL | 3275 | 42.38 | 3.7 | 0.36 | -56.7 | -2.7 | 0.48 | -0.6 | 98.1 |
| 1553 | SHELF | E1 | TRAIN | 2850 | 13.83 | 4.0 | 0.43 | -54.0 | 1.9 | 0.50 | -4.3 |  |
| 1553 | SHELF | E1 | VAL | 1096 | 14.18 | -5.9 | -0.41 | -61.1 | -8.0 | 0.47 | -6.8 | 63.4 |
| 1554 | REVERSE_SPLIT | E5 | TRAIN | 3018 | 14.64 | 2.9 | 0.15 | -105.8 | -16.7 | 0.48 | 52.7 |  |
| 1554 | REVERSE_SPLIT | E5 | VAL | 1381 | 17.87 | 5.6 | 0.19 | -116.3 | -24.6 | 0.47 | nan | 92.7 |
| 1555 | AUDITOR | E5 | TRAIN | 323 | 1.58 | 72.6 | 1.01 | -103.9 | 0.3 | 0.49 | 58.8 |  |
| 1555 | AUDITOR | E5 | VAL | 153 | 2.03 | 18.6 | 0.27 | -128.8 | -31.2 | 0.47 | nan | 63.6 |
| 1556 | NON_RELIANCE | E1 | TRAIN | 137 | 0.69 | 4.9 | 0.12 | -52.1 | 4.9 | 0.47 | -8.2 |  |
| 1556 | NON_RELIANCE | E1 | VAL | 54 | 0.84 | -2.4 | -0.04 | -56.7 | -2.4 | 0.56 | 3.4 | 67.2 |
| 1557 | LATE_FILING | E5 | TRAIN | 373 | 1.86 | 112.2 | 1.59 | -44.1 | 60.0 | 0.54 | 54.3 |  |
| 1557 | LATE_FILING | E5 | VAL | 180 | 2.40 | 179.5 | 1.63 | 12.9 | 111.5 | 0.59 | 173.3 | 100.0 |
| 1558 | OFFICER_EXIT | E1 | TRAIN | 19333 | 93.72 | 2.5 | 0.55 | -41.9 | 1.7 | 0.50 | -2.7 |  |
| 1558 | OFFICER_EXIT | E1 | VAL | 7486 | 96.86 | -0.3 | -0.06 | -37.5 | -0.5 | 0.50 | 0.3 | 100.0 |
| 1559 | CONTRACT | E5 | TRAIN | 6000 | 29.09 | 25.0 | 1.08 | -126.6 | -26.8 | 0.50 | -22.4 |  |
| 1559 | CONTRACT | E5 | VAL | 1940 | 25.10 | 13.2 | 0.45 | -136.0 | -36.2 | 0.46 | nan | 76.0 |
| 1560 | ACTIVIST | E1 | TRAIN | 1253 | 6.08 | -32.7 | -1.61 | -102.2 | -37.9 | 0.46 | -27.6 |  |
| 1560 | ACTIVIST | E1 | VAL | 447 | 5.81 | 55.0 | 0.85 | -67.7 | -2.1 | 0.48 | 51.9 | 100.0 |
