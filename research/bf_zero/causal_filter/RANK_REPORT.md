# RANK_REPORT — attention-rank filter, cells 1,351-1,352 (PREREG_RANK.md)


Rows scored: 12135 (TRAIN 7390, VAL 4745). Program count: 1,352.


## rank_mkt VOID share

Days with bar coverage < 80% (vs data/cache.db daily_bars that day): 344/344 days = 100.0% of days, 100.0% of signals VOID.


## Cell 1,351 -- rank_cand <= 10 vs > 10

```
              cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1351 rank_cand<=10    kept band     22.0        0.4       -0.269    -0.85       -0.1         0.13    NaN      NaN        NaN    NaN      NaN        NaN
1351 rank_cand<=10    kept meas      NaN        NaN          NaN      NaN        NaN          NaN    NaN      NaN        NaN    NaN      NaN        NaN
1351 rank_cand<=10 dropped band   2279.0       43.0       -0.356   -11.80      -15.3         0.11 1032.0     44.9     -0.312  -7.04    -14.0       0.13
1351 rank_cand<=10 dropped meas   2207.0       41.6       -0.216    -7.14       -9.0         0.21 1029.0     44.7     -0.185  -4.12     -8.3       0.17
```

TRAIN halves (meas, obtainable): {'kept_H1': -0.562, 'kept_H1_n': 8, 'kept_H2': 0.394, 'kept_H2_n': 8, 'dropped_H1': -0.132, 'dropped_H1_n': 3177, 'dropped_H2': -0.186, 'dropped_H2_n': 3032}


Verdict 1351: {'g1': False, 'halves_ok': False, 'dropped_ok': True, 'c4': False, 'c5': False, 'passed': False}


## Cell 1,352 -- rank_mkt <= 10 vs > 10 (VOID days excluded)

```
             cell    side  arm
1352 rank_mkt<=10    kept band
1352 rank_mkt<=10    kept meas
1352 rank_mkt<=10 dropped band
1352 rank_mkt<=10 dropped meas
```

TRAIN halves (meas, obtainable): {'kept_H1': nan, 'kept_H1_n': 0, 'kept_H2': nan, 'kept_H2_n': 0, 'dropped_H1': nan, 'dropped_H1_n': 0, 'dropped_H2': nan, 'dropped_H2_n': 0}


Verdict 1352: {'g1': False, 'halves_ok': False, 'dropped_ok': False, 'c4': False, 'c5': False, 'passed': False}


## Diagnostics (report-only): <=5, <=20 thresholds

### 1351_le5

```
          cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1351 diag <= 5    kept band      NaN        NaN          NaN      NaN        NaN          NaN    NaN      NaN        NaN    NaN      NaN        NaN
1351 diag <= 5    kept meas      NaN        NaN          NaN      NaN        NaN          NaN    NaN      NaN        NaN    NaN      NaN        NaN
1351 diag <= 5 dropped band   2289.0       43.2       -0.352   -11.68      -15.2         0.13 1032.0     44.9     -0.299  -6.74    -13.4       0.13
1351 diag <= 5 dropped meas   2214.0       41.8       -0.208    -6.90       -8.7         0.23 1027.0     44.7     -0.173  -3.84     -7.7       0.17
```

### 1352_le5

```
          cell    side  arm
1352 diag <= 5    kept band
1352 diag <= 5    kept meas
1352 diag <= 5 dropped band
1352 diag <= 5 dropped meas
```

### 1351_le20

```
           cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1351 diag <= 20    kept band       65        1.2       -0.286    -1.59       -0.4         0.26     55      2.4     -0.529  -3.02     -1.3       0.17
1351 diag <= 20    kept meas       54        1.0       -0.096    -0.47       -0.1         0.28     47      2.0     -0.462  -2.37     -0.9       0.22
1351 diag <= 20 dropped band     2268       42.8       -0.357   -11.79      -15.3         0.09   1031     44.8     -0.317  -7.16    -14.2       0.17
1351 diag <= 20 dropped meas     2203       41.6       -0.212    -7.03       -8.8         0.21   1026     44.6     -0.189  -4.22     -8.4       0.22
```

### 1352_le20

```
           cell    side  arm
1352 diag <= 20    kept band
1352 diag <= 20    kept meas
1352 diag <= 20 dropped band
1352 diag <= 20 dropped meas
```

## Monotone decile table (TRAIN, meas arm, obtainable), mean net_meas by decile

### rank_cand deciles (D1 = lowest rank number = strongest)

```
            mean  count
rank_cand              
D1        -0.137    621
D2        -0.302    620
D3        -0.232    619
D4        -0.247    616
D5        -0.076    621
D6        -0.235    614
D7        -0.204    616
D8        -0.112    613
D9         0.109    617
D10       -0.144    622
```

### rank_mkt deciles

```
Empty DataFrame
Columns: []
Index: []
```

## Forward check

`[HOD DRY] WOULD BUY` lines since 2026-09-18 (60s bound exceeded for --since 2026-09-14, PREREG fallback date used). n=9 (rank<=10: 9, rank>10: 0).

Mean nominal book R: rank<=10 = 1.558, rank>10 = None (R taken from the WOULD BUY line itself; no EOD spec re-simulation run for this n).
