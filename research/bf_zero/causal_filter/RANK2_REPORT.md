# RANK2_REPORT — attention-rank filter, cells 1,353-1,354 (PREREG_RANK2.md)


Rows scored: 12135 (TRAIN 7390, VAL 4745). Program count: 1,354.


## rank_active universe: all symbols in bars_sip.db (100% coverage by construction)

Days scanned: 344.


## Cell 1,353 -- rank_sig <= 10 vs > 10

```
             cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1353 rank_sig<=10    kept band     2079       39.2       -0.374   -11.92      -14.7         0.09    902     39.2     -0.318  -6.69    -12.5       0.13
1353 rank_sig<=10    kept meas     1971       37.2       -0.208    -6.55       -7.7         0.26    897     39.0     -0.185  -3.81     -7.2       0.30
1353 rank_sig<=10 dropped band     1101       20.8       -0.371    -8.48       -7.7         0.19    796     34.6     -0.359  -7.22    -12.4       0.09
1353 rank_sig<=10 dropped meas     1048       19.8       -0.202    -4.57       -4.0         0.26    767     33.3     -0.192  -3.85     -6.4       0.17
```

TRAIN halves (meas, obtainable): {'kept_H1': -0.224, 'kept_H1_n': 1600, 'kept_H2': -0.206, 'kept_H2_n': 1854, 'dropped_H1': -0.041, 'dropped_H1_n': 1585, 'dropped_H2': -0.151, 'dropped_H2_n': 1186}


Verdict 1353: {'g1': False, 'halves_ok': False, 'dropped_ok': True, 'c4': False, 'c5': True, 'passed': False}


## Cell 1,354 -- rank_active <= 10 vs > 10

```
                cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1354 rank_active<=10    kept band      401        7.6       -0.980   -20.93       -7.4         0.00    481     20.9     -0.957 -21.63    -20.0        0.0
1354 rank_active<=10    kept meas      336        6.3       -0.836   -16.20       -5.3         0.00    400     17.4     -0.852 -17.24    -14.8        0.0
1354 rank_active<=10 dropped band     1441       27.2       -1.000   -38.79      -27.2         0.02    755     32.8     -0.930 -25.18    -30.5        0.0
1354 rank_active<=10 dropped meas     1348       25.4       -0.827   -30.96      -21.0         0.02    729     31.7     -0.788 -20.13    -25.0        0.0
```

TRAIN halves (meas, obtainable): {'kept_H1': -0.911, 'kept_H1_n': 125, 'kept_H2': -0.787, 'kept_H2_n': 216, 'dropped_H1': -0.749, 'dropped_H1_n': 1144, 'dropped_H2': -0.78, 'dropped_H2_n': 1150}


Verdict 1354: {'g1': False, 'halves_ok': False, 'dropped_ok': True, 'c4': False, 'c5': True, 'passed': False}


## Diagnostics (report-only): <=5, <=20 thresholds

### 1353_le5

```
          cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1353 diag <= 5    kept band     1820       34.3       -0.361   -10.76      -12.4         0.13    810     35.2     -0.344  -6.87    -12.1       0.17
1353 diag <= 5    kept meas     1675       31.6       -0.217    -6.30       -6.9         0.28    790     34.3     -0.232  -4.51     -8.0       0.26
1353 diag <= 5 dropped band     1664       31.4       -0.374   -10.53      -11.8         0.09    940     40.9     -0.373  -8.04    -15.3       0.04
1353 diag <= 5 dropped meas     1585       29.9       -0.210    -5.87       -6.3         0.26    910     39.6     -0.225  -4.70     -8.9       0.09
```

### 1354_le5

```
          cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1354 diag <= 5    kept band      100        1.9       -0.957   -10.03       -1.8         0.09    209      9.1     -1.007 -15.32     -9.1       0.09
1354 diag <= 5    kept meas       80        1.5       -0.815    -7.80       -1.2         0.09    163      7.1     -0.920 -13.14     -6.5       0.09
1354 diag <= 5 dropped band     1576       29.7       -0.985   -39.69      -29.3         0.02    824     35.8     -0.924 -26.29    -33.1       0.00
1354 diag <= 5 dropped meas     1481       27.9       -0.826   -32.59      -23.1         0.02    803     34.9     -0.782 -20.87    -27.3       0.00
```

### 1353_le20

```
           cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1353 diag <= 20    kept band     2248       42.4       -0.351   -11.53      -14.9         0.13    968     42.1     -0.350  -7.68    -14.7       0.00
1353 diag <= 20    kept meas     2173       41.0       -0.204    -6.68       -8.3         0.25    960     41.7     -0.213  -4.57     -8.9       0.17
1353 diag <= 20 dropped band      417        7.9       -0.204    -2.82       -1.6         0.23    605     26.3     -0.407  -7.15    -10.7       0.00
1353 diag <= 20 dropped meas      394        7.4       -0.073    -1.01       -0.5         0.38    583     25.3     -0.262  -4.64     -6.6       0.17
```

### 1354_le20

```
           cell    side  arm  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_t  TRAIN_wkR  TRAIN_green  VAL_n  VAL_tpw  VAL_meanR  VAL_t  VAL_wkR  VAL_green
1354 diag <= 20    kept band     1027       19.4       -1.008   -33.73      -19.5         0.04    691     30.0     -0.962 -25.77    -28.9        0.0
1354 diag <= 20    kept meas      915       17.3       -0.834   -26.70      -14.4         0.02    627     27.3     -0.864 -21.82    -23.5        0.0
1354 diag <= 20 dropped band      993       18.7       -0.994   -32.06      -18.6         0.00    602     26.2     -0.913 -21.28    -23.9        0.0
1354 diag <= 20 dropped meas      920       17.4       -0.822   -25.00      -14.3         0.00    576     25.0     -0.739 -16.14    -18.5        0.0
```

## Monotone decile table (TRAIN, meas arm, obtainable), mean net_meas by decile

### rank_sig deciles (D1 = lowest rank number = strongest)

```
           mean  count
rank_sig              
D1       -0.270    619
D2       -0.176    617
D3       -0.202    621
D4       -0.196    616
D5       -0.220    617
D6       -0.227    619
D7       -0.200    613
D8       -0.057    620
D9       -0.004    615
D10      -0.030    622
```

### rank_active deciles

```
              mean  count
rank_active              
D1          -0.831    263
D2          -0.947    260
D3          -0.794    262
D4          -0.722    262
D5          -0.901    262
D6          -0.815    261
D7          -0.805    260
D8          -0.743    261
D9          -0.602    259
D10         -0.574    264
```

## Forward check

`[HOD DRY] WOULD BUY` lines since 2026-09-18 (60s bound exceeded for --since 2026-09-14, PREREG fallback date used). n=9 (rank<=10: 9, rank>10: 0).

Mean nominal book R: rank<=10 = 1.558, rank>10 = None (R taken from the WOULD BUY line itself; no EOD spec re-simulation run for this n).

## Verdict (main session, 2026-09-21) — cells 1,353–1,354 FAIL; the attention-rank frame is refuted on HOD breaks
Rank among the day's fired breakouts (≤ 10 vs > 10): kept −0.208 / −0.185 R (n 1,971 / 897) vs dropped −0.202 /
−0.192 on TRAIN / VAL — no separation, both TRAIN halves negative. Rank among today's active setup universe:
kept −0.84 / −0.85 vs dropped −0.83 / −0.79 — no separation either. Power is adequate (SE ≈ 0.02–0.03 R on the
signal rank; a ±0.06 R difference would show). Caveat: the active-universe rank was computable only on the
signals whose symbol had bars at the signal minute (n 1,684 / 1,129), a coverage-selected subset with a much
worse base rate (−0.8 R) than the full population (−0.2 R); within it rank still carries nothing.
Being the day's top mover does not change what an HOD break does next, in either definition. Programme count 1,354.
