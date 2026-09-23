# KEPT_DIAG.md — post-hoc diagnostics of cell 1,412 kept book (risk-on tape, all signals)

## TRAIN: kept 1048 trades on 91 days, mean +0.076 R (t +0.64)

- day R: best +42.3, worst -82.6, median -0.51, green days 42%; top 10 % of days = 289% of R
- week R: worst -75.8, P10 -10.9, green weeks 48% of 46
- order within the day: first 4 kept signals +0.061 R (n 290) vs later ones +0.082 R (n 758)
- placebo (random-minute long, same name-day, risk-on minutes, C1): -0.038 R (n 926) → signal − placebo +0.115 R

```
CADENCE BAR  (unknown, ALL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.5 wk  P90 7.4 wk        [fail]   gaps: [2, 1, 1, 3, 1, 1, 7, 11, 4, 6]
C2 bleed     P90 0.17 R     cycles net>0 80% [pass]
C3 reds      P10 -5.59 R  min -75.83 R  MDD 76.16 R   under-water max 59 wk   [fail]
C4 green     48%  null 50%                   [fail]
C5 fills/wk  14.16                             [pass]
C6 tail      C6 not audited
C7 power     cycles 10   bootstrap P90-gap 75% UB 17.2 wk    [fail]
diagnostics  ex-top-5% -46.77 R   capped -103.39 R   top-5 share 158.6%   weekly P&L histogram: [-75.8, -23.5, -13.6, -12.0, -12.0, -9.9, -8.1, -5.9, -4.8, -3.7, -2.4, -2.0, -1.8, -1.5, -1.3, -1.3, -1.2, -1.1, -1.1, -1.1, -0.9, -0.6, -0.3, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 1.7, 2.2, 2.4, 2.5, 3.2, 3.2, 3.7, 3.8, 4.3, 5.3, 6.4, 6.9, 7.8, 18.2, 20.0, 21.7, 25.3, 26.5, 35.5, 64.6]
```

## VAL: kept 839 trades on 52 days, mean +0.245 R (t +1.31)

- day R: best +174.0, worst -23.7, median -1.09, green days 46%; top 10 % of days = 146% of R
- week R: worst -21.2, P10 -19.5, green weeks 43% of 21
- order within the day: first 4 kept signals +0.132 R (n 170) vs later ones +0.274 R (n 669)
- placebo (random-minute long, same name-day, risk-on minutes, C1): +0.143 R (n 739) → signal − placebo +0.102 R

```
CADENCE BAR  (unknown, ALL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 4.0 wk  P90 5.0 wk        [fail]   gaps: [5, 1, 3, 4, 5]
C2 bleed     P90 -1.56 R     cycles net>0 80% [pass]
C3 reds      P10 -2.71 R  min -21.24 R  MDD 34.85 R   under-water max 52 wk   [fail]
C4 green     40%  null 50%                   [fail]
C5 fills/wk  11.34                             [pass]
C6 tail      C6 not audited
C7 power     cycles 5   bootstrap P90-gap 75% UB 24.2 wk    [fail]
diagnostics  ex-top-5% -49.18 R   capped -59.71 R   top-5 share 123.9%   weekly P&L histogram: [-21.2, -21.0, -19.5, -6.8, -5.2, -4.6, -4.3, -3.0, -2.0, -1.9, -1.4, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.5, 9.4, 15.5, 15.7, 41.5, 48.8, 164.8]
```

## TEST: kept 557 trades on 29 days, mean +0.321 R (t +3.13)

- day R: best +58.9, worst -12.9, median +0.53, green days 52%; top 10 % of days = 85% of R
- week R: worst -7.6, P10 -5.4, green weeks 53% of 15
- order within the day: first 4 kept signals -0.030 R (n 107) vs later ones +0.404 R (n 450)
- placebo (random-minute long, same name-day, risk-on minutes, C1): +0.168 R (n 499) → signal − placebo +0.152 R

```
CADENCE BAR  (unknown, ALL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 0.00 R  min 0.00 R  MDD 0.00 R   under-water max 74 wk   [fail]
C4 green     0%  null 0%                   [fail]
C5 fills/wk  0.00                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% 0.00 R   capped 0.00 R   top-5 share 0.0%   weekly P&L histogram: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

