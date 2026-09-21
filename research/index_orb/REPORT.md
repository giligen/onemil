# Index ORB walk — REPORT (research/index_orb/PREREG.md)

Generated 2026-09-21T18:38:02.948895+00:00

## Data quality

QQQ: 27 thin session days (<380 bars), excluded from walk
  sample: 2016-02-22, 2016-11-25, 2017-07-03, 2017-11-24, 2018-05-02, 2018-05-03, 2018-07-03, 2018-11-23, 2018-12-24, 2019-07-03, 2019-11-29, 2019-12-24, 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18, 2020-11-27, 2020-12-24, 2021-11-26, 2022-11-25, 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-12-24
SPY: 24 thin session days (<380 bars), excluded from walk
  sample: 2016-11-25, 2017-07-03, 2017-11-24, 2018-07-03, 2018-11-23, 2019-07-03, 2019-08-12, 2019-11-29, 2019-12-24, 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18, 2020-11-27, 2020-12-24, 2022-11-25, 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24


## Cell 1329: QQQ W=5 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1178 | 2.51 | 0.112 | 0.040 | -0.103 | 1.770 | 0.78 | 33.5 | -0.222 | 0.019 | 63.4 | 36.6 | 0.9 | 32.58 | 10 |
| TRAIN half 2016-2019 | 541 | 2.59 | 0.081 | -0.007 | -0.182 | 1.753 | -0.09 | 33.3 | -0.277 | -0.024 | 63.8 | 36.2 | 1.7 | 32.58 | 10 |
| TRAIN half 2020-2024 | 637 | 2.44 | 0.139 | 0.080 | -0.036 | 1.785 | 1.14 | 33.8 | -0.181 | 0.057 | 63.1 | 36.9 | 0.3 | 20.88 | 7 |
| VAL 2025-2026/05 | 168 | 2.28 | 0.069 | 0.005 | -0.123 | 1.738 | 0.03 | 36.3 | -0.258 | -0.040 | 62.5 | 37.5 | 0.0 | 11.95 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=133 mean=-0.066 total=-8.80
- 2017: n=128 mean=-0.100 total=-12.81
- 2018: n=142 mean=0.031 total=4.43
- 2019: n=138 mean=0.098 total=13.51
- 2020: n=126 mean=-0.078 total=-9.78
- 2021: n=133 mean=0.091 total=12.07
- 2022: n=114 mean=0.161 total=18.31
- 2023: n=141 mean=0.203 total=28.57
- 2024: n=123 mean=0.017 total=2.06

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.04 R  min -3.12 R  MDD 7.30 R   under-water max 13 wk   [fail]
C4 green     38%  null 51%                   [fail]
C5 fills/wk  2.32                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -3.65 R   capped 0.33 R   top-5 share 1216.1%   weekly P&L histogram: [-3.1, -2.2, -2.1, -1.1, -1.1, -1.1, -1.0, -1.0, -0.9, -0.7, -0.2, -0.2, -0.1, 0.0, 0.2, 0.2, 1.7, 1.7, 2.0, 2.1, 3.2, 4.0]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=True 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.268 R

## Cell 1330: QQQ W=5 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1061 | 2.26 | 0.022 | -0.047 | -0.186 | 1.886 | -0.81 | 27.4 | -0.354 | -0.091 | 67.9 | 32.1 | 0.5 | 95.91 | 11 |
| TRAIN half 2016-2019 | 453 | 2.17 | -0.020 | -0.106 | -0.279 | 1.830 | -1.23 | 26.0 | -0.414 | -0.148 | 66.9 | 33.1 | 0.7 | 73.15 | 11 |
| TRAIN half 2020-2024 | 608 | 2.33 | 0.054 | -0.003 | -0.117 | 1.927 | -0.04 | 28.5 | -0.310 | -0.049 | 68.6 | 31.4 | 0.3 | 48.61 | 9 |
| VAL 2025-2026/05 | 182 | 2.47 | 0.098 | 0.033 | -0.096 | 1.833 | 0.24 | 34.6 | -0.290 | -0.010 | 62.1 | 37.9 | 0.5 | 24.87 | 13 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=117 mean=-0.159 total=-18.66
- 2017: n=121 mean=-0.344 total=-41.61
- 2018: n=104 mean=0.147 total=15.31
- 2019: n=111 mean=-0.027 total=-3.03
- 2020: n=121 mean=-0.256 total=-30.94
- 2021: n=118 mean=0.097 total=11.40
- 2022: n=136 mean=0.181 total=24.60
- 2023: n=107 mean=-0.212 total=-22.64
- 2024: n=126 mean=0.125 total=15.71

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.19 R  min -3.15 R  MDD 10.86 R   under-water max 8 wk   [fail]
C4 green     29%  null 50%                   [fail]
C5 fills/wk  2.32                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -13.27 R   capped -8.27 R   top-5 share -82.1%   weekly P&L histogram: [-3.2, -2.6, -2.2, -2.2, -2.1, -2.1, -1.6, -1.5, -1.5, -1.1, -1.1, -1.1, 0.0, 0.1, 0.2, 0.4, 0.4, 0.9, 2.2, 2.3, 2.3, 6.0]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.272 R

## Cell 1331: QQQ W=5 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2239 | 4.77 | 0.070 | -0.001 | -0.142 | 1.826 | -0.03 | 30.6 | -0.283 | -0.033 | 65.5 | 34.5 | 0.7 | 109.11 | 8 |
| TRAIN half 2016-2019 | 994 | 4.77 | 0.035 | -0.052 | -0.226 | 1.788 | -0.92 | 30.0 | -0.335 | -0.081 | 65.2 | 34.8 | 1.2 | 95.06 | 6 |
| TRAIN half 2020-2024 | 1245 | 4.77 | 0.097 | 0.040 | -0.075 | 1.856 | 0.75 | 31.2 | -0.245 | 0.005 | 65.8 | 34.2 | 0.3 | 52.86 | 8 |
| VAL 2025-2026/05 | 350 | 4.76 | 0.084 | 0.020 | -0.109 | 1.786 | 0.20 | 35.4 | -0.269 | -0.025 | 62.3 | 37.7 | 0.3 | 31.70 | 5 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=250 mean=-0.110 total=-27.46
- 2017: n=249 mean=-0.219 total=-54.42
- 2018: n=246 mean=0.080 total=19.74
- 2019: n=249 mean=0.042 total=10.48
- 2020: n=247 mean=-0.165 total=-40.72
- 2021: n=251 mean=0.094 total=23.47
- 2022: n=250 mean=0.172 total=42.92
- 2023: n=248 mean=0.024 total=5.93
- 2024: n=249 mean=0.071 total=17.77

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.0 wk  P90 2.0 wk        [pass]   gaps: [2]
C2 bleed     P90 0.61 R     cycles net>0 100% [pass]
C3 reds      P10 -3.05 R  min -3.65 R  MDD 14.88 R   under-water max 13 wk   [fail]
C4 green     35%  null 51%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 1   bootstrap P90-gap 75% UB 9.0 wk    [fail]
diagnostics  ex-top-5% -15.01 R   capped -10.51 R   top-5 share -115.6%   weekly P&L histogram: [-3.6, -3.2, -3.1, -2.9, -2.7, -2.7, -2.6, -2.1, -1.9, -1.7, -1.2, -1.1, -0.8, -0.1, 0.1, 0.6, 0.7, 1.5, 2.0, 4.0, 5.5, 8.0]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=True 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.191 R

## Cell 1332: QQQ W=15 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1153 | 2.46 | 0.102 | 0.052 | -0.047 | 1.322 | 1.34 | 43.5 | -0.123 | 0.050 | 47.2 | 52.8 | 0.3 | 23.65 | 7 |
| TRAIN half 2016-2019 | 502 | 2.41 | 0.072 | 0.012 | -0.108 | 1.276 | 0.21 | 42.6 | -0.165 | 0.012 | 45.2 | 54.8 | 0.8 | 23.65 | 7 |
| TRAIN half 2020-2024 | 651 | 2.50 | 0.124 | 0.083 | -0.000 | 1.357 | 1.56 | 44.1 | -0.095 | 0.079 | 48.7 | 51.3 | 0.0 | 16.56 | 7 |
| VAL 2025-2026/05 | 178 | 2.42 | 0.113 | 0.068 | -0.023 | 1.177 | 0.77 | 49.4 | -0.079 | 0.068 | 42.1 | 57.9 | 0.0 | 10.53 | 5 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=128 mean=-0.024 total=-3.08
- 2017: n=126 mean=-0.004 total=-0.53
- 2018: n=123 mean=0.070 total=8.63
- 2019: n=125 mean=0.009 total=1.11
- 2020: n=133 mean=0.057 total=7.56
- 2021: n=142 mean=0.132 total=18.75
- 2022: n=120 mean=0.079 total=9.43
- 2023: n=140 mean=0.078 total=10.91
- 2024: n=116 mean=0.062 total=7.22

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -1.98 R  min -2.75 R  MDD 6.42 R   under-water max 6 wk   [pass]
C4 green     60%  null 50%                   [pass]
C5 fills/wk  2.45                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% 0.77 R   capped 5.77 R   top-5 share 89.2%   weekly P&L histogram: [-2.8, -2.3, -2.1, -1.0, -1.0, -1.0, -0.3, -0.1, -0.1, 0.2, 0.2, 0.2, 0.4, 0.6, 0.6, 0.7, 1.3, 1.6, 1.7, 1.9, 2.0, 6.4]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=True 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.176 R

## Cell 1333: QQQ W=15 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1079 | 2.30 | 0.002 | -0.045 | -0.140 | 1.487 | -0.99 | 35.7 | -0.275 | -0.066 | 52.0 | 48.0 | 0.5 | 73.49 | 10 |
| TRAIN half 2016-2019 | 488 | 2.34 | -0.031 | -0.087 | -0.201 | 1.546 | -1.25 | 33.0 | -0.342 | -0.124 | 51.0 | 49.0 | 1.0 | 52.68 | 10 |
| TRAIN half 2020-2024 | 591 | 2.27 | 0.030 | -0.010 | -0.089 | 1.437 | -0.17 | 37.9 | -0.226 | -0.019 | 52.8 | 47.2 | 0.0 | 29.05 | 9 |
| VAL 2025-2026/05 | 172 | 2.34 | 0.068 | 0.023 | -0.067 | 1.304 | 0.23 | 42.4 | -0.163 | 0.023 | 47.1 | 52.9 | 0.0 | 19.30 | 8 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=121 mean=-0.063 total=-7.66
- 2017: n=123 mean=-0.178 total=-21.84
- 2018: n=123 mean=-0.045 total=-5.58
- 2019: n=121 mean=-0.063 total=-7.62
- 2020: n=114 mean=-0.176 total=-20.03
- 2021: n=109 mean=0.081 total=8.82
- 2022: n=129 mean=0.135 total=17.43
- 2023: n=107 mean=-0.095 total=-10.14
- 2024: n=132 mean=-0.014 total=-1.91

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.12 R  min -3.34 R  MDD 15.84 R   under-water max 8 wk   [fail]
C4 green     40%  null 50%                   [fail]
C5 fills/wk  2.18                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -9.55 R   capped -6.36 R   top-5 share -50.0%   weekly P&L histogram: [-3.3, -2.9, -2.1, -2.1, -1.8, -1.6, -1.5, -1.4, -1.1, -1.1, -1.0, -1.0, -0.4, 0.0, 0.6, 1.1, 1.3, 2.0, 2.1, 2.3, 2.5, 3.2]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.199 R

## Cell 1334: QQQ W=15 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2232 | 4.75 | 0.054 | 0.005 | -0.092 | 1.405 | 0.17 | 39.7 | -0.198 | -0.006 | 49.5 | 50.5 | 0.4 | 60.38 | 7 |
| TRAIN half 2016-2019 | 990 | 4.75 | 0.021 | -0.037 | -0.153 | 1.416 | -0.82 | 37.9 | -0.251 | -0.055 | 48.1 | 51.9 | 0.9 | 47.90 | 7 |
| TRAIN half 2020-2024 | 1242 | 4.76 | 0.079 | 0.039 | -0.042 | 1.395 | 0.98 | 41.1 | -0.158 | 0.032 | 50.6 | 49.4 | 0.0 | 25.34 | 5 |
| VAL 2025-2026/05 | 350 | 4.76 | 0.091 | 0.046 | -0.044 | 1.239 | 0.69 | 46.0 | -0.122 | 0.046 | 44.6 | 55.4 | 0.0 | 15.09 | 4 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=249 mean=-0.043 total=-10.74
- 2017: n=249 mean=-0.090 total=-22.37
- 2018: n=246 mean=0.012 total=3.05
- 2019: n=246 mean=-0.026 total=-6.51
- 2020: n=247 mean=-0.050 total=-12.47
- 2021: n=251 mean=0.110 total=27.57
- 2022: n=249 mean=0.108 total=26.86
- 2023: n=247 mean=0.003 total=0.77
- 2024: n=248 mean=0.021 total=5.31

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -3.06 R  min -3.97 R  MDD 12.58 R   under-water max 13 wk   [fail]
C4 green     43%  null 50%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -3.47 R   capped 0.76 R   top-5 share 556.8%   weekly P&L histogram: [-4.0, -3.9, -3.1, -2.7, -2.2, -1.5, -1.4, -1.0, -0.9, -0.9, -0.8, -0.8, 0.0, 0.8, 1.0, 1.6, 1.7, 2.6, 3.7, 3.9, 4.2, 4.2]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.132 R

## Cell 1335: QQQ W=30 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1179 | 2.51 | 0.051 | 0.011 | -0.068 | 1.028 | 0.38 | 47.3 | -0.116 | 0.011 | 35.5 | 64.5 | 0.3 | 41.99 | 7 |
| TRAIN half 2016-2019 | 525 | 2.52 | 0.002 | -0.046 | -0.141 | 0.992 | -1.05 | 43.4 | -0.173 | -0.046 | 34.5 | 65.5 | 0.6 | 36.83 | 7 |
| TRAIN half 2020-2024 | 654 | 2.51 | 0.090 | 0.057 | -0.009 | 1.055 | 1.38 | 50.5 | -0.073 | 0.057 | 36.4 | 63.6 | 0.0 | 12.18 | 6 |
| VAL 2025-2026/05 | 192 | 2.61 | 0.058 | 0.022 | -0.050 | 0.949 | 0.32 | 54.2 | -0.098 | 0.022 | 32.3 | 67.7 | 0.0 | 11.44 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=132 mean=-0.051 total=-6.69
- 2017: n=138 mean=-0.007 total=-0.93
- 2018: n=134 mean=-0.118 total=-15.86
- 2019: n=121 mean=-0.003 total=-0.41
- 2020: n=145 mean=-0.014 total=-2.01
- 2021: n=134 mean=0.089 total=11.90
- 2022: n=121 mean=0.007 total=0.87
- 2023: n=127 mean=0.087 total=11.08
- 2024: n=127 mean=0.121 total=15.42

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -0.86 R  min -1.68 R  MDD 3.26 R   under-water max 9 wk   [fail]
C4 green     45%  null 49%                   [fail]
C5 fills/wk  2.64                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 10.3 wk    [fail]
diagnostics  ex-top-5% -1.47 R   capped 3.53 R   top-5 share 128.5%   weekly P&L histogram: [-1.7, -1.0, -0.9, -0.8, -0.7, -0.5, -0.4, -0.2, -0.2, 0.0, -0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.5, 0.7, 0.8, 0.9, 1.5, 6.6]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.137 R

## Cell 1336: QQQ W=30 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1024 | 2.18 | 0.021 | -0.016 | -0.092 | 1.250 | -0.42 | 40.5 | -0.208 | -0.027 | 37.9 | 62.1 | 0.3 | 50.95 | 9 |
| TRAIN half 2016-2019 | 452 | 2.17 | -0.039 | -0.084 | -0.174 | 1.288 | -1.38 | 35.8 | -0.287 | -0.102 | 37.8 | 62.2 | 0.7 | 42.22 | 9 |
| TRAIN half 2020-2024 | 572 | 2.19 | 0.069 | 0.037 | -0.027 | 1.218 | 0.73 | 44.2 | -0.145 | 0.033 | 37.9 | 62.1 | 0.0 | 23.94 | 6 |
| VAL 2025-2026/05 | 156 | 2.12 | 0.068 | 0.034 | -0.035 | 1.149 | 0.37 | 45.5 | -0.130 | 0.023 | 32.7 | 67.3 | 0.0 | 11.26 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=115 mean=-0.061 total=-7.00
- 2017: n=109 mean=-0.176 total=-19.16
- 2018: n=110 mean=0.027 total=3.00
- 2019: n=118 mean=-0.125 total=-14.72
- 2020: n=99 mean=-0.067 total=-6.61
- 2021: n=112 mean=0.029 total=3.29
- 2022: n=125 mean=0.150 total=18.74
- 2023: n=115 mean=-0.044 total=-5.01
- 2024: n=121 mean=0.089 total=10.80

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -1.59 R  min -3.72 R  MDD 9.27 R   under-water max 8 wk   [fail]
C4 green     47%  null 50%                   [fail]
C5 fills/wk  2.00                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -5.38 R   capped -2.62 R   top-5 share -105.4%   weekly P&L histogram: [-3.7, -2.0, -1.6, -1.5, -1.4, -1.2, -1.0, -1.0, -0.6, -0.4, -0.2, 0.0, 0.0, 0.3, 0.6, 0.8, 1.1, 1.3, 1.3, 1.6, 2.1, 2.8]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.184 R

## Cell 1337: QQQ W=30 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2203 | 4.69 | 0.037 | -0.002 | -0.079 | 1.137 | -0.06 | 44.2 | -0.160 | -0.006 | 36.6 | 63.4 | 0.3 | 79.65 | 9 |
| TRAIN half 2016-2019 | 977 | 4.68 | -0.017 | -0.063 | -0.157 | 1.138 | -1.74 | 39.9 | -0.224 | -0.071 | 36.0 | 64.0 | 0.6 | 67.30 | 9 |
| TRAIN half 2020-2024 | 1226 | 4.70 | 0.080 | 0.048 | -0.017 | 1.134 | 1.47 | 47.6 | -0.109 | 0.046 | 37.1 | 62.9 | 0.0 | 16.13 | 6 |
| VAL 2025-2026/05 | 348 | 4.73 | 0.062 | 0.027 | -0.043 | 1.041 | 0.49 | 50.3 | -0.115 | 0.022 | 32.5 | 67.5 | 0.0 | 18.44 | 5 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=247 mean=-0.055 total=-13.69
- 2017: n=247 mean=-0.081 total=-20.10
- 2018: n=244 mean=-0.053 total=-12.87
- 2019: n=239 mean=-0.063 total=-15.13
- 2020: n=244 mean=-0.035 total=-8.63
- 2021: n=246 mean=0.062 total=15.19
- 2022: n=246 mean=0.080 total=19.60
- 2023: n=242 mean=0.025 total=6.07
- 2024: n=248 mean=0.106 total=26.22

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.21 R  min -3.72 R  MDD 5.26 R   under-water max 10 wk   [fail]
C4 green     53%  null 51%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -4.09 R   capped 0.91 R   top-5 share 260.8%   weekly P&L histogram: [-3.7, -2.3, -2.2, -2.0, -1.5, -1.4, -1.0, -0.9, -0.7, -0.4, -0.3, 0.4, 0.6, 0.9, 1.0, 1.1, 1.3, 1.3, 1.9, 2.0, 2.0, 6.6]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.112 R

## Cell 1338: SPY W=5 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1123 | 2.39 | 0.114 | -0.004 | -0.239 | 1.973 | -0.06 | 29.8 | -0.316 | -0.058 | 68.2 | 31.8 | 0.1 | 54.24 | 8 |
| TRAIN half 2016-2019 | 499 | 2.39 | 0.067 | -0.070 | -0.345 | 1.958 | -0.80 | 28.5 | -0.382 | -0.121 | 69.1 | 30.9 | 0.0 | 54.24 | 8 |
| TRAIN half 2020-2024 | 624 | 2.39 | 0.151 | 0.050 | -0.154 | 1.985 | 0.63 | 30.9 | -0.262 | -0.008 | 67.5 | 32.5 | 0.2 | 52.97 | 7 |
| VAL 2025-2026/05 | 176 | 2.39 | -0.035 | -0.143 | -0.358 | 1.700 | -1.12 | 28.4 | -0.402 | -0.160 | 69.9 | 30.1 | 0.0 | 33.84 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=122 mean=-0.237 total=-28.91
- 2017: n=118 mean=-0.086 total=-10.12
- 2018: n=128 mean=-0.116 total=-14.82
- 2019: n=131 mean=0.144 total=18.81
- 2020: n=119 mean=-0.100 total=-11.89
- 2021: n=128 mean=0.336 total=43.01
- 2022: n=110 mean=0.379 total=41.66
- 2023: n=144 mean=-0.009 total=-1.25
- 2024: n=123 mean=-0.329 total=-40.51

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.31 R  min -3.35 R  MDD 15.84 R   under-water max 13 wk   [fail]
C4 green     39%  null 50%                   [fail]
C5 fills/wk  2.36                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -13.00 R   capped -10.51 R   top-5 share -23.7%   weekly P&L histogram: [-3.4, -3.3, -2.3, -2.2, -2.2, -2.2, -2.2, -2.1, -1.2, -1.1, -0.7, -0.4, 0.0, 0.4, 0.5, 0.8, 1.3, 1.5, 1.8, 2.1, 2.1, 2.5]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.256 R

## Cell 1339: SPY W=5 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1120 | 2.39 | 0.009 | -0.105 | -0.334 | 2.114 | -1.67 | 25.2 | -0.470 | -0.199 | 72.1 | 27.9 | 0.2 | 148.60 | 10 |
| TRAIN half 2016-2019 | 498 | 2.39 | -0.056 | -0.195 | -0.473 | 2.042 | -2.13 | 24.7 | -0.548 | -0.281 | 71.7 | 28.3 | 0.0 | 105.42 | 10 |
| TRAIN half 2020-2024 | 622 | 2.38 | 0.061 | -0.034 | -0.223 | 2.169 | -0.39 | 25.6 | -0.411 | -0.134 | 72.5 | 27.5 | 0.3 | 61.47 | 8 |
| VAL 2025-2026/05 | 173 | 2.35 | 0.006 | -0.101 | -0.316 | 2.086 | -0.64 | 25.4 | -0.458 | -0.195 | 71.7 | 28.3 | 0.6 | 35.10 | 11 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=129 mean=-0.113 total=-14.54
- 2017: n=131 mean=-0.471 total=-61.66
- 2018: n=121 mean=-0.092 total=-11.13
- 2019: n=117 mean=-0.085 total=-9.90
- 2020: n=128 mean=-0.215 total=-27.53
- 2021: n=124 mean=0.215 total=26.64
- 2022: n=140 mean=0.030 total=4.19
- 2023: n=104 mean=-0.138 total=-14.38
- 2024: n=126 mean=-0.078 total=-9.84

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -3.31 R  min -3.40 R  MDD 27.45 R   under-water max 15 wk   [fail]
C4 green     14%  null 50%                   [fail]
C5 fills/wk  2.27                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 12.0 wk    [fail]
diagnostics  ex-top-5% -29.91 R   capped -24.91 R   top-5 share -75.8%   weekly P&L histogram: [-3.4, -3.3, -3.3, -3.3, -2.3, -2.2, -2.2, -2.2, -2.1, -2.1, -1.2, -1.2, -1.1, -1.1, -1.1, -1.1, -1.0, -0.6, -0.5, 1.7, 3.6, 12.9]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.317 R

## Cell 1340: SPY W=5 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2243 | 4.78 | 0.062 | -0.054 | -0.286 | 2.045 | -1.26 | 27.5 | -0.393 | -0.129 | 70.2 | 29.8 | 0.1 | 181.59 | 10 |
| TRAIN half 2016-2019 | 997 | 4.78 | 0.005 | -0.133 | -0.409 | 2.000 | -2.09 | 26.6 | -0.465 | -0.201 | 70.4 | 29.6 | 0.0 | 145.75 | 10 |
| TRAIN half 2020-2024 | 1246 | 4.78 | 0.106 | 0.008 | -0.188 | 2.078 | 0.14 | 28.3 | -0.334 | -0.071 | 70.0 | 30.0 | 0.2 | 95.93 | 7 |
| VAL 2025-2026/05 | 349 | 4.74 | -0.015 | -0.122 | -0.337 | 1.898 | -1.20 | 26.9 | -0.430 | -0.178 | 70.8 | 29.2 | 0.3 | 63.60 | 10 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=251 mean=-0.173 total=-43.45
- 2017: n=249 mean=-0.288 total=-71.78
- 2018: n=249 mean=-0.104 total=-25.95
- 2019: n=248 mean=0.036 total=8.91
- 2020: n=247 mean=-0.160 total=-39.42
- 2021: n=252 mean=0.276 total=69.66
- 2022: n=250 mean=0.183 total=45.85
- 2023: n=248 mean=-0.063 total=-15.63
- 2024: n=249 mean=-0.202 total=-50.35

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -5.48 R  min -5.65 R  MDD 41.09 R   under-water max 14 wk   [fail]
C4 green     22%  null 50%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -42.16 R   capped -37.16 R   top-5 share -53.2%   weekly P&L histogram: [-5.6, -5.5, -5.5, -5.4, -4.6, -4.5, -4.4, -3.3, -2.8, -2.1, -1.5, -1.2, -1.1, -0.8, -0.5, -0.3, -0.1, -0.1, 0.8, 1.4, 5.0, 14.6]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.203 R

## Cell 1341: SPY W=15 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1124 | 2.39 | 0.117 | 0.035 | -0.127 | 1.491 | 0.80 | 40.5 | -0.171 | 0.027 | 53.3 | 46.7 | 0.0 | 41.10 | 7 |
| TRAIN half 2016-2019 | 491 | 2.35 | 0.034 | -0.060 | -0.249 | 1.469 | -0.91 | 37.3 | -0.272 | -0.073 | 54.4 | 45.6 | 0.0 | 41.10 | 6 |
| TRAIN half 2020-2024 | 633 | 2.43 | 0.181 | 0.110 | -0.033 | 1.505 | 1.83 | 43.0 | -0.091 | 0.105 | 52.4 | 47.6 | 0.0 | 26.89 | 7 |
| VAL 2025-2026/05 | 188 | 2.56 | 0.054 | -0.022 | -0.175 | 1.357 | -0.22 | 42.0 | -0.208 | -0.030 | 53.2 | 46.8 | 0.0 | 16.76 | 4 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=120 mean=-0.111 total=-13.28
- 2017: n=120 mean=-0.013 total=-1.54
- 2018: n=123 mean=-0.006 total=-0.71
- 2019: n=128 mean=-0.110 total=-14.12
- 2020: n=134 mean=0.106 total=14.21
- 2021: n=133 mean=0.291 total=38.72
- 2022: n=118 mean=0.160 total=18.85
- 2023: n=134 mean=0.060 total=8.05
- 2024: n=114 mean=-0.091 total=-10.37

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.19 R  min -3.23 R  MDD 5.98 R   under-water max 6 wk   [fail]
C4 green     45%  null 50%                   [fail]
C5 fills/wk  2.64                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 10.3 wk    [fail]
diagnostics  ex-top-5% -2.89 R   capped 2.11 R   top-5 share 162.7%   weekly P&L histogram: [-3.2, -2.6, -2.2, -1.9, -1.1, -1.1, -1.1, -1.0, -1.0, -0.8, -0.6, -0.1, 0.2, 0.7, 0.8, 1.0, 1.2, 1.8, 1.9, 2.2, 4.0, 7.5]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.198 R

## Cell 1342: SPY W=15 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1114 | 2.37 | 0.001 | -0.075 | -0.229 | 1.640 | -1.54 | 32.5 | -0.335 | -0.106 | 59.6 | 40.4 | 0.2 | 117.73 | 9 |
| TRAIN half 2016-2019 | 502 | 2.41 | -0.080 | -0.169 | -0.348 | 1.556 | -2.44 | 30.1 | -0.425 | -0.201 | 58.4 | 41.6 | 0.2 | 94.46 | 9 |
| TRAIN half 2020-2024 | 612 | 2.35 | 0.068 | 0.002 | -0.132 | 1.702 | 0.02 | 34.5 | -0.263 | -0.029 | 60.6 | 39.4 | 0.2 | 43.86 | 7 |
| VAL 2025-2026/05 | 161 | 2.19 | 0.135 | 0.059 | -0.091 | 1.604 | 0.47 | 36.0 | -0.190 | 0.045 | 54.0 | 46.0 | 0.0 | 17.70 | 8 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=131 mean=-0.161 total=-21.07
- 2017: n=128 mean=-0.295 total=-37.74
- 2018: n=125 mean=-0.056 total=-7.02
- 2019: n=118 mean=-0.163 total=-19.20
- 2020: n=113 mean=-0.065 total=-7.37
- 2021: n=118 mean=-0.037 total=-4.37
- 2022: n=132 mean=0.201 total=26.48
- 2023: n=114 mean=-0.027 total=-3.09
- 2024: n=135 mean=-0.079 total=-10.73

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -3.18 R  min -3.77 R  MDD 16.65 R   under-water max 8 wk   [fail]
C4 green     28%  null 50%                   [fail]
C5 fills/wk  2.00                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.2 wk    [fail]
diagnostics  ex-top-5% -15.32 R   capped -10.32 R   top-5 share -103.8%   weekly P&L histogram: [-3.8, -3.6, -3.2, -3.1, -2.1, -2.1, -1.8, -1.1, -1.1, -1.1, -1.1, -1.0, -0.6, -0.3, -0.3, 0.0, 0.0, 2.5, 2.7, 2.9, 2.9, 7.8]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.253 R

## Cell 1343: SPY W=15 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2238 | 4.77 | 0.059 | -0.020 | -0.178 | 1.567 | -0.60 | 36.5 | -0.252 | -0.039 | 56.4 | 43.6 | 0.1 | 129.18 | 8 |
| TRAIN half 2016-2019 | 993 | 4.76 | -0.024 | -0.115 | -0.299 | 1.514 | -2.40 | 33.6 | -0.347 | -0.138 | 56.4 | 43.6 | 0.1 | 122.70 | 8 |
| TRAIN half 2020-2024 | 1245 | 4.77 | 0.125 | 0.057 | -0.081 | 1.606 | 1.24 | 38.8 | -0.177 | 0.039 | 56.5 | 43.5 | 0.1 | 53.48 | 8 |
| VAL 2025-2026/05 | 349 | 4.74 | 0.091 | 0.016 | -0.136 | 1.474 | 0.20 | 39.3 | -0.193 | 0.004 | 53.6 | 46.4 | 0.0 | 20.42 | 4 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=251 mean=-0.137 total=-34.35
- 2017: n=248 mean=-0.158 total=-39.29
- 2018: n=248 mean=-0.031 total=-7.73
- 2019: n=246 mean=-0.135 total=-33.32
- 2020: n=247 mean=0.028 total=6.84
- 2021: n=251 mean=0.137 total=34.35
- 2022: n=250 mean=0.181 total=45.33
- 2023: n=248 mean=0.020 total=4.97
- 2024: n=249 mean=-0.085 total=-21.10

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 4.5 wk  P90 6.5 wk        [fail]   gaps: [2, 7]
C2 bleed     P90 -1.54 R     cycles net>0 50% [fail]
C3 reds      P10 -4.54 R  min -5.34 R  MDD 14.11 R   under-water max 13 wk   [fail]
C4 green     39%  null 51%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 2   bootstrap P90-gap 75% UB 9.2 wk    [fail]
diagnostics  ex-top-5% -10.14 R   capped -6.67 R   top-5 share -247.8%   weekly P&L histogram: [-5.3, -4.8, -4.7, -3.2, -3.1, -3.0, -2.8, -2.0, -1.1, -1.1, -0.8, 0.1, 0.3, 0.3, 0.5, 0.8, 2.0, 2.8, 3.7, 5.1, 6.4, 7.2]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.158 R

## Cell 1344: SPY W=30 long

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1175 | 2.50 | 0.070 | 0.005 | -0.125 | 1.193 | 0.14 | 45.4 | -0.154 | 0.002 | 42.0 | 58.0 | 0.0 | 43.75 | 7 |
| TRAIN half 2016-2019 | 521 | 2.50 | 0.049 | -0.027 | -0.179 | 1.200 | -0.51 | 43.2 | -0.194 | -0.034 | 40.9 | 59.1 | 0.0 | 32.67 | 5 |
| TRAIN half 2020-2024 | 654 | 2.51 | 0.087 | 0.030 | -0.083 | 1.187 | 0.65 | 47.1 | -0.125 | 0.030 | 42.8 | 57.2 | 0.0 | 22.84 | 7 |
| VAL 2025-2026/05 | 193 | 2.62 | -0.014 | -0.071 | -0.184 | 1.004 | -0.98 | 47.7 | -0.196 | -0.071 | 41.5 | 58.5 | 0.0 | 22.74 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=125 mean=-0.008 total=-1.02
- 2017: n=137 mean=0.025 total=3.40
- 2018: n=130 mean=-0.057 total=-7.39
- 2019: n=129 mean=-0.069 total=-8.86
- 2020: n=148 mean=-0.068 total=-10.13
- 2021: n=138 mean=0.111 total=15.34
- 2022: n=115 mean=0.042 total=4.78
- 2023: n=124 mean=0.140 total=17.38
- 2024: n=129 mean=-0.059 total=-7.65

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -1.07 R  min -2.21 R  MDD 6.80 R   under-water max 20 wk   [fail]
C4 green     38%  null 50%                   [fail]
C5 fills/wk  2.68                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -7.11 R   capped -2.11 R   top-5 share -298.0%   weekly P&L histogram: [-2.2, -2.1, -1.1, -1.1, -1.0, -1.0, -1.0, -0.9, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.4, 0.7, 0.9, 1.1, 1.2, 1.4, 5.3]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.144 R

## Cell 1345: SPY W=30 short

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 1046 | 2.23 | 0.019 | -0.041 | -0.160 | 1.337 | -0.99 | 36.9 | -0.245 | -0.050 | 46.0 | 54.0 | 0.0 | 69.82 | 9 |
| TRAIN half 2016-2019 | 462 | 2.22 | -0.056 | -0.125 | -0.263 | 1.199 | -2.24 | 33.3 | -0.307 | -0.125 | 43.3 | 56.7 | 0.0 | 61.86 | 9 |
| TRAIN half 2020-2024 | 584 | 2.24 | 0.078 | 0.025 | -0.079 | 1.435 | 0.43 | 39.7 | -0.198 | 0.009 | 48.1 | 51.9 | 0.0 | 29.66 | 7 |
| VAL 2025-2026/05 | 156 | 2.12 | 0.102 | 0.047 | -0.064 | 1.299 | 0.45 | 42.3 | -0.141 | 0.041 | 39.1 | 60.9 | 0.0 | 11.49 | 6 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=125 mean=-0.176 total=-22.03
- 2017: n=108 mean=-0.165 total=-17.83
- 2018: n=115 mean=0.085 total=9.78
- 2019: n=114 mean=-0.243 total=-27.70
- 2020: n=99 mean=0.075 total=7.43
- 2021: n=111 mean=-0.078 total=-8.71
- 2022: n=131 mean=0.149 total=19.54
- 2023: n=123 mean=0.002 total=0.25
- 2024: n=120 mean=-0.031 total=-3.69

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -2.60 R  min -3.69 R  MDD 11.49 R   under-water max 8 wk   [fail]
C4 green     40%  null 50%                   [fail]
C5 fills/wk  1.95                             [fail]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -9.12 R   capped -5.55 R   top-5 share -64.3%   weekly P&L histogram: [-3.7, -3.4, -2.6, -2.2, -2.1, -1.1, -1.0, -1.0, -0.6, -0.4, 0.0, -0.0, 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.3, 1.7, 3.5, 3.6]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.208 R

## Cell 1346: SPY W=30 both

| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2016-2024 | 2221 | 4.73 | 0.046 | -0.017 | -0.142 | 1.263 | -0.62 | 41.4 | -0.198 | -0.023 | 43.9 | 56.1 | 0.0 | 91.51 | 8 |
| TRAIN half 2016-2019 | 983 | 4.71 | -0.000 | -0.073 | -0.218 | 1.200 | -1.90 | 38.6 | -0.245 | -0.077 | 42.0 | 58.0 | 0.0 | 76.32 | 7 |
| TRAIN half 2020-2024 | 1238 | 4.75 | 0.082 | 0.028 | -0.081 | 1.309 | 0.75 | 43.6 | -0.159 | 0.020 | 45.3 | 54.7 | 0.0 | 33.08 | 8 |
| VAL 2025-2026/05 | 349 | 4.74 | 0.038 | -0.018 | -0.131 | 1.145 | -0.30 | 45.3 | -0.176 | -0.021 | 40.4 | 59.6 | 0.0 | 19.94 | 8 |

TRAIN year-by-year (net 1bp mean R, total R):
- 2016: n=250 mean=-0.092 total=-23.04
- 2017: n=245 mean=-0.059 total=-14.43
- 2018: n=245 mean=0.010 total=2.38
- 2019: n=243 mean=-0.150 total=-36.56
- 2020: n=247 mean=-0.011 total=-2.70
- 2021: n=249 mean=0.027 total=6.64
- 2022: n=246 mean=0.099 total=24.31
- 2023: n=247 mean=0.071 total=17.63
- 2024: n=249 mean=-0.046 total=-11.33

```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -4.37 R  min -4.78 R  MDD 9.31 R   under-water max 10 wk   [fail]
C4 green     38%  null 51%                   [fail]
C5 fills/wk  4.64                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -12.66 R   capped -7.66 R   top-5 share -72.6%   weekly P&L histogram: [-4.8, -4.7, -4.5, -3.2, -1.8, -1.4, -1.3, -1.1, -1.0, -0.8, -0.4, -0.2, 0.1, 0.1, 0.1, 0.5, 0.9, 2.5, 2.6, 2.8, 3.1, 5.3]
```

**Pass bar**: 1(VAL net1bp+t)=False 2(TRAIN+halves)=False 3(>=6/9 yrs)=False 4(ex-top5/cap5R>0 both)=False 5(cadence C3/C4/C5 VAL)=False 6(3bp VAL>0)=False -> **NO PASS**
MDE (VAL, sd/sqrt(n)*2) = 0.123 R

## Summary
0 of 18 cells pass all 6 criteria.

## Verdict (main session, 2026-09-21) — 0 of 18 cells pass; line closed at pass 1
* Every cell fails criterion 1 (VAL net ≥ +0.08 R, t ≥ 2) and 2 (TRAIN net ≥ +0.08 both halves). Best by
  structure: SPY W=15 long (TRAIN net +0.117 R, halves +0.02/+0.18, 8/9 years, VAL +0.054 t 0.57) and QQQ W=15
  long (TRAIN +0.052, 7/9 years, VAL +0.068 t 0.77). Shorts are negative on TRAIN in 8 of 9 cells.
* Mechanism of failure: R = the opening range ≈ 0.1–0.2 % of price, so a 1 bp cost per side is 7–10 % of R and a
  3 bp cost turns every cell negative; the gross edge is tail-carried (ex-top-5 % negative everywhere, 2022–2023
  carry the decade). The published result is leverage on that thin, tail-dependent gross edge in trend years.
* Adequacy: VAL n ≈ 170 per cell, sd ≈ 1.7 R → MDE ≈ 0.26 R at t = 2; VAL cannot resolve a +0.05 R effect. TRAIN
  (n ≈ 1,180, MDE ≈ 0.10 R) does, and only SPY W=15 long clears +0.10 there. What is excluded: a net edge ≥ +0.1 R
  per trade at 1 bp on any cell except SPY W=15 long, where +0.12 on TRAIN did not repeat at size on VAL.
* Not re-opened without a new mechanism (e.g. a day-type filter with a causal trace); 1,346 cells on the programme.
