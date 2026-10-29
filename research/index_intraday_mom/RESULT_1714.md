# RESULT 1,714-1,716 - market intraday momentum SPY/QQQ (PREREG_1714.md)

Built by `cell_1714.py`. TEST (2023-01+) never loaded (hard cut at load). TRAIN thresholds were frozen (`thresholds_<SYM>.json`) by the TRAIN runs before any VAL run. Entry = open of the 15:31 bar, exit = close of the 15:59 bar (proxy for the MOC print, NOT the official closing auction price). Cost 1 bp/side (sensitivity 3 bp/side). 3x column = 3 x gross - 6 bp (3 bp/side), never traded.

## SPY TRAIN (cost 1.0 bp/side, round trip 2.0 bp)

### Data check

| calendar session days | eligible | excluded <380 bars | excluded missing key bar | first | last |
|---|---|---|---|---|---|
| 1006 | 997 | 9 | 0 | 2016-01-04 | 2019-12-31 |

(Exclusion criteria exactly as PREREG: <380 bars, or no 09:30/09:59/15:29/15:31/15:59 bar; half-day sessions fall under <380 bars.)

### Frozen TRAIN thresholds (SPY, from 997 TRAIN days)

| |r1| median | |rday| P50 | |rday| P70 | |rday| P85 |
|---|---|---|---|
| 12.15 bp | 26.54 bp | 46.09 bp | 78.12 bp |

### Cell table (net bps per trade; R = 25 bp)

| cell | side | n | trades/wk | gross | net | sd | iid t | yr-boot CI | mo-boot CI | ex-top-5% | cap+50 | win% | net@3bp/side | 3x net | green wk | null mean/P95 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 986 | 4.72 | -0.40 | -2.40 | 20.8 | -3.64 | [-3.33,-1.72] | [-3.81,-1.15] | -4.80 | -2.65 | 44.1 | -6.40 | -7.21 | 37.8% | 36.9/42.1 (0.426) |
| 1714A | LO | 495 | 2.37 | -0.83 | -2.83 | 18.8 | -3.35 | [-4.89,-1.41] | [-4.97,-0.87] | -4.69 | -2.84 | 45.3 | -6.83 | -8.48 | 40.7% | 40.4/45.0 (0.483) |
| 1714B | LS | 997 | 4.77 | 0.84 | -1.16 | 20.7 | -1.77 | [-2.24,0.16] | [-2.65,0.34] | -3.90 | -1.70 | 44.1 | -5.16 | -3.47 | 40.7% | 37.0/42.1 (0.132) |
| 1714B | LO | 543 | 2.60 | 0.45 | -1.55 | 18.9 | -1.91 | [-2.71,-0.70] | [-3.61,0.66] | -3.87 | -1.89 | 44.9 | -5.55 | -4.65 | 40.7% | 41.1/45.5 (0.612) |
| 1714C | LS | 498 | 2.38 | -1.20 | -3.20 | 24.9 | -2.88 | [-5.54,-1.22] | [-5.33,-1.25] | -6.01 | -3.66 | 44.2 | -7.20 | -9.61 | 32.1% | 38.6/44.0 (0.981) |
| 1714C | LO | 255 | 1.22 | -1.95 | -3.95 | 22.9 | -2.75 | [-8.18,-0.84] | [-7.60,-0.80] | -6.06 | -3.95 | 46.3 | -7.95 | -11.84 | 32.1% | 33.0/36.8 (0.682) |
| 1715P50 | LS | 498 | 2.38 | 1.82 | -0.18 | 26.0 | -0.16 | [-2.80,2.87] | [-2.91,2.68] | -3.83 | -1.24 | 45.4 | -4.18 | -0.55 | 42.1% | 38.5/44.0 (0.145) |
| 1715P50 | LO | 278 | 1.33 | 1.39 | -0.61 | 22.7 | -0.45 | [-2.82,1.95] | [-3.87,3.18] | -3.51 | -1.28 | 45.7 | -4.61 | -1.84 | 31.6% | 34.9/38.8 (0.926) |
| 1715P70 | LS | 299 | 1.43 | 2.66 | 0.66 | 31.6 | 0.36 | [-2.27,4.06] | [-3.80,4.84] | -3.81 | -1.11 | 47.2 | -3.34 | 1.97 | 34.4% | 31.9/36.4 (0.218) |
| 1715P70 | LO | 164 | 0.78 | 1.41 | -0.59 | 27.8 | -0.27 | [-4.86,2.29] | [-5.83,5.93] | -4.51 | -1.72 | 47.0 | -4.59 | -1.77 | 23.0% | 23.7/26.8 (0.684) |
| 1715P85 | LS | 150 | 0.72 | 6.99 | 4.99 | 36.9 | 1.66 | [-2.47,12.65] | [-3.49,12.66] | -0.99 | 1.61 | 52.0 | 0.99 | 14.97 | 20.1% | 17.8/21.1 (0.166) |
| 1715P85 | LO | 74 | 0.35 | 4.98 | 2.98 | 31.4 | 0.81 | [-6.66,11.07] | [-4.63,12.32] | -2.02 | 0.77 | 51.4 | -1.02 | 8.93 | 13.4% | 12.3/14.8 (0.292) |

Cells: 1714A = sign(r1); 1714B = sign(r1+r12); 1714C = sign(r1) if |r1|>TRAIN median; 1715P50 = sign(rday) if |rday|>TRAIN P50; 1715P70 = sign(rday) if |rday|>TRAIN P70 (PRIMARY); 1715P85 = sign(rday) if |rday|>TRAIN P85. LS = long-short, LO = long-only. yr-boot = year-block bootstrap (few blocks, crude); mo-boot = month-block.

### Weekly table (weekly sum of net in R = 25 bp; empty weeks count 0)

| cell | side | weeks | P10 (R) | green | >= +5R share | n strong | median gap (wk) | P90 gap (wk) |
|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 209 | -1.98 | 37.8% | 0.0% | 0 | nan | nan |
| 1714A | LO | 209 | -1.47 | 40.7% | 0.0% | 0 | nan | nan |
| 1714B | LS | 209 | -2.10 | 40.7% | 1.9% | 4 | 11.0 | 27.8 |
| 1714B | LO | 209 | -1.52 | 40.7% | 0.5% | 1 | nan | nan |
| 1714C | LS | 209 | -1.53 | 32.1% | 0.0% | 0 | nan | nan |
| 1714C | LO | 209 | -0.89 | 32.1% | 0.0% | 0 | nan | nan |
| 1715P50 | LS | 209 | -1.41 | 42.1% | 1.9% | 4 | 11.0 | 27.8 |
| 1715P50 | LO | 209 | -0.88 | 31.6% | 0.5% | 1 | nan | nan |
| 1715P70 | LS | 209 | -1.19 | 34.4% | 1.9% | 4 | 11.0 | 27.8 |
| 1715P70 | LO | 209 | -0.64 | 23.0% | 0.5% | 1 | nan | nan |
| 1715P85 | LS | 209 | -0.74 | 20.1% | 1.9% | 4 | 11.0 | 27.8 |
| 1715P85 | LO | 209 | -0.13 | 13.4% | 0.5% | 1 | nan | nan |

`scripts/cadence_bar.py` takes a trades CSV (date+pnl_R) with its own split/TEST guards and a full multi-criterion spec; not invoked, the weekly table is computed here.

### 1,716 report-only

OLS r13 (bps) on r1 (bps) and r12 (bps), HC1 t: | const | b(r1) | t | b(r12) | t | R2 | n |
|---|---|---|---|---|---|---|
| -0.45 | 0.0124 | 0.31 | 0.0502 | 2.00 | 0.01823 | 997 |

Decile of r1 (own-split deciles) -> mean r13 (gross bps); corr(decile, mean r13) = -0.45 (monotonic = +1):

| decile | mean r1 bp | n | mean r13 bp | t |
|---|---|---|---|---|
| 1 | -38.1 | 100 | -0.04 | -0.01 |
| 2 | -19.2 | 100 | 0.37 | 0.16 |
| 3 | -12.0 | 99 | 1.06 | 0.55 |
| 4 | -6.9 | 100 | -1.47 | -0.82 |
| 5 | -2.4 | 103 | 0.63 | 0.38 |
| 6 | 2.5 | 96 | -1.52 | -1.17 |
| 7 | 7.7 | 100 | 2.07 | 1.59 |
| 8 | 12.7 | 99 | -0.17 | -0.13 |
| 9 | 19.5 | 100 | -1.28 | -0.53 |
| 10 | 42.9 | 100 | -3.26 | -1.26 |

Per-year (net bps, LS): unconditional mean r13 beside 1714A and 1715P70:

| year | days | mean r13 (all days) | 1714A n | net | t | 1715P70 n | net | t |
|---|---|---|---|---|---|---|---|---|
| 2016 | 251 | 0.59 | 249 | -1.89 | -1.79 | 87 | -1.89 | -0.81 |
| 2017 | 249 | -0.77 | 245 | -1.54 | -2.30 | 30 | -2.17 | -0.85 |
| 2018 | 249 | -2.18 | 248 | -3.82 | -1.90 | 108 | 5.52 | 1.31 |
| 2019 | 248 | 0.95 | 244 | -2.36 | -2.03 | 74 | -2.31 | -0.84 |

SPY 20-day realised vol terciles (TRAIN cut points 7.5% / 11.3% annualised), LS net bps:

| cell | vol tercile | n | net | t |
|---|---|---|---|---|
| 1714A | low | 323 | -1.17 | -1.60 |
| 1714A | mid | 317 | -1.83 | -2.33 |
| 1714A | high | 325 | -4.87 | -2.99 |
| 1715P70 | low | 49 | 3.83 | 1.17 |
| 1715P70 | mid | 85 | -4.57 | -2.30 |
| 1715P70 | high | 150 | 3.26 | 1.05 |

## QQQ TRAIN (cost 1.0 bp/side, round trip 2.0 bp)

### Data check

| calendar session days | eligible | excluded <380 bars | excluded missing key bar | first | last |
|---|---|---|---|---|---|
| 1006 | 994 | 12 | 0 | 2016-01-04 | 2019-12-31 |

(Exclusion criteria exactly as PREREG: <380 bars, or no 09:30/09:59/15:29/15:31/15:59 bar; half-day sessions fall under <380 bars.)

### Frozen TRAIN thresholds (QQQ, from 994 TRAIN days)

| |r1| median | |rday| P50 | |rday| P70 | |rday| P85 |
|---|---|---|---|
| 19.85 bp | 37.39 bp | 63.81 bp | 107.02 bp |

### Cell table (net bps per trade; R = 25 bp)

| cell | side | n | trades/wk | gross | net | sd | iid t | yr-boot CI | mo-boot CI | ex-top-5% | cap+50 | win% | net@3bp/side | 3x net | green wk | null mean/P95 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 989 | 4.73 | -0.08 | -2.08 | 23.3 | -2.80 | [-2.34,-1.81] | [-3.24,-0.99] | -4.91 | -2.63 | 44.9 | -6.08 | -6.23 | 42.1% | 39.1/44.5 (0.200) |
| 1714A | LO | 515 | 2.46 | -0.29 | -2.29 | 22.3 | -2.32 | [-3.65,-1.10] | [-4.47,-0.16] | -4.74 | -2.57 | 47.6 | -6.29 | -6.86 | 44.5% | 43.7/47.8 (0.413) |
| 1714B | LS | 994 | 4.76 | 1.22 | -0.78 | 23.2 | -1.06 | [-2.58,1.16] | [-2.41,0.99] | -3.93 | -1.53 | 44.8 | -4.78 | -2.34 | 41.1% | 39.1/44.5 (0.294) |
| 1714B | LO | 536 | 2.56 | 0.93 | -1.07 | 19.9 | -1.24 | [-1.87,-0.22] | [-3.07,1.02] | -3.56 | -1.40 | 45.9 | -5.07 | -3.20 | 44.0% | 44.2/48.3 (0.554) |
| 1714C | LS | 497 | 2.38 | -0.38 | -2.38 | 27.3 | -1.94 | [-2.76,-1.96] | [-4.58,-0.27] | -5.77 | -3.31 | 44.9 | -6.38 | -7.15 | 35.4% | 39.5/45.0 (0.911) |
| 1714C | LO | 260 | 1.24 | -1.21 | -3.21 | 26.1 | -1.98 | [-4.75,-1.87] | [-6.62,0.15] | -5.96 | -3.61 | 46.9 | -7.21 | -9.62 | 34.4% | 34.4/38.3 (0.535) |
| 1715P50 | LS | 497 | 2.38 | 2.55 | 0.55 | 28.5 | 0.43 | [-2.40,4.32] | [-2.38,3.62] | -3.50 | -0.84 | 47.7 | -3.45 | 1.65 | 43.1% | 40.5/45.9 (0.247) |
| 1715P50 | LO | 279 | 1.33 | 2.10 | 0.10 | 23.2 | 0.07 | [-2.94,3.36] | [-3.27,3.97] | -2.96 | -0.55 | 48.7 | -3.90 | 0.29 | 37.3% | 36.2/40.2 (0.372) |
| 1715P70 | LS | 298 | 1.43 | 3.75 | 1.75 | 32.9 | 0.92 | [-2.94,6.83] | [-2.79,6.32] | -2.94 | -0.28 | 49.3 | -2.25 | 5.26 | 35.4% | 32.6/36.9 (0.179) |
| 1715P70 | LO | 162 | 0.78 | 2.82 | 0.82 | 26.3 | 0.39 | [-4.33,5.69] | [-3.59,5.22] | -2.73 | 0.10 | 50.6 | -3.18 | 2.45 | 27.8% | 25.7/29.2 (0.197) |
| 1715P85 | LS | 149 | 0.71 | 4.90 | 2.90 | 38.3 | 0.92 | [-3.13,8.10] | [-2.94,9.11] | -2.90 | -0.23 | 46.3 | -1.10 | 8.71 | 23.0% | 20.6/23.9 (0.171) |
| 1715P85 | LO | 73 | 0.35 | 4.39 | 2.39 | 32.0 | 0.64 | [-9.36,9.68] | [-5.54,11.32] | -2.00 | 0.84 | 47.9 | -1.61 | 7.16 | 15.3% | 13.3/15.8 (0.131) |

Cells: 1714A = sign(r1); 1714B = sign(r1+r12); 1714C = sign(r1) if |r1|>TRAIN median; 1715P50 = sign(rday) if |rday|>TRAIN P50; 1715P70 = sign(rday) if |rday|>TRAIN P70 (PRIMARY); 1715P85 = sign(rday) if |rday|>TRAIN P85. LS = long-short, LO = long-only. yr-boot = year-block bootstrap (few blocks, crude); mo-boot = month-block.

### Weekly table (weekly sum of net in R = 25 bp; empty weeks count 0)

| cell | side | weeks | P10 (R) | green | >= +5R share | n strong | median gap (wk) | P90 gap (wk) |
|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 209 | -2.35 | 42.1% | 1.4% | 3 | 72.5 | 129.7 |
| 1714A | LO | 209 | -1.58 | 44.5% | 0.5% | 1 | nan | nan |
| 1714B | LS | 209 | -2.31 | 41.1% | 1.4% | 3 | 23.0 | 39.0 |
| 1714B | LO | 209 | -1.38 | 44.0% | 0.5% | 1 | nan | nan |
| 1714C | LS | 209 | -1.59 | 35.4% | 1.0% | 2 | 1.0 | 1.0 |
| 1714C | LO | 209 | -1.08 | 34.4% | 0.5% | 1 | nan | nan |
| 1715P50 | LS | 209 | -1.54 | 43.1% | 1.4% | 3 | 23.0 | 39.0 |
| 1715P50 | LO | 209 | -0.85 | 37.3% | 0.5% | 1 | nan | nan |
| 1715P70 | LS | 209 | -1.38 | 35.4% | 1.0% | 2 | 46.0 | 46.0 |
| 1715P70 | LO | 209 | -0.62 | 27.8% | 0.5% | 1 | nan | nan |
| 1715P85 | LS | 209 | -0.93 | 23.0% | 1.4% | 3 | 23.0 | 36.6 |
| 1715P85 | LO | 209 | -0.24 | 15.3% | 0.5% | 1 | nan | nan |

`scripts/cadence_bar.py` takes a trades CSV (date+pnl_R) with its own split/TEST guards and a full multi-criterion spec; not invoked, the weekly table is computed here.

### 1,716 report-only

OLS r13 (bps) on r1 (bps) and r12 (bps), HC1 t: | const | b(r1) | t | b(r12) | t | R2 | n |
|---|---|---|---|---|---|---|
| -0.30 | 0.0080 | 0.28 | 0.0414 | 1.95 | 0.01696 | 994 |

Decile of r1 (own-split deciles) -> mean r13 (gross bps); corr(decile, mean r13) = -0.05 (monotonic = +1):

| decile | mean r1 bp | n | mean r13 bp | t |
|---|---|---|---|---|
| 1 | -62.4 | 100 | -2.74 | -0.74 |
| 2 | -31.1 | 99 | 0.91 | 0.41 |
| 3 | -18.9 | 99 | 1.13 | 0.62 |
| 4 | -10.7 | 100 | 0.55 | 0.23 |
| 5 | -2.6 | 99 | -1.44 | -1.21 |
| 6 | 5.0 | 99 | -0.97 | -0.60 |
| 7 | 12.7 | 100 | 3.37 | 1.89 |
| 8 | 21.1 | 99 | -0.23 | -0.08 |
| 9 | 32.6 | 99 | 0.59 | 0.32 |
| 10 | 67.0 | 100 | -3.29 | -1.13 |

Per-year (net bps, LS): unconditional mean r13 beside 1714A and 1715P70:

| year | days | mean r13 (all days) | 1714A n | net | t | 1715P70 n | net | t |
|---|---|---|---|---|---|---|---|---|
| 2016 | 250 | -0.17 | 249 | -2.40 | -1.95 | 80 | -1.20 | -0.43 |
| 2017 | 249 | 0.30 | 246 | -1.90 | -2.40 | 36 | -1.74 | -0.57 |
| 2018 | 246 | -2.15 | 245 | -2.28 | -1.01 | 107 | 8.83 | 2.08 |
| 2019 | 249 | 1.14 | 249 | -1.72 | -1.37 | 75 | -3.51 | -1.19 |

SPY 20-day realised vol terciles (TRAIN cut points 7.5% / 11.3% annualised), LS net bps:

| cell | vol tercile | n | net | t |
|---|---|---|---|---|
| 1714A | low | 324 | -1.26 | -1.45 |
| 1714A | mid | 323 | -1.26 | -1.40 |
| 1714A | high | 321 | -4.16 | -2.30 |
| 1715P70 | low | 56 | 1.63 | 0.43 |
| 1715P70 | mid | 96 | -3.05 | -1.42 |
| 1715P70 | high | 133 | 5.53 | 1.64 |

## SPY VAL (cost 1.0 bp/side, round trip 2.0 bp)

### Data check

| calendar session days | eligible | excluded <380 bars | excluded missing key bar | first | last |
|---|---|---|---|---|---|
| 756 | 749 | 7 | 0 | 2020-01-02 | 2022-12-30 |

(Exclusion criteria exactly as PREREG: <380 bars, or no 09:30/09:59/15:29/15:31/15:59 bar; half-day sessions fall under <380 bars.)

### Frozen TRAIN thresholds (SPY, from 997 TRAIN days)

| |r1| median | |rday| P50 | |rday| P70 | |rday| P85 |
|---|---|---|---|
| 12.15 bp | 26.54 bp | 46.09 bp | 78.12 bp |

### Cell table (net bps per trade; R = 25 bp)

| cell | side | n | trades/wk | gross | net | sd | iid t | yr-boot CI | mo-boot CI | ex-top-5% | cap+50 | win% | net@3bp/side | 3x net | green wk | null mean/P95 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 749 | 4.77 | -1.61 | -3.61 | 43.2 | -2.29 | [-6.10,-2.00] | [-6.34,-0.96] | -8.74 | -5.77 | 42.6 | -7.61 | -10.84 | 40.1% | 41.9/48.4 (0.707) |
| 1714A | LO | 386 | 2.46 | -2.34 | -4.34 | 30.1 | -2.84 | [-6.25,-0.60] | [-7.46,-1.12] | -8.40 | -5.43 | 39.9 | -8.34 | -13.01 | 31.2% | 39.0/44.6 (0.994) |
| 1714B | LS | 749 | 4.77 | 1.48 | -0.52 | 43.2 | -0.33 | [-1.81,1.59] | [-3.86,2.60] | -5.67 | -2.88 | 46.9 | -4.52 | -1.57 | 48.4% | 41.9/48.4 (0.058) |
| 1714B | LO | 404 | 2.57 | 0.63 | -1.37 | 39.4 | -0.70 | [-4.55,1.32] | [-4.28,1.89] | -6.62 | -3.68 | 44.3 | -5.37 | -4.11 | 40.1% | 39.3/44.6 (0.419) |
| 1714C | LS | 509 | 3.24 | -2.17 | -4.17 | 47.0 | -2.00 | [-4.78,-3.10] | [-8.48,-0.06] | -9.39 | -6.38 | 43.6 | -8.17 | -12.50 | 42.0% | 44.1/51.0 (0.730) |
| 1714C | LO | 268 | 1.71 | -2.06 | -4.06 | 30.7 | -2.16 | [-8.37,0.08] | [-7.89,0.04] | -8.22 | -5.19 | 38.8 | -8.06 | -12.18 | 33.1% | 36.0/40.8 (0.863) |
| 1715P50 | LS | 528 | 3.36 | 2.52 | 0.52 | 47.1 | 0.25 | [-0.79,1.80] | [-3.15,4.05] | -4.88 | -2.24 | 49.4 | -3.48 | 1.56 | 54.1% | 44.1/50.3 (0.004) |
| 1715P50 | LO | 295 | 1.88 | 0.99 | -1.01 | 41.1 | -0.42 | [-3.47,1.54] | [-4.06,1.99] | -6.05 | -3.20 | 47.1 | -5.01 | -3.03 | 43.9% | 37.1/42.0 (0.014) |
| 1715P70 | LS | 395 | 2.52 | 1.33 | -0.67 | 51.8 | -0.26 | [-4.04,1.34] | [-4.65,3.52] | -6.63 | -4.17 | 46.3 | -4.67 | -2.02 | 45.2% | 40.1/45.9 (0.100) |
| 1715P70 | LO | 218 | 1.39 | -0.59 | -2.59 | 45.7 | -0.84 | [-6.79,0.40] | [-6.11,1.19] | -8.30 | -5.37 | 43.6 | -6.59 | -7.77 | 35.0% | 32.2/36.9 (0.176) |
| 1715P85 | LS | 264 | 1.68 | 0.35 | -1.65 | 59.8 | -0.45 | [-8.18,2.55] | [-6.98,3.84] | -8.73 | -6.19 | 45.8 | -5.65 | -4.96 | 36.3% | 34.8/40.1 (0.353) |
| 1715P85 | LO | 140 | 0.89 | -1.29 | -3.29 | 54.3 | -0.72 | [-12.49,0.35] | [-8.33,2.08] | -10.05 | -7.33 | 41.4 | -7.29 | -9.87 | 22.9% | 22.5/26.8 (0.460) |

Cells: 1714A = sign(r1); 1714B = sign(r1+r12); 1714C = sign(r1) if |r1|>TRAIN median; 1715P50 = sign(rday) if |rday|>TRAIN P50; 1715P70 = sign(rday) if |rday|>TRAIN P70 (PRIMARY); 1715P85 = sign(rday) if |rday|>TRAIN P85. LS = long-short, LO = long-only. yr-boot = year-block bootstrap (few blocks, crude); mo-boot = month-block.

### Pass bar on VAL (every item PASS/FAIL; sleeve = SPY, QQQ must agree)

| cell | side | pass | verdict | mean>=+2bp | t>=2 | ex-top-5%>0 | cap+50>0 | >=2 trades/wk | green>=55% & >null | TRAIN same sign, t>=1 (value=TRAIN mean) | QQQ VAL mean>0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 1/8 | REJECT | -3.61 FAIL | -2.29 FAIL | -8.74 FAIL | -5.77 FAIL | 4.77 PASS | 40.13 FAIL | -2.40 FAIL | -2.03 FAIL |
| 1714A | LO | 1/8 | REJECT | -4.34 FAIL | -2.84 FAIL | -8.40 FAIL | -5.43 FAIL | 2.46 PASS | 31.21 FAIL | -2.83 FAIL | -1.97 FAIL |
| 1714B | LS | 2/8 | REJECT | -0.52 FAIL | -0.33 FAIL | -5.67 FAIL | -2.88 FAIL | 4.77 PASS | 48.41 FAIL | -1.16 FAIL | 0.20 PASS |
| 1714B | LO | 1/8 | REJECT | -1.37 FAIL | -0.70 FAIL | -6.62 FAIL | -3.68 FAIL | 2.57 PASS | 40.13 FAIL | -1.55 FAIL | -0.10 FAIL |
| 1714C | LS | 1/8 | REJECT | -4.17 FAIL | -2.00 FAIL | -9.39 FAIL | -6.38 FAIL | 3.24 PASS | 42.04 FAIL | -3.20 FAIL | -3.46 FAIL |
| 1714C | LO | 0/8 | REJECT | -4.06 FAIL | -2.16 FAIL | -8.22 FAIL | -5.19 FAIL | 1.71 FAIL | 33.12 FAIL | -3.95 FAIL | -3.26 FAIL |
| 1715P50 | LS | 2/8 | REJECT | 0.52 FAIL | 0.25 FAIL | -4.88 FAIL | -2.24 FAIL | 3.36 PASS | 54.14 FAIL | -0.18 FAIL | 0.80 PASS |
| 1715P50 | LO | 1/8 | REJECT | -1.01 FAIL | -0.42 FAIL | -6.05 FAIL | -3.20 FAIL | 1.88 FAIL | 43.95 FAIL | -0.61 FAIL | 0.72 PASS |
| 1715P70 | LS | 1/8 | REJECT | -0.67 FAIL | -0.26 FAIL | -6.63 FAIL | -4.17 FAIL | 2.52 PASS | 45.22 FAIL | 0.66 FAIL | -0.45 FAIL |
| 1715P70 | LO | 0/8 | REJECT | -2.59 FAIL | -0.84 FAIL | -8.30 FAIL | -5.37 FAIL | 1.39 FAIL | 35.03 FAIL | -0.59 FAIL | -0.28 FAIL |
| 1715P85 | LS | 0/8 | REJECT | -1.65 FAIL | -0.45 FAIL | -8.73 FAIL | -6.19 FAIL | 1.68 FAIL | 36.31 FAIL | 4.99 FAIL | -1.43 FAIL |
| 1715P85 | LO | 0/8 | REJECT | -3.29 FAIL | -0.72 FAIL | -10.05 FAIL | -7.33 FAIL | 0.89 FAIL | 22.93 FAIL | 2.98 FAIL | -2.27 FAIL |

Green-week gate = observed >= 55% AND observed > null mean (null P95 and p shown in the cell table). Verdict rule (PREREG): all 8 gates -> PASS; mean<=0 or ex-top-5%<=0 -> REJECT; mean>0 with t<2 -> UNDETERMINED.

### Weekly table (weekly sum of net in R = 25 bp; empty weeks count 0)

| cell | side | weeks | P10 (R) | green | >= +5R share | n strong | median gap (wk) | P90 gap (wk) |
|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 157 | -3.96 | 40.1% | 1.9% | 3 | 51.5 | 91.9 |
| 1714A | LO | 157 | -2.67 | 31.2% | 1.9% | 3 | 2.0 | 2.8 |
| 1714B | LS | 157 | -3.28 | 48.4% | 5.1% | 8 | 4.0 | 43.2 |
| 1714B | LO | 157 | -2.35 | 40.1% | 2.5% | 4 | 4.0 | 76.8 |
| 1714C | LS | 157 | -3.13 | 42.0% | 2.5% | 4 | 16.0 | 72.0 |
| 1714C | LO | 157 | -2.47 | 33.1% | 1.3% | 2 | 1.0 | 1.0 |
| 1715P50 | LS | 157 | -2.67 | 54.1% | 3.2% | 5 | 10.0 | 71.7 |
| 1715P50 | LO | 157 | -1.78 | 43.9% | 1.3% | 2 | 4.0 | 4.0 |
| 1715P70 | LS | 157 | -2.56 | 45.2% | 3.2% | 5 | 3.5 | 68.7 |
| 1715P70 | LO | 157 | -1.74 | 35.0% | 1.3% | 2 | 4.0 | 4.0 |
| 1715P85 | LS | 157 | -2.18 | 36.3% | 3.2% | 5 | 3.0 | 69.1 |
| 1715P85 | LO | 157 | -1.49 | 22.9% | 1.3% | 2 | 4.0 | 4.0 |

`scripts/cadence_bar.py` takes a trades CSV (date+pnl_R) with its own split/TEST guards and a full multi-criterion spec; not invoked, the weekly table is computed here.

### 1,716 report-only

OLS r13 (bps) on r1 (bps) and r12 (bps), HC1 t: | const | b(r1) | t | b(r12) | t | R2 | n |
|---|---|---|---|---|---|---|
| -0.81 | -0.1356 | -1.11 | 0.0263 | 0.72 | 0.01787 | 749 |

Decile of r1 (own-split deciles) -> mean r13 (gross bps); corr(decile, mean r13) = -0.49 (monotonic = +1):

| decile | mean r1 bp | n | mean r13 bp | t |
|---|---|---|---|---|
| 1 | -70.9 | 75 | 7.95 | 0.74 |
| 2 | -33.3 | 75 | 1.83 | 0.39 |
| 3 | -19.2 | 75 | -4.25 | -1.16 |
| 4 | -10.0 | 75 | 1.65 | 0.39 |
| 5 | -2.7 | 75 | -3.90 | -0.84 |
| 6 | 5.1 | 74 | -1.90 | -0.55 |
| 7 | 13.0 | 75 | -0.08 | -0.03 |
| 8 | 21.8 | 75 | -6.26 | -1.95 |
| 9 | 34.9 | 75 | -4.78 | -1.39 |
| 10 | 65.5 | 75 | 1.74 | 0.40 |

Per-year (net bps, LS): unconditional mean r13 beside 1714A and 1715P70:

| year | days | mean r13 (all days) | 1714A n | net | t | 1715P70 n | net | t |
|---|---|---|---|---|---|---|---|---|
| 2020 | 247 | 1.34 | 247 | -2.00 | -0.51 | 126 | -0.92 | -0.13 |
| 2021 | 252 | -3.47 | 252 | -2.72 | -1.90 | 95 | -4.04 | -1.44 |
| 2022 | 250 | -0.21 | 250 | -6.10 | -2.67 | 174 | 1.34 | 0.49 |

SPY 20-day realised vol terciles (TRAIN cut points 7.5% / 11.3% annualised), LS net bps:

| cell | vol tercile | n | net | t |
|---|---|---|---|---|
| 1714A | low | 33 | -7.34 | -1.92 |
| 1714A | mid | 137 | -3.40 | -1.97 |
| 1714A | high | 579 | -3.45 | -1.73 |
| 1715P70 | low | 9 | -7.63 | -1.41 |
| 1715P70 | mid | 50 | -9.55 | -2.74 |
| 1715P70 | high | 336 | 0.83 | 0.28 |

## QQQ VAL (cost 1.0 bp/side, round trip 2.0 bp)

### Data check

| calendar session days | eligible | excluded <380 bars | excluded missing key bar | first | last |
|---|---|---|---|---|---|
| 756 | 748 | 8 | 0 | 2020-01-02 | 2022-12-30 |

(Exclusion criteria exactly as PREREG: <380 bars, or no 09:30/09:59/15:29/15:31/15:59 bar; half-day sessions fall under <380 bars.)

### Frozen TRAIN thresholds (QQQ, from 994 TRAIN days)

| |r1| median | |rday| P50 | |rday| P70 | |rday| P85 |
|---|---|---|---|
| 19.85 bp | 37.39 bp | 63.81 bp | 107.02 bp |

### Cell table (net bps per trade; R = 25 bp)

| cell | side | n | trades/wk | gross | net | sd | iid t | yr-boot CI | mo-boot CI | ex-top-5% | cap+50 | win% | net@3bp/side | 3x net | green wk | null mean/P95 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 745 | 4.75 | -0.03 | -2.03 | 47.4 | -1.17 | [-3.96,0.39] | [-4.73,0.55] | -7.85 | -5.06 | 45.6 | -6.03 | -6.10 | 40.8% | 43.4/49.7 (0.774) |
| 1714A | LO | 404 | 2.57 | 0.03 | -1.97 | 40.0 | -0.99 | [-6.01,2.72] | [-5.37,1.52] | -7.54 | -4.63 | 44.8 | -5.97 | -5.90 | 40.8% | 43.1/48.4 (0.808) |
| 1714B | LS | 748 | 4.76 | 2.20 | 0.20 | 47.3 | 0.12 | [-2.11,3.92] | [-2.84,3.48] | -5.51 | -2.97 | 48.7 | -3.80 | 0.61 | 51.0% | 43.2/49.7 (0.033) |
| 1714B | LO | 417 | 2.66 | 1.90 | -0.10 | 44.1 | -0.05 | [-3.42,3.65] | [-2.95,3.19] | -6.00 | -3.37 | 47.5 | -4.10 | -0.30 | 47.8% | 42.6/47.8 (0.071) |
| 1714C | LS | 506 | 3.22 | -1.46 | -3.46 | 53.7 | -1.45 | [-7.01,1.86] | [-7.48,0.44] | -10.23 | -7.39 | 43.9 | -7.46 | -10.37 | 37.6% | 44.8/51.6 (0.975) |
| 1714C | LO | 266 | 1.69 | -1.26 | -3.26 | 43.6 | -1.22 | [-9.27,4.40] | [-8.22,2.02] | -9.55 | -6.55 | 42.9 | -7.26 | -9.77 | 36.3% | 39.1/43.9 (0.854) |
| 1715P50 | LS | 554 | 3.53 | 2.80 | 0.80 | 51.4 | 0.37 | [-1.94,4.38] | [-3.56,5.28] | -5.33 | -3.02 | 49.5 | -3.20 | 2.40 | 55.4% | 44.1/50.3 (0.001) |
| 1715P50 | LO | 304 | 1.94 | 2.72 | 0.72 | 47.3 | 0.26 | [-2.54,4.86] | [-3.44,5.46] | -5.74 | -3.07 | 48.4 | -3.28 | 2.16 | 46.5% | 41.2/46.5 (0.066) |
| 1715P70 | LS | 427 | 2.72 | 1.55 | -0.45 | 55.3 | -0.17 | [-3.36,2.35] | [-5.52,4.50] | -6.94 | -4.56 | 49.2 | -4.45 | -1.35 | 46.5% | 42.9/49.0 (0.197) |
| 1715P70 | LO | 225 | 1.43 | 1.72 | -0.28 | 50.7 | -0.08 | [-5.66,3.17] | [-4.86,4.33] | -7.07 | -4.46 | 48.0 | -4.28 | -0.83 | 39.5% | 37.1/42.0 (0.251) |
| 1715P85 | LS | 272 | 1.73 | 0.57 | -1.43 | 65.2 | -0.36 | [-8.72,2.17] | [-9.17,6.80] | -9.02 | -7.12 | 48.5 | -5.43 | -4.28 | 39.5% | 39.0/45.2 (0.482) |
| 1715P85 | LO | 142 | 0.90 | -0.27 | -2.27 | 60.2 | -0.45 | [-11.86,1.40] | [-9.25,5.78] | -10.72 | -8.27 | 44.4 | -6.27 | -6.81 | 24.8% | 26.1/30.6 (0.725) |

Cells: 1714A = sign(r1); 1714B = sign(r1+r12); 1714C = sign(r1) if |r1|>TRAIN median; 1715P50 = sign(rday) if |rday|>TRAIN P50; 1715P70 = sign(rday) if |rday|>TRAIN P70 (PRIMARY); 1715P85 = sign(rday) if |rday|>TRAIN P85. LS = long-short, LO = long-only. yr-boot = year-block bootstrap (few blocks, crude); mo-boot = month-block.

### Pass bar on VAL (every item PASS/FAIL; sleeve = SPY, QQQ must agree)

| cell | side | pass | verdict | mean>=+2bp | t>=2 | ex-top-5%>0 | cap+50>0 | >=2 trades/wk | green>=55% & >null | TRAIN same sign, t>=1 (value=TRAIN mean) | SPY VAL mean>0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 1/8 | REJECT | -2.03 FAIL | -1.17 FAIL | -7.85 FAIL | -5.06 FAIL | 4.75 PASS | 40.76 FAIL | -2.08 FAIL | -3.61 FAIL |
| 1714A | LO | 1/8 | REJECT | -1.97 FAIL | -0.99 FAIL | -7.54 FAIL | -4.63 FAIL | 2.57 PASS | 40.76 FAIL | -2.29 FAIL | -4.34 FAIL |
| 1714B | LS | 1/8 | REJECT | 0.20 FAIL | 0.12 FAIL | -5.51 FAIL | -2.97 FAIL | 4.76 PASS | 50.96 FAIL | -0.78 FAIL | -0.52 FAIL |
| 1714B | LO | 1/8 | REJECT | -0.10 FAIL | -0.05 FAIL | -6.00 FAIL | -3.37 FAIL | 2.66 PASS | 47.77 FAIL | -1.07 FAIL | -1.37 FAIL |
| 1714C | LS | 1/8 | REJECT | -3.46 FAIL | -1.45 FAIL | -10.23 FAIL | -7.39 FAIL | 3.22 PASS | 37.58 FAIL | -2.38 FAIL | -4.17 FAIL |
| 1714C | LO | 0/8 | REJECT | -3.26 FAIL | -1.22 FAIL | -9.55 FAIL | -6.55 FAIL | 1.69 FAIL | 36.31 FAIL | -3.21 FAIL | -4.06 FAIL |
| 1715P50 | LS | 3/8 | REJECT | 0.80 FAIL | 0.37 FAIL | -5.33 FAIL | -3.02 FAIL | 3.53 PASS | 55.41 PASS | 0.55 FAIL | 0.52 PASS |
| 1715P50 | LO | 0/8 | REJECT | 0.72 FAIL | 0.26 FAIL | -5.74 FAIL | -3.07 FAIL | 1.94 FAIL | 46.50 FAIL | 0.10 FAIL | -1.01 FAIL |
| 1715P70 | LS | 1/8 | REJECT | -0.45 FAIL | -0.17 FAIL | -6.94 FAIL | -4.56 FAIL | 2.72 PASS | 46.50 FAIL | 1.75 FAIL | -0.67 FAIL |
| 1715P70 | LO | 0/8 | REJECT | -0.28 FAIL | -0.08 FAIL | -7.07 FAIL | -4.46 FAIL | 1.43 FAIL | 39.49 FAIL | 0.82 FAIL | -2.59 FAIL |
| 1715P85 | LS | 0/8 | REJECT | -1.43 FAIL | -0.36 FAIL | -9.02 FAIL | -7.12 FAIL | 1.73 FAIL | 39.49 FAIL | 2.90 FAIL | -1.65 FAIL |
| 1715P85 | LO | 0/8 | REJECT | -2.27 FAIL | -0.45 FAIL | -10.72 FAIL | -8.27 FAIL | 0.90 FAIL | 24.84 FAIL | 2.39 FAIL | -3.29 FAIL |

Green-week gate = observed >= 55% AND observed > null mean (null P95 and p shown in the cell table). Verdict rule (PREREG): all 8 gates -> PASS; mean<=0 or ex-top-5%<=0 -> REJECT; mean>0 with t<2 -> UNDETERMINED.

### Weekly table (weekly sum of net in R = 25 bp; empty weeks count 0)

| cell | side | weeks | P10 (R) | green | >= +5R share | n strong | median gap (wk) | P90 gap (wk) |
|---|---|---|---|---|---|---|---|---|
| 1714A | LS | 157 | -3.92 | 40.8% | 5.7% | 9 | 4.5 | 38.6 |
| 1714A | LO | 157 | -2.73 | 40.8% | 3.2% | 5 | 14.5 | 79.2 |
| 1714B | LS | 157 | -3.73 | 51.0% | 7.6% | 12 | 7.0 | 27.0 |
| 1714B | LO | 157 | -2.38 | 47.8% | 3.8% | 6 | 7.0 | 53.2 |
| 1714C | LS | 157 | -3.45 | 37.6% | 5.1% | 8 | 4.0 | 42.0 |
| 1714C | LO | 157 | -2.49 | 36.3% | 3.8% | 6 | 3.0 | 56.2 |
| 1715P50 | LS | 157 | -3.16 | 55.4% | 7.0% | 11 | 7.0 | 21.4 |
| 1715P50 | LO | 157 | -1.98 | 46.5% | 3.2% | 5 | 18.0 | 57.7 |
| 1715P70 | LS | 157 | -3.14 | 46.5% | 6.4% | 10 | 4.0 | 27.4 |
| 1715P70 | LO | 157 | -1.64 | 39.5% | 3.2% | 5 | 18.0 | 57.7 |
| 1715P85 | LS | 157 | -2.40 | 39.5% | 5.1% | 8 | 3.0 | 45.2 |
| 1715P85 | LO | 157 | -1.50 | 24.8% | 2.5% | 4 | 7.0 | 80.6 |

`scripts/cadence_bar.py` takes a trades CSV (date+pnl_R) with its own split/TEST guards and a full multi-criterion spec; not invoked, the weekly table is computed here.

### 1,716 report-only

OLS r13 (bps) on r1 (bps) and r12 (bps), HC1 t: | const | b(r1) | t | b(r12) | t | R2 | n |
|---|---|---|---|---|---|---|
| -0.07 | -0.0449 | -0.73 | 0.0342 | 1.53 | 0.00882 | 748 |

Decile of r1 (own-split deciles) -> mean r13 (gross bps); corr(decile, mean r13) = -0.18 (monotonic = +1):

| decile | mean r1 bp | n | mean r13 bp | t |
|---|---|---|---|---|
| 1 | -97.4 | 75 | 4.68 | 0.45 |
| 2 | -52.8 | 75 | -0.52 | -0.09 |
| 3 | -29.7 | 75 | 0.24 | 0.05 |
| 4 | -14.8 | 74 | -2.49 | -0.72 |
| 5 | -1.8 | 75 | -0.19 | -0.06 |
| 6 | 10.4 | 75 | -2.53 | -0.66 |
| 7 | 20.7 | 74 | -2.41 | -0.63 |
| 8 | 33.6 | 75 | 7.93 | 1.38 |
| 9 | 52.9 | 75 | -6.67 | -1.38 |
| 10 | 93.2 | 75 | 1.06 | 0.21 |

Per-year (net bps, LS): unconditional mean r13 beside 1714A and 1715P70:

| year | days | mean r13 (all days) | 1714A n | net | t | 1715P70 n | net | t |
|---|---|---|---|---|---|---|---|---|
| 2020 | 247 | 2.38 | 245 | 0.39 | 0.09 | 133 | -1.79 | -0.25 |
| 2021 | 251 | -2.42 | 251 | -3.96 | -2.45 | 113 | -3.36 | -1.24 |
| 2022 | 250 | -0.17 | 249 | -2.48 | -0.93 | 181 | 2.35 | 0.79 |

SPY 20-day realised vol terciles (TRAIN cut points 7.5% / 11.3% annualised), LS net bps:

| cell | vol tercile | n | net | t |
|---|---|---|---|---|
| 1714A | low | 32 | -7.56 | -1.61 |
| 1714A | mid | 137 | -3.31 | -1.66 |
| 1714A | high | 576 | -1.42 | -0.65 |
| 1715P70 | low | 10 | -8.12 | -1.33 |
| 1715P70 | mid | 60 | -10.67 | -2.99 |
| 1715P70 | high | 357 | 1.48 | 0.47 |

## Verdict

All 12 VAL rows (6 cells x LS/LO) on SPY and QQQ are REJECT (mean net <= 0 or ex-top-5 % <= 0). Primary cells on SPY VAL: 1714A LS net -3.61 bp (t -2.29, gross -1.61 bp), 1715P70 LS net -0.67 bp (t -0.26, gross +1.33 bp). QQQ agrees in sign (1714A -2.03 bp, 1715P70 -0.45 bp). Only 1715P50 LS and 1714B LS on QQQ show a positive VAL mean (+0.5 to +0.8 bp, t < 0.4): below the +2 bp bar and far below the MDE. TEST never loaded. This is a statement about this test: SPY/QQQ 1-min Alpaca SIP, 2016-2022, entry 15:31 open, exit 15:59 close, 1 bp/side, daily event design; it is not a statement about other indices, other windows or the published 1993-2013 sample.

## Caveats (read as an adversary)

* **Sign of the effect is not momentum here.** 1714A gross is -0.40 bp (TRAIN, SPY) and -1.61 bp (VAL, SPY): the sign of r1 does not predict the last half-hour in 2016-2022; the OLS slope on r1 is +0.012 (t 0.3) on TRAIN and -0.136 (t -1.1) on VAL, R2 1.8 % either way (the TRAIN r12 coefficient t 2.0 is the day-so-far, not the first half-hour). Decile monotonicity corr is -0.45 (TRAIN) / -0.49 (VAL): mildly reversing, not significant per decile (VAL decile 1 +7.95 bp t 0.74). The published effect (t ~ 4 in 1993-2013) did not replicate in this window; the published window ends before it, so this is not a contradiction of the paper, only of persistence.
* **Cost is not what kills it.** Gross is already <= +1.3 bp on every cell at VAL; the 2 bp round trip turns a zero into a negative. With zero cost 1715P70 LS would be +1.33 bp (t ~ 0.5): inside the noise (se ~ 2.6 bp at n 395).
* **15:31 open obtainability.** A market order at 15:30:30 fills at the next print, not at the bar open; the 1 bp/side charge is an assumption, not a measurement (no NBBO in this cache). In 2020 (March) and 2022 the SPY spread at 15:30 was wider than 1 bp on stress days; a measured cost can only push the net lower. The 15:31 bar OPEN is a price offered inside that bar, but not necessarily reachable at the instant the signal bar (15:29, r12) closes at 15:30:00.
* **15:59 close is a proxy for the MOC print.** The official close auction price can differ from the 15:59 bar close by several bp on imbalance days (esp. index rebalances, quad-witching, month-ends). Not measured here (no auction prints in the cache). Direction of the error is unknown; it would add noise, not a systematic gain, so it cannot rescue a negative mean.
* **Early closes / half days.** Excluded via the < 380-bar rule (SPY: 9 TRAIN, 7 VAL; QQQ: 12 TRAIN, 8 VAL). The QQQ TRAIN count is higher than SPY by 3 days: QQQ has a few full days with < 380 bars in the cache (cache gaps, not half-days); no day was excluded for a missing key bar. Exclusion is based on bar count only, not outcome; availability gap not winner/loser-tested because the rows are tiny (< 1.2 % of days).
* **Year concentration.** Per-year LS net for 1714A is negative in all three VAL years (2020 -2.0, 2021 -2.7, 2022 -6.1 bp); 2022 carries the t (t -2.67 alone). 1715P70 is -0.9 / -4.0 / +1.3 bp, sign flips with the year. No positive year carries a result (there is none to carry).
* **Regime coverage.** VAL is almost all in the top SPY-vol tercile of the TRAIN cut points (579 of 749 days for 1714A): the TRAIN-frozen tercile cuts (7.5 % / 11.3 % annualised) describe the 2016-19 low-vol period, so the vol table in VAL is nearly one cell. VAL is a high-vol regime (COVID, 2022 bear); the hedging-flow story predicts the effect to be strongest there and it is not.
* **Statistics.** iid t beside a year-block bootstrap on 3 blocks (crude, shown only because the PREREG asks) and a month-block bootstrap (36 blocks, the better one). Day-level dependence is small at one trade per day. Green-week null: observed 40 % vs null 42 % (p 0.71) for 1714A SPY LS: no better than random direction. Long-only vs the null subset: LO is below the null in every cell (buy the drift only on the same days earns ~ nothing after 2 bp).
* **Programme multiplicity.** This builder adds 6 cells x 2 sides x 2 symbols = 24 rows on VAL (programme count 1,713 -> 1,716 by PREREG; the extra variants are reported, not chosen). Nothing here is selected; every row fails.
* **Weekly table.** Weekly P10 -2.2 to -4.0 R on all LS rows (R = 25 bp, one sd of r13): the book would bleed -2 to -4 R in a bad week with no strong weeks (3 weeks >= +5R in 157 for 1714A SPY); cadence bar not approachable. `scripts/cadence_bar.py` not run (takes its own trades CSV and TEST guard; weekly stats computed in this builder).
* **Independent rebuild.** Not done: the PREREG ladder requires a pass before a rebuild; there is no pass, so no ladder step, no TEST read, nothing to the owner as an edge. The reject is a claim about this one design; an adequacy review (e.g. entry at 15:30 instead of 15:31, other half-hour definitions, an earlier window) is NOT done and would be a different PREREG.
