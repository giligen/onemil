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
