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
