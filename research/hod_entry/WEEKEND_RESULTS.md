# WEEKEND_RESULTS — cells 1,429 / 1,430 / 1,431 / 1,439 / 1,440 / 1,441 / 1,442

Frozen per `research/hod_entry/PREREG_WEEKEND.md` (2026-09-25 18:35 UTC). One table appended per
cell as it completes; full detail in each cell's own RESULT_*.md.

## 1,430 — exits on the winning fills

| variant | n (TRAIN/VAL) | ΔR TRAIN-H2 | ΔR VAL | VAL t | ex-top-5 % (VAL) | worst week (var / B0, R) | verdict |
|---|---|---|---|---|---|---|---|
| (a) 90-min time stop | 1165/1443 | -0.119 (-0.051) | -0.132 (-0.065) | -6.56 | -0.220 | -12.10 / -17.17 | FAIL |
| (b) breakeven lock (+1R→BE) | 1163/1442 | -0.153 (-0.054) | -0.170 (-0.071) | -8.64 | -0.229 | -26.14 / -17.17 | FAIL |
| (c) ORB lock (+1.5R→+0.5R) | 1164/1440 | -0.108 (-0.015) | -0.135 (-0.039) | -8.14 | -0.206 | -17.68 / -17.17 | FAIL |
| (d) 50% scale-out at +2R | 1157/1433 | -0.008 (+0.083) | -0.040 (+0.050) | -1.49 | -0.168 | -20.35 / -20.49 | FAIL |
| (e) VWAP-close after +0.5R | 1163/1442 | -0.091 (-0.021) | -0.096 (-0.029) | -5.92 | -0.152 | -20.93 / -17.17 | FAIL |
| (f) close at 14:30 | 1165/1443 | -0.076 (-0.002) | -0.076 (-0.001) | -6.98 | -0.130 | -24.51 / -17.17 | FAIL |

Parenthetical ΔR = without the new 30 bps stop-slip charge. All six FAIL; (d) closest (see
`RESULT_1430.md`). Full report: `research/hod_entry/RESULT_1430.md`.

## 1,429 — fill-quality sizing

| holdout | charge | n | weighted R/risk | flat R | ΔR | VAL t (diff) | worst wk (weighted/flat) |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | no stop-slip | 1165 | +0.299 | +0.285 | +0.013 | 4.13 | -1.11 / -1.11 |
| TRAIN-H2 | +30bps slip | 1165 | +0.220 | +0.207 | +0.013 | 3.01 | -1.19 / -1.19 |
| VAL | no stop-slip | 1443 | +0.255 | +0.238 | +0.017 | 4.42 | -0.22 / -0.24 |
| VAL | +30bps slip | 1443 | +0.177 | +0.161 | +0.017 | 3.19 | -0.32 / -0.34 |

FAIL: ΔR positive with strong day-clustered t (3.0–4.4) but under the +0.05 pass bar on both
holdouts. Report-only: VAL odd-lot trigger prints (n=917) mean net_R +0.332 vs round-lot (n=526)
+0.074. Full report: `research/hod_entry/RESULT_1429.md`.

## 1,431 — no-fill cohort short

| holdout | D (post break-bar-close filter) | scored n | mean net R (no-slip) | mean net R (30bps slip) | VAL t | share shortable | fills/wk |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 237 | 64 | -1.375 | -1.866 | -9.42 | 0.375 | 2.67 |
| VAL | 324 | 93 | -1.058 | -1.438 | -7.90 | 0.376 | 4.43 |

FAIL, decisively: mean net R deeply negative both holdouts (bar +0.10), VAL t strongly negative
(bar +2), shortable share 0.375/0.376 misses the 60% bar too. R had to be floored at 0.5% of price
(unfloored PREREG text let R collapse to 1-2 ticks for signals whose next-bar open passed the
break-bar high, exploding R-multiples to ±10^13 — 501 rows dropped, WARNING-logged). No shortable
column existed in the PREREG's named asset file; used `borrow_flags.csv` instead (documented in
`RESULT_1431.md`). Full report: `research/hod_entry/RESULT_1431.md`.

## 1,442 — tape-triggered override

| holdout | (a) BROKER n / mean R (slip) | (b) OVERRIDE n / mean R (slip) | fill rate base/a/b | (b)−(a) paired n | ΔR | VAL t |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 994 / +0.236 | 813 / +0.278 | 0.335 / 0.292 / 0.240 | 723 | -0.016 | -5.77 |
| VAL | 1233 / +0.183 | 985 / +0.226 | 0.306 / 0.270 / 0.215 | 879 | -0.016 | -5.13 |

FAIL, decisively: (b)-(a) is negative both holdouts (bar was +0.03), t < -5, and OVERRIDE's own
fill rate (0.24/0.22) is below both BROKER's and E1's baseline — 300ms of waiting loses more fills
than it improves prices. Full report: `research/hod_entry/RESULT_1442.md`.
