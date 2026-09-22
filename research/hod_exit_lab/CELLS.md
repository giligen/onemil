# CELLS — HOD Exit Lab pass 1 (X1-X12, D1-D5, W1-W2)

B0 population: 12010 trades (TRAIN=7315, VAL=4695). B0 TRAIN mean net R = -0.2212, VAL mean net R = -0.3043.

## Exit cells (X1-X12), paired ΔR vs B0
| cell | rule | train ΔR | val ΔR | val t | H1 ΔR | H2 ΔR | ex-top5 ok | MDD ok | PASS |
|---|---|---|---|---|---|---|---|---|---|
| X1 | no target: stop or 15:55 only | -0.000 | -0.005 | -1.026 | -0.003 | 0.003 | False | True | fail |
| X2 | target +1R | -0.042 | -0.016 | 0.159 | -0.052 | -0.031 | True | True | fail |
| X3 | target +3R | 0.014 | 0.003 | -0.024 | 0.012 | 0.016 | False | True | fail |
| X4 | target +5R | -0.001 | 0.006 | -0.299 | -0.009 | 0.008 | False | True | fail |
| X5 | breakeven lock @ +1R | -0.017 | -0.018 | -1.290 | -0.022 | -0.012 | False | True | fail |
| X5b | ORB-style lock: +1.5R trig -> +0.5R | -0.006 | -0.014 | -1.446 | -0.009 | -0.003 | False | True | fail |
| X6 | trailing stop, MFE-1R once armed @1R | -0.028 | -0.020 | -0.770 | -0.036 | -0.019 | False | True | fail |
| X7 | time stop 60min, exit if close<entry | -0.027 | -0.004 | -0.323 | -0.047 | -0.004 | False | True | fail |
| X7b | time stop 30min, exit if close<entry+0.25R | -0.036 | -0.011 | -0.586 | -0.039 | -0.034 | False | True | fail |
| X8 | 50% @+1R, runner=breakeven-lock no target | -0.033 | -0.009 | -0.379 | -0.046 | -0.019 | False | True | fail |
| X9 | VWAP exit: close<VWAP -> next open | -0.002 | -0.016 | -1.128 | 0.006 | -0.010 | False | True | fail |
| X10 | clock exit @ 12:00 open | -0.097 | -0.001 | -0.606 | -0.182 | -0.002 | False | False | fail |
| X11 | stop widened 1.5x | -0.008 | 0.000 | -0.827 | -0.006 | -0.011 | False | True | fail |
| X12 | stop tightened 0.75x | -0.008 | -0.012 | -0.474 | -0.015 | -0.000 | False | True | fail |

## Cohort cells (D1-D5, W1-W2), kept cohort vs full-B0-mean ΔR
| cell | rule | train ΔR | val ΔR | val t | H1 ΔR | H2 ΔR | val fills/wk | dropped R (tr/val) | ex-top5 ok | MDD ok | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | SPY above 09:30 open at signal minute | 0.024 | 0.014 | -2.965 | 0.014 | 0.031 | 147.591 | -0.281/-0.335 | True | True | fail |
| D2 | breadth <= TRAIN-H1 median (21 prior mkt-wide signals) | -0.061 | 0.026 | 0.492 | -0.091 | -0.028 | 101.682 | -0.131/-0.328 | False | True | fail |
| D3 | exclude entries 12:00-14:00 ET | -0.109 | 0.011 | -1.351 | -0.227 | 0.003 | 190.045 | 0.184/-0.394 | False | True | fail |
| D4 | HMM calm state (hmm_state==0) on the day | -0.031 | -0.021 | -1.748 | -0.060 | -0.000 | 204.200 | -0.162/-0.162 | False | True | fail |
| D5 | first signal of day per symbol only (n_prior==0) | -0.761 | 0.001 | 0.001 | -0.788 | NA | 1.000 | -0.208/-0.304 | False | True | fail |

> **D5 CAVEAT (VOID)**: n_prior is capped at 20 for 81% of features.csv rows (only 0.08% of VAL rows show n_prior==0), so the kept cohort is near-empty (VAL n=4, 1.0 fills/wk) — not a meaningful test of the first-of-day-only hypothesis. Data-quality issue in the n_prior field, not a clean null.
| W1 | week gate: trade wk t only if wk t-1 B0 net R > 0 | -0.024 | 0.002 | -0.596 | -0.247 | 0.096 | 150.000 | -0.219/-0.304 | False | True | fail |
| W2 | daily kill: no new entries after day realized R <= -3 | -0.121 | 0.034 | -2.529 | -0.226 | -0.012 | 83.091 | -0.090/-0.326 | False | True | fail |

## Cadence bar (scripts/cadence_bar.py --split VAL), B0 + passing cells
### B0
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -103.61 R  min -163.06 R  MDD 1428.58 R   under-water max 22 wk   [fail]
C4 green     5%  null 50%                   [fail]
C5 fills/wk  213.41                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -1430.10 R   capped -1428.58 R   top-5 share -0.1%   weekly P&L histogram: [-163.1, -118.7, -103.9, -100.9, -98.4, -98.1, -90.2, -83.2, -76.6, -60.9, -58.2, -57.7, -53.6, -48.4, -48.2, -45.4, -35.8, -33.4, -32.1, -18.9, -4.4, 1.5]

```

## Report-only: day-of-week (B0, all splits kept)
| day | mean net R | n |
|---|---|---|
| Monday | -0.353 | 2604 |
| Tuesday | -0.297 | 2023 |
| Wednesday | -0.092 | 2908 |
| Thursday | -0.239 | 2217 |
| Friday | -0.322 | 2258 |

## Report-only: time-of-day buckets
| bucket | mean net R | n |
|---|---|---|
| 09:30-09:45 | -0.262 | 780 |
| 09:45-10:00 | -0.261 | 2405 |
| 10:00-11:00 | -0.358 | 5189 |
| 11:00-13:00 | -0.263 | 2263 |
| 13:00+ | 0.174 | 1373 |

## Report-only: entry-minute deciles
| decile (entry_m range) | mean net R | n |
|---|---|---|
| (576.999, 587.0] | -0.276 | 1382 |
| (587.0, 594.0] | -0.242 | 1123 |
| (594.0, 602.0] | -0.265 | 1143 |
| (602.0, 612.0] | -0.170 | 1205 |
| (612.0, 620.0] | -0.679 | 1179 |
| (620.0, 635.4] | -0.386 | 1174 |
| (635.4, 661.0] | -0.243 | 1215 |
| (661.0, 705.0] | -0.287 | 1203 |
| (705.0, 798.0] | -0.238 | 1193 |
| (798.0, 841.0] | 0.245 | 1193 |
