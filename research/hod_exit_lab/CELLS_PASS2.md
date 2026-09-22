# CELLS PASS 2 -- HOD-break trade-level filters, entry mechanics, overnight runner, slots

Cells 1,380-1,388 per `PREREG_PASS2.md`. Scored via `score_pass2.py`, importing B0/paired-stat/cadence helpers from `score_cells.py` (pass 1) unchanged. Programme count after this pass: 1,388.

B0 population: 12010 trades (TRAIN=7315, VAL=4695). B0 TRAIN mean net R = -0.2212, B0 VAL mean net R = -0.3043.

## Cells
| id | name | kind | train_dR | val_dR | val_t_dR | h1_dR | h2_dR | extop5_ok | mdd_ok | val_fills_wk | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | drop r_pct < 1.5% (cost-gate) | cohort | 0.040 | 0.055 | -0.27 | 0.044 | 0.030 | True | True | 135.55 | fail |
| T2 | drop r_pct < 2.5% (cost-gate, dose check) | cohort | 0.057 | 0.129 | 0.57 | 0.076 | 0.004 | True | True | 49.68 | fail |
| M1 | symbol had >=1 HOD-break signal in prior 5 sessions (any out | cohort | 0.055 | 0.107 | 1.11 | 0.119 | -0.000 | True | True | 84.27 | fail |
| M2 | symbol's prior-5-session signals net POSITIVE under B0 | cohort | 0.039 | 0.064 | 1.12 | 0.080 | 0.007 | True | True | 32.86 | fail |
| E1 | retest entry within 15 min, else no trade | exit | 0.130 | 0.118 | 18.83 | 0.148 | 0.109 | True | True | n/a | **PASS** |
| O1 | overnight hold on close-at-high strength (else B0) | exit | -0.023 | 0.007 | 1.41 | -0.041 | -0.003 | False | True | n/a | fail |
| O2 | O1 with exit at next day 10:00 open instead of 09:30 | exit | 0.000 | 0.000 | n/a | 0.000 | 0.000 | True | True | n/a | fail |
| S1 | 8 concurrent slots instead of 4 (same first-12/day cap) | cohort | -0.081 | -0.000 | 0.02 | -0.090 | -0.067 | False | True | 55.55 | fail |
| S2 | daily cap 20 instead of 12, 4 concurrent (same as baseline) | cohort | -0.048 | -0.015 | 0.31 | -0.032 | -0.055 | False | True | 53.86 | fail |

- **O1** note: eod_pop=2239 held=530 missing_daily_bar=0 gap_anomaly=0 (fell back to B0)
- **O2** note: VOID -- not computable under permitted data (no fresh DB pass allowed): bars_sip.db is scoped to (symbol,day) pairs that had a HOD-break signal, not a full intraday market cache, so the next trading day is essentially never itself a signal-day for the same symbol. 0/530 next-day 10:00 bars were found; all 530 fell back to B0, which is why train_dR/val_dR read exactly 0.0 -- this is a data gap, not a tested negative result. O2 needs a genuine next-day minute-bar source before it can be scored.
- **S1** note: rule5 (dropped-cohort<0) NOT applied: PREREG scopes the D/W dropped clause to T/M cells; slot overflow is random-equivalent by mechanism.
- **S2** note: rule5 (dropped-cohort<0) NOT applied: PREREG scopes the D/W dropped clause to T/M cells; slot overflow is random-equivalent by mechanism.

## Report-only: T1/T2 dose curve by r_pct decile (B0, all trades)
| decile (r_pct range) | mean r_pct | mean net_R | n |
|---|---|---|---|
| (0.999, 1.129] | 1.06 | -0.341 | 1202 |
| (1.129, 1.264] | 1.19 | -0.413 | 1200 |
| (1.264, 1.419] | 1.34 | -0.274 | 1201 |
| (1.419, 1.599] | 1.51 | -0.332 | 1201 |
| (1.599, 1.802] | 1.70 | -0.252 | 1201 |
| (1.802, 2.051] | 1.92 | -0.203 | 1201 |
| (2.051, 2.38] | 2.21 | -0.194 | 1201 |
| (2.38, 2.779] | 2.57 | -0.175 | 1201 |
| (2.779, 3.352] | 3.05 | -0.206 | 1201 |
| (3.352, 4.528] | 3.77 | -0.145 | 1201 |

## Report-only: M1 by count of prior-5-session signals (B0 net_R)
| prior_count (capped 5) | mean net_R | n |
|---|---|---|
| 0 | -0.297 | 7576 |
| 1 | -0.185 | 2956 |
| 2 | -0.148 | 1069 |
| 3 | -0.129 | 337 |
| 4 | -0.638 | 68 |
| 5 | -1.100 | 4 |

## Report-only: E1 unfilled-counterfactual cohort (B0 outcome of signals that never retested)
- TRAIN: n=1426, mean B0 net_R=0.119
- VAL: n=992, mean B0 net_R=-0.039

## Report-only: E1 full-book comparison (retest book vs B0 book, unpaired)
- TRAIN: E1 n=5870 mean=-0.175  |  B0 n=7315 mean=-0.221
- VAL: E1 n=3682 mean=-0.258  |  B0 n=4695 mean=-0.304

## Report-only: O1 overnight gap distribution (held subset, gap dR = variant - B0)
- n held = 530, mean dR = -0.255, P1 = -4.854, worst night = ETHT 2025-10-13 dR=-10.452

## Report-only: O2 overnight gap distribution (held subset, gap dR = variant - B0)
- n held = 0, mean dR = n/a, P1 = n/a, worst night = None None dR=n/a

## Report-only: S1/S2 fills/week and weekly P10 (VAL, kept cohort)
- S1: fills/week=55.55, weekly P10=-33.138
- S2: fills/week=53.86, weekly P10=-31.217

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
### E1
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -71.78 R  min -119.25 R  MDD 950.07 R   under-water max 22 wk   [fail]
C4 green     5%  null 50%                   [fail]
C5 fills/wk  167.36                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -951.99 R   capped -950.07 R   top-5 share -0.2%   weekly P&L histogram: [-119.2, -79.6, -72.1, -69.0, -66.2, -64.6, -60.6, -50.2, -45.3, -44.6, -37.7, -37.5, -37.3, -33.0, -27.8, -25.1, -24.5, -22.7, -19.3, -12.8, -2.9, 1.9]

```