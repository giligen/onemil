# CELLS.md -- PMH-break (PREREG_PMH.md, cells 1,389-1,392)

Population: 4,810 PMH-break candidates over the HOD-universe symbol-days; TRAIN=2025 (halves split at 2025-07-02, no prior convention found in this repo for a fresh population -- stated explicitly, a deviation to flag), VAL=2026-01..05, TEST sealed and untouched. Base cost = measured half-spread at the HOD signal minute of the same name-day (100% NBBO coverage per build.log); the price x hour-bucket band cost (research/fuckup_audit/probe_costs.py COST_CURVE_BPS) is reported ONLY as a sensitivity line below, never as the pass-bar cost, per PREREG_PMH.md.

## D1/D3 placebo caveat

paths.parquet caches ONLY each signal's own entry_m -> 15:55 bars (no pre-break bars are cached for any signal -- confirmed on the raw parquet). D1 is therefore a "random LATER minute on the same name-day" (drawn from the signal's own cached post-break window), not a free uniform-intraday-minute placebo -- WEAKER than the PREREG's literal spec because the post-break window still carries some of the same move; flagged, not silently narrowed. D3 holds entry_m (clock time) fixed and swaps in a random OTHER signal's own name/price path from the first cached bar at or after that clock time; skipped when no candidate has coverage there (coverage % reported per cell). Both average N_DRAWS=8 seeded draws per trade.

## Cells

| cell | name | train_n | val_n | train_R | val_R | val_t | h1_R | h2_R | extop5_tr | extop5_val | fills/wk(VAL,4c/12d) | D1(VAL) | D3(VAL) | band_R(VAL) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1389 | P1 target +2R, stop, 15:55 | 2014 | 1557 | 0.624 | 0.767 | 11.563 | 0.544 | 0.703 | 0.553 | 0.704 | 27.50 | 0.062 | 0.346 | 0.654 | True |
| 1390 | P2 no target, stop, 15:55 | 2012 | 1555 | 0.866 | 0.986 | 9.110 | 0.787 | 0.944 | 0.544 | 0.697 | 21.18 | 0.068 | 0.651 | 0.872 | True |
| 1391 | P3 breakeven lock at +1R, no target, 15:55 | 2013 | 1561 | 0.750 | 0.900 | 9.352 | 0.669 | 0.830 | 0.437 | 0.612 | 22.86 | 0.071 | 0.559 | 0.786 | True |
| 1392 | P4 = P1 restricted to breaks in 09:31-10:00 ET | 1608 | 1265 | 0.706 | 0.822 | 11.319 | 0.654 | 0.756 | 0.639 | 0.761 | 23.18 | 0.065 | 0.375 | 0.699 | True |

## Pass-bar rule detail per cell

- **1389** (P1 target +2R, stop, 15:55): {'rule_R': True, 'rule_t': True, 'rule_halves': True, 'rule_et5': True, 'rule_fills': True, 'rule_d1': True, 'rule_d3': True} | D1 coverage 99.9% | D3 coverage 100.0%
- **1390** (P2 no target, stop, 15:55): {'rule_R': True, 'rule_t': True, 'rule_halves': True, 'rule_et5': True, 'rule_fills': True, 'rule_d1': True, 'rule_d3': True} | D1 coverage 99.9% | D3 coverage 100.0%
- **1391** (P3 breakeven lock at +1R, no target, 15:55): {'rule_R': True, 'rule_t': True, 'rule_halves': True, 'rule_et5': True, 'rule_fills': True, 'rule_d1': True, 'rule_d3': True} | D1 coverage 99.8% | D3 coverage 100.0%
- **1392** (P4 = P1 restricted to breaks in 09:31-10:00 ET): {'rule_R': True, 'rule_t': True, 'rule_halves': True, 'rule_et5': True, 'rule_fills': True, 'rule_d1': True, 'rule_d3': True} | D1 coverage 99.9% | D3 coverage 100.0%

## 1392 (P4) dropped cohort -- breaks OUTSIDE 09:31-10:00 ET

train_R=0.299 (n=406), val_R=0.531 (n=292)

## Cadence bar (scripts/cadence_bar.py --split VAL), per cell

### 1389
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 1.0 wk  P90 1.0 wk        [pass]   gaps: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 100% [pass]
C3 reds      P10 19.38 R  min 13.16 R  MDD 0.00 R   under-water max 0 wk   [pass]
C4 green     100%  null 50%                   [pass]
C5 fills/wk  70.77                             [pass]
C6 tail      C6 not audited
C7 power     cycles 21   bootstrap P90-gap 75% UB 1.0 wk    [pass]
diagnostics  ex-top-5% 1070.33 R   capped 110.00 R   top-5 share 10.4%   weekly P&L histogram: [13.2, 17.8, 19.1, 22.1, 27.9, 34.6, 36.5, 36.8, 39.9, 39.9, 40.9, 43.6, 48.4, 55.6, 64.2, 66.0, 70.5, 89.7, 96.8, 98.2, 108.6, 124.6]
```
### 1390
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 1.0 wk  P90 1.0 wk        [pass]   gaps: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 100% [pass]
C3 reds      P10 22.49 R  min 14.37 R  MDD 0.00 R   under-water max 0 wk   [pass]
C4 green     100%  null 50%                   [pass]
C5 fills/wk  70.68                             [pass]
C6 tail      C6 not audited
C7 power     cycles 21   bootstrap P90-gap 75% UB 1.0 wk    [pass]
diagnostics  ex-top-5% 1363.39 R   capped 110.00 R   top-5 share 11.1%   weekly P&L histogram: [14.4, 15.9, 21.6, 30.9, 32.3, 33.1, 34.5, 37.8, 39.3, 50.1, 51.0, 52.5, 65.9, 72.0, 73.9, 85.7, 111.8, 118.7, 119.8, 133.6, 168.7, 169.6]
```
### 1391
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 1.0 wk  P90 1.0 wk        [pass]   gaps: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 100% [pass]
C3 reds      P10 16.70 R  min 8.48 R  MDD 0.00 R   under-water max 0 wk   [pass]
C4 green     100%  null 50%                   [pass]
C5 fills/wk  70.95                             [pass]
C6 tail      C6 not audited
C7 power     cycles 21   bootstrap P90-gap 75% UB 1.0 wk    [pass]
diagnostics  ex-top-5% 1235.82 R   capped 110.00 R   top-5 share 12.0%   weekly P&L histogram: [8.5, 15.1, 16.3, 20.7, 29.7, 29.8, 33.3, 38.0, 40.1, 45.1, 49.3, 49.6, 61.7, 72.8, 73.8, 78.7, 94.5, 101.6, 115.5, 125.5, 136.1, 168.3]
```
### 1392
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 1.0 wk  P90 1.0 wk        [pass]   gaps: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 100% [pass]
C3 reds      P10 21.98 R  min 17.07 R  MDD 0.00 R   under-water max 0 wk   [pass]
C4 green     100%  null 50%                   [pass]
C5 fills/wk  57.50                             [pass]
C6 tail      C6 not audited
C7 power     cycles 21   bootstrap P90-gap 75% UB 1.0 wk    [pass]
diagnostics  ex-top-5% 930.12 R   capped 110.00 R   top-5 share 10.6%   weekly P&L histogram: [17.1, 17.2, 22.0, 22.1, 22.2, 26.1, 29.9, 31.1, 32.1, 33.3, 36.8, 37.9, 49.9, 53.2, 54.2, 54.9, 56.7, 80.9, 82.2, 85.0, 85.3, 109.8]
```
