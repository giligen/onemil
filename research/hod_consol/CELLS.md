# CELLS.md -- base-under-the-high (PREREG.md, cells 1,400-1,402)

Population: research/hod_pmh_causal/pm_candidates.csv (23,767 ORB wide-seed causal symbol-days, gap>=3%/open $3-50/prior-vol>=500K, all knowable at 09:30:00 ET). Signal: FIRST 1-min bar 10:00-14:00 ET pressing a >=20-min-old high with a base <=6% deep, cost-gated >=1.5%. TRAIN=2025 (halves split 2025-07-02), VAL=2026-01..05, TEST sealed. Cost = PROXY 15bp half-spread both legs + 2bp/side slip (no measured NBBO for these minutes, per PREREG) -- any pass is re-scored on measured Alpaca NBBO before being reported as a pass.

## Placebos

D1: same name-day, seeded random minute in 10:00-14:00, stop = min low of the PRIOR 20 bars (excludes the draw bar itself), same 1.5% floor, <=5 redraws per draw else that draw is dropped, C1 exit. D3: a random OTHER signal-producing symbol on the SAME DATE (pool = other signal symbol-days that date -- paths are only cached where a signal fired, not the full raw candidate universe), entered at the real signal's own clock minute, its own prior-20-bar-low stop and floor, C1 exit. Both placebos ALWAYS use the C1 (target +2R) exit rule regardless of which cell is being tested, per PREREG. N_DRAWS=8 seeded draws per VAL trade.

## Cells

| cell | name | train_n | val_n | train_R | val_R | val_t | h1_R | h2_R | extop5_tr | extop5_val | fills/wk(VAL,4c/12d) | D1(VAL) | D3(VAL) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1400 | C1 target +2R, stop, 15:55 | 3958 | 2289 | -0.060 | 0.111 | -0.169 | -0.092 | -0.031 | -0.164 | 0.016 | 33.59 | 0.080 | 0.015 | False |
| 1401 | C2 no target, stop, 15:55 | 3946 | 2264 | -0.015 | 0.096 | -0.926 | -0.064 | 0.030 | -0.278 | -0.131 | 27.64 | 0.080 | 0.046 | False |
| 1402 | C3 breakeven lock at +1R, no target, 15:55 | 3955 | 2276 | -0.037 | 0.074 | -1.208 | -0.098 | 0.018 | -0.288 | -0.147 | 31.18 | 0.075 | 0.021 | False |

## Pass-bar rule detail per cell

- **1400** (C1 target +2R, stop, 15:55): {'rule_R': False, 'rule_t': False, 'rule_halves': False, 'rule_et5': False, 'rule_fills': True, 'rule_d1': False, 'rule_d3': False} | D1 coverage 87.6% | D3 (universe pool, pass bar) coverage 96.0% | D3 signal-pool diagnostic 0.235 | stop-at-open VAL trades 0
- **1401** (C2 no target, stop, 15:55): {'rule_R': False, 'rule_t': False, 'rule_halves': False, 'rule_et5': False, 'rule_fills': True, 'rule_d1': False, 'rule_d3': False} | D1 coverage 88.2% | D3 (universe pool, pass bar) coverage 95.8% | D3 signal-pool diagnostic 0.246 | stop-at-open VAL trades 0
- **1402** (C3 breakeven lock at +1R, no target, 15:55): {'rule_R': False, 'rule_t': False, 'rule_halves': False, 'rule_et5': False, 'rule_fills': True, 'rule_d1': False, 'rule_d3': False} | D1 coverage 88.3% | D3 (universe pool, pass bar) coverage 95.8% | D3 signal-pool diagnostic 0.240 | stop-at-open VAL trades 0

## Cadence bar (scripts/cadence_bar.py --split VAL), per cell

### 1400
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 1.0 wk  P90 3.0 wk        [pass]   gaps: [1, 3, 1, 1, 2, 1, 4, 3, 1, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 82% [pass]
C3 reds      P10 -21.69 R  min -45.20 R  MDD 108.46 R   under-water max 9 wk   [fail]
C4 green     55%  null 50%                   [fail]
C5 fills/wk  104.05                             [pass]
C6 tail      C6 not audited
C7 power     cycles 11   bootstrap P90-gap 75% UB 3.4 wk    [pass]
diagnostics  ex-top-5% 99.45 R   capped -121.05 R   top-5 share 61.0%   weekly P&L histogram: [-45.2, -37.1, -21.9, -19.9, -18.1, -10.3, -8.3, -7.4, -6.9, -6.0, 7.4, 8.6, 10.3, 12.1, 12.3, 18.7, 28.5, 33.7, 44.6, 45.4, 58.8, 155.3]
```
### 1401
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.0 wk  P90 4.3 wk        [pass]   gaps: [1, 3, 1, 3, 5, 4, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 50% [fail]
C3 reds      P10 -34.54 R  min -43.77 R  MDD 142.19 R   under-water max 15 wk   [fail]
C4 green     43%  null 51%                   [fail]
C5 fills/wk  102.91                             [pass]
C6 tail      C6 not audited
C7 power     cycles 8   bootstrap P90-gap 75% UB 4.6 wk    [fail]
diagnostics  ex-top-5% 9.77 R   capped -217.86 R   top-5 share 95.5%   weekly P&L histogram: [-43.8, -36.6, -35.3, -27.3, -26.4, -22.6, -20.2, -19.1, -15.7, -7.5, -6.0, -2.7, 0.4, 9.1, 14.2, 15.9, 21.1, 26.3, 51.0, 54.7, 80.3, 207.4]
```
### 1402
```
CADENCE BAR  (unknown, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 2.0 wk  P90 4.3 wk        [pass]   gaps: [1, 3, 1, 3, 5, 4, 1, 1]
C2 bleed     P90 0.00 R     cycles net>0 50% [fail]
C3 reds      P10 -33.37 R  min -46.23 R  MDD 157.85 R   under-water max 16 wk   [fail]
C4 green     48%  null 50%                   [fail]
C5 fills/wk  103.45                             [pass]
C6 tail      C6 not audited
C7 power     cycles 8   bootstrap P90-gap 75% UB 4.8 wk    [fail]
diagnostics  ex-top-5% -22.54 R   capped -215.98 R   top-5 share 113.3%   weekly P&L histogram: [-46.2, -40.4, -33.6, -31.2, -26.8, -21.4, -19.9, -15.6, -12.3, -9.0, -5.3, 0.1, 0.5, 12.7, 12.7, 15.6, 17.6, 18.8, 35.9, 49.6, 75.5, 191.8]
```
