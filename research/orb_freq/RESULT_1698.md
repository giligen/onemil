# RESULT 1,698: six further layers on the ORB stack (base + L1 tilt + L2 add)

Reconstruction: 478 usable fills of 478 in 1694_runners.csv (failures={'no_bars': 0, 'no_i0_match': 0}, idea43 cross-check mismatches=2). OOS2024H2 UNTESTED (n=0, not in this book).

## Layer table (paired dR per fill vs the stack below it; both directions)

| Layer | Dir | n | mean dR | day_t | ex_top5 dR | worst wk $ (base $) | risk x | pass |
|---|---|---|---|---|---|---|---|---|
| L3a_add2R_origStop | TRAIN2025 | 212 | -0.125 | -1.22 | -0.303 | $-6,669 ($-5,156) | n/a | n |
| L3a_add2R_origStop | VAL2026 | 259 | +0.182 | 1.97 | -0.130 | $-5,883 ($-4,739) | n/a | n |
| L3b_add2R_add1BE_stop | TRAIN2025 | 212 | -0.108 | -2.54 | -0.241 | $-6,314 ($-5,156) | n/a | n |
| L3b_add2R_add1BE_stop | VAL2026 | 259 | +0.074 | 1.48 | -0.173 | $-5,508 ($-4,739) | n/a | n |
| L4b_addStop_BE | TRAIN2025 | 212 | -0.053 | -0.36 | -0.211 | $-2,197 ($-5,156) | n/a | n |
| L4b_addStop_BE | VAL2026 | 259 | -0.244 | -1.65 | -0.375 | $-2,428 ($-4,739) | n/a | n |
| L4c_addStop_liveLock | TRAIN2025 | 212 | +0.048 | 0.23 | -0.120 | $-5,999 ($-5,156) | n/a | n |
| L4c_addStop_liveLock | VAL2026 | 259 | -0.022 | -0.97 | -0.091 | $-4,629 ($-4,739) | n/a | n |
| L8_basePartial2R | TRAIN2025 | 212 | +0.058 | 1.14 | -0.002 | $-4,416 ($-5,156) | n/a | Y |
| L8_basePartial2R | VAL2026 | 259 | -0.095 | -2.01 | -0.154 | $-4,446 ($-4,739) | n/a | n |
| L5_budget2x | TRAIN2025 | 212 | +0.066 | 0.76 | -0.064 | $-2,381 ($-5,156) | 0.87x | n |
| L5_budget2x | VAL2026 | 259 | -0.204 | -1.15 | -0.300 | $-3,451 ($-4,739) | 0.75x | n |
| L5_budget3x | TRAIN2025 | 212 | +0.065 | 2.53 | +0.000 | $-3,384 ($-5,156) | 0.96x | Y |
| L5_budget3x | VAL2026 | 259 | -0.110 | -1.58 | -0.166 | $-4,358 ($-4,739) | 0.90x | n |
| L5_budget4x | TRAIN2025 | 212 | +0.012 | 1.64 | +0.000 | $-4,710 ($-5,156) | 0.99x | n |
| L5_budget4x | VAL2026 | 259 | -0.024 | -0.97 | -0.047 | $-4,739 ($-4,739) | 0.96x | n |
| L6_tilt2D | VAL2026 | 259 | -0.038 | 0.04 | -0.427 | $-3,155 ($-4,739) | 1.00x | n |
| L6_tilt2D | TRAIN2025 | 212 | +0.227 | 1.55 | -0.097 | $-5,019 ($-5,156) | 1.05x | n |
| L7_tiltSPY | VAL2026 | 259 | -0.061 | 0.16 | -0.333 | $-5,778 ($-4,739) | 1.01x | n |
| L7_tiltSPY | TRAIN2025 | 212 | +0.241 | 1.57 | -0.031 | $-6,456 ($-5,156) | 0.96x | n |

L6 2D-tilt rank robustness (9 cells, TRAIN vs VAL fit): spearman=+0.10 (bar >= 0.6)
L7 SPY-tilt rank robustness (3 cells, TRAIN vs VAL fit): spearman=+0.50 (bar >= 0.6)
L5 drawdown candidates (worst-wk cut>=25% at EV cost<=0.02R): none

**Layers joining the final stack:** base+L1 tilt+L2 add (frozen)

## Q3 2026 weekly $: BASE (stack: base+L1 tilt+L2 add) vs FINAL STACK
| Wk Mon | Fills | $Base | $Final |
|---|---|---|---|
| 2026-06-29 | 0 | $0 | $0 |
| 2026-07-06 | 10 | $-4,739 | $-4,739 |
| 2026-07-13 | 12 | $3,215 | $3,215 |
| 2026-07-20 | 15 | $-450 | $-450 |
| 2026-07-27 | 14 | $729 | $729 |
| 2026-08-03 | 7 | $-1,183 | $-1,183 |
| 2026-08-10 | 5 | $-1,068 | $-1,068 |
| 2026-08-17 | 0 | $0 | $0 |
| 2026-08-24 | 6 | $4,543 | $4,543 |
| 2026-08-31 | 7 | $629 | $629 |
| 2026-09-07 | 0 | $0 | $0 |
| 2026-09-14 | 9 | $4,000 | $4,000 |
| 2026-09-21 | 6 | $-654 | $-654 |
| 2026-09-28 | 1 | $121 | $121 |

**Q3 totals**: Base $5,142 (6/14 green, worst wk $-4,739, max DD $4,739) vs Final $5,142 (6/14 green, worst wk $-4,739, max DD $4,739).

**Caveats**: single quarter for the weekly table (n not an OOS claim); L6/L7 multipliers are selection-half cell means on <=9 cells -- thin per-cell n; when >1 layer passes the final stack sums each layer's OWN delta additively (not a jointly re-optimized combination); L5 is reported as a drawdown layer, never counted toward the EV pass bar; tilt and add are layered on the SAME fills throughout (lever-isolation, not independently-selected edges).
