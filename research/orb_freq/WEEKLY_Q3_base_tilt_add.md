# Q3 2026 weekly: BASE vs BASE+TILT vs BASE+TILT+ADD(idea43), ORB production book

| Wk | Mon | Fills | SumR base | $Base | $Tilt | $Add | GrnB | GrnT | GrnA |
|---|---|---|---|---|---|---|---|---|---|
| 27 | 06-29 | 0 | 0.00 | $0 | $0 | $0 | - | - | - |
| 28 | 07-06 | 10 | -5.54 | -$2,077 | -$2,385 | -$4,739 | N | N | N |
| 29 | 07-13 | 12 | +2.71 | $1,015 | $2,031 | $3,215 | Y | Y | Y |
| 30 | 07-20 | 15 | -2.25 | -$842 | -$674 | -$450 | N | N | N |
| 31 | 07-27 | 14 | +6.87 | $2,576 | $1,508 | $729 | Y | Y | Y |
| 32 | 08-03 | 7 | -3.20 | -$1,199 | -$1,303 | -$1,183 | N | N | N |
| 33 | 08-10 | 5 | -1.32 | -$494 | -$793 | -$1,068 | N | N | N |
| 34 | 08-17 | 0 | 0.00 | $0 | $0 | $0 | - | - | - |
| 35 | 08-24 | 6 | +2.78 | $1,042 | $1,928 | $4,543 | Y | Y | Y |
| 36 | 08-31 | 7 | -0.26 | -$96 | $456 | $629 | N | Y | Y |
| 37 | 09-07 | 0 | 0.00 | $0 | $0 | $0 | - | - | - |
| 38 | 09-14 | 9 | +2.44 | $914 | $1,887 | $4,000 | Y | Y | Y |
| 39 | 09-21 | 6 | -1.48 | -$556 | -$264 | -$654 | N | N | N |

**Quarter totals (91 fills, 13 ISO weeks)**

| Variant | Total $ | Mean R-equiv/fill | Green wks/13 | Worst wk $ | Max DD $ |
|---|---|---|---|---|---|
| Base | $283 | +0.0083 | 4 | -$2,077 | $2,077 |
| +Tilt | $2,390 | +0.0700 | 5 | -$2,385 | $2,385 |
| +Tilt+Add(idea43) | $5,021 | +0.1471 | 5 | -$4,739 | $4,739 |

Tilt realised risk: Sum(multiplier)/n = **1.0714x** base (91 fills: 34 low=1.5x, 33 mid=1.0x, 21 high=0.5x, 3 blank-tercile=1.0x neutral).

**Definitions/caveats**
- BASE R = `A3_noTarget_liveLock` in `1694_runners.csv`, confirmed byte-identical to production `E1_production` (verified in `1694_money.py`); $ = R x $375.
- Tilt edges applied in **research-ratio units** (0.03716/0.07649) via the precomputed, 100%-parity-checked `bin_rvol_0935` column (`PARITY_1697_tilt.md`) -- the per-fill file carries the raw ratio, not engine units (2.898/5.966). 1.5x low / 1.0x mid / 0.5x high; 3/91 fills NaN-ADV20 (blank tercile) given neutral 1.0x, flagged not imputed.
- Clamp (Q5, total <=1.5x) is a **no-op** here: no `adaptive_mult`/`pm_mult` column exists in the per-fill data to multiply against, and tilt's own ceiling (1.5x) already equals the cap.
- Idea43 (add 1 unit at +1R, original-R units: `2*r1-1` if the +1R level touches before a stop-out) was **recomputed**, not read from a per-fill file (`1687_reads.csv` is aggregates-only) -- via `1687_cells.py`'s unchanged `idea43_walk`, walked read-only off `bars_sip.db` (sqlite `mode=ro`) for the 91 Q3 fills. 0 reconstruction failures.
- Green = weekly $ > 0; weeks 27/34/37 are true zero-fill weeks (no production entries), shown as "-", not counted green or red.
- Max DD = peak-to-trough on the cumulative weekly-$ curve starting from 0, in week order 27->39.
- Single quarter, n=91/13wk, not an OOS claim; tilt and idea43 are both layered on the SAME 91 fills (not independently selected) -- a lever-isolation read, not a new independent edge.
