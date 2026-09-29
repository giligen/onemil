# REBUILD_1643 -- independent rebuild of cells 1,643-1,645

Built from PREREG_1643.md alone (cell_1643.py / RESULT_1643_build.md not read
until after this file and events_1643_rebuild.csv were written).
TRAIN=2016-2020, VAL=2021-2023, TEST(2024+) sealed -- not computed.

Sign convention check (1643 raw_r sign == sign(exit-entry)): 1.0000 (expect 1.0000)
Early-close sessions excluded: 17 (Thanksgiving+1, Jul-3-if-weekday, Dec-24-if-weekday, 2016-2023)

| cell | split | n | days | ev/wk | mean bps | clust t | ex-top5% bps | #pos/9 |
|---|---|---|---|---|---|---|---|---|
| 1643 | TRAIN | 2187 | 847 | 8.38 | 0.81 | 0.31 | -6.16 | 6/9 |
| 1643 | VAL | 1631 | 549 | 10.43 | -3.16 | -2.29 | -7.57 | 2/9 |
| 1645 | TRAIN | 1805 | 740 | 6.92 | 3.16 | 1.18 | -4.78 | 6/9 |
| 1645 | VAL | 1563 | 523 | 9.99 | -1.46 | -0.82 | -5.86 | 3/9 |
| 1644 | TRAIN | 3992 | 1122 | 15.30 | 7.35 | 1.62 | -12.34 | 8/9 |
| 1644 | VAL | 3194 | 728 | 20.42 | -0.49 | -0.14 | -13.85 | 3/9 |

## |r_t| tercile table (mean net bps, low/mid/high tercile among that cell's fired events)

| cell | split | T1_low | T2_mid | T3_high | monotone? |
|---|---|---|---|---|---|
| 1643 | TRAIN | 0.81 | 0.18 | 1.43 | no (rising) |
| 1643 | VAL | -3.52 | -1.97 | -4.00 | no (falling) |
| 1645 | TRAIN | 4.00 | 4.22 | 1.27 | no (falling) |
| 1645 | VAL | -2.24 | -0.06 | -2.08 | no (rising) |
| 1644 | TRAIN | 3.98 | 4.35 | 13.73 | YES (rising) |
| 1644 | VAL | -2.41 | 0.95 | 0.01 | no (rising) |

## Per-underlying mean net bps (VAL)

| symbol | 1643 VAL | 1645 VAL |
|---|---|---|
| SPY | -3.27 | -1.13 |
| QQQ | -2.57 | 0.16 |
| IWM | -2.17 | -0.70 |
| XLF | -6.80 | -5.00 |
| XLE | -8.56 | -4.67 |
| GDX | -7.99 | -7.78 |
| XBI | 1.97 | 5.28 |
| TLT | 5.54 | -0.71 |
| SMH | -1.54 | 0.06 |

## Cost / assumption notes
- Entry cost: 1bp half-spread SPY/QQQ/IWM/TLT, 2bp SMH/XLF/XLE/GDX/XBI (per spec).
- Exit cost: 0.5bp MOC on the official-close leg (1643/1645), per spec.
- Cell 1644 (report-only) applies the SAME entry cost + assumes another 0.5bp on the
  next-open exit leg -- the spec does not price this leg explicitly; flagged as an
  assumption, not a spec fact. 1644 is not scored against the pass bar.
- FOMC-day exclusion line and the SPY-only / worst-day / MDE lines from the full PREREG
  report spec were NOT computed here (out of scope for this rebuild's deliverable list).
