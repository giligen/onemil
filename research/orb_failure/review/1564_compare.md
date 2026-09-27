# 1564/1565/1566 builder vs. independent rebuild — row-level compare

Keyed by (symbol, date, cell); builder = `cell_1564_events.csv`, rebuild = `rebuild_1564_events.csv`.
Builder's raw CSV carries an "entered" attempt for every population row regardless of the row's own
`event` label; the population that actually corresponds to each cell's signal is
`entered==True & event==(FAILURE for 1564/1565, SUCCESS for 1566)` — that filter is applied on both
sides below (rebuild's file already contains only that filtered population).

## Event-set Jaccard (signal-event rows, entered)
| cell | builder n | rebuild n | intersection | Jaccard |
|---|---|---|---|---|
| 1564 failed-break short | 635 | 638 | 634 | **0.992** |
| 1565 failed-break short (VWAP target) | 635 | 86 | 85 | **0.134** |
| 1566 held-break long | 2893 | 2851 | 2851 | **0.986** |

1564 and 1566 agree almost perfectly on WHICH symbol-days qualify. **1565 does not** — rebuild's
gated population is 7.4x smaller than builder's. Builder applies the same base FAILURE population to
1565 as to 1564 (identical `not_shortable`/`price_lt_5`/`no_1030_bar`/`ssr` skip counts on both cells)
and only changes the exit target; nothing in `cell_1564.py`'s entry gate for 1565 differs from 1564.
The rebuild evidently adds a necessary-condition filter for the VWAP target (e.g. VWAP must sit on the
correct side of entry to be a valid short target) that builder never enforces — builder's 1565 hits
`why=target` 4,186/5,253 times (80%), an implausibly high rate consistent with an ungated, often-already-there
VWAP target rather than a real trade condition.

## Share of common entered rows within 0.01 R (net_R)
| cell | matched rows | share within 0.01R | mean |diff| R |
|---|---|---|---|
| 1564 | 634 | 13.3% | 0.174 |
| 1565 | 85 | 3.5% | 0.606 |
| 1566 | 2851 | 23.3% | 0.047 |
| **all cells (pooled)** | 3570 | **~21.0%** | — |

Row-level net_R agreement is poor everywhere, worst on 1565.

## VAL mean net_R per cell
| cell | builder VAL mean | rebuild VAL mean (REBUILD_1564.md) | diff (b − r) |
|---|---|---|---|
| 1564 | −0.325 | −0.275 | −0.050 |
| 1565 | −0.562 | −0.881 | +0.319 |
| 1566 | −0.050 | −0.014 | −0.037 |

## Passing cells
Builder (`RESULT_1564.md`): `PASS BAR (VAL): False` on all three cells → **passing = []**.
Rebuild (`REBUILD_1564.md`): VAL mean_net_R negative on 1564 (−0.275, t −3.9) and 1565 (−0.881, t −3.0),
and indistinguishable from zero on 1566 (−0.014, t −0.24) → **passing = []**.
**Same passing cells: yes, both empty** — but the agreement is only in the headline verdict, not in
the underlying rows (see above and below).

## Dominant cause of the 10 largest |net_R| differences
Ranked top 10 by |net_R_builder − net_R_rebuild| across all three cells, all at nearly identical entry
prices (builder vs rebuild entries differ by a few cents on 9/10 rows):

| symbol/date/cell | entry_b vs entry_r | exit_b vs exit_r | why_b / why_r | net_R_b | net_R_r |
|---|---|---|---|---|---|
| ARIS 2026-03-25 (1564,1565) | 17.795 / 17.840 | 17.850 / 17.850 (same) | stop/stop | −2.21 | −7.80 |
| ING 2025-05-02 (1564,1565) | 20.705 / 20.710 | 20.725 / 20.725 (same) | stop/stop | −2.68 | −6.53 |
| NVTS 2025-06-10 (1565) | 8.255 / 8.260 | 7.981 / 7.948 | target/target | +7.52 | +2.69 |
| CRML 2025-10-06 (1566) | 13.19 / 13.16 | 10.90 / 16.61 | **stop/target** | −1.33 | +1.96 |
| KEEL 2025-10-15 (1564) | 6.180 / 6.195 | 5.759 / 6.520 | **target/stop** | +1.85 | −1.08 |
| PRTA 2025-08-06 (1564) | 7.830 / 7.850 | 7.944 / 7.661 | **stop/target** | −1.27 | +1.57 |
| ETHE 2025-03-19 (1566) | 16.882 / 16.877 | 16.720 / 17.190 | **stop/target** | −1.17 | +1.46 |
| PLTM 2025-12-26 (1565) | 23.730 / 23.790 | 23.320 / 23.275 | target/target | +2.17 | +5.23 |

**Dominant cause: range_low (R-unit) reconstruction.** On the `stop/stop` and `target/target` rows the
exit *price* matches or nearly matches between builder and rebuild (ARIS, ING both exit at the exact
same stop price), yet net_R differs by 3.5x–4x. That is only possible if the risk-per-share
denominator (distance from entry to the reconstructed range_low, i.e. "1 R") differs sharply between
the two implementations — not a cost/spread effect (spread cannot move R by 3-5x) and not SSR (`ssr`
is False on every one of the top 10 rows). `half_src` on 5/10 rows is `flat_fallback` (no measured
NBBO for that symbol-day) vs `real` on the others, but the size of the gap is the same order in both,
so spread source is a minor, not dominant, contributor.

**Secondary cause: exit-path divergence (day-high/day-low stop vs target ordering).** 4/10 rows
(CRML, KEEL, PRTA, ETHE) don't just disagree on net_R, they disagree on **which exit fired first**
(`why` flips stop↔target) — builder and rebuild are walking the same bars to a different conclusion
about whether the stop or the target level is touched first intraday, most likely from a different
day-high/day-low or stop-trail reconstruction feeding the two engines' path-walkers.

No 10:30-boundary or SSR disagreement appears among the top 10 (event-set Jaccard on 1564/1566 is
already 0.99, and `ssr` is False on every top-10 row) — the 10:30 declaration itself is not implicated;
the R-unit computed off that declaration is.

## Bottom line
- 1564 and 1566: same population (Jaccard ~0.99), same headline FAIL verdict, but row-level net_R
  agrees only 13-23% of the time within 0.01R — the two engines are computing a materially different
  R-unit and/or path-walk on the same trade population. Not yet an independent-check PASS on the
  numbers, only on the top-line direction.
- 1565: population itself disagrees (Jaccard 0.13) — the VWAP-target gate is under-specified in the
  builder (looks ungated, ~80% target-hit rate is a red flag) and needs its own necessary-condition
  check before any 1565 number is usable.
- Recommendation: do not report 1565 in its current form; re-derive the shared range_low/R-unit
  helper for 1564/1566 and re-run before treating the row-level numbers (not just the pass/fail
  verdict) as agreeing.
