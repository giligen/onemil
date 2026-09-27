# 1,552 builder vs. independent rebuild — comparison

Builder: `cell_1552.py` / `cell_1552_events.csv` (144,264 event-rows) / `cell_1552_stats.csv`.
Rebuild: `rebuild_1552_full.py` / `rebuild_1552_events.csv` (119,492 event-rows) / `rebuild_1552_stats.csv`,
written from PREREG_1552.md + Amendment 1 prose only, parsed independently from the raw submissions
gzip cache. Both restricted to TRAIN (2019-2022) / VAL (2023-2024H1) only — no TEST row exists in
either stats file, so nothing here touches the sealed split. 9 of 10 classes are computable
(BUYBACK_OR_INSIDER / cell 1,561 needs Form-4 XML, out of scope for both sides, per Amendment 1).

## 1. Event-set Jaccard per cell (keys: date, symbol; cell number ≡ class, mapping below)

| cell | class | split | builder n | rebuild n | intersection | union | **Jaccard** |
|---|---|---|---|---|---|---|---|
| 1552 | OFFERING | TRAIN | 10,481 | 9,948 | 7,614 | 12,815 | **0.594** |
| 1552 | OFFERING | VAL | 3,180 | 3,116 | 2,710 | 3,586 | **0.756** |
| 1553 | SHELF | TRAIN | 3,119 | 2,790 | 1,891 | 4,018 | **0.471** |
| 1553 | SHELF | VAL | 1,138 | 1,062 | 928 | 1,272 | **0.730** |
| 1554 | REVERSE_SPLIT | TRAIN | 3,523 | 3,008 | 2,370 | 4,161 | **0.570** |
| 1554 | REVERSE_SPLIT | VAL | 1,639 | 1,381 | 1,285 | 1,735 | **0.741** |
| 1555 | AUDITOR | TRAIN | 511 | 322 | 199 | 634 | **0.314** |
| 1555 | AUDITOR | VAL | 185 | 153 | 127 | 211 | **0.602** |
| 1556 | NON_RELIANCE | TRAIN | 221 | 136 | 69 | 288 | **0.240** |
| 1556 | NON_RELIANCE | VAL | 67 | 54 | 40 | 81 | **0.494** |
| 1557 | LATE_FILING | TRAIN | 407 | 373 | 192 | 588 | **0.327** |
| 1557 | LATE_FILING | VAL | 201 | 180 | 136 | 245 | **0.555** |
| 1558 | OFFICER_EXIT | TRAIN | 22,809 | 19,170 | 16,617 | 25,362 | **0.655** |
| 1558 | OFFICER_EXIT | VAL | 8,932 | 7,453 | 7,117 | 9,268 | **0.768** |
| 1559 | CONTRACT | TRAIN | 9,046 | 5,967 | 4,858 | 10,155 | **0.478** |
| 1559 | CONTRACT | VAL | 3,017 | 1,935 | 1,836 | 3,116 | **0.589** |
| 1560 | ACTIVIST | TRAIN | 1,600 | 1,126 | 810 | 1,916 | **0.423** |
| 1560 | ACTIVIST | VAL | 530 | 384 | 349 | 565 | **0.618** |

Jaccard is materially below 1.0 everywhere: 0.24–0.66 on TRAIN, 0.49–0.77 on VAL (VAL always
higher — consistent with fewer late-arriving-filing edge cases as the raw submissions cache gets
more stable closer to present). This is not a rounding-level agreement; it is a real population
disagreement between the two independent builds.

## 2. VAL net bps on the TRAIN-named leg, side by side

| cell | class | builder leg | rebuild leg | **legs agree?** | builder VAL net bps | rebuild VAL net bps | abs diff (bps) |
|---|---|---|---|---|---|---|---|
| 1552 | OFFERING | E1 | E1 | yes | −5.43 | 3.66 | 9.09 |
| 1553 | SHELF | E1 | E1 | yes | −1.29 | −5.90 | 4.61 |
| 1554 | REVERSE_SPLIT | E5 | E5 | yes | 21.41 | 5.61 | 15.80 |
| 1555 | AUDITOR | **E1** | **E5** | **NO** | −77.98 | 18.61 | 96.59 |
| 1556 | NON_RELIANCE | **E5** | **E1** | **NO** | 185.96 | −2.45 | 188.41 |
| 1557 | LATE_FILING | E5 | E5 | yes | 155.92 | 179.45 | 23.53 |
| 1558 | OFFICER_EXIT | E1 | E1 | yes | 2.29 | −0.34 | 2.62 |
| 1559 | CONTRACT | E5 | E5 | yes | −44.82 | 13.18 | 58.00 |
| 1560 | ACTIVIST | **E5** | **E1** | **NO** | −101.30 | 54.98 | 156.29 |

**Named-leg agreement: 6/9 cells.** The three disagreements (AUDITOR, NON_RELIANCE, ACTIVIST) are
exactly the three lowest-n, lowest-Jaccard classes (n≈150–1,900 TRAIN events) — small samples where
the TRAIN-leg pick is unstable and sensitive to exactly which events each build includes, not a
sign either implementation mis-scored a leg it agrees is in the population.

**Max abs difference (VAL net bps), same leg on both sides regardless of which was named** (the
apples-to-apples number, across all 9 cells × 2 splits × 2 legs = 36 matched cells):
**163.71 bps** (NON_RELIANCE, TRAIN, leg E5: builder +115.8 vs rebuild −47.9 bps). Runners-up:
AUDITOR TRAIN E5 (151.2 bps), NON_RELIANCE VAL E5 (145.4 bps), LATE_FILING TRAIN E1 (88.3 bps).
The **max abs difference on the named-leg-only comparison in row 2 above is 188.41 bps**
(NON_RELIANCE) — larger because that row compares two *different* legs when the named leg
disagrees; treat 163.71 bps as the honest single number.

None of the 9 cells clears the PREREG pass bar (net ≥ +15 bps, t ≥ 2.5, ≥3 events/week VAL, etc.)
on either build's own numbers — this comparison is about builder/rebuild agreement, not a pass/fail
call, which RESULT_1552.md and REBUILD_1552.md already both report as universal fails on VAL.

## 3. Dominant cause of the differences

Ruled out (checked directly, both sides match):
- **Acceptance-time / entry-session rule.** Both `cell_1552.py::entry_session` and
  `rebuild_1552_full.py::build_events` resolve the same convention (before 09:00 ET → same-day
  MOO if a session exists else next session; at/after 09:00 ET → the next session strictly after
  the calendar date). Diagnostic: for OFFERING TRAIN, date-only Jaccard is 0.989 (984 vs 981
  distinct filing dates) — session resolution is not where the disagreement lives.
- **Serial-issuer cap mechanics (Amendment 1).** Both count ALL 424B1–B5 forms toward the
  trailing-365-day cap (not just 424B3/B5), and both compute the count causally, as of each
  candidate filing's own `filingDate` using only filings dated ≤ that date
  (`cell_1552.py:is_serial_issuer`/`compute_serial_flags`; `rebuild_1552_full.py` judgment-call
  #2). No look-ahead on either side — this is a shared design choice, not a discrepancy.
- **Dual share classes / CIK→symbol join mechanism.** Both sides explode a CIK into every symbol
  `symbol_cik_map.csv` maps it to and score each mapped symbol independently as its own tradable
  member (builder's `fetch_submissions.py`: "a CIK can map to >1 symbol... each gets its own row";
  rebuild's judgment-call #1: "each mapped symbol is scored independently"). The *counts* of
  multi-symbol CIKs differ slightly (builder explodes off the full 7,754-CIK map; rebuild's prose
  cites 371/7,135 *primary*-source CIKs), a candidate for a small share of the gap but not the
  dominant one (see date-only diagnostic below).

Confirmed as the dominant cause — **population-filter / classification disagreement at the
individual-event level, worse on the low-frequency classes**:
- For OFFERING (highest-n class), date-coverage agreement is near-total (0.989) but the
  event-set Jaccard is only 0.594 — of 977 shared filing dates, 970 share ≥1 symbol but the
  per-date *symbol sets* still differ enough to pull overall Jaccard down, i.e. one side keeps
  an extra same-day filing/symbol the other side's price-liquidity gate ($1 / $1M) or serial-cap
  reclassification drops.
- For AUDITOR and NON_RELIANCE (item 4.01 / non-reliance restatement flags, the lowest-n, most
  item-code-dependent classes), date-only Jaccard itself collapses to 0.46 / 0.44 — entire
  candidate filings present on one side are simply absent on the other. This points at
  **item-code / form-type classification** (which exact item strings and forms trigger AUDITOR
  vs NON_RELIANCE) and **SSR / price-floor population filters** computed off independently-parsed
  prior-day bars, not at a shared join or timing bug. Both scripts note SSR/sub-$5 exclusion is
  SHORT-cell-only (`SSR_DROP`/`SSR_PRIOR_RET`), consistent between the two, but the raw
  candidate counts feeding that filter (`class_counts_raw_vs_amended.csv`'s builder OFFERING raw
  count of 1,149,657 vs the rebuild doc's cited raw count of 44,368) are measured at different
  pipeline checkpoints (builder's "raw" is pre-form-restriction across the whole file; rebuild's
  "raw" is already form/item-restricted) and are **not apples-to-apples** — a caveat for reading
  that file, not itself the discrepancy driver.

**Bottom line:** builder and rebuild agree on order-of-magnitude population sizes (final n within
~5–35% per cell) and on direction/mechanism most of the time, but individual-event identity
agreement is only fair-to-moderate (Jaccard 0.24–0.77), three of nine TRAIN-named legs disagree,
and VAL net-bps differences reach 160+ bps on the smallest classes. Given every cell already fails
the pass bar on both builds, this does not change any ship/no-ship call, but it means neither
build's per-cell VAL number should be quoted to the owner as precise — only the shared qualitative
finding (every 1552-class cell fails VAL under either independent implementation) is safe to
report. A third, adjudicating pass (or a diff of the two classification outputs row-by-row on the
raw submissions) would be needed before trusting either side's point estimate on AUDITOR,
NON_RELIANCE, or ACTIVIST specifically.
