# REBUILD 1,550-1,551 — independent rebuild of PREREG_1550.md, from prose only

Built by a fresh agent (no access to `cell_1550.py`, `build_panel.py`, `test_cell_1550.py`,
`RESULT_1550.md`, `cell_1550_nights.csv`). Code: `research/overnight_high/rebuild_1550.py`.
Outputs: `rebuild_1550_stats.csv` (all splits x years x N), `rebuild_1550_nights.csv`
(row per name-night: sample, cell, split, date, symbol, ret_on_next, ret_net_5bp, vol_ratio).
Row-level Jaccard / net-bps cross-check against the frozen build is NOT done here (this rebuild
never opened that file) — do it as a separate step by diffing the two nights CSVs on (date,symbol).

## Rule implemented (verbatim from the PREREG's "Rule" section)
Signal at day-t close: `close_t >= max(close_{t-252..t-1})`, `volume_t >= 1.5 x ADV20`,
`close_t >= $5`, `dvol20 >= $10M`, test tickers (`^Z[A-Z]ZZT$`, `^ZZ`) excluded. ADV20 and
dvol20 both computed on a shift(1)-then-roll(20) window (never see day t). Rank by
`volume_t/ADV20`, take top N. Return = `open_{t+1}/close_t - 1`. Cost 5 bps round trip primary
(2/10 bps also computed, in the stats CSV, not repeated below for brevity).

**Divergence found vs. the disclosed prior code** (`research/lit_review_2026/build_daily_panel.py`):
that script's `high52` is `shift(1).rolling(250, min_periods=60).max()` of **HIGH**, not CLOSE,
and requires only 60 (not 252) prior sessions. The PREREG's own prose says "highest CLOSE of the
prior 252 sessions" — this rebuild follows the PREREG prose literally, which is a *mechanically
different, stricter* filter than the one that produced the disclosed `overnight_auction.md`
numbers (HIGH >= CLOSE always, so the disclosed rule fires on a superset of names, and its 60-day
minimum lets it fire during the first year of a sample where this rebuild cannot). The two are
not literally the same test; flagging this is exactly what an independent-from-prose rebuild is
for — it is a spec ambiguity, not a coding bug in either implementation.

## Data-limitation caveat (reported per the PREREG's own instruction to log the deviation)
A literal 252-*session* high needs 252 trading days of history before the signal day. Both raw
files start at their sample's first date, so the first ~12-13 months of each sample cannot carry
a fully-populated window and are dropped (`min_periods=252`, no shortcut). Effect: EXTENSION's
usable nights start ~2020-01 (2019 contributes 0 signal days); PANEL's usable nights start
~2025-07 (2024-07..2025-06 contributes 0 signal days) — TRAIN is therefore 2025H2 only, not the
full 18 months the split label implies. This is a data-coverage constraint of the parquet inputs,
not a rule choice, and it materially thins TRAIN.

## EXTENSION (2019-01-02..2024-06-28 Alpaca daily, decisive read — never seen before this rebuild)
Usable nights 2020-01..2024-06 (1,109 distinct signal nights).

| N (cell) | nights | gross bps | net@5bp | t (day-clustered) | ex-top-5% bps | winner-capped bps | placebo margin bps (t) | null pctile |
|---|---|---|---|---|---|---|---|---|
| 10 (1550) | 9,364 | +22.22 | **+17.22** | 2.37 | +0.45 | -24.61 | +14.47 (t 1.74) | 99.5 |
| 25 (1551) | 17,071 | +18.52 | **+13.52** | **1.94** | +2.71 | -14.60 | +8.01 (t 1.18) | 99.8 |

Per-calendar-year net@5bp (day-clustered t): 2020 **+64.8 (t 3.36)**, 2021 +22.4 (t 1.17), 2022
**-17.2 (t -0.36)**, 2023 **-6.7 (t -0.09)**, 2024H1 +5.8 (t 0.32) — for cell 1550; cell 1551 same
pattern (2020 +42.5 t 3.21, 2021 +19.9 t 1.46, 2022 -18.2 t -0.50, 2023 -10.9 t -0.48, 2024H1
+10.1 t 0.33). **2 of 5 years net-positive**, both cells; the whole-sample number is a 2020 carry.
Survivorship: 8,436 symbols present in 2019 rising to 11,517 by 2024 (Alpaca+PIT union already
applied upstream) — long-only overnight rule on >=$10M-ADV names, upward-biased, bounded.

**Tail dependence is severe**: ex-top-5% collapses from +17.2/+13.5 bps to +0.4/+2.7 bps (~98%/80%
of the net edge lives in the top 5% of nights), and winner-capping at +5% flips both cells
**negative** (-24.6/-14.6 bps). This is the signature of a lottery-ticket book, not a repeatable
edge (per the project's tail-dependence protocol).

**EXTENSION vs. its own pass bar (frozen in PREREG):**
- net >= +8bps @5bp: **PASS** (both cells)
- day-clustered t >= 2.5: FAIL (1550: 2.37; 1551: 1.94)
- ex-top-5% > 0: PASS nominally, but economically ~nil (both cells)
- >=4 of 5.5 years positive: **FAIL** (2/5, both cells)
- placebo margin >= +5bps AND t >= 2: margin clears (+14.5/+8.0bps) but **t fails** (1.74/1.18)
- null percentile >= 99: PASS (99.5/99.8)

**EXTENSION fails its own pre-registered pass bar** on 3 of 5 sub-criteria (t, years-positive,
placebo-margin-t), and the one metric that nominally clears (ex-top-5% > 0) is a rounding-level
pass that reverses under winner-capping. Cell 1550 (N=10) is the better of the two on every
metric (t, ex-top-5 direction, placebo margin), consistent with the PREREG's instruction to name
the better extension cell before reading the panel.

## PANEL (rebuilt, 2024-07-01..2026-09-04 Databento EQUS.SUMMARY, delisted included)
TRAIN (usable 2025H2 only, see caveat above; 126 days) / VAL (2026-01..05, 102 days) / TEST
(>=2026-06, 67 days, disclosed as spent, 14 weeks).

| N (cell) | split | nights | gross bps | net@5bp | t (day-clustered) |
|---|---|---|---|---|---|
| 10 (1550) | TRAIN | 1,229 | +18.04 | +13.04 | 0.60 |
| 10 (1550) | VAL | 1,003 | +62.51 | +57.51 | 1.63 |
| 10 (1550) | TEST | 647 | -15.72 | -20.72 | -0.86 |
| 25 (1551) | TRAIN | 2,762 | +22.00 | +17.00 | 1.14 |
| 25 (1551) | VAL | 2,271 | +35.80 | +30.80 | 1.09 |
| 25 (1551) | TEST | 1,266 | -13.70 | -18.70 | -1.31 |

**TRAIN+VAL pooled** (the PREREG's pass-bar quantity): cell 1550 net@5bp **+33.02 bps, t 1.66**;
cell 1551 net@5bp **+23.23 bps, t 1.57**. Magnitude clears +8bps easily; **t fails the >=2 bar**
on both cells. TEST (already-spent, reported for completeness only, never gates a decision here):
both cells **negative** (-20.7/-18.7 bps), same sign as the disclosed literature note's top-25/
top-50 TEST reads (-13.6/-15.2 bps net@5bp there) — directionally consistent even though the two
rules differ mechanically (see divergence note above); this rebuild's top-10 TEST is negative
where the disclosed note's top-10 TEST was positive (+4.3 net@5bp), which is expected given the
HIGH-vs-CLOSE / 250-vs-252 / min_periods-60-vs-252 differences changing which names populate a
thin (n~10) daily slate the most.

## Refuters
- **Price-scale**: nights with |ret_on_next| > 30% flagged, not dropped (counts in the stats CSV,
  `flagged_gt30pct` column) — ~0.3-1.2% of nights per cell/split, consistent with occasional
  genuine large gaps on a new-high/volume-shock population rather than systematic unadjusted
  corporate-action artifacts; not independently confirmed against a corporate-actions feed in
  this budget.
- **Look-ahead**: ADV20, dvol20 and the 252-session high are all `shift(1)`'d before rolling —
  none can see day t's own bar. Confirmed by construction in `build_features()`.
- **Survivorship**: reported above for EXTENSION (rising symbol count 2019->2024, upward bias
  bounded and disclosed, per PREREG).
- **MOC/MOO 15:49-order variant**: **UNTESTED** — this rebuild has only daily OHLCV, no 15:45/
  15:50 intraday snapshots, so the executable-order refuter cannot be run from these inputs.
  Reported as untested, not as a pass.

## Verdict against the frozen pass bar
**FAIL, both cells.** EXTENSION (the decisive, never-seen read) fails 3 of its 5 named
sub-criteria and its edge is tail-carried to the point of flipping sign under a standard +5%
winner cap. PANEL TRAIN+VAL clears the magnitude bar but fails significance (t 1.66/1.57 < 2) on
both cells — consistent with, not independent evidence against, the EXTENSION read. Per the
PREREG's own consequence rule ("FAIL -> the overnight family is closed with the extension numbers
on record"), this independent rebuild's numbers support closing cells 1,550-1,551 as money-book
candidates; no dry-run ledger is warranted from this rebuild's numbers. The disclosed TEST window
is negative in both the original note and this rebuild, for whatever residual value that has now
that it is spent.

## Caveats an adversary would raise
1. Two different operational definitions of "252-day high" exist across the programme (this
   rebuild's literal CLOSE/252/min_periods-252 vs. the disclosed code's HIGH/250/min_periods-60);
   neither is definitively "the" PREREG rule until the owner picks one, and the sub-2.5/2.0
   t-stat failures here would need re-checking under the other definition.
2. Full battery (ex-top-5%, placebo, null, cadence) was only computed at the whole-sample level
   per split, not per calendar year, to stay inside the step budget — a per-year tail check could
   reveal 2020 alone drives the null/placebo passes too.
3. `flagged_gt30pct` nights were not manually inspected for split/dividend artifacts.
4. Cadence-bar green-week/gap fields are blank for most rows because `n < 10` per-day sample
   guards suppressed them, or `weekly_cadence` degenerated (0 strong weeks in every slice at
   $3,000/name risk — R=ret_net directly — meaning no week in any split/cell ever cleared +5R
   at this notional; this itself is informative and consistent with the FAIL verdict but was not
   the primary decision quantity).
