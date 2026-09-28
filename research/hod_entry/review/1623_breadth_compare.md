# Independent-check compare — cell 1,624 (break breadth / "the crowd")

Builder: `cell_1624.py`, `cell_1624_fills.csv`, `RESULT_1624.md`, `cell_1624_arm_events_cache.csv`.
Rebuild: `rebuild_1624.py` (written blind to the builder files, per CLAUDE.md independent-reimplementation
rule #1), `rebuild_1624_fills.csv`, `REBUILD_1624.md`. Both files carry the identical 9,911-row
population (day,symbol) — 0 keys only-in-builder, 0 keys only-in-rebuild, `outcome_R` matches
exactly (100.00%) on every merged row, confirming both draw from the same underlying fill source and
this is a genuine independent-implementation comparison, not a population mismatch.

## Bar (Jaccard >= 0.99, means within 0.01 R) — NOT MET

| check | value | bar | result |
|---|---|---|---|
| Kept-set Jaccard, VAL | 0.7624 (inter 2653 / union 3480; builder n 2940, rebuild n 3193) | >= 0.99 | **FAIL** |
| Kept-set Jaccard, overall (TRAIN-H2+VAL) | 0.7340 (inter 3923 / union 5345; builder n 4400, rebuild n 4868) | >= 0.99 | FAIL |
| Kept-set Jaccard, TRAIN-H2 only | 0.6810 (inter 1270 / union 1865; builder n 1460, rebuild n 1675) | >= 0.99 | FAIL |
| VAL kept mean net R | builder −0.1726 vs rebuild −0.1689, \|diff\| = **0.0037 R** | within 0.01 R | **PASS** |
| TRAIN-H2 kept mean net R (extra, not the ask) | builder −0.1644 vs rebuild −0.1384, diff 0.0260 R | within 0.01 R | fails, out of scope of the VAL-only ask |
| Passing set (ship/no-ship on the frozen 1623/1624 bar) | builder: [] (Overall: FAIL, 4/9 criteria met); rebuild: not cleared, sign-consistent (−0.169, t −3.10, both far below +0.15R / t 2.5) | equal | **MATCH — both empty** |

Row-level detail (VAL, n=5,513 merged): 3-way tercile (bottom/mid/top) exact agreement 75.2%;
binary kept-flag (top vs not-top) agreement 85.0%. Crosstab (builder rows × rebuild columns):

| builder \ rebuild | bottom | mid | top |
|---|---|---|---|
| bottom | 694 | 184 | 18 |
| mid | 359 | 796 | 522 |
| top | 55 | 232 | 2653 |

## Dominant cause of the Jaccard failure: arm-event population scope, not a coding bug

B30 raw counts differ by a **median 5.33x / mean 6.63x** between builds (VAL: builder mean 74.99,
median 53; rebuild mean 12.90, median 10), correlated at **r = 0.86** — directionally related, not
equivalent. Source of the gap:

- **Builder** replays cell 1,438's `armed_crossing_bars` over the *full causal superset* — the
  67-row-per-day-ish scan universe, not just fills — via `cell_1624_arm_events_cache.csv`
  (**65,974 rows**, columns `day,symbol,m_hi`). RESULT_1624.md's own text: B30/B60 = "any status, all
  names," verified 100.00% exact agreement against `causal_arming_causal.csv`'s `n_cross` column. This
  is the literal prose reading of "any status."
- **Rebuild** pre-declares (in REBUILD_1624.md, written before comparing to the builder) that the
  sanctioned inputs hand it no per-event timestamp for `nofill`/`not_armed` arm attempts — `arm_m`
  exists only on the 9,911 `fill` rows in `features_1478_A.csv` — so B30/B60 there count "other
  FILLS' `arm_m` only," a disclosed, explained undercount of the literal spec, not an invented
  shortcut.

This ~5x scale difference moves enough fills across the TRAIN-H2-frozen tercile cutoffs to hold
Jaccard at 0.68–0.76: the two counts rank similarly often (agreement 75–85% row-level) but not
consistently enough to pass a 0.99 identity bar. The aggregate VAL kept mean still lands within
0.0037 R of each other because both counts are still measuring the same underlying "how busy is the
tape right now" construct and the errors partly cancel in aggregate — row membership disagrees far
more than the aggregate statistic does. This is a real data-access asymmetry between what the builder
could reach (an internal replay artifact, `cell_1624_arm_events_cache.csv`) and what the rebuild task
was sanctioned to open, not an arithmetic error in either script.

**Secondary, smaller cause**: `exit_m` (used only for fills/week concurrency slotting, not the R
statistics) matches within 1e-6 on only 73.8% of VAL rows (5,016/5,513 both non-null) — rebuild
sources it from `rebuild_1481_fills.csv` joined on (day,symbol,fill_min) per its own disclosed
caveat, rather than cell 1438's own value. Does not affect kept-set membership or any R mean.

## Sign-flip on a secondary sub-criterion (does not change the overall verdict)

kept-minus-dropped, VAL: builder **−0.0043** (kept slightly *worse* than dropped) vs rebuild
**+0.0040** (kept slightly *better* than dropped) — opposite sign, both near zero. This flips the
"dropped < kept, VAL" pass-bar line: builder marks it FAIL; rebuild's own prose calls it
"marginal-yes." Both builds still fail the two binding headline gates (kept mean >= +0.15 R,
day-clustered t >= 2.5) by a wide margin regardless, so this flip changes a secondary checkbox, not
the FAIL conclusion.

## Verdict

Cell 1,624 is **CLOSED, and the two independent builds agree on that** (passing set matches: both
empty; headline sign and magnitude close, VAL mean diff 0.0037 R). The comparison's own bar is **not
met** on the Jaccard leg (0.73–0.76 vs required 0.99) — the row-level kept SET differs by roughly a
quarter to a third of its members, traced to a disclosed, explainable difference in which arm-event
population ("any status" full causal-superset replay vs fill-only `arm_m`) each build could access,
not to a defect in either script. Do not treat the shared FAIL/negative headline as validated
row-for-row; do treat it as validated in aggregate direction and magnitude. Which arm-event
population 1623/1624's mechanism *should* use if this line is ever revisited is a live open question
for whoever owns `PREREG_1623.md` — out of scope for this comparison task.
