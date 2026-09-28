# PREREG_1610 — Builder vs Rebuild comparison

Executes `research/exec_quality/PREREG_1610.md`'s own "Independent check" gate: Part A first-passage
probabilities within 0.02, Part B ≥99% of per-fill rows within 0.01R. Compares builder
(`cell_1610.py` → `cell_1610_driftmap.csv` / `cell_1610_fills.csv`, per `RESULT_1610.md`) against the
from-prose rebuild (`rebuild_1610.py` → `rebuild_1610_driftmap.csv` / `rebuild_1610_fills.csv`, per
`REBUILD_1610.md`). Cells covered: 1,610–1,616 (cost-aware HOD-break barriers).

## Part A — driftmap: PASSES (max abs diff 0.000931 vs the 0.02 bar)

432/432 (holdout × quintile × k% × m%) rows matched on key in both files. Builder's
`p_up_before_down` is documented **unconditional** (own caveat: "not renormalized over resolved
paths only"); rebuild independently reports both `p_up_unconditional` and `p_up_conditional`.
Comparing like-for-like (unconditional ↔ unconditional): **max abs diff = 0.000931**, 0/432 pairs
exceed 0.02. `driftless_p` matches **exactly** (0.000000 diff) in both, confirming the `m/(k+m)`
driftless-formula ambiguity was resolved identically by both builds.

(For contrast, builder-unconditional vs rebuild-**conditional** gives max diff 0.222 with 133/432
over bar — a pure column-pairing/definitional issue if done naively, not a real discrepancy; both
docs' own caveats already flag builder=unconditional.)

The 10 largest residual (still tiny, 0.0003–0.00093) diffs all cluster in TRAIN quintiles 2–3 at wide
(k,m) pairs, where the two builds' `n_neither` (15:55 censoring count) differs by 1 fill — a minor
bar-boundary rounding edge case, not a systematic error.

## Part B — fills: FAILS in every one of the 7 cells (bar 99% within 0.01R)

Builder's `cell_1610_fills.csv` is long-format (one row per (cell, fill), `cell` ∈ {1610..1616},
69,029 rows); rebuild's `rebuild_1610_fills.csv` is wide-format (one row per fill, `cN_*`-suffixed
columns per cell, 9,911 rows). Joined per cell on (day, symbol): builder `net_R` vs rebuild
`cN_net_R_base`.

| cell | builder n | rebuild-only unmatched | **share within 0.01R** | mean abs diff (R) | max abs diff (R) | why-mismatches |
|---|---|---|---|---|---|---|
| 1610 | 9,911 | 0 | 0.23% | 0.348 | 2.55 | 0 / 9,911 |
| 1611 | 9,911 | 0 | 0.26% | 0.371 | 2.38 | 1 / 9,911 |
| 1612 | 9,911 | 0 | **13.39%** | 0.040 | 2.90 | 3 / 9,911 |
| 1613 | 9,911 | 0 | 2.26% | 0.114 | 3.42 | 9 / 9,911 |
| 1614 | 9,911 | 0 | 2.43% | 0.118 | 3.28 | 10 / 9,911 |
| 1615 | 9,737 | 174 | 2.08% | 0.281 | 2.11 | 2 / 9,737 |
| 1616 | 9,737 | 174 | 2.24% | 0.462 | 11.49 | 4 / 9,737 |

No cell reaches the 99% bar; the best is cell 1612 at 13.4%.

### Dominant cause (pervasive, explains the population-wide failure): R-unit denominator mismatch

Rebuild's own doc states its convention explicitly: *"Every cell's net R is reported in the BASE R
unit (fill − stop, the UNSCALED original R) even where the cell's own stop distance differs …
avoids a smaller-denominator artifact."* The cross-cell pattern is consistent with **builder instead
normalizing net_R by each cell's OWN (rescaled) stop distance**:

* **Cell 1612** (stop = fill − R, *identical* to the base stop) has by far the best agreement
  (13.4%, 5–50× every other cell) — when a cell's own R equals base R, the normalization choice
  cannot matter.
* **Cells 1610/1611** (stop rescaled to 0.75×R — the most different from base among the "clean"
  rescale cells) are the *worst* (0.23% / 0.26%); their mean abs diff (0.35–0.37R) is quantitatively
  consistent with a ~25% multiplicative gap between a 0.75×R-denominated number and a
  base-R-denominated one.
* **Cell 1616** (quintile-optimal, k or m up to 3% — the widest, most variable rescale) has the
  single largest outlier (11.49R) and the highest mean abs diff (0.462R).
* Crucially, for 1610/1611 the **walk outcome** (`why`) agrees on effectively every row (0 and 1
  mismatches out of 9,911) — the sub-1% agreement rate there cannot be a walk-logic bug; it has to be
  in how dollar P&L is converted to an "R" multiple.

This is not confirmed against `cell_1610.py`'s source in this task (out of scope / step budget) —
it is the best-supported hypothesis from the CSV pattern and should be checked directly against the
two scripts' `net_R` formulas before relying on either file's per-fill R value.

### Secondary cause (a handful of individually-large outliers): walk-outcome (`why`) disagreement

5 of the pooled top-10 largest single-fill diffs have builder `why=target` vs rebuild `why=stop` (or
vice versa), with exit-minute timing differing by 32–212 minutes — concentrated in the cost-eaten
cells 1612–1614 (why-mismatch counts 3, 9, 10 — all <0.1% of rows) and reproduced on the *same*
symbol/day across multiple cells (DNTH 2025-11-14 and WDCX 2026-02-09 mismatch in both 1613 and
1614; VIST 2025-10-09 appears in 4 of the top-10 rows, across cells 1612/1613/1614/1616). This points
to a small difference in how the cost-widened barrier **level** itself (c_in/c_out baked into the
stop/target price, per cells 1612–1614's formulas) is computed between the two builds — enough to
flip the walk order on genuinely borderline paths. This layer is much smaller in row-count than the
denominator issue above but produces some of the single-largest per-fill diffs.

### Secondary, already self-documented cause: cells 1615/1616 row-count gap

Builder drops all 174 fills with NaN `spread_bps_at_arm` entirely from cells 1615/1616 (9,737 vs the
full 9,911 population); rebuild instead computes a degenerate EOD-exit fallback for those rows (its
own caveat: *"`walk_single` then falls through to an EOD exit for those 174 rows"*). This is a
second, independent, already-documented divergence specific to those two cells (174 rebuild-only
unmatched rows each), additive to — not a cause of — the normalization issue above.

## Passing cells: MATCH — both empty

* Builder (`RESULT_1610.md` verdict table): `passes_val_bar` = False on every scored VAL row for
  cells 1610–1615; 1616 is report-only. **Builder passing cells = []**.
* Rebuild (`REBUILD_1610.md` pass-bar table): all 6 scored cells **FAIL**; 1616 report-only, and its
  own TRAIN-selected pair reverses sign on VAL (+0.062% → −0.086%), the expected overfit signature.
  **Rebuild passing cells = []**.
* `builder: []` == `rebuild: []` → **same_passing_cells = TRUE**.

## Overall verdict

Part A clears its 0.02 bar cleanly (max diff 0.00093). Part B does **not** clear its 99%-within-0.01R
bar in any cell (0.23%–13.4%) — this is the PREREG's own pre-committed Independent-check gate, and
per its "Independent check and consequences" section a PASS/FAIL call is only meant to be acted on
once this agreement is achieved. The headline **qualitative** conclusion is robust either way — both
builds independently find zero cells clearing the VAL pass bar, and VAL mean-net% for every cell is
hundreds of bps away from the +0.15%/+0.10% thresholds, an order of magnitude bigger than the
per-fill disagreement found here — so a flip to PASS from fixing the normalization mismatch is very
unlikely. But the row-level check the PREREG itself requires has **not** been satisfied, so this FAIL
should not yet be relayed as fully checked at the per-fill level; **agreement_ok = FALSE** pending a
reconciliation of the net_R denominator convention (base R vs. each cell's own R) between
`cell_1610.py` and `rebuild_1610.py`, and a rerun of this comparison.

## Method notes

Per-cell fills join key = (day, symbol), builder filtered to `cell==N` per row, rebuild's `cN_*`
suffix columns used for the matched cell. Driftmap join key = (holdout, quintile [normalized
ALL/Q1–Q5 ↔ ALL/1–5], k_pct, m_pct, both rounded to 6dp). Ad hoc analysis script (not a repo
artifact, per this task's write-scope restriction) at
`/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/compare_1610.py`
— reproducible in under a minute against the same six CSVs.
