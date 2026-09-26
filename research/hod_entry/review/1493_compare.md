# Cell 1,493 independent-rebuild check

Builder: `cell_1493_fills.csv` (493,515 fill×cell rows, from `cell_1493.py`).
Rebuild: `rebuild_1493_fills.csv` (493,515 rows, from `rebuild_1493.py`; ran fresh in this
session — no rebuild output existed before this check, generated over the full 8,973-fill
population, ~34 min walk). Both share the identical population (8,973 fills, TRAIN=3,957 /
VAL=5,016) and the identical 55 cells; joined 1:1 on (day, symbol, cell) with zero orphans on
either side.

## Cell-name mapping (stated, not assumed)
Builder `STOP%|EXIT` ↔ rebuild `sSTOP_EXIT`, with `tgt` → `t`:
`0.5%|NONE↔s0.5_NONE`, `0.5%|T30↔s0.5_T30`, `0.5%|T60↔s0.5_T60`, `0.5%|tgt0.5↔s0.5_t0.5`
… same pattern through `3.0%`, `CL%|*↔sCL_*`, and `M ↔ M_mirror`. 55/55 cells matched
one-to-one, confirmed by listing both unique cell sets.

## Row-level agreement
- Matched rows: 493,515 / 493,515 (100%, no left/right-only keys).
- **Share with |net_pct diff| ≤ 0.01% of price: 79.98%**
- **Share with |net_pct diff| ≤ 0.05% of price: 83.78%**
- Mean |diff| 0.099pp, median 0.0014pp, max 25.12pp.

**PREREG bar (≥99% within 0.01% AND same selected cell): FAILS on agreement rate.**

## Selection outcome
Both implementations independently fail the TRAIN-H2 gate (stop≥1.0%, ≥3 fills/wk,
day-clustered t≥2) on every cell — **both select NONE**, so the pass-bar's second
condition (same selected cell) holds, but only because both sides land on the same null
outcome, not because the surfaces agree:

| cell (norm) | TRAIN mean_b | TRAIN t_b | TRAIN mean_r | TRAIN t_r |
|---|---|---|---|---|
| best-by-mean illustration `3.0%\|NONE` / `s3.0_NONE` | −0.1159% | −0.77 | −0.1326% | −2.80 |
| rebuild's own raw-best `1.5%\|NONE` / `s1.5_NONE` | −0.1459% | −1.39 | −0.1309% | −3.01 |

Rebuild's raw ranking picks a *different* top TRAIN cell (`s1.5_NONE`) than the builder's
report-only illustration cell (`3.0%|NONE`); both cells still fail t≥2 on both sides, so the
gate outcome is unaffected, but the surfaces are not close enough to trust the ranking.

VAL mean_pct diff on the illustration cell `3.0%|NONE`/`s3.0_NONE`: builder −0.0776% vs
rebuild −0.0637%, **diff 0.0139pp** — small in isolation. The **max abs per-cell VAL mean
diff across all 55 cells is 0.2036pp** (`s0.5_t3.0`: builder −0.4326% vs rebuild −0.2290%),
concentrated in the 0.5%-stop cells (excluded from the gate anyway, but they drive the
row-level disagreement share down).

## Diagnosis of the 20 largest row diffs
All 20 of the largest (fill, cell) diffs (13pp–25pp) are the same pattern: **builder
`why=stop` (correctly stopped out for a loss) vs rebuild `why=eod`/`time`/`target` (rode the
position through a huge subsequent move)** — e.g. OKLL 2025-12-04 builder −0.69% (stop) vs
rebuild +24.42% (eod, diff 25.12pp); MRNA 2026-01-13 builder −10.94% (stop) vs rebuild
+8.24% (eod, diff 19.18pp); LUNR, EXAS, QPUX, RGTU/RGTX, AVEX, COHX, IPGP, BAIG same shape.
65% of all rows with diff > 1.0pp (12,220 / 493,515 = 2.5% of rows) show this exact
builder-stop / rebuild-not-stop signature.

**Root cause: fill-minute (tape-minute) handling.** Builder's `fill_min` is fractional
(e.g. 620.111892, sub-minute tape time); rebuild's `minute_r` is a bare integer bar minute
(626) — a difference of several whole minutes on these rows, not a rounding artifact.
Across **all** matched rows (not just the large-diff ones), 55.6% have |fill_min_b −
fill_min_r| > 0.5 minutes — this is a systemic, pervasive divergence in how the two
implementations resolve the entry tape-minute into a bar index, not an edge case. Most of
the time the shifted entry bar doesn't change the outcome (median diff is only 0.0014pp),
but on the volatile/illiquid low-float names in the top-20 (OKLL, RGTU, QPUX, RGTX, MRNA,
LUNR, etc.) the shifted entry lands the walk in a different bar sequence relative to the
stop level, so the builder's walk crosses the stop intrabar while the rebuild's walk (from
a different starting bar) does not, and rides to EOD/time/target instead. `store_served_1438`
(bar-store provenance) is not the driver — big-diff rows are 80/20 split on it, same as
background.

**This is a real implementation divergence, not a naming or cost-unit artifact**: same
cost convention (net_pct/net_R units align, e.g. MRNA `s3.0_t3.0` shows a clean 3.000% net_pct
on the rebuild's target-hit vs builder's stop-hit at −10.94%, so both apply the same %-of-price
scale). The fix belongs in the fill-minute → bar-index conversion shared by both walks, then
this cell should be rebuilt from a single one of them (parity by construction) before any
further row-level check.

## Bottom line
- Agreement bar (≥99% within 0.01%): **FAIL** (79.98%).
- Same TRAIN-H2 selection outcome: **PASS** (both NONE — but for a mostly-negative surface
  that itself doesn't closely agree cell-to-cell).
- Dominant cause: fill/tape-minute resolution differs between builder and rebuild on ~56%
  of rows; this only produces large P&L divergence (2.5% of rows, but the entire tail of
  the largest diffs) when the shifted entry bar changes whether the walk crosses the stop
  before a subsequent large move on a volatile/illiquid name.
- Conclusion for the PREREG: cell 1,493's headline (no cell clears the TRAIN-H2 gate) is
  directionally corroborated (rebuild also finds every cell net-negative on TRAIN and fails
  the same gate), but the row-level and per-cell magnitudes are not independently confirmed
  at the 99% bar — do not read the per-cell VAL numbers (surface, ex-top-5%, obtainability
  follow-ons) as settled until the fill-minute discrepancy is fixed and the two
  implementations are re-run and re-compared.
