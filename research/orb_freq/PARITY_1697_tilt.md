# Parity 1697: engine rel_volume_0935 tercile vs research bin_rvol_0935

**Scope**: read-only parity check of `trading/orb_rvol_tilt.py` (cell 1,694's
rvol tilt) against `research/orb_freq/1694_runners.csv` (478 production
fills, per-fill `rvol_0935` raw ratio + `bin_rvol_0935` research tercile
label). 462/478 usable (16 excluded: NaN ADV20 on both sides, consistent).

**Method**: imported `tercile_for_rvol` and `DEFAULT_EDGES` from
`trading/orb_rvol_tilt.py` directly (not reimplemented). Engine input
`rel_volume_0935` was obtained via the exact algebraic identity
`research_ratio x (SESSION_MINUTES/RANGE_MINUTES)` = `research_ratio x 78.0`
(SESSION_MINUTES=390.0, RANGE_MINUTES=5.0 in `trading/orb_addon_gates.py`) —
mathematically identical to calling `resolve_rel_volume_0935(range_total_volume,
adv20)` since 1694_runners.csv only carries the pre-divided ratio, not the
two raw inputs separately; division/multiplication by the same adv20 and
the fixed 78x constant commute exactly, so no raw-column lookup was needed.

**Result — agreement**:
- 2025 (TRAIN2025, n=204): **100.00%** (204/204), confusion matrix diagonal
  (68/68/68 low/mid/high).
- 2026 (VAL2026, n=258): **100.00%** (258/258), confusion matrix diagonal
  (96/84/78 low/mid/high).
- Pooled n=462: **100.00%**, zero mismatches.

**Edge cross-check**: recomputed the TRAIN2025 1/3 and 2/3 quantiles
directly from the 204 raw-ratio values: e1=0.037160, e2=0.076488 vs the
hardcoded `_RESEARCH_EDGES_RAW_RATIO = (0.03716, 0.07649)` — diff
-0.0000003 / -0.0000018, i.e. the constants are the live quantiles rounded
to 5 dp with no effect on any fill's tercile (no fill sits in the rounding
gap). `DEFAULT_EDGES` in the engine's own units is the exact product
(2.89848, 5.96622), not a separately-rounded constant, so no double-rounding
risk exists at runtime.

**Verdict**: edges are right as shipped. No recompute needed. The x78
rescale and the hardcoded edges reproduce the research's per-fill tercile
labels byte-for-bit on both TRAIN2025 and VAL2026 at 100% agreement.
