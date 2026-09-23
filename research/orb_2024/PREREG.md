# PREREG — Does the LIVE ORB edge survive a different regime? 2024-07-02 .. 2024-12-31. Cells 1,415–1,416

Why: tonight's risk-on base entry held on TRAIN, VAL and the sealed 2026 quarter and then reversed on the untouched
2024H2 half-year (1,413: −0.131 R) — all its evidence came from one market regime. ORB is the only book with a
measured edge, and every ORB number (2025-01 .. 2026-09) comes from that same regime. Its selection literals
(z-params, quintile cutoffs, mults in `orb.yaml`) were fit on 2025–26, so 2024H2 is out of sample for selection
too. Frozen before any 2024 ORB number exists.

## Universe (survivorship-free, knowable by 09:35 ET)
From `research/day_breadth/y2024/candidates.csv` (EQUS point-in-time, gap ≥ 3 %, $3–50, prior volume ≥ 500K, class K,
'+'→'.WS', preferreds out) and its minute bars `research/day_breadth/y2024/bars.db`, with the 09:30–09:35 RTH volume
≥ 15,000 floor. Cell 1,415 = production seed (gap ≥ 5 %, open $3–30). Cell 1,416 = the `addon_p30` pool (gap 3–5 %,
open $30–50), each walked ALONE like `research/orb_seed_wide/run_quarter_pools.sh`.

## Book (identical to the live config)
`study_orb_features.py` (features) + `study_orb_pipeline_static_lock.py` (selection + exits) with the `orb.yaml`
literals as they stand on 2026-09-23 (no refit), 8 shared slots, risk $375, **catalyst veto OFF**
(`ORB_CATALYST_VETO=0`, as live since 9/21; no 2024 news data exists), PDR / G1 / range-size vetoes ON, static lock +
touchgo + ATR floor + scale-out as configured. No threshold or parameter changes.

## Pass bar (per cell) — the question is survival, not proof (≈ 40–60 fills expected, underpowered for t ≥ 2)
SURVIVES iff net R per fill > 0 AND total $ > 0 AND ex-top-5 % net R per fill > 0. RED FLAG iff net R per fill ≤ −0.10
(the edge may be regime-bound: pause any scaling of the ORB ramp and report). Also reported: n fills, fills/week,
t (day-clustered), monthly $, best/worst month, the no-fill share, and the same numbers for 2025 (runB_true) beside.

## Verification
Universe causality (Opus) if SURVIVES is used to justify scaling or the p30 pool; independent rebuild of the book.

## Not allowed
Refitting anything on 2024; any other cell on this data; changing the pass bar after the number exists.
Programme count 1,416.
