# ORB multi-window — pre-registered 2026-09-18 17:05 UTC (before any bar is scanned)

## Why
Owner 9/18: "ORB is not enough — multiply its frequency or add strategies." The honest ORB book is $12.2K / 21 mo at
stage size (≈ 10 picks/month, 1.0 fill/day). More SLOTS do not help (D1_orb: edge gone by rank 9). More SIZE is the
ramp, gated on realized P&L. The untested frequency lever is more ENTRY WINDOWS: the 5-minute opening range is the
only one the book has ever fired on; 15- and 30-minute opening ranges are the standard alternatives in the
literature and select a partly different set of names/days (later confirmation, fewer noise breaks). Zero data
cost: features rebuild from cache.db.

## Rule
Same detector, same selection stack (composite z-score with TRAIN-refit params per window, quintile cutoffs,
Q1 filter, PDR/range-size/catalyst vetoes, touchgo, static lock 1.75R → +0.5R, 15:45 flat), same $10K-stage
sizing and 8 slots, applied to an opening range of W ∈ {15, 30} minutes: range = 09:30 → 09:30+W, entry =
stop-limit at range_high × (1 + 30 bps) placed at 09:30+W, cancel after 60 min, stop = range_low. The
entered-inclusive convention (no-fill rows win slots at $0) applies. `range_size_pct` veto threshold is re-derived
per window as the worst raw quintile in BOTH years, exactly as the V1 veto study did for W=5 (it is a % of price
over a longer range and cannot be inherited).

## Cells (declared)
- W=15 alone, W=30 alone (8 slots, stage sizing) — 2 cells.
- Combined books: 5+15, 5+30, 5+15+30 with ONE rule for overlap — a symbol already ordered (filled OR resting) by an
  earlier window is skipped by later windows (no double exposure, no refill); slots are shared (8 total) — 3 cells.
- Sensitivity, not gated: combined with 12 shared slots — 1 cell.
**6 cells.** Splits TRAIN 2025 / VAL 2026-01..05 / TEST 2026-06+. Gates PLAN §1: G1 t ≥ 2 on TRAIN for the ADDED
picks (the 15/30-window picks not already in the 5-min book), G2 VAL same sign + ≥ 55% weeks green, TEST once behind
a written FREEZE.md. Ship bar for a combined book: ≥ +30% picks vs the 5-min book AND R/pick of the added picks
≥ +0.30 on every split AND MDD and worst month not worse than the 5-min book's. Tail tests, permutation p across the
6, availability audit (every feature at 09:30+W from bars ≤ 09:30+W), obtainability (Stage Q's measured cap fill:
ask > cap ⇒ no fill), price-scale check, cell count (6 here; cumulative ORB-line count stated).

## Deliverable
`research/orb_multiwindow/REPORT.md`: per-window pick counts and overlap with the 5-min book; the 6-cell table per
split; the added-picks R; MDD/worst month/red months; the verdict line in PLAN §1 phrasing. A survivor → independent
rebuild from prose before any engine work (the engine's range window is one constant, `range_minutes`, but a second
window means a second submission burst and a shared-slot rule — a real change, not a knob).
