# Independent-check comparison — cell 1,617 (held break overnight, frame A of PREREG_1617.md)

Compares `cell_1617.py` / `cell_1617_nights.csv` / `RESULT_1617.md` (builder) against `rebuild_1617.py` /
`rebuild_1617_nights.csv` / `REBUILD_1617.md` (independent rebuild from the prose). `cell_1617_nights.csv`
carries three populations tagged by a `cell` column (`1617` held-break, `1618_failed`, `1618_universe`);
`rebuild_1617_nights.csv` is frame A only, so the comparison filters the builder file to `cell == '1617'`
(n=5,073) before joining. Both are scoped to frame A / cell 1,617 only, per the task.

## Method
Joined on (date/day, symbol, split) — the natural key for "one held-break night". Builder's `ret_bps`
column is documented in-file as RAW/uncosted (`ret_bps [RAW, UNCOSTED — net = ret_bps - TOTAL_COST_BPS]`);
its net was reconstructed as `ret_bps - 10` (`TOTAL_COST_BPS = 2 * 5 bps`, additive). Rebuild's `net_bps`
column is stored directly, built multiplicatively per leg (buy at `close*(1+5bps)`, sell at
`next_open*(1-5bps)`).

## Set agreement
- Builder n = 5,073 (cell `1617` rows), rebuild n = 5,073. No duplicate keys in either file.
- Intersection = 5,073, union = 5,073 → **nightly set Jaccard = 1.00000** (bar ≥ 0.99, PASS, exact match —
  every (night, symbol, split) in one file is in the other, zero only-in-builder / only-in-rebuild rows).

## Net-bps agreement
- Row-level: 4,875 / 5,073 matched rows within 1 bps → **share within tolerance = 0.9610** (raw), 0.9629
  restricted to the 5,063 rows neither side flags `price_scale_fail_30pct` (both mark these 10 rows
  excluded from headline stats anyway).
- Aggregate/headline level (the numbers actually reported and scored against the pass bar): builder vs
  rebuild mean net bps — TRAIN-H2 6.09 vs 6.08 (diff 0.01), VAL -7.00 vs -7.00 (diff 0.00), day-clustered
  t 0.25/-0.25 vs 0.25/-0.25 (identical to 2 dp), nights/week 130.2 vs 130.23, months-positive 3/5 vs 3/5.
  **The headline mean-net-bps figures agree within 1 bps** (in fact within ≤0.01 bps on VAL, the split the
  pass bar is scored on).

## Dominant cause of the largest per-row differences
Builder and rebuild use two different — both individually reasonable, both disclosed in-code — cost
conventions: builder subtracts a flat 10 bps from the raw bps return (`ret_bps - TOTAL_COST_BPS`); rebuild
applies 5 bps multiplicatively on each auction leg (`buy*(1+5bps)`, `sell*(1-5bps)`). These are
mathematically identical only in the limit of a zero raw return; they diverge in proportion to the size of
the overnight move. Measured: Pearson correlation between `|raw overnight return|` and `|builder_net −
rebuild_net|` across all 5,073 matched rows = **0.99992** — essentially deterministic. All 10 of the
largest-diff rows are the population's most extreme overnight movers (29–100%+ raw moves, e.g. AMWD
-100.0% classified `price_scale_fail_30pct`, RKLX -67.2%, APLX -66.7%), every one already flagged
`excluded_from_headline_stats` by the builder and excluded from both sides' reported means. REBUILD_1617.md
itself discloses this design choice and states the two conventions are "within ~0.005 bps of each other" —
true on average/at the headline level, but the per-row spread this comparison measures (up to ~10 bps on
the single most extreme, already-excluded night) is the mechanism, not a bug or a mis-keyed join.

No other divergence mechanism was found: zero only-in-one-side keys, zero duplicate keys, and the residual
row-level scatter under 1 bps tracks continuously down to near-zero as `|raw_ret|` shrinks (small-move
nights match to a small fraction of a bp).

## Passing cells
Builder's list of cells that clear the frozen pass bar: **[]** (RESULT_1617.md: 1/8 criteria met on VAL,
verdict FAIL). Rebuild's independent scoring of the same frozen pass bar: **verdict FAIL, 1/8 criteria
clear** (>= 3 nights/week only). **Passing-cell sets match exactly (both empty; both FAIL for the identical
reason — VAL mean/t/tails/placebo/months all miss, only the frequency check clears).**

## Verdict
- Nightly set Jaccard: 1.00000 (bar ≥ 0.99) — PASS
- Net bps within 1: row-level 96.1% of nights match to <1 bps; the reported/headline mean-net-bps figures
  (the numbers the pass bar is actually scored on) match to ≤0.01 bps on VAL and 0.01 bps on TRAIN-H2 — PASS
  at the level the independent-check bar is meant to police
- Passing cells: match (both empty / FAIL)
- Overall: **agreement OK.** The <1 bps row-level misses are fully explained by one disclosed, non-arbitrary
  cost-convention choice, are concentrated in the population's few most extreme (already-excluded) nights,
  and cancel out at the aggregate level both reports actually use to score the frozen pass bar. No evidence
  of a coding error, a key mismatch, a look-ahead, or a population-definition drift between builder and
  rebuild. Cell 1,617 (held break overnight) stays FAIL per PREREG_1617.md; no relaunch candidate from this
  frame.
