# PREREG_1623 (cell 1,623) — builder vs. independent rebuild, compare

Inputs: `cell_1623_fills.csv` (builder, 9,911 rows) vs `rebuild_1623_fills.csv` (rebuild, 9,911 rows),
keyed on `day+symbol`. 100% key match (9,911/9,911, 0 left-only, 0 right-only), 0 duplicate keys on
either side. Split labels reconcile 1:1 after normalizing builder's `TRAIN-H2` to rebuild's `TRAIN`
(REBUILD_1623.md documents this naming choice itself; 0 mismatches after normalization, VAL 5,513 / TRAIN
4,398 both sides). REBUILD_1623.md states it was written from `PREREG_1623.md` prose only, without
opening `cell_1623.py` / `cell_1623_fills.csv` / `RESULT_1623.md` — a genuine independent build.

## Bar: Jaccard >= 0.99, kept means within 0.01 R — CLEARS, essentially exact agreement

| metric | value |
|---|---|
| kept-set (G+) Jaccard, ALL splits | **1.0000** (687/687 kept both sides, 0 disagreements) |
| kept-set (G+) Jaccard, VAL | **1.0000** (356/356 kept both sides) |
| kept-set (G+) Jaccard, TRAIN-H2 | **1.0000** (331/331 kept both sides) |
| VAL kept mean net R, builder | 0.0490 (n=356) |
| VAL kept mean net R, rebuild | 0.0490 (n=356) |
| VAL kept mean diff (rebuild − builder) | **0.0000 R** |
| TRAIN-H2 kept mean diff | **0.0000 R** (−0.3899 both sides) |
| outcome_R row-level agreement (9,911 matched rows) | max abs diff 0.000000, 0 rows differ |
| F row-level agreement (8,384 rows with F defined both sides) | max abs diff 0.000000, 0 rows differ |
| n_res row-level agreement (9,911 rows) | max abs diff 0, 0 rows differ |
| gate_plus (kept flag) row-level disagreements | **0 / 9,911** |

Passing set (frozen pass-bar, scored on VAL): **builder = [] (Verdict: FAIL, RESULT_1623.md line 62);
rebuild, scored the same way = [] too.** `same_passing` = MATCH.

## Passing-set reconciliation, criterion by criterion (VAL, frozen bar in PREREG_1623.md)

| criterion | builder (RESULT_1623.md) | rebuild (REBUILD_1623.md) | agree? |
|---|---|---|---|
| kept mean net R >= 0.15 | FAIL 0.0490 | FAIL 0.0490 | yes, exact |
| day-clustered t >= 2.5 | FAIL 0.34 | FAIL 0.34 | yes, exact |
| ex-top-5% > 0 | FAIL −0.0534 | FAIL −0.0534 | yes, exact |
| fills/wk (12/4) >= 3.0 | PASS 4.09 | PASS 4.09 | yes, exact |
| dropped < kept, VAL half | PASS (−0.1857 < 0.0490) | PASS (−0.1857 < 0.0490) | yes, exact |
| dropped < kept, TRAIN-H2 half | FAIL (−0.1486 not < −0.3899) | FAIL (−0.1486 not < −0.3899) | yes, exact |
| TRAIN-H2 same sign as VAL, t >= 1.0 | FAIL (mean −0.3899, t −3.51) | FAIL (mean −0.3899, t −3.51) | yes, exact |
| kept cache-only share within 5pp of 19.5% | PASS 17.4% | PASS 17.4% | yes, exact |
| placebo margin >= 0.10 R and t >= 2.0 | FAIL (margin 0.1548, t −0.61) | PASS (margin 0.6215, t 3.70) | **NO — sole disagreement** |

(Builder writes "dropped < kept on BOTH holdouts" as one AND'd checklist line; rebuild reports the VAL and
TRAIN-H2 halves as two rows. Same underlying numbers on both sides either way — not a real split, just a
table-layout difference.)

8 of 9 rows (7 of 8 builder-numbered criteria) agree exactly, to the reported decimal, on numbers built
from a fully independent re-derivation of the population, the exit-minute source, F, n_res, and the gate.
The 9th disagrees in direction, but even flipping it to PASS does not change the cell's verdict, because 5
other criteria (kept mean, t, ex-top-5%, TRAIN-H2 sign+t, TRAIN-H2 half of dropped<kept) already fail
identically on both builds. **Verdict agrees: cell 1,623 does not clear the frozen bar under either build
— nothing ships.**

## Dominant cause of the one disagreement (placebo, VAL: t −0.61 vs t 3.70)

Both scripts call `np.random.RandomState(1623)` — same class, same seed integer — but permute a
**different object**, confirmed by reading the code directly:

- **Builder** (`cell_1623.py:328-330`): `rng.permutation(days)` permutes the array of **unique day
  values**, then maps `old_day -> new_day` through a dict applied per row. This is a day-BLOCK relabel:
  every fill that was on day X stays grouped with the same fill-mates, just wearing a different day's
  name. Within-day co-occurrence (which fills can resolve which) is fully preserved.
- **Rebuild** (`rebuild_1623.py:237-238`): `rng.permutation(other['day'].to_numpy())` permutes the
  **per-row day column directly**. Each individual fill gets an independently reassigned day label, so
  fills that were never on the same real day can now share a shuffled "day," and fills that were on the
  same real day get scattered apart. Within-day co-occurrence is destroyed and replaced by an arbitrary
  new grouping.

Same seed, same RNG class, structurally different permutation target -> different random-stream
consumption -> different shuffled populations (VAL-conditioned-on-shuffled-TRAIN-H2: builder n=208 vs
rebuild n=75; TRAIN-H2-conditioned-on-shuffled-VAL: builder n=186 vs rebuild n=92) -> different placebo
means and t-stats. Both documents flag this exact risk in their own caveats before this comparison was
run: RESULT_1623.md calls it "a SINGLE permutation draw... noisier than the 1,000-draw count-matched null
used elsewhere in this line... a different draw could move the margin/t meaningfully"; REBUILD_1623.md
calls its own reading "this rebuild's own interpretation of a terse prose line... a different, equally
defensible reading... would give a different placebo number." PREREG_1623.md's wording ("days shuffled")
does not disambiguate day-block relabeling from per-fill day reassignment — this is a spec ambiguity that
independently produced two different draws, not a coding error in either script.

## Bottom line

Everything load-bearing for the cell — population, split assignment, exit-minute source, F, n_res, the
G+ kept set itself, and every reported mean/t built on it — reproduces EXACTLY between two independent
implementations: Jaccard 1.0000 on both VAL and TRAIN-H2, 0.0000 R mean diff, zero row-level
disagreements across 9,911 matched fills. Both bars in this task (Jaccard >= 0.99, means within 0.01 R)
clear with no slack. The one real disagreement is confined to a single, explicitly-flagged-as-noisy
placebo draw whose shuffle mechanism the PREREG prose left ambiguous; it flips one of nine checklist rows
but not the cell's verdict. **Cell 1,623 is confirmed CLOSED (FAIL) by an independent rebuild — no ship
action follows.** If this placebo criterion is ever load-bearing on its own for a future cell, pin the
exact shuffle operation (block-relabel vs. per-row reassignment) in the PREREG prose, and re-run with
several seeds per RESULT_1623.md's own caveat.
