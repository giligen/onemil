# Predicting the next "1,427" before it ships — creative brainstorm (2026-09-26)

Trigger: owner override — "All bs, HOD think out of the box, zoom out, how can you predict the
lookahead that led to the 0.3R, be creative." This replaces the queued mechanical build of cells
1,479/1,480 (PYRAMID / FAILED-BREAK SHORT) for this run; that PREREG is untouched and still queued.

## What actually happened (grounding, not restating)
Cell 1,427's headline +0.33R was not a look-ahead in the classic sense (future info at decision
time) — it was a **silent-fallback data-completeness artifact**. The HOD level was computed from
`data/cache.db`, which is sparse for many symbol-days (e.g. FUN 2025-07-01: 7 bars → level 30.77,
true level from `bars_sip.db` = 31.84). Sparse-but-*correct* cache.db entries turned out to
correlate with real trade outcome, so the rule was quietly selecting on "how complete was our
cache" rather than on price action. Same shape as three other 2026-09 incidents in this repo: the
ORB corpse-gate (stale snapshot silently served), the fetch-completeness bug (retry loop silently
dropped 5,000 tickers), and the config whitelist silently dropping a live key. **Pattern: a cheap/
degraded path silently substitutes for the authoritative one, and nothing ever diffs the two.**

## Ideas to *predict* this class of bug, not just find it after the fact

1. **Twin-source auto-diff as a build step, not a review step.** Every cell that derives a
   decision quantity (level, price, volume) from one store must recompute it from the other
   available store and emit `source_agree_pct` / max delta automatically into the cell's own output
   — before anyone reads a headline R. Turns lens-review (done manually, days later) into a CI gate.

2. **Coverage-conditioned edge test (generic, reusable).** Bucket every fill by bar-count-in-source
   at signal time. A real edge is flat across bucket; an edge concentrated in the low-coverage
   bucket (or with a step-function jump at the completeness threshold) IS the artifact's fingerprint
   — this generalizes past HOD to any cell built on a store with known gaps.

3. **Sparsity-injection placebo.** Take the authoritative store, synthetically thin it to match the
   sparse store's bar-count distribution, and re-run. If the edge reappears under synthetic
   sparsity and dies on the dense feed, that's a mechanical proof, not a judgment call — cheap,
   ~10 lines, could be a standard step in every PREREG that touches cache.db.

4. **Make "bar_count_at_signal" a first-class feature and test whether IT predicts R directly.**
   If it does, the population is reading off vendor/cache completeness, full stop — regardless of
   which "real" feature is being sold as the edge.

5. **Governance cheap-shot: a PREREG footnote requirement.** Any PREREG whose decision level can be
   computed from more than one store must name the store, and state whether it has been diffed
   against the alternative for this exact population. Costs one sentence; would have flagged 1,427
   before it was ever walked.

6. **One shared "shadow-diff harness" utility** (not HOD-specific) — register a fast/cached path and
   a slow/authoritative path for any critical derived field anywhere in the repo (BF, ORB, MACD
   included); sample-diff N% of rows on every run; hard-fail the run above a divergence threshold.
   This is the general fix for the recurring "silent degraded fallback" failure mode, applied as
   routine CI rather than as an adversarial review that only happens after a number looks good
   enough to show the owner.

## Worked example: idea #5 applied to cell 1,478 itself (same run, second pass)
Spawned separately as "the independent rebuilder of cell 1,478" (PREREG_1478.md's own protocol:
Jaccard ≥ 0.95 / kept-mean within 0.03 R vs the builder). Two blockers found before any rebuild
code was written, in order of severity:

1. **PREREG_1478.md re-imports the exact bug it exists to answer.** Item 3 (universe/sector
   breadth) is sourced "from bars_sip.db + cache.db across the universe" and item 5 (symbol
   persistence, 60 sessions) "from bars_sip.db / cache.db" — both name TWO stores for one derived
   quantity with no stated single-source rule or diff, the precise gap idea #5's footnote
   requirement targets. Before either feature's importance is trusted, it needs the idea #1
   twin-source diff and the idea #2 coverage-conditioned edge test, specifically on those two
   feature groups (breadth, persistence) — not on the fill-level book generically.
2. **The builder side does not exist yet to compare against.** `research/hod_entry/` has
   `features_1478_B.csv` / `features_1478_C.csv` (partial, from `build_features_1478_C.py`,
   11:45/12:06) but no `model_1478_predictions.csv` or `RESULT_1478.md`; the build process
   (`run_1478C.pid`) is no longer running. A faithful independent rebuild of all ~30 causal
   features (universe breadth, SIC-2 peer panel, tick-level odd-lot/trade-size, 60-session
   persistence) is a multi-store, multi-hour engineering job in its own right — exactly the kind
   the harness expects as ONE resumable nohup script, not something to rush inline. Doing it
   carelessly to produce *a* number would recreate the same failure mode this whole thread is
   about (a plausible-looking result nobody diffed against its source). Not attempted this pass;
   flagging the blocker is the honest output, per "a null is a claim about MY test first."

## What this does NOT do
Does not touch cells 1,479/1,480, config.yaml, cache.db or bars_sip.db. No PREREG frozen, no
population walked, no number reported. This is process/tooling ideation only, per the override.
