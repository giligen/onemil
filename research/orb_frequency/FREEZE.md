# FREEZE — the TEST seal for `research/orb_frequency/`

TEST = **2026-06-01 → 2026-09-16** (~16 market weeks, ~50 picks at the shipped 8-slot book).

## The seal

1. `score.py` refuses to print any TEST number unless `--reveal-test` is passed **and**
   this file already carries the commit hash of the sealed recommendation (§ Opening below).
2. TEST is scored **exactly once**, for **exactly two** cells:
   - the shipped B+ baseline (F0, N=8), and
   - the **single** recommended frontier point.
3. The recommendation is written into `REPORT.md` and committed **before** the seal is opened.
   The commit hash is then recorded here and TEST is revealed in a **separate** commit.
4. If TEST disagrees with the recommendation, the recommendation is **not** re-chosen. The
   disagreement is reported as-is. TEST is a sanity check on ~16 weeks; at that n its
   green-week MDE₈₀ is ≳ 30pp and it cannot select anything.

## State

- **Sealed at**: 2026-09-19, before any cell was scored.
- **Recommendation commit**: `a12bc6e` (2026-09-19) — REPORT.md §8 committed before the seal was opened.
- **Opened at**: 2026-09-19, immediately after `a12bc6e`, for exactly two cells:
  the shipped B+ baseline (F0, N=8) and the rule's survivor (F4, N=8).

## Why this matters here

`research/green_weeks/` and `research/bf_frequency/` both ran with a sealed TEST and both
survived it. ORB's 2026-06+ split is the only window where the shipped B+ stack has ever been
the live configuration; spending it on grid search would destroy the one piece of near-live
evidence this book has.
