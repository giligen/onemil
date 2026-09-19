# FREEZE — the TEST seal for `research/orb_gates2/`

TEST = **2026-06-01 → 2026-09-16** (~16 market weeks; ~57 picks at the shipped 8-slot book).

## The seal

1. `research/orb_frequency/score.py` refuses to print any TEST number unless `--reveal-test` is
   passed **and** this file already carries the commit hash of the sealed recommendation.
2. TEST is scored **exactly once**, for **exactly two** cells:
   - the shipped B+ baseline (**G0**, N=8), and
   - the **single** recommended cell — or, if the recommendation is PREREG §4's null finding,
     the highest-green-week cell, labelled as a sanity check that selected nothing.
3. The recommendation is written into `REPORT.md` and committed **before** the seal is opened.
   The commit hash is recorded here and TEST is revealed in a **separate** commit.
4. If TEST disagrees, the recommendation is **NOT** re-chosen. The disagreement is reported
   as-is. At ~16 weeks TEST's green-week MDE₈₀ is ≳ 38pp: it cannot select anything.

## State

- **Sealed at**: 2026-09-19, before any cell of this stage was scored. PREREG commit: *(recorded on commit — see git log for `research/orb_gates2/PREREG.md`)*.
- **Recommendation commit**: `91137a8` (2026-09-19) — REPORT.md §7 committed before the seal was opened.
- **Opened at**: 2026-09-19, immediately after `91137a8`, for exactly two cells: the shipped B+
  baseline (G0, N=8) and — the recommendation being the null finding — the highest-green-week
  cell, G5/G8 tied and broken by PREREG §4's tie-break on pooled total R -> **G8**. Both fill
  models. Results in REPORT.md §8a and `test_reveal.csv`.

## Why this matters here

2026-06+ is the only window in which the shipped B+ stack has ever been the live configuration.
Stage 1 (`research/orb_frequency/`) already spent one look at it for two cells. Spending it on
grid search would destroy the one piece of near-live evidence this book has.
