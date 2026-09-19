# FREEZE — the TEST seal

**TEST = 2026-06-01 → 2026-08-31 is SEALED.**

## The seal, in operational terms

1. `score.py` computes TRAIN (2025) and VAL (2026-01..05) for every cell and **refuses to
   emit any TEST statistic** unless `--reveal-test` is passed on the command line.
2. `--reveal-test` is passed **exactly once**, and only for **two** cells:
   - the shipped-P1 baseline, and
   - the single frontier point selected by PREREG §5's pre-committed rule.
3. It is passed **after** the recommendation has been written into `REPORT.md`. The
   recommendation text is committed before the reveal; the reveal is a separate commit.
4. No cell is added, removed, re-tuned or re-ranked after a TEST number has been seen.
   If TEST disagrees with the recommendation, the disagreement is **reported**, not
   engineered away, and the recommendation stands as written (a survivor still needs
   its own pre-registration and the owner's word before anything ships).

## Why TEST is weak here, stated before it is opened

TEST is 3 calendar months and contains **~3 shipped-P1 picks** (bf_decay §6: the whole
2026H2 P1 book is 3 trades, mean −0.588R, t = −1.06 against zero). Its MDE₈₀ at the
book's own variance is far above 1R. TEST cannot confirm anything at P1's frequency; it
can only contradict loudly. At the higher-frequency frontier points it carries more
trades and is correspondingly more informative — which is itself part of the finding
(PREREG §5, "resolution inside a quarter").

## Reveal log

| when | cells revealed | committed before reveal |
|---|---|---|
| 2026-09-19, after REPORT.md §11 was committed (`173c88f`) | shipped-P1 baseline; recommended point **F7** | `173c88f` — PREREG, FREEZE, grid, separation, frontier and REPORT §0–§11 all committed before `score.py --reveal-test` was run |

Result of the reveal is recorded in REPORT.md. Nothing is re-ranked afterwards.
