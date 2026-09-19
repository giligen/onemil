# FREEZE — the TEST seal for `research/green_weeks`

**TEST is SEALED** until the recommendation is written and committed.

| book | TRAIN | VAL | TEST (sealed) |
|---|---|---|---|
| ORB | 2025-01-01 → 2025-12-31 | 2026-01-01 → 2026-05-31 | 2026-06-01 → 2026-09-16 |
| HOD | 2025-01-01 → 2025-12-31 | 2026-01-01 → 2026-05-31 | 2026-06-01 → 2026-09-11 |
| BF  | 2025-01-01 → 2025-12-31 | 2026-01-01 → 2026-05-31 | 2026-06-01 → 2026-08-31 |

## The seal, operationally

1. `score.py` computes TRAIN and VAL for every cell and **refuses to emit any TEST
   statistic** unless `--reveal-test` is passed with an explicit list of cell ids.
2. `--reveal-test` is passed **exactly once per book**, and only for **two** cells:
   the shipped exit (E0) and the single cell selected by PREREG §2's ranking rule and
   §8's gate.
3. It is passed **after** the recommendation has been written into `REPORT.md`. The
   recommendation is committed first; the reveal is a separate commit.
4. No cell is added, removed, re-tuned or re-ranked after a TEST number has been seen.
   If TEST disagrees with the recommendation, the disagreement is **reported**, and the
   recommendation stands as written.
5. If the pre-committed gate selects **no** cell for a book, TEST is **not opened** for
   that book at all.

## Why TEST is weak here, stated before it is opened

TEST is 14–15 weeks. §7's unpaired half-width on a green-week share at that size is
**±25pp**. TEST cannot confirm a week-shape finding; it can only contradict a large
one. It is opened for honesty, not for power.
