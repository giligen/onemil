# FREEZE — hod_frames6

TEST (2026-06-01 → 2026-09-11) is SEALED for this pass.

No cell in `hod_frames6` computes, prints or stores a TEST number. Every score script filters to
`split in ('TRAIN', 'VAL')` before any statistic is taken. The seal is lifted only by a committed
recommendation in `FRAMES.md` naming the single cell to be read on TEST.

Splits (unchanged, `hod_break/score.py::SPLIT_RANGE`):

| split | range |
|---|---|
| TRAIN | 2025-01-02 → 2025-12-31  (H1 = < 2025-07-01, H2 = >= 2025-07-01) |
| VAL   | 2026-01-01 → 2026-05-31 |
| TEST  | 2026-06-01 → 2026-09-11  — **SEALED** |
