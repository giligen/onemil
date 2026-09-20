# frames13 — FREEZE

TEST is `day >= 2026-06-01`. It is NOT opened in this pass.

* F40's panel scan cuts at `2026-06-01` for the scored 2025/2026 cells. The **2016-01 → 2024-12** era
  cell (A16) is entirely before TRAIN and is not TEST.
* F41 re-reads pass-7 artifacts, which were built TRAIN/VAL only.
* F42 re-walks TRAIN/VAL trades only.

Declared exception: none. If one becomes necessary it is logged here BEFORE it is taken.
