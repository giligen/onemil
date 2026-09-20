# frames16 — FREEZE

**TEST is `day >= 2026-06-01`. It is NOT opened in this pass, under any result.**

Splits (PLAN §1, unchanged): TRAIN `2025-01-02 … 2025-12-31` · VAL `2026-01-01 … 2026-05-31` ·
TEST `2026-06-01 … 2026-09-11`.

How each arm is prevented from touching TEST:

* **Arm 1 (instrument calibration)** scores no book and consumes no split. Its single session is
  `2026-03-11`, inside VAL. It produces R², β and event counts — never a return, never a trade,
  never a cell. Nothing it prints can be a TEST reading.
* **Arm 2 (lambda residual)** scores `hod_filter_stack/b2.pkl`, which is TRAIN/VAL-only by
  construction (`hod_frames6/book6.csv`, max day < 2026-06-01). The loader additionally filters
  `split in ('TRAIN','VAL')` before any join, as `ofi.py` already does.
* **Arm 3 (the mirror short)** cuts its signal population at `2026-06-01` **in the builder**, not in
  the scorer, so a TEST row cannot be produced by accident. The hourly profile
  (`frames15/hourly15.parquet`) and the daily panel (`frames15/daily15_*.parquet`) contain rows on
  and after `2026-06-01`; they are used **only** as trailing denominators and as the SSR prior-close
  lookup for decision sessions strictly before `2026-06-01`. No signal, no walk, no trade and no
  cell is produced from a session on or after the TEST boundary.

The SSR rail reaches one session FORWARD (a trigger on day t blocks day t+1). For a decision session
on `2026-05-29` (the last VAL session) the trigger lookup reads `2026-05-28` and `2026-05-29` only —
it never reads forward across the boundary, because SSR is a backward-looking condition.

**Declared exception: none.** If one becomes necessary it is logged here, in writing, BEFORE it is
taken.
