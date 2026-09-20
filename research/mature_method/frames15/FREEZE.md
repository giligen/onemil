# frames15 — FREEZE

TEST is `day >= 2026-06-01`. It is **NOT** opened in this pass, under any result.

* Arm A scores HOD-break's `B2` population, which is TRAIN/VAL-only by construction
  (`hod_frames6/book6.csv`, max day < 2026-06-01).
* Arm B's standalone books cut their signal population at `2026-06-01` in the builder, not in the
  scorer, so a TEST row cannot be produced by accident. The multi-day holds additionally drop any
  decision session whose exit session would land on/after `2026-06-01`.
* The per-stock hourly profile and the dense daily panel are built over 2024-2026 for the trailing
  windows; rows on/after `2026-06-01` exist in the intermediate parquet **only** as denominators for
  nothing — no signal, no trade and no cell is scored from them.

Declared exception: none. If one becomes necessary it is logged here BEFORE it is taken.
