# frames14 — FREEZE

TEST is `day >= 2026-06-01`. It is NOT opened in this pass.

* **F45** is pure MEASUREMENT of quoted spreads, not a book score. Its minute-of-day sample is drawn
  from the HOD measured-NBBO population, which is itself TRAIN/VAL-only (`bf_zero/causal_filter/nbbo.csv`,
  max day < 2026-06-01). The BF re-pricing uses `bf_frequency/runs/P1.csv` and the regen-7 raw cache,
  both of which end 2026-04. **The ORB books re-read here (Stage P/Q) print a TEST column because
  Stage P selects nothing** — that column is quoted verbatim from those committed reports and no new
  TEST number is produced.
* **F43** walks pass-6's arm-d controls and pass-12's floor keys, both cut at `2026-06-01`.
* **F44** scores the overnight panel with `TEST_FROM = 2026-06-01` (frames13 `f40.load`). The
  **2016-01 → 2024-12** era check is entirely before TRAIN and is not TEST.

Declared exception: none. If one becomes necessary it is logged here BEFORE it is taken.
