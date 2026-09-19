# hod_bleed — TEST FREEZE

TEST = **2026-06-01 → 2026-09-11**, sealed from the first line of code.

* `walk3.py` walks only days whose signals are `split ∈ {TRAIN, VAL}`; no TEST-dated bar was read by
  the walk, by `part1.py` or by `cells.py`.
* `hod_break/score.py::SPLITS` resolves to `('TRAIN', 'VAL')` unless `--test` is passed **and** this
  file exists. `--test` was never passed.
* PREREG §6: TEST is opened only if a cell clears the ship bar on TRAIN **and** VAL.

**0 of 32 cells cleared the ship bar. 0 of 32 have a positive VAL ΔnetR. TEST was never opened.**

This file is written at the end of the pass so the record shows the seal was in force throughout,
and it does NOT authorise a TEST read — nothing in this pass earned one.
