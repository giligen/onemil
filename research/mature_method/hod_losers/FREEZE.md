# FREEZE — hod_losers

**TEST (2026-06-01 → 2026-09-11) WAS NEVER OPENED.**

PREREG §3 seals TEST behind the claim bar G1. `cells.py` scored all 20 declared cells on TRAIN and
VAL only (`score.SPLITS = ('TRAIN','VAL')`; `--test` was never passed and this file did not exist
while the cells ran). **0 of 18 cell-rows passed G1** — every TRAIN mean net R is negative, the best
is −0.0018 R (P14 / B0) at t −0.03, and the best TRAIN gross in the pass is **+0.0572 R** against the
**+0.25 R** the bar asks and the **+0.2151 R** measured cost the book must clear. G2 was therefore
never evaluated and TEST was never read.

No TEST-dated bar was loaded by `walk.py` or `walk2.py` either: both walk only the days of the
TRAIN+VAL signal sets (`sg.split.isin(('TRAIN','VAL'))`), 344 sessions, last day 2026-05-29.

Written after scoring, before the report.
