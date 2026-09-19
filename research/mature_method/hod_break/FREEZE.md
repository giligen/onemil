# FREEZE — mature_method / HOD-break

Written 2026-09-19, at the moment the recommendation in `REPORT.md` was committed.

**Recommendation committed: STAY DRY AS INSTRUMENT. Recommended action: NONE.**
No config, `orb.yaml`, systemd unit, cron, order or cache was written; `hod_break.enabled: true,
dry_run: true` is left exactly as the owner set it on 2026-09-19.

**TEST (2026-06-01 → 2026-09-11) WAS NOT OPENED.** `PREREG.md` §3 states that only a cell clearing
G1 (TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week) AND G2 (VAL same sign, >= 55 % green weeks)
opens TEST, once. **0 of 23 cells cleared G1** — every TRAIN mean net R is negative and the best
TRAIN t is -1.76 — so G2 was never evaluated and TEST was never read. `score.py` refuses to compute
a TEST number without `--test`, and `--test` additionally requires this file to exist.

The only TEST figures anywhere in `REPORT.md` are in §1, where they are `research/bf_zero/REPORT.md`
§6a's own already-published reproduction row (43.7/wk, +0.016 R, green 9/15, worst -17.5) being
reproduced to the digit as the runbook's step-1 gate. No cell was scored on TEST.

If a future pre-registration on this book wants TEST, it must declare its cells first and carry the
disclosure that this pass already consumed 46 cell x split decisions on TRAIN and VAL.
