# FREEZE — hod_preopen_regime

TEST (2026-06-01 → 2026-09-11) was **NEVER OPENED** in this pass.

No cell cleared the claim bar (G1: 0 of 162), so per `PREREG.md` §5 the TEST split was not read.
`score3.py` / `supp.py` / `halves.py` compute TRAIN and VAL only unless `--test` is passed AND this
file carries a committed recommendation, which it does not.

Committed recommendation: **STAY DRY**. `config.yaml hod_break` unchanged
(`enabled: true, dry_run: true`). See `REPORT.md`.
