# FREEZE — hod_fresh

**TEST (2026-06-01 → 2026-09-11) is SEALED.**

`pass3.py` walks bars for **2025-01-02 → 2026-05-31 only** (TRAIN + VAL). No TEST-dated bar is read
by this study's bar pass, so no TEST number can be computed by accident.

`score4.py` additionally refuses to compute a TEST number unless BOTH of the following hold:
this file exists AND `--test` is passed on the command line.

TEST is opened **once**, and only for a cell that has already passed G1 on TRAIN and G2 on VAL, with
the recommendation committed to the repository before the seal is broken.

## Seal status

| date | action | by |
|---|---|---|
| 2026-09-19 | sealed at pre-registration (`PREREG.md`, commit `b1e5470`) | hod_fresh pass |

**The seal was never broken. No TEST-dated bar was read by `pass3.py`; no TEST number appears in
`REPORT.md`.**
