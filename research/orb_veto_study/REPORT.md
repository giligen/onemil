# ORB V1 veto study — REPORT

## Run 1 (2026-09-06 21:20 UTC) — INVALID, discarded
The first pass used `research/orb_signal_study/features_base.csv` as the exit-resim dump. Its `pnl` is the features CSV's fixed +2R / −1R exit (the trap CLAUDE.md warns about), not the shipped static lock: the baseline came out $3,244 / 130 picks against the honest $6,085 / 130, with `exit_reason=target` rows. The pipeline accepted it because the keys matched. Nothing from that pass is evidence. Fix: a full bar-walk dump with `ORB_BT_DUMP_CANDIDATES` (static-lock exits for every candidate), baseline must reproduce $6,085 / 130 before any veto row counts. Era columns now read `_sized_pnl` (stage scale, same as the monthly table).
