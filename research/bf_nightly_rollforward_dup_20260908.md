# BF nightly roll-forward: duplicate rows + one divergent exit (found 2026-09-08 EOD)

- 9/7 22:30 nightly built the gap 9/5→9/7 with monthly chunking → re-walked 9/1–9/4 and appended three rows already in the (regen-7) cache: BBCP, FCUV, PATX 9/4.
- PATX 9/4 came back with a DIFFERENT exit: `stop` 11:34 @13.43 (−$368) vs regen-7 `trail_stop` 10:08 @14.47 (+$268). Same entry/stop/shares.
- Reproduction today (9/8 23:xx) of the exact nightly path for 9/4 alone, with and without `--no-cache`, and the single-symbol backtest with the partial on/off: ALL give the 10:08 @14.47 trail exit (with the P1 partial on: `pp+trail_stop`, +$752 at $2K). The 9/7 row is not reproducible → hypothesis: a transient/truncated fresh bar fetch under `--no-cache` on 9/7 (bars 10:02–10:08 missing → trail never armed). Not proven.
- Fixes: cache deduped keeping regen-7's rows (backup `data/bull_flag_cache_e50_x30.csv.bak.pre_dedupe_20260908`); `scripts/cache_append_dedupe.py` now runs in the nightly before the append (production row is the reference, a re-walk never overwrites it). Follow-up (not done): a bar-count sanity guard on `--no-cache` fetches, or drop `--no-cache` for the roll-forward since the day's bars are already backfilled.
- Day-1 P1 parity: Stage-2 on the live config for 9/8 takes 0 trades (the one raw row, SMU, is a 2x wrapper — removed by the live universe rule); live took 0. Clean.
