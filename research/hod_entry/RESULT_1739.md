# RESULT 1,739 — STOPPED at the arm-match gate, NOT SCORED (2026-11-12)

Programme count 1,739 (no score read). Files: `1739_gen.py`, `1739_gen_check.txt`, `1739_arms.csv` (TEST, cache.db rule), `1739_arms_sipLevel.csv` (TEST, true-HOD level from bars_sip.db). `1739_score.py` is written and untested (never run).

**How replay_signals.csv was made.** `entry_replay.py` (cells 1,423-1,425): one row per HOD-break B0 signal of `hod_exit_lab/signals.parquet` (entry_m = break minute), level = `compute_level` = max high of CLOSED bars before the break minute, taken from `data/cache.db` `intraday_bars_1min` (not bars_sip.db). Covers 2025-07-01..2026-05-29 (8,248 rows). TEST arms = the 3,521 TEST rows of signals.parquet (same detector population, entry_m identical in 100 % of TRAIN/VAL rows).

**Gate (>= 90 %) FAILED on every reading.**
- Generator (same rule, cache.db as of today) vs replay_signals levels on TRAIN/VAL: 62.1 % within 0.1 % (TRAIN 60.1, VAL 63.7). cache.db has been backfilled since 9/25, so the old levels are not reproducible.
- True-HOD level (bars_sip.db) vs replay_signals levels: 17.8 % agree; in 82 % of arms the replay level is BELOW the true running HOD (median gap 1.9 % when they differ; e.g. GUSH 2025-07-08: 23.65 vs true 24.62, cache.db held 5 bars).
- TEST: cache.db level vs bars_sip level agree 49 % (TRAIN/VAL 55 %).
- The PREREG's literal gate (level vs "fill entry minus chase cap") has no tick data for TEST (hod_ofi/raw ends 2026-05-29) and is not operationalizable.

**Implication.** The 1,737 M1 TRAIN/VAL numbers (+0.18 R) were measured on sparse-cache.db levels, mostly under the true HOD, so M1 as scored is not "bid at the HOD". A valid TEST read needs a decision: (a) re-score TRAIN/VAL M1 on the true-HOD level first, or (b) freeze a TEST rule on `1739_arms_sipLevel.csv` under a new PREREG. No TEST number was computed.
