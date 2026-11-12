# PREREG 1,740 — M1 retest bid RE-SCORED on TRUE high-of-day levels (replaces 1,737 M1; 1,739 found the level artifact)

FROZEN 2026-11-12 23:20 UTC before any number. Programme count on the HOD line: 1,739 → 1,740. Rules as PREREG_1736
(no git / service / orders; read-only everywhere; write only `research/hod_entry/1740_*` and `RESULT_1740.md`;
`walker.fetch_day_bars` on a read-only `bars_sip.db` URI; `nice -n 19`; background scripts with log polling; never
Read a file > 300 lines in full; budget ≤ 60 calls). This task IS the owner's request; do not pivot.

## Why
`RESULT_1739.md`: `replay_signals.csv` levels come from the sparse `cache.db`; in 82 % of arms they sit BELOW the
true running HOD (median 1.9 %). 1,737 M1 (+0.18 R) therefore bid under the real high at a level that is not knowable
live — withdrawn. This cell scores the SAME mechanism on the true level, from `bars_sip.db`, on TRAIN and VAL; TEST
stays sealed and is read ONCE only if TRAIN and VAL pass.

## Population and the true level
Signals = `research/hod_exit_lab/signals.parquet` (day, symbol, entry_m = the break minute, split, half, half_spread;
TRAIN 7,390 / VAL 4,745 / TEST 3,521). Level = the running high of day BEFORE the break bar: max(h) over the bars
with m < entry_m from 09:30 (reuse `level_from_bars` in `1739_gen.py` if it does exactly this — state its definition;
`1739_arms_sipLevel.csv` already holds TEST). Write `1740_arms.csv` (day, symbol, entry_m, level, split, half). Sanity
(report): the share of signals whose break bar has h ≥ level (must be ≈ 100 %), and the distribution of
(B0 `entry` − level)/level (the chase paid by the break buyer).

## M1 on the true level (conventions of REBUILD_1737, unchanged)
C = first bar at/after entry_m with close ≥ level; bid at `level` from C+1; cancel on a close ≥ level × 1.02 or at
15:30; fill at `level` on the first bar with low < level (strict) — AND the 10 bp through-trade variant (low <
level × 0.999); stop = min low of the 5 bars before the break bar, raised to level × 0.985 if closer (floor 1.5 %);
target fill + 2 R; stop fills at the stop (gap-through at the open) + 13 bp; both in one bar = stop; else the 15:55
open; costs entry 2 bp, target 2 bp, stop / 15:55 exits half_spread + 2 bp; a missing bar → the next available.
Secondary (reported, no pass): the same bid at level × 0.99 (1 % under the true high — close to what 1,737
accidentally measured, so the owner can see what the artifact was worth).

## Scoring (TRAIN H1, TRAIN H2, TRAIN, VAL; TEST sealed)
n, fill share, mean net R, day-clustered t, iid t, ex-top-5 %, winner-capped, stop share, green-week share, weekly
P10, max drawdown, fills/week; the capped book (8/day, 4 concurrent, first-come); the filled vs unfilled cohorts'
B0 net R (adverse-selection direction); placebo 300 draws (same geometry from a random minute after C+1).

## Pass bar (pre-committed)
Strict fill: mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL, both TRAIN halves same sign; 10 bp rule ≥ +0.05
on both; ex-top-5 % ≥ 0; placebo ≥ 95 on VAL; ≥ 3 fills/week. Pass → (same agent) read TEST ONCE with the identical
code and report it beside TRAIN/VAL (TEST bar: ≥ +0.10 strict, t ≥ 2, 10 bp ≥ +0.05). Fail → stop, report, TEST
untouched.

## Output
`1740_gen.py`, `1740_score.py`, `1740_arms.csv`, `1740_orders.csv`, `1740_summary.csv`, `RESULT_1740.md` ≤ 350 words:
the sanity numbers, TRAIN/VAL tables for strict / 10 bp / 1 %-under, capped book, PASS/FAIL, and TEST if read.
Return ≤ 120 words: fill share, mean R / day-t / n per read (strict, 10 bp), capped book, PASS/FAIL, TEST if read.
An independent rebuild from this prose follows before the number reaches the owner.
