# PREREG 1,739 — sealed out-of-sample read of the M1 retest bid (cell 1,737) on 2026-06..09 arms — ONE SHOT

FROZEN 2026-11-12 22:40 UTC before any number. Programme count on the HOD line: 1,738 → 1,739. Rules as PREREG_1736
(no git / service / orders; read-only everywhere; write only `research/hod_entry/1739_*` and `RESULT_1739.md`;
`walker.fetch_day_bars` on a read-only `bars_sip.db` URI; `nice -n 19`; background script + log polling if > 5 min;
never Read a file > 300 lines in full; budget ≤ 60 calls). This task IS the owner's request; do not pivot.

## Why
1,737 M1 (`RESULT_1737.md` verdict section, `REBUILD_1737.md`) passed on TRAIN 2025H2 and VAL 2026H1 — two halves of
the same era. The exit lab's sealed TEST window is 2026-06-01..2026-09-04 (`research/hod_exit_lab/signals.parquet`
split == 'TEST', 3,521 fills; `bars_sip.db` covers it). The arm table `replay_signals.csv` stops at 2026-05-29, so
the TEST arms must be GENERATED with the same rule — this is the only new work, and it is scored once.

## Arms for the TEST window
Find how `replay_signals.csv` was produced: grep `research/hod_entry/PREREG_170*.md`, `RESULT_170*.md` and
`research/hod_entry/*.py` for "replay_signals" and "causal_arming"; use `research/hod_entry/causal_arming.py` (the
cell 1,438 causal arming: `HodBreakParams` defaults from `trading/hod_break.py`, universe = the exit lab's TEST
(day, symbol) set — `signals.parquet` split TEST gives the symbol-days that FILLED; the arm population also needs the
UNFILLED arms, so if the generator needs the universe of symbol-days, use `research/hod_exit_lab/pm_candidates.csv`
(symbol, bar_date) restricted to the TEST dates, or the same universe file the 1,700-series replay used — state
which). Output `1739_arms.csv` with the SAME columns as replay_signals (day, symbol, entry_m = the break minute,
level, split = 'TEST'). Sanity: on the TEST symbol-days that are in `signals.parquet`, ≥ 90 % must get an arm whose
level is within 0.1 % of the fill's `entry` − the chase cap; report the match share; < 90 % = STOP and report, no
score.

## Score (exactly 1,737 M1, the rebuild's conventions)
C = first bar at/after entry_m with close ≥ level; bid at level from C+1, cancel on close ≥ level × 1.02 or 15:30;
fill at level on the first bar with low < level (strict) — AND the 10 bp through-trade variant; stop = min low of
the 5 bars before the break bar, raised to level × 0.985 if closer; target fill + 2 R; stop fills at stop (gap at
the open) + 13 bp; both-in-one-bar = stop; else 15:55 open; costs entry 2 bp, target 2 bp, stop / 15:55 half_spread
(signals.parquet match on day+symbol, decile-median fallback) + 2 bp; missing bar → next available. Report: n, fill
share, mean net R, day-clustered t, ex-top-5 %, winner-capped, stop share, green-week share, weekly P10, fills/week;
the capped book (8/day, 4 concurrent, first-come); placebo 300 draws (same geometry from a random minute after C+1).

## Pass bar (pre-committed, sealed: the number is read ONCE)
TEST mean net R ≥ +0.10 with day-t ≥ 2.0 (strict fill) AND ≥ +0.05 with the 10 bp rule AND ex-top-5 % ≥ 0 AND
placebo ≥ 95. Pass → the dry/paper retest-bid mode spec goes ahead. Fail → M1 is a 2025H2/2026H1 regime fact and
stops here (report the TEST number beside TRAIN/VAL, no re-cut).

## Output
`1739_arms.csv`, `1739_gen.py`, `1739_score.py`, `1739_orders.csv`, `RESULT_1739.md` ≤ 300 words (arm-match share,
the table TRAIN/VAL (from RESULT_1737) beside TEST, capped book, PASS/FAIL, caveats). Return ≤ 100 words: arm
match share, TEST mean R / t / n strict and 10 bp, capped book, PASS/FAIL. A second agent rebuilds the TEST score
from this prose on `1739_arms.csv` (compare on day, symbol, entry_m to 0.001 R) before the number reaches the owner.
