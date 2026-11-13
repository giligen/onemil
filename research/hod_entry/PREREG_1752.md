# PREREG 1,752 — ANTICIPATION entry: buy BEFORE the HOD break with a strict stop, filtered by the statistical
# probability of the break (volume, time-to-HOD, daily moving averages) — data-driven on winners vs losers

FROZEN 2026-11-13 19:10 UTC before any number. Programme cell count: 1,751 → 1,752. Owner's ask 11/13: "Buy
before HOD? With strict SL? Buy with filters on statistical probability based on volume or time to HOD … and based
on moving avg of x days". Rules: write only `research/hod_entry/1752_*` and `RESULT_1752.md`; no git, no service,
no orders, no network; `research/bf_zero/bars_sip.db` read-only URI (`file:...?mode=ro`), opened ONLY after 20:05
UTC (`walker.wait_for_db_window`), `ulimit -v 2500000; nice -n 19`, peak RSS < 1.5 GB (per-day chunks, parquet
checkpoints in `1752_cache/`); budget ≤ 70 calls. This task IS the owner's request; do not pivot.

## Why this is not 1,481 / 1,742 / 1,751
Those bought the RETEST (after the break) or the confirmed hold. 1,752 buys the APPROACH: price below the running
HOD, before any break, so the entry is cheaper than the break price and the stop can sit under the approach bar —
the question is whether a causal filter makes P(break | approach) and the break's follow-through large enough to
pay for the approaches that fail. Every feature is computed at the approach bar; nothing from the break.

## Population (the honest one — INCLUDES approaches that never broke)
All (day, symbol) of the HOD-break population's universe: the symbol-days in `research/hod_entry/1735_per_signal.csv`
PLUS, for the same days, every symbol in the scanner-qualified universe that the 1,438 build used
(`cell_1428_population.csv` / the 1,438 universe source — the agent locates it and states the file; if only the
arm-day list exists, say so and use the arms' symbol-days as the population, reporting that it is break-conditioned
and therefore an UPPER bound). 1-min bars from bars_sip.db. TRAIN/VAL/TEST splits as 1,735 (`split` column);
TEST sealed.

## Approach event (causal)
Running HOD = max high from 09:30 to minute t−1, set ≥ 10 minutes before t (a level exists). Approach at t = the
first minute after 09:45 ET where high_t ≥ HOD × (1 − d) and close_t < HOD, d ∈ {0.5 %, 1.0 %} (one event per
symbol-day per d; secondary: a second approach after a failed first). Entry = close_t + half_spread + 2 bp (market
at the approach bar close; obtainable). Strict stop S ∈ {low of the approach bar, min low of the last 3 bars,
entry − 0.5 %} (stop slip as PREREG_1740). Exits: (a) no break within K ∈ {10, 20, 30} minutes → exit at close_{t+K};
(b) break → the LIVE rule from the break (stop stays S, target = entry + 2 × (entry − S), else close 15:55).
R unit = entry − S (the strict stop).

## Features at t (all causal; the owner's three plus the engine's)
Volume: vol_5 / vol_day_avg, vol_t / vol_5, cumulative volume vs the 20-day ADV share by minute. Time: minute of
day, minutes since the HOD was set, number of prior approaches today (attempts), minutes since the open move.
Daily MAs from the daily bars on disk (`research/signal_search/1748_cache` / `1746_cache`, ranks 1–3,000; else the
`daily_bars` table): prev_close vs MA5 / MA10 / MA20 / MA50, MA20 slope (5-day), HOD vs the 20-day high (is the HOD
a multi-day breakout level?), gap % and prior-day range. Market: SPY return since the open.

## Stage A — anatomy on TRAIN only
Two outcomes per approach: BREAK within K (binary) and net R (the trade). Winners vs losers by net R sign: the
distribution of every feature (deciles, AUC, day-clustered t). P(break | feature decile). Then the ONE-rule grid:
each single feature threshold × d × S × K → TRAIN mean net R, n/wk, ex-top-5 %, both halves same sign; rank by
TRAIN-H2; pick the best with ≥ 3 fills/wk; record the FULL grid (multiplicity visible). Also the unfiltered book
(every approach) so the filter's lift is explicit.

## Stage B — frozen rule on VAL (one read)
n/wk, net mean R, day-clustered t, ex-top-5 %, winner-capped +3 R, weekly P10, green-week share vs count-matched
null, year signs, P(break) on VAL vs TRAIN (calibration), placebo = the same rule on a random minute of the same
symbol-day away from the HOD (the signal vs the name-day), cost ×1.5, MDE at 80 % power.

## Kill (pre-committed — any one kills)
VAL net ≤ 0 or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % ≤ 0; < 3 fills/wk; placebo ≥ the rule on
VAL; P(break) under the rule on VAL < TRAIN − 10 pp (filter not stable). Survive → TEST once → independent rebuild
→ dry-run entry mode spec (`entry_mode: anticipation`, d, S, K, the filter).

## Output
`1752_build.py` (bars → approach events + features, chunked), `1752_score.py`, `1752_events.parquet`,
`1752_anatomy.csv`, `1752_grid.csv`, `1752_val.csv`, `RESULT_1752.md` ≤ 350 words (winners-vs-losers reading
first: which of volume / time-to-HOD / daily-MA features separate, then P(break) calibration, then the rule and
VAL). Return ≤ 120 words.
