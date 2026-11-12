# PREREG 1,736 — short the FAILED TOUCH of the HOD (owner 2026-11-12 "do all tonight")

FROZEN 2026-11-12 19:30 UTC before any number. Programme count on the HOD line: 1,735 → 1,736. One cell, one
primary exit, one secondary exit, two universes (all shortable / ETB-now). Rules: no git, no service action, no
orders, read-only on every DB/CSV; write only `research/hod_entry/1736_*` and `RESULT_1736.md`. Heavy-DB blackout:
call `research/hod_exit_lab/walker.wait_for_db_window()` (blocks until 20:05 UTC) BEFORE the first bars_sip.db
query; query by (symbol, day) through `walker.fetch_day_bars(con, symbol, day)` (imports from that module), never a
table scan; open the DB read-only (`file:...?mode=ro`); `nice -n 19`. Never Read a file > 300 lines in full. Budget
≤ 60 calls. This task IS the owner's request; do not pivot.

## Prior (disclose in the RESULT)
Cell 1,675 (`RESULT_1675.md`): shorting OUR OWN FILLED long after a post-entry failure model fires — the flagged longs
lose −0.5 to −0.8 R but "the short does not capture the decline" — closed. This cell is a different event: the level
is TOUCHED and REJECTED without the long ever filling (the arm's resting stop-limit never fills); the trapped buyers
are the ones who chased through the level. Nobody has scored this event. The old "unfilled breaks −0.57 R" (1,427)
was measured on the sparse-cache artifact book and is NOT evidence; this cell measures the event fresh on SIP bars.

## Population
Arms = `research/hod_entry/replay_signals.csv` (8,248 rows, 2025-07-01..2026-05-29; columns day, symbol, entry_m,
level, split, half, usable, void_reason, e1_fill, …). Keep every row with a `level` (drop void_reason == no_level);
`e1_fill` is NOT used to define the event (the event is defined from bars below) but is reported beside it.
`entry_m` is the minute-of-day (ET) of the ARM bar j = the last closed bar before the break bar; verify on 20 random
rows that bar j's high < level ≤ max(high) of the bars that follow, and state the result. Reads: TRAIN = 2025H2
(split TRAIN), VAL = 2026-01..05 (split VAL). Bars: `research/bf_zero/bars_sip.db` (ET minute-of-day `m` from
walker.fetch_day_bars; session VWAP from `walker.session_vwap`). Prior close: `data/cache.db` `daily_bars`
(symbol, bar_date, close), the last bar_date < day. Shortability: ONE read-only call with the HOD paper key
(`ALPACA_HOD_API_KEY/SECRET`, `GET https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity`),
keep (symbol, shortable, easy_to_borrow) in `1736_shortable.csv` — TODAY's flags, not point-in-time: disclosed, and
the score is reported on "all arms" AND "shortable-now" AND "ETB-now".

## Event (per arm, from bars at or after entry_m + 1)
1. Touch bar T = the first bar with h ≥ level. No touch → no event.
2. Failure bar F = the first bar in T..T+3 whose CLOSE ≤ level × (1 − 0.002), provided no bar in T..F has a close ≥
   level × 1.006 (that would be a successful break — the long's cap — not a failure). No such F → no event.
3. Rails at F: SSR proxy — the close at F ≥ 0.90 × prior close (unknown prior close = fail closed, no trade); time —
   F's minute ≤ 15:30 ET; one event per (day, symbol) (the first).
4. Short entry = the OPEN of bar F+1, hit the bid: price − half_spread − 2 bp. Stop (buy) = max(h over T..F) ×
   1.002; R = stop − entry; require R ≥ 1.0 % of entry else no trade (R-must-exceed-spread; report how many drop).
5. Exits, primary **E1**: cover at target = entry − 2 R if a bar's l ≤ target (fills at the target); stop if a bar's
   h ≥ stop (fills at stop + 13 bp slip, gap-through at the bar open); both in one bar = stop; else cover at the
   15:55 open. Secondary **E2**: cover at the open of the bar after the first CLOSE ≥ session VWAP (the reclaim), else
   stop / 15:55 as E1. Cover legs pay half_spread + 2 bp (stop leg: + the 13 bp above instead).
Cost input: half_spread per (day, symbol) from `research/hod_exit_lab/signals.parquet` when the pair matches (it does
for every arm that filled in the exit lab); for unmatched arms use the median half_spread of matched arms in the same
price decile (deciles cut on `level`) and report the unmatched share.

## Scoring (E1 and E2 × {all, shortable-now, ETB-now} × {TRAIN, VAL})
n arms, n touches, n events (event rate), n trades after rails, events/week; mean net R, day-clustered t (cluster =
day), iid t, ex-top-5 % (drop the 5 % best trades), winner-capped (+3 R), stop share, green-week share, weekly P10,
max drawdown (R), mean R as % of price. Placebo (300 draws, VAL and TRAIN): for each trade, the same short geometry
(R, exits) started at a random minute AFTER F+1 and before 15:00 on the same (day, symbol) — placebo minutes must come
after the signal window (`feedback_placebo_window_selection`) — percentile of the observed mean under the placebo
distribution. Also report the B0-counterfactual: what the long would have made had it filled at the level (from
`research/hod_exit_lab/b0_trades.csv` when the pair exists) — descriptive only.

## Pass bar (pre-committed)
E1 on the ETB-now universe PASSES only if: mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL; ex-top-5 % ≥ 0 on
both; placebo percentile ≥ 95 on VAL; ≥ 3 trades/week on VAL at the ETB-now universe; green-week share ≥ 55 %.
E2 and the other universes are secondary (reported; a lone pass there = "secondary, needs its own PREREG").
A pass = a PAPER instrument in the HOD engine's existing failure-short order path (dry first), never live.

## Output
`research/hod_entry/1736_score.py` (end-to-end), `1736_events.csv` (one row per arm: day, symbol, entry_m, level,
split, touch_m, fail_m, event flag, rails, entry, stop, R, r_pct, e1_exit, e1_why, e1_net_R, e2_exit, e2_why,
e2_net_R, shortable, etb, half_spread, matched_spread flag, e1_fill of the long, b0_net_R), `1736_summary.csv`,
`RESULT_1736.md` ≤ 350 words with the tables, PASS/FAIL against the bar, the caveats (flags not PIT, SSR proxy,
borrow cost 0 assumed, spread fallback share, uptick rule on SSR names), the arm-bar verification, regime-episode
count. Return ≤ 120 words: event rate, E1 ETB-now mean R / day-t / n per read, PASS/FAIL.

## Independent rebuild (second agent, has not read 1736_score.py)
From this prose: recompute the event flags, entry, stop and E1 net R for every arm on TRAIN and VAL; compare to
`1736_events.csv` on (day, symbol, entry_m): flags identical, prices to 1e-6, net R to 0.001; write `REBUILD_1736.md`.
Mismatches > 0 rows = the number does not reach the owner.
