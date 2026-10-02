# ORB open-tick stall 2026-10-01/02 (09:35 decision lost two days running)

## Timeline (UTC; ET = UTC-4), 2026-10-02
* 12:32:00 scanner warms the daily_bars sqlite prefilter (2,473 symbols) - a DIFFERENT cache from the engine's snapshot cache.
* 13:30:41 scanner fetch 34.0 s; 13:30:46 `ORB prewarm cache MISS 3041/3041 ... 5.53s (cache already had 0)`.
* 13:30:47-13:36 312 `stale-snapshot reject` lines 1-10 s apart; 13:31 on `Engine tick TIMEOUT`; 13:32:24 `prewarm
  cache STALE (ALL cached snapshots flipped)`; `Cycle overrun: WORK took 226.3s`; no decision at 13:35.

## Root causes
1. (a) `_snapshot_cache` (`trading/orb_engine.py` `_get_snapshots_warm`) is not the sqlite prefilter
   (`scanner/realtime_scanner.py` ~L854). Pre-open snapshots have open<=0 / yesterday-dated bars, so the engine cache
   cannot be usefully warmed before 09:30; first fetch (5.5 s) is cheap and is NOT the stall.
2. (b) `_is_complete` (open>0 and daily_bar_date == today ET) correctly flips every entry stale at the open: a bar still
   dated yesterday carries YESTERDAY's open, serving it would feed the gap gate a wrong gap (9/25 review finding).
   Decision: rule kept; the cost is one more REST batch (2.5 s measured 13:32:27), not a stall. Not a bug.
3. (c) THE stall = N+1 SQL: `build_orb_universe_from_snapshots` called `Database.get_intraday_bars_cached(sym, today)`
   per symbol; without an index hint SQLite served ORDER BY from the (symbol,timestamp) autoindex and scanned each
   symbol's whole history on the 84M-row `intraday_bars_1min`: 212 ms/query (replay, 50-symbol sample, node loaded),
   so ~3,000 x = 226 s live. The "1 symbol/s" log spacing is this query, not the corpse gate (pure in-memory).
4. (d) The 09:35 decision shares the engine tick thread; the universe build ran inline with no deadline, so tick
   TIMEOUTs piled up and the decision waited behind it.

## Fixes (selection / vetoes / sizing / orders untouched)
* `persistence/database.py`: `INDEXED BY idx_intraday_bars_symbol_date` on the per-symbol query (212 -> 0.9 ms) and new
  `get_intraday_bars_for_date(symbols, date)` - chunked `symbol IN (...) AND bar_date=?`, index seeks only.
* `trading/orb_engine.py`: one batched lookup before the candidate loop; `deadline` (default
  `execution.open_tick_budget_sec`, absent = 20 s): past it `_get_snapshots_warm` defers the REST refresh (WARNING with
  count) and the loop stops (ERROR `N of M candidates not evaluated ... deferred, not rejected`).
* Parity: same data, same admission code; `test_bulk_open_equals_per_symbol_open` asserts the batched 09:30 open equals
  the per-symbol result on a fixture (admitted sets identical with and without batching).

## Tests (`nice -n 19`, targeted files)
`tests/test_orb_open_tick_stall_20261002.py` (9) + `tests/test_orb_open_tick_replay.py` (1) + prewarm/fast-submit
files: 29 passed. Covers: yesterday-dated snapshots at the open re-fetched not served; warm cache hit = 0 REST; one
bulk query / zero per-symbol; batched == per-symbol; expired deadline WARNING; scan-cut deferred count; slow scan
returns inside budget. Full suite: pending (run after 20:05 UTC).

## Replay (`python scripts/orb_open_tick_replay.py --date 2026-10-02`, cache.db `mode=ro`, REST stubbed at 5.53 s/3041)
* BEFORE: per-symbol legacy SQL x 3,041 = 644.7 s (node under load; 226 s live).
* AFTER: real engine tick 8.2 s (5.5 s stub REST + batched lookup + gate), budget 20 s, 1 REST call.
* Caveat: cache.db has no 2026-10-02 daily bar/minute bars, so opens fall back to the prior close (0 admitted); timing
  is valid, the admitted set is not (parity is shown by the fixture test).

## Monday boot check
`journalctl -u onemil-trader --since 13:29 | grep -E "open-tick budget|prewarm cache (MISS|STALE)|WORK took"` - expect
no `open-tick budget` line and the first-tick `WORK took` well under 20 s; service restart needed to load (not done).
