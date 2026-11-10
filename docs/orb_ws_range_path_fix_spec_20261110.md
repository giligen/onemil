# ORB websocket range path: seed the bar window with the session-open bars — fix spec 2026-11-10

Diagnosis: `docs/orb_ws_range_path_diagnosis_20261110_RESULT.md` (read it first) and the failing reproduction
`tests/test_orb_ws_range_path_20261110.py::test_ws_range_completes_when_candidate_admitted_after_the_open_bar`.
Root cause (confirmed on the real StopMonitor): a symbol's rolling bar window in `trading/stop_monitor.py` starts
EMPTY when ORB calls `subscribe_bars(symbol)` (~994–1009), which happens only after the gap gate admits the symbol
(always after the 09:30 bar). Bulk-streamed symbols (`subscribe_bars_many`, HOD) get NO window (~1011–1020, ~1142).
So `_first_session_open_ts_utc` never finds the 14:30Z/13:30Z bar and the websocket has never completed an ORB
range; the REST sweep at 09:35 has been the only source (11/10 it failed → no ranges all day).

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env / data/*.db /
crontab / logs state files. pytest only on the listed files, `--ignore=tests/integration`, one run at a time, never
piped through head/tail. Never Read a file > 300 lines in full. This task IS the owner's request; do not pivot.
Budget ≤ 40 calls. `fast_submit` stays false; do not touch it.

## 1. StopMonitor keeps the session-open bars for EVERY streamed symbol (bounded)
- New `self._open_bars: Dict[str, List[dict]]` — for each bar received (bulk or windowed) whose timestamp is on
  today's session date, keep at most `OPEN_BARS_KEEP = 8` bars per symbol **starting from the first bar at/after
  09:30 ET** (ignore pre-market bars; use the existing ET helpers in the module — grep how the module computes ET /
  the session date today; do not invent a new offset). Memory bound: 8 × ~4,000 symbols × ~100 B.
- Clear `_open_bars` wherever the per-day bar state is reset (find the daily reset path; if none exists for bars,
  clear it when the first bar of a NEW session date arrives — one place, logged once at INFO).
- `subscribe_bars(symbol)` (both branches: new symbol and bulk→windowed switch): when creating/initialising the
  window, seed it with `_open_bars.get(symbol, [])` (copy, chronological, de-duplicated by timestamp against anything
  already in the window). Log at DEBUG "seeded {n} session-open bars".
- Bar handler fan-out is unchanged.

## 2. ORB bulk-subscribes its universe seed before the open
In `trading/orb_engine.py`, where the daily universe seed is loaded (grep "ORB universe seed" in
scanner/realtime_scanner.py:~946 and the engine's prewarm/`build_universe` path): once per session date, if
`self.stop_monitor` has `subscribe_bars_many`, call it with the seed symbols (ZZZT test tickers excluded as the
universe already does) and log INFO "[ORB] bulk-subscribed {n} seed symbols for session-open bars". Must run before
09:30 ET on a normal boot (12:30 UTC boot → prewarm) and be idempotent (a second call the same day subscribes only
new names — `subscribe_bars_many` already returns the count of new ones; log only when > 0).

## 3. Tests (unit + integration per the bug protocol)
- The existing failing reproduction must PASS unchanged (do not edit its assertions).
- `tests/test_stop_monitor_open_bars_20261110.py`: unit — bulk symbol receives 14:30Z..14:36Z bars, then
  `subscribe_bars` → window holds the 14:30..14:34 bars in order (and later bars appended live); cap at 8; pre-market
  bar ignored; window already holding a bar with the same timestamp is not duplicated; new-session-date reset clears.
- `tests/test_orb_ws_range_path_20261110.py` (append): integration with the REAL StopMonitor + REAL ORBEngine
  (mocked Alpaca client `spec=`): seed bulk-subscribed pre-open, 09:30–09:34 bars streamed, gap gate admits at
  09:31:40, bar 09:34 closes → "range complete" from the WS path with exact H/L, BEFORE any REST sweep; then the
  sweep runs and counts the symbol as `superseded` (the first non-zero superseded in the codebase's history — assert
  `counts`/log).
- ORB engine unit: the bulk-subscribe call happens once per day with the seed, idempotent.
- Run: `python3 -m pytest tests/test_stop_monitor_open_bars_20261110.py tests/test_orb_ws_range_path_20261110.py
  tests/test_stop_monitor*.py tests/test_orb_engine*.py tests/test_orb_post_open_sweep.py
  tests/test_orb_range_sweep_hardening_20261110.py tests/test_hod_break*.py -q --ignore=tests/integration` → 0 failures.

## Output
`docs/orb_ws_range_path_fix_20261110_RESULT.md` ≤ 200 words: file:line of the changes, where the daily reset lives,
test counts, anything you could not do. Return ≤ 120 words.
