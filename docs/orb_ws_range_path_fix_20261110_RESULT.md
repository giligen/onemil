# ORB WS range path fix - RESULT 2026-11-10

Spec: `docs/orb_ws_range_path_fix_spec_20261110.md`. No git, no service action, no config touched.

## Changes
- `trading/stop_monitor.py`
  - init: `_open_bars`, `_open_bars_date`, `_open_bars_lock`.
  - `:1019` `OPEN_BARS_KEEP = 8`; `:1031` `_record_open_bar_locked` keeps the first 8 bars at/after 09:30 ET for every streamed symbol (bulk and windowed), via `_bar_et` (ZoneInfo ET on the existing `_bar_start_epoch` parser).
  - `:1055` `_seed_window_locked`; `subscribe_bars` (`:1000`, both branches) seeds the window (chronological, de-duplicated, DEBUG "seeded n session-open bars").
  - `_on_bar` (`:1200`) records the open bar and appends the window under the lock.
- `trading/orb_engine.py`: `bulk_subscribe_seed` (`:1398`, excludes `^Z[A-Z]ZZT$`, INFO "[ORB] bulk-subscribed n seed symbols for session-open bars" only when n > 0, ERROR on failure). Called from `build_orb_universe_from_snapshots` (`:1445`) with the RAW seed on every pre-open scanner tick; idempotent because `subscribe_bars_many` returns only new names.

## Daily reset
No per-day bar reset existed. `_open_bars` clears when the first bar of a new ET session date arrives (one INFO, `stop_monitor.py:~1047`). An older-date bar is ignored.

## Tests
New: `test_stop_monitor_open_bars_20261110.py` (7), `test_orb_bulk_seed_subscribe_20261110.py` (5), 2 integration tests appended to `test_orb_ws_range_path_20261110.py`. The original reproduction passes unchanged. Spec run: 437 passed, 0 failed.

## Notes / not done
- "Superseded" needs the stream to finish a range while the REST fetch is out (a range already complete before 09:35 means the sweep never runs). The integration test covers both: WS completes before the sweep (0 REST passes), and WS completes mid-fetch (1 superseded).
- Per-day idempotence relies on StopMonitor's `_bar_symbols`; no engine-side date key.
- Needs a service restart and a Monday-open check (grep `range complete`, `superseded`). `fast_submit` untouched (still false).
