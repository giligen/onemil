# HOD force close vs exit-coverage race — result 2026-11-11

Changes (no commit, no service action):
- `trading/stop_monitor.py`: `mark_closing`/`clear_closing`/`coverage_order_ids`/`_closing_in_flight` (~807-845); `_coverage_skip` yields to a live mark (DEBUG "close in flight"); 15 min expiry logs WARNING and coverage acts again; `remove_watch` clears the mark; the cycle records order ids it placed (~927).
- `trading/hod_break_engine.py`: `force_close_all(only=None)` marks BEFORE any leg cancel (`_mark_closing` ~3206), clears marks once the position left the registry; `_settle_exit_legs` returns "timed out"; `_cancel_own_sells` (~3229) runs on attempt >= 2 or after a leg-poll timeout (own = `hod-` coid, our leg/close ids, coverage-placed ids; foreign sells untouched; waits <= `FC_LEG_POLL_S`); `_adopt_prior_day_open_rows` flags `overnight_from` + WARNING; `_close_overnight_leftovers` (~3758, called from process_tick) closes at >= 09:31 ET with the market ladder (never MOC).
- `trading/orb_engine.py:7539`: mark calls only (`_mark_closing_for_coverage`), cleared by `remove_watch`/expiry.

Tests: `tests/test_hod_force_close_coverage_race_20261111.py` 12 pass (unit + AXSM replay with a real StopMonitor, 0 ERROR). Regression set (exit coverage, held-qty, stop_monitor*, test_hod_*, orb routing) 646 passed / 1 skipped; ORB force-close tests 217 passed.

Deviation: a held_for_orders error on attempt 1 stays WARNING even after a sweep (existing 10/22 test, next attempt re-sweeps); on attempt >= 2 after the sweep it is ERROR.
Not done: no live-API probe, no weekend boot rehearsal. The sqlite integration uses the existing mocked-DB fixtures.
