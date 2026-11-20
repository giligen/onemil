# ORB: exit coverage books our own rested target fill as `external` and the target poll then raises a false ERROR (2026-11-20)

## The incident (verified read-only: journal, data/trades.db row 848, broker GET on the ORB paper account)
* 14:35:30 UTC SPCG entry bracket submitted (295 sh, limit 16.91, SL 15.22, TP 50.73); filled 14:35:47 @ 16.8837.
* 14:36:00 the touch-and-go target rested: `[ORB TP] rested SPCG @ $16.89 (leg b46723bf -> a682c8f9, coid=orb-tp-SPCG-20261120)`;
  StopMonitor watching (stop 16.42), `mark_target_resting` set `watch.tp_leg_id = a682c8f9`, `target_resting=True`, `target_price=16.89`.
* 14:36:19 the TP leg a682c8f9 FILLED 295 @ 16.910983 (broker), the SL leg was cancelled by the OCO. Broker flat.
* 14:36:45 the StopMonitor exit-coverage cycle (`_exit_coverage_cycle` → `_book_external_flat`, trading/stop_monitor.py ~940-968)
  saw the position flat, read `get_last_filled_sell` = order a682c8f9 @ 16.9110 and emitted the exit with
  `exit_reason=ExitReason.EXTERNAL`, `pricing_method="external"`, retiring the watch:
  `EXIT COVERAGE: SPCG closed externally — booking 295 sh @ 16.9110 (order a682c8f9), watch removed`.
* 14:37:05 the engine tick's `_poll_target_fills` (trading/orb_engine.py ~6977-7030) saw the TP leg `filled` on the order
  stream and called `stop_monitor.book_target_rested_fill(...)`, which found no watch → WARNING
  `no active watch (already retired by a race?)` → returned False → engine ERROR
  `SPCG book_target_rested_fill failed for an order-stream-confirmed FULL fill (295sh @ $16.91) — DB row will be
  reconciled by the next sync_positions pass instead` (that sentence is also wrong: `sync_positions` runs at boot only).
* 14:37:06 the drained coverage event closed row 848: `exit_reason='external'`, exit 16.910983, pnl +8.06. Money is right;
  the REASON is wrong (it was our own target fill, `target_rested`) and the ERROR reached the owner for nothing.
* First occurrence (no `book_target_rested_fill failed` in logs/session_archive). Structural: a target that fills just
  after an engine tick is seen by the 60-s coverage cycle before the next tick's poll, every time. Precedent for this
  family: docs/orb_scaleout_coverage_conflict_20261113.md (D5 added `_book_external_flat`).

## The fix (TDD: failing tests first). Two parts, both in trading/stop_monitor.py; no engine, strategy or constant change.
### Part 1 — coverage recognises our own rested target
In `_book_external_flat`, after the "last filled sell belongs to this watch" check: when
`watch.target_resting and str(fill.get("id") or "") == str(watch.tp_leg_id or "") and watch.tp_leg_id`, the flat IS the
rested target's fill. Book it through the SAME event the order-stream path emits: factor the event construction of
`book_target_rested_fill` (~1996-2020: `StopExitEvent(... exit_reason=ExitReason.TARGET_RESTED.value,
exit_branch=ExitBranch.LIMIT.value, pricing_method='target_rested_limit', exit_limit_price=limit_price, confirmed=True ...)`)
into one private helper `_queue_target_rested_event(symbol, watch_fields, filled_qty, fill_price, limit_price, order_id)`
used by both callers (the order-stream path passes `order_id=''` today — pass the real id when known, keep '' acceptable).
Log INFO `[ORB TP] {symbol} target FILLED (exit coverage) — {qty}sh @ ${px:.4f} (limit ${watch.target_price:.2f}, order {id[:8]})`
instead of the "closed externally" line; retire the watch and clear `_exit_in_progress` exactly as `book_target_rested_fill` does.
Any other flat (id differs, not target_resting, empty tp_leg_id) stays `external` exactly as today.
### Part 2 — the late poll is not an error when the exit is already booked
* Record every exit booking: in `_emit_stop_exit_event` and in the target-rested helper, set
  `self._booked_exits[symbol] = (exit_reason, qty, fill_price, time.time())` (new dict, created in `__init__`, guarded by
  `_watch_lock` or `_exit_lock` — pick the lock the surrounding code already holds and say which).
* In `book_target_rested_fill`, the `watch is None` branch: look up `_booked_exits.get(symbol)`; if it exists, is younger
  than `_BOOKED_EXIT_TTL_S = 600` and its reason is `ExitReason.TARGET_RESTED.value` → log INFO
  `[ORB TP] {symbol} target fill already booked by exit coverage ({qty}sh @ ${px:.4f}) — nothing to do` and return True
  (the engine then sets `target_last_booked_qty` and stops polling). Any other case (no record, stale record, a different
  reason such as a stop exit — that would be a genuine double-fill anomaly) keeps today's WARNING and returns False, and
  the engine ERROR stays.
* Fix the engine ERROR text only (trading/orb_engine.py ~7020-7030): replace "DB row will be reconciled by the next
  sync_positions pass instead" with "row NOT booked by this path — check the row by hand (sync_positions runs at boot only)".
  No other engine change.
### Tests — `tests/test_orb_target_coverage_race_20261120.py` (fixtures: `mon`, `client` from
`tests/test_orb_scaleout_coverage_conflict_20261113.py` — import them like that file imports from test_orb_add_on; read
`test_stale_sell_before_the_watch_is_not_booked` there for how `get_last_filled_sell`/`get_open_positions` are mocked, and
`tests/test_orb_target_limit.py` for how a watch gets `mark_target_resting`):
1. SPCG replay: watch added, `mark_target_resting(tp_leg_id='a682c8f9', target_price=16.89)`, broker flat,
   `get_last_filled_sell` → {'id':'a682c8f9','price':16.910983,'filled_at': now} → `_book_external_flat` returns True, the
   drained event has `exit_reason == ExitReason.TARGET_RESTED.value`, `pricing_method == 'target_rested_limit'`,
   `exit_limit_price == 16.89`, `shares == 295`, `confirmed is True`; caplog has the "(exit coverage)" INFO and NO
   "closed externally" line; watch gone.
2. Same but the sell id differs ('zzz') → `external` exactly as today (existing behaviour pinned).
3. Same but `target_resting=False` (bracket TP never rested) → `external`.
4. After test 1's booking, `book_target_rested_fill('SPCG', 295, 16.910983, 16.89, rested_at)` → True, INFO "already
   booked", no WARNING record.
5. No booking recorded → WARNING + False (existing behaviour pinned); a recorded `stop_triggered`-type reason → WARNING + False.
6. A record older than 600 s → WARNING + False.
7. Engine level (engine fixture style from the 11/13 file): `_poll_target_fills` with the order stream reporting `filled`
   and `stop_monitor.book_target_rested_fill` returning True → no ERROR record and `pos.target_last_booked_qty == filled_qty`;
   returning False → exactly one ERROR whose text contains "sync_positions runs at boot only".
8. Integration (real `StopMonitor` with the mocked client + real `ORBEngine` drain, as the 11/13 file's DB round-trip test
   does): coverage books the SPCG target → the engine drains it → `db.update_trade` is called with `exit_reason ==
   'target_rested'` and `exit_price == 16.910983`.

## Run (these files only; the main session runs the full suite post-close — NEVER start it yourself)
`python3 -m pytest tests/test_orb_target_coverage_race_20261120.py tests/test_orb_target_limit.py
tests/test_orb_scaleout_coverage_conflict_20261113.py tests/test_stop_monitor*.py tests/test_orb_exit_coverage*.py
-q -p no:cacheprovider --ignore=tests/integration` → all green (list the files that exist; skip a glob that matches nothing).

## Rules
Sonnet, ≤ 60 calls. The LIVE service is running: no `git commit`/`git stash`, no service restart, no orders, no writes to
`data/*.db`, `logs/*`, `orb.yaml`, `config.yaml`, `.env`. Files you may edit: `trading/stop_monitor.py`,
`trading/orb_engine.py` (the one ERROR string only), the new test file, and a "Result" section appended to THIS file.
Do not change `_coverage_skip`, the coverage cadence, `ExitReason`, or any constant other than adding `_BOOKED_EXIT_TTL_S`.
Docstrings on every new function; every fallback logs WARNING. Return ≤ 150 words: files changed, test counts, which
lock guards `_booked_exits`, anything you could not do, verified facts separated from guesses.

## Result (2026-11-20)
Implemented: `_book_external_flat` books our own rested target as `target_rested` via new `_queue_target_rested_event` (shared with
`book_target_rested_fill`); `_booked_exits` (guarded by `_watch_lock`, written in the same hold that pops the watch) + `_BOOKED_EXIT_TTL_S = 600`;
late poll returns True with INFO when a young target_rested record exists; engine ERROR text fixed. New tests: 9 (file
`tests/test_orb_target_coverage_race_20261120.py`); run set 277 passed (no `test_orb_exit_coverage*` file exists).

## Review (main session, 2026-11-20 15:0x UTC)
`_queue_target_rested_event` now CLAIMS the watch first (pop under `_watch_lock` together with the `_booked_exits` write)
and queues nothing when the pop returns None — the coverage thread and the engine's order-stream poll may both hold
the same watch; one fill must produce one event. `tests/test_orb_target_coverage_race_20261120.py` has 10 tests
(`test_watch_claimed_once_when_both_paths_race` added); run set 278 green. Commit after the post-close full suite
(the service runs the working tree at the next boot, Monday 11/23). Row 848 (SPCG 11/20) keeps `exit_reason='external'`
in the DB — P&L correct, label wrong; a one-row relabel is the owner's call.
