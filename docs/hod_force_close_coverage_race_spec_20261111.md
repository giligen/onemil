# HOD force close vs exit-coverage race (AXSM 2026-11-11) — spec 2026-11-11

Incident (paper HOD account, journal 2026-11-11 20:55–20:57 UTC, AXSM 9 sh):
1. 20:55:00 `force_close_all` cancels the AXSM stop leg, polls ≤ `FC_LEG_POLL_S` 10 s, the leg is not terminal, sells
   anyway → Alpaca 40310000 `held_for_orders` → attempt 1 WARNING "retrying in 60s" (`hod_break_engine.py:3352`).
2. 20:55:33 the StopMonitor exit-coverage cycle (`stop_monitor.py:_exit_coverage_cycle` → `exit_coverage.repair`)
   sees the position with no live stop and places a NEW stop-sell @ 144.90 → ERROR on Telegram (`exit_coverage.py:212`).
3. 20:57:23 force-close attempt 2 → `held_for_orders` again (the coverage stop holds the 9 sh) → ERROR
   "FORCE CLOSE AXSM FAILED" (`hod_break_engine.py:3354`); no further attempt before the 16:00 close; the DAY stop
   expired; AXSM 9 sh is held OVERNIGHT (−$9 paper). Two guards fought each other; both alarms were self-inflicted.

Rules: no git (no commit/stash), no service action, no orders (never `--submit`, never a broker write — the only
broker calls allowed are reads with the HOD paper keys), never write orb.yaml / config.yaml / .env / data/*.db /
crontab / logs/*. pytest only on files that exist, `--ignore=tests/integration`, one run at a time, never piped
through head/tail. Never Read a file > 300 lines in full (grep / sed ranges). `MagicMock(spec=...)` for domain
classes. Budget ≤ 40 calls. **The live service boots from this working tree at 12:30 UTC — leave no file
half-edited; every edit must leave the module importable.** This task IS the owner's request ("Errors on
telegram!"); do not pivot.

## Fix (mechanism, ONE path each)
1. **Coverage yields to a close in flight.** The engine marks the symbol as closing on the StopMonitor BEFORE it
   cancels any leg (`stop_monitor.mark_closing(strategy, symbol)` / `clear_closing`; the mark is cleared when the
   position leaves `self.positions` or the close order is terminal-filled). `_coverage_skip` returns True for a
   marked symbol with a DEBUG line "close in flight"; the mark expires on its own after 15 min (WARNING when it
   expires with the position still open — then coverage acts). Same for the ORB engine's force close if it shares
   the coverage thread (grep `mark_closing` call sites you add; ORB code is frozen except kill/risk rails — this is
   a risk rail, keep the ORB edit to the mark/clear calls only).
2. **Force-close retry cancels what holds the shares.** On every attempt ≥ 2 (and on attempt 1 after the leg poll
   timed out), before submitting: list open orders for the symbol on this account, cancel every open SELL that is
   ours (client_order_id prefix `hod-` or the watch's `sl_leg_id` / coverage order id — never a foreign order),
   poll ≤ `FC_LEG_POLL_S` until terminal, then submit. A `held_for_orders` error AFTER that cancel-and-wait is a
   genuine ERROR; before it, WARNING only.
3. **Overnight leftover closes at the open.** `sync_positions` at boot: a restored open position whose
   `session_date` < today logs WARNING "{sym}: overnight leftover from {session_date} — closing at the open" and is
   force-closed by the existing path at the first eligible tick ≥ 09:31 ET (same pricing ladder), not at 15:55.
   Its StopMonitor watch is re-armed as today so coverage keeps it covered until then.

## Tests
`tests/test_hod_force_close_coverage_race_20261111.py`: unit — (a) `_coverage_skip` True while marked, False after
clear and after expiry (WARNING asserted); (b) attempt ≥ 2 cancels the open own sell orders first and waits for
terminal before `submit_limit_sell_order` (call order asserted on `MagicMock(spec=AlpacaClient)`), foreign sells
untouched; (c) a persisted position dated yesterday is closed at 09:31, not 15:55. Integration — temp sqlite DB +
mocked client replaying the AXSM sequence (cancel → leg pending → coverage cycle runs → no new stop → retry →
sell submitted → position closed, 0 ERROR records in caplog). Run the new file + `tests/test_exit_coverage_20261029.py
tests/test_hod_force_close_held_qty_20261023.py tests/test_stop_monitor*.py tests/test_hod_*.py` (files that exist)
→ 0 failures.

## Output
`docs/hod_force_close_coverage_race_20261111_RESULT.md` ≤ 200 words: file:line per change, test counts, anything
you could not do. Return ≤ 100 words.
