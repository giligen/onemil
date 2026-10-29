# Exit-coverage invariant + alert flood cap — RESULT (2026-10-29)

Spec: `docs/exit_coverage_invariant_spec_20261029.md`. Nothing committed, no service restarted (the running trader
keeps the old code until its next restart).

## Files
* NEW `trading/exit_coverage.py` — `snapshot`, `check`, `repair`, `covered_qty`, `CoverageResult`.
* `trading/stop_monitor.py` — `WatchEntry.added_at`; ctor kwarg `exit_coverage_enabled` (default True);
  `_start_exit_coverage_thread` (own daemon thread, started from `start()`, works in WS and polling mode);
  `_exit_coverage_loop` (90 s first run, 60 s period, exits at once if disabled); `_exit_coverage_cycle`.
* `config.py` (`exit_coverage_enabled` -> `stop_monitor.exit_coverage_enabled`, default true) + `main.py` wiring.
  config.yaml NOT touched.
* `data_sources/alpaca_client.py` — `get_open_orders` rows gain `order_class` and `type` (additive).
* `notifications/telegram_notifier.py` — `message_pattern`, `_pattern_cap_gate`, wired in `send_message`.
* Tests: `tests/test_exit_coverage_20261029.py` (21), `tests/test_telegram_pattern_cap_20261029.py` (12).

## Client calls
position `AlpacaClient.get_open_positions()`; open sells `AlpacaClient.get_open_orders()` (flat, includes bracket
legs); old-leg status `get_order(id)`; repair `submit_stop_sell_order(symbol, qty, stop_price)` (same call as the
EMERGENCY stop-market path). Stop = `watch.stop_price` (the lock stop is written there when armed).

## Deviations / judgement calls
* Each account is read ONCE per cycle (2 REST calls per client, not 2 per watch); a symbol that looks naked is
  re-checked fresh before any order. Well inside the spec's call budget.
* OCO / bracket / OTO legs count ONCE (largest leg), other open sells add up — this is what makes the LABX shape
  (OCO 200 of 423) repair 223.
* Exit-in-progress uses the existing `_exit_in_progress` + `_exit_started_at` (stale after 300 s), no new flag.
* Pattern gate runs BEFORE `_rate_cap_gate` (not after): a flooded pattern must not eat the 20/min budget.
* Correction (coordinator): repair qty is capped at `watch.shares` (`min(watch.shares, broker qty) - covered`); broker
  excess is never repaired, WARNING once per symbol per session (`... excess N sh not ours, not touched`).

## Tests
Slice (spec command): 238 passed. Full `tests/` `--ignore=tests/integration`: 5341 passed, 6 skipped, 0 failed.

## Not done
No real-API probe / weekend boot rehearsal (agents may not touch the account); `get_open_orders` limit is 500 rows.
