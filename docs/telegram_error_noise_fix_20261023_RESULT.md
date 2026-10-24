# Telegram ERROR noise fix 2026-10-23 — RESULT (not committed)

## Files changed
* `trading/orphan_reconciler.py` — `peer_strategies_for()`, `peer_strategies` kwarg on `reconcile_strategy_orphans`, `_recent_any_strategy_trade(..., peer_strategies)` (SQL `strategy IN` on the legacy path, row filter on the public-API path; None = old behaviour).
* `trading/trading_engine.py`, `trading/orb_engine.py`, `trading/macd_wave_engine.py` — callers pass `peer_strategies_for(<strategy>, self.peer_clients_by_strategy)`.
* `main.py` — one shared `account_peer_clients` dict (strategy -> broker client: bull_flag, macd_wave, orb, hod_break, ignition) attached to the three engines as `peer_clients_by_strategy`; resolved lazily at scan time.
* `data_sources/alpaca_client.py` — cached read-only `account_number` property (one REST call, None + WARNING on failure, failure not cached); timeout log is WARNING when `timeout_retries == 0`, ERROR when allowed retries are exhausted.
* `trading/hod_break_engine.py` — `FC_LEG_POLL_S = 10.0`; attempt-1 `40310000` / `insufficient qty available` -> WARNING, no `_notify`; everything else unchanged.
* `trading/stop_monitor.py` — `_HELD_QTY_RETRY_BACKOFFS_S = (0.5, 1.0, 2.0, 3.0, 4.0)` (10.5 s).
* `trading/orb_engine.py` — `_report_open_tick_budget_exhausted` + `open_tick_exhaustion_level` (WARNING first tick, ERROR on 2nd consecutive or at/after 09:34 ET); streak reset by a tick finishing in budget and by `_reset_daily_locked`.
* `trading/exit_reasons.py` — added `ExitReason.PREPLACE_RECONCILE` (the pre-existing full-suite failure `test_exit_reasons ... orb_engine.py` came from commit aac8ee7's inline literal).
* New tests: `tests/test_orphan_reconciler_peer_strategies_20261023.py` (7), `tests/test_hod_force_close_held_qty_20261023.py` (4), `tests/test_stop_monitor_held_qty_backoffs_20261023.py` (1), `tests/test_orb_open_tick_budget_level_20261023.py` (4), `TestTimeoutLogLevel20261023` appended to `tests/test_alpaca_client.py` (2).

## Tests (`--ignore=tests/integration`, `-p no:randomly`)
* Spec slice (+ `tests/test_alpaca_client.py`): 401 passed.
* Full suite: 5301 passed, 6 skipped, 0 failed.

## Peer-set account-number source
`AlpacaClient.account_number` (new cached property over `get_account_info()['account_number']`). Peer = another strategy whose client reports an equal, non-None number. HOD's own `_account_number_cached()` was not reused (returns '' + ERROR on failure).

## Deviations / not done
* Peer-set integration test uses `MagicMock(spec=AlpacaClient)` client doubles with differing/matching `account_number`, not engine doubles (the map holds clients).
* `Database` has no `get_trades_in_window`, so production takes the SQL path; both paths are tested.
* Alpaca timeout rule: "+ 0 retry" means the single attempt is also the last one, so the literal spec would leave it ERROR. Interpreted as: a caller that disabled retries owns severity (WARNING here, exception still raised); ERROR only for exhausted allowed retries.
* Boot rehearsal / real-API probe NOT done (agent may not restart services). `account_peer_clients` wiring in `main.py` is covered only by AST parse + unit tests.
