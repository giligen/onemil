# HOD OCO persistence fix: RESULT (2026-10-17, spec `docs/hod_oco_persistence_fix_20261017.md`)
Not deployed: no restart, no orders, no config/cache/trades.db writes, no git. Live at the next boot after owner GO.

| Item | Change (`trading/hod_break_engine.py` unless noted) |
|---|---|
| 1 persist | `_on_live_fill` existing-trade_id branch now writes `pattern_data` (new tp/sl ids, merged with the registry's) to the row; registry and DB agree |
| 2 settle | `_cancel_and_settle_safety_net` (cancel, poll REST <= `oco_settle_timeout_s` 5 s until terminal), `_book_settled_exit` (a leg that sold is booked, no re-place), `_place_safety_net` (WARNING, retry once after `oco_retry_delay_s` 2 s, then ERROR + ungated `_notify` with symbol/qty/broker msg) |
| 3 reconcile | `_reconcile_by_broker_sells`: CLOSED sell orders of the symbol, filled_at >= entry fill, up to open qty; limit -> `target`, stop/stop_limit -> `stop_loss`, else `eod`; `_book_leg_fill` accepts the `stop_loss` alias |
| 4 carry-over | `_adopt_prior_day_open_rows` (from `sync_positions`; `Database.get_open_trades_between` in `persistence/database.py`): 7-day open HOD rows held by the broker are adopted (qty bounded by row shares, stale legs settled, OCO re-placed + persisted, watch, WARNING + `_notify`); not held -> `exit_pending_verification` -> item 3. Docstring states the HOD-account-only assumption |
| 5 breaker | `StopMonitor.last_error/last_error_ts/last_msg_ts` (`trading/stop_monitor.py`); `_stop_monitor_breaker_reason` logged before `sys.exit(1)` (`scanner/realtime_scanner.py`) |
| 6 queue | `on_mover` skips candidates/seen_today/positions and names already queued; `_admit_movers` drains the whole queue (no 500 cap), lock held only for the pops |
| 7 ledger | `_rotate_stale_ledger` (old file -> `<name>.<YYYYMMDD>.csv`, WARNING once) in `_append_csv_row` and `_append_dry_ledger`; `_arm_live_order` stores `idx` + `arm_features` in `lo` (`_compute_arm_feature_vals` factored out) |

Tests: new `tests/test_hod_oco_persistence.py` (21: persistence sequence 58/63 -> 60/63 -> 63/63 with real Database and restart rehydrate, settle ordering, filled-meanwhile, retry accepted, retry failed + notify, reconcile limit/stop/ignored/lookup-fail, carry-over adopt / not-held / >7 d, queue 10,000 dups / 1,200 drain, ledger rotation, arm idx + features) and 4 in `tests/test_stop_monitor_circuit_breaker.py`.
Superseded tests updated to the rotate rule: `test_hod_arm_features.py`, `test_hod_resting_entry.py`, `test_hod_counterfactuals.py` (old-header "never rewritten"). `tests/conftest.py` `hod_live_alpaca`: cancelled order ids read back `canceled`.

Gates: hod + stop_monitor 529 passed / 1 skipped; full suite minus integration 5230 passed / 4 skipped / 0 failed. Integration dir not run (needs real API).
Caveats: the breaker-path test checks the call site by source inspection plus the helper; no real-API rehearsal done. A partial entry whose leg sold mid-replace leaves later-bought shares to the periodic adoption (documented in `_book_settled_exit`). RBRK row 544 still needs the owner's UPDATE (or the Monday boot reconcile now matches it by broker truth).
