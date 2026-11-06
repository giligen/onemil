# RESULT: orb_preplace_late_fill spec 2026-11-06

Files touched: `trading/orb_engine.py` (`_await_terminal`, `_order_field`, `_confirm_late_provisional_fill`, reconcile pre-pass cancel+shared 15 s wait, REPLACED path, `_cancel_preplaced_order`), `trading/exit_coverage.py` (`report_unledgered_excess`), `trading/stop_monitor.py` (calls it), new `tests/test_orb_preplace_late_fill.py` (7 tests: a-f + DB integration), `get_order` mocks given a terminal status in `tests/test_orb_preplace_budget_20261006.py`, `..._cancel_db_status_20261020.py`, `test_orb_preplace_at_close.py`.

Tests: preplace*, exit_coverage*, stop_monitor*, orb_engine* = 341 passed, 0 failed (`--ignore=tests/integration`).

Deviations: (1) safety net keys on the StopMonitor watch's strategy == 'orb' (StopMonitor has no DB handle) instead of a trades-row query; once per (symbol, day). (2) Code was written before the tests (not strict TDD). (3) Late fill also increments `n_filled_before_reconcile`. (4) `_confirm_fill` gets one extra `get_order` read for the final order.
