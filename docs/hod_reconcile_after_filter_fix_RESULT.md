# HOD reconcile after-filter fix, RESULT (2026-10-18)
Spec: `docs/hod_reconcile_after_filter_fix_20261018.md`.

## Diff (trading/hod_break_engine.py, `_reconcile_by_broker_sells`)
* Broker query: `after = since - timedelta(minutes=30)` (Alpaca `after` filters on submitted_at, not filled_at).
* In-code filter: sell `filled_at >= since - 120 s` (tolerance for a fast stop-out before the engine books the entry).
* DEBUG log: orders returned / sells / candidates. Docstring documents both windows and the RBRK 10/15 incident.

## Tests (tests/test_hod_oco_persistence.py, class TestReconcileByBrokerTruth)
* test_sell_submitted_before_since_but_filled_81_min_after_is_booked_and_the_broker_query_is_wide (asserts get_orders `after <= since - 30 min`)
* test_sell_filled_10_minutes_before_since_is_not_booked
* test_sell_filled_60_s_before_since_inside_the_tolerance_is_booked

## Run
`pytest tests/test_hod_oco_persistence.py tests/test_hod*.py --ignore=tests/integration`: 375 passed, 1 skipped, 0 failed.
Note: `reconcile_pending_exits()` returns rows closed (1), not shares; the 17 sh is checked via the row (closed/target).
