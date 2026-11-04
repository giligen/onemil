# RESULT: HOD partial-fill `shares` fix (2026-11-04)

**Change:** `trading/hod_break_engine.py` (`_on_live_fill`, existing-trade_id branch): the `update_trade` call now
also writes `'shares': filled_qty` (the cumulative broker fill). Nothing else changed.

**Tests (both fail without the fix, pass with it):**
- Unit: `tests/test_hod_live_resting.py::TestPartialFillCompletionUpdatesShares::test_partial_then_filled_writes_cumulative_shares_and_filled_qty` (21 then 34, asserts shares and filled_qty == 34).
- Integration (real `Database`, tempdir): `tests/test_hod_partial_fill_shares_20261104.py::test_partial_then_filled_leaves_one_row_with_shares_34_filled_qty_34` (one row, (34, 34)).

**Runs (`--ignore=tests/integration`):** test_hod_partial_fill_shares_20261104 + test_hod_live_resting + test_hod_break_engine = 127 passed, 0 failed. Not committed.
