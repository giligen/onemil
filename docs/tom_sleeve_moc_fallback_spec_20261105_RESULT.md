# TOM sleeve MOC fallback fix: RESULT 2026-11-05

Changes (`scripts/tom_sleeve.py`, `data_sources/alpaca_client.py`):
- `get_open_orders` now returns `time_in_force`; `chain_state` exposes `st['tif']` for live orders.
- New `is_live_cls`: the 15:58 late fallback skips a live CLS order on both legs (INFO "MOC live", left alone). If the broker object lacks tif it infers from the ledger (QQQ, not `-mkt`, row written in the MOC window) and logs a WARNING.
- Exit leg: MOC only for `MOC_RELIABLE_SYMBOLS` (QQQ); SPY/IWM do nothing in the MOC window and exit MARKET in the late window.
- Message text now "unfilled before the close (15:58 ET)".

Tests (`tests/test_tom_sleeve.py`): 7 new unit tests plus the 11/4 sequence integration test (real tempdir ledger). Existing cancel/market tests now supply tif=day, and two SPY MOC exit tests moved to QQQ. 56 passed; exit_coverage tests pass.
