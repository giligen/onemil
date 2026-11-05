# Result: transient Alpaca error noise (2026-11-05)

**Changed**
- `data_sources/alpaca_client.py`: module `_is_transient(exc)` (requests ConnectionError, RemoteDisconnected, urllib3 ProtocolError, AlpacaAPITimeoutError; walks cause/context chain) and `AlpacaClient._retry_once(fn, label)` (WARNING, 1.0 s, one retry). Applied only to `get_latest_trades`, `get_snapshots`, `get_current_bars`, `get_account_info`; their wrapper log is now WARNING.
- `scanner/realtime_scanner.py`: new `_run_guarded_cycle` (called from `run`) with `_consecutive_cycle_failures`. First AlpacaAPIError = WARNING, no notifier; 2nd+ consecutive = ERROR + notifier "N consecutive cycles failed"; non-Alpaca exception = ERROR at once; success resets.

**Tests** (121 passed, 0 failed): `tests/test_alpaca_client_transient.py` (11), `tests/test_realtime_scanner_cycle_errors.py` (6, includes a real-AlpacaClient integration test with the first HTTP call dropped), plus existing `tests/test_alpaca_client*.py`.

**Note**: AlpacaAPITimeoutError is retried per spec; `_call_with_timeout` already retried it once, so a timeout now gets a second attempt (worst case longer cycle).
