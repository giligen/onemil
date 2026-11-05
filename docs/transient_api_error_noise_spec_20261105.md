# Transient Alpaca disconnects reach Telegram as ERROR twice — spec 2026-11-05

## Evidence
- 11/5 15:30:20 UTC `data_sources/alpaca_client.py:793` `ERROR Failed to get latest trades: ('Connection aborted.',
  RemoteDisconnected(...))` then 15:31:16 `scanner/realtime_scanner.py:420` `ERROR Scanner intraday cycle failed at
  10:30 ET: AlpacaAPIError: Failed to get latest trades: ...`. The next cycle (15:32) ran fine. 11/3: the same on
  account info. 3 events in the journal window, every one self-healed on the next cycle. Each event = TWO Telegram
  ERROR pings (the wrapper logs ERROR and raises; the caller logs ERROR again). Owner rule: no harmless ERRORs on Telegram.

## Fix (root cause, two places; keep every real failure loud)
1. `AlpacaClient`: in the generic `except Exception` of `get_latest_trades`, `get_snapshots`, `get_account`/account
   info and `get_current_bars` (grep the identical `logger.error(...); raise AlpacaAPIError(...)` pattern — change
   ONLY those four): classify `requests.exceptions.ConnectionError`, `http.client.RemoteDisconnected`,
   `urllib3.exceptions.ProtocolError` and `AlpacaAPITimeoutError` as TRANSIENT → one retry after 1.0 s with a
   `logger.warning("... transient (<type>) — retrying once")`. If the retry succeeds, return normally. If it fails, or
   the exception is not transient, keep raising `AlpacaAPIError` but log at WARNING in the wrapper (the caller owns
   the ERROR decision). One shared helper `_is_transient(exc)` + `_retry_once(fn, label)`, no copy-paste.
2. `realtime_scanner.run` cycle wrapper (l.417–430): count consecutive failed cycles. First failure → WARNING
   (`exc_info=True` kept) and no notifier call. Second consecutive failure or more → ERROR + notifier as today, and
   the message says "N consecutive cycles failed". Reset the counter on a successful cycle. A failure that is not an
   `AlpacaAPIError` (a code bug) stays ERROR on the first occurrence.

## Tests (unit + integration, both required)
- unit `tests/test_alpaca_client_transient.py`: data_client mock raising RemoteDisconnected once then returning →
  one WARNING, result returned; raising twice → AlpacaAPIError raised, WARNING only (no ERROR record, use caplog);
  a ValueError → raised as AlpacaAPIError without retry.
- unit `tests/test_realtime_scanner_cycle_errors.py`: cycle raising AlpacaAPIError once → WARNING, no notifier;
  twice in a row → ERROR + notifier once; success resets; a RuntimeError → ERROR immediately.
- integration: the existing scanner cycle test with a real `AlpacaClient` whose HTTP session is patched to drop the
  first connection (`requests` adapter mock) → the cycle completes on the retry, 0 ERROR records.
- Run the touched test files + `tests/test_alpaca_client*.py` + `tests/test_realtime_scanner*.py`; zero failures.

## Rules
No git commit, no service restart, no orders, no DB/config/log writes. `--ignore=tests/integration`. `MagicMock(spec=...)`
for domain classes. Budget ≤ 35 calls. RESULT → `docs/transient_api_error_noise_spec_20261105_RESULT.md` (≤ 150
words); return ≤ 100 words.
