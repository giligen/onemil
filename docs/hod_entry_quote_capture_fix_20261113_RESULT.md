# HOD entry-quote capture fix — RESULT 2026-11-13 (in-session; the agent cascade was refused by the API edge, 3 × 403)

## What was actually broken (the spec's premise corrected)
`entry_quote_bid/ask/spread` ARE captured since 2026-11-02 (76 of 78 November HOD rows; commit 64c6cc5 10/31). The
two 11/06 KTB / 11/12 WOLF misses are the "no cross snapshot" case the 10/31 WARNING covers. What was NULL on 100 %
of the 325 rows: `entry_quote_bid_size`, `entry_quote_ask_size` and `entry_l2_depth`.

## Root cause
HOD symbols are PRINT-watched (`StopMonitor.subscribe_trades_quotes` → `_print_quotes[sym] = (bid, ask, ts)`), never
QuoteWatch-ed. `_snapshot_cross_quote` took the sizes from `get_quote_watch_snapshot(sym)`, which is None for every
HOD symbol in production; the 10/31 unit test mocked that accessor with sizes, so it passed. L2 depth
(`data_sources.l2_depth.log_l2_async`, the bull-flag path's helper) was never called from the HOD engine.

## Change
* `trading/stop_monitor.py`: `_print_quote_sizes[sym] = (bid_size, ask_size)` written beside the quote in `_on_quote`,
  popped with it on unsubscribe; `get_print_quote_sizes(sym)` (the 3-tuple `get_print_quote` is unchanged).
* `trading/hod_break_engine.py`: `_snapshot_cross_quote` falls back to the print-quote sizes when the QuoteWatch
  snapshot has none (0 → NULL, never a fabricated size); `_launch_l2_at_fill` fires `log_l2_async` once per order
  from `_write_entry_quote` (independent of the NBBO columns), ONE WARNING per symbol-day on a launch failure,
  behind `hod_break.l2_at_fill` (default OFF — the snapshot is a PAID Databento MBP-10 request; the owner decides).
* Tests: `tests/test_hod_entry_quote_sizes_l2_20261113.py` (10: StopMonitor cache + real `_on_quote` path, size
  fallback, zero sizes → NULL, old-monitor tolerance, L2 off/on/once/failure/independent of the quote).

## Verification
`pytest tests/test_hod_entry_quote_sizes_l2_20261113.py tests/test_hod_exec_instruments_20261031.py
tests/test_hod_live_resting.py tests/test_hod_break_engine.py tests/test_hod_resting_entry.py tests/test_stop_monitor*.py`
→ 340 passed. Full pre-boot command (`pytest tests/ -q -x --no-header -p no:randomly`, the 11:27 UTC cron's) → see
the commit message. Ships at the 12:30 UTC weekday start; first proof = `entry_quote_bid_size IS NOT NULL` on the
first HOD fill of 11/13 (`scripts/hod_dry_ledger.py` / trades.db read-only).

## Not changed
No order, config, service or crontab change. `l2_at_fill` stays off until the owner says GO on the spend.
