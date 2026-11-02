# Journal flood fix - RESULT (2026-11-02)

Not committed, service not restarted: picked up by tomorrow's boot. Journald retention (`SystemMaxUse`) is the owner's line (system config), untouched.

| Item | Before (today) | After (estimate) | Location |
|---|---|---|---|
| A1 `CLOSE CALL` DEBUG | 1.06M | only real near-misses (one missing criterion; if `intraday_change`, move >= 0.5 x `intraday_change_pct_min`); expected a few K/day | `scanner/criteria.py` `is_close_call()` + `ScannerCriteria.is_close_call`, used at the DEBUG line |
| A2 verbose `print` table | 1.1M | <= 21 rows (top 20 by `intraday_change_pct` + `(+N more)`) x 272 cycles ~ 5.9K | `scanner/realtime_scanner.py` `CLOSE_CALLS_PRINT_CAP`, `close_calls` build uses same predicate, `_print_intraday_output` |
| B alpaca stream ack (46 x hundreds of KB) | ~46 entries | 0 (`alpaca`, `asyncio` pinned INFO; `websockets/urllib3/httpx` were already WARNING) | `monitoring/logger.py` `THIRD_PARTY_INFO_LOGGERS` |
| C1 `orb_gap_gate:155` GAP_GATE echo (input echo, no state change; ledger row persisted by callers, unaffected) | 76K | once per (symbol, source) per session ~ symbols seen x <= 2 (low thousands) | `trading/orb_gap_gate.py` `_gap_gate_logged` |
| C2 `criteria:167` INTRADAY QUALIFIED (re-reported every cycle while qualified) | 29,980 | transition-only (re-logs after a non-qualifying evaluation); ~ hundreds | `scanner/criteria.py` `_qualified_logged` |
| C3 `trading_engine:1351` "Already traded today, skipping" (no state change) | 26K | once per symbol per day (cleared in daily reset) ~ tens | `trading/trading_engine.py` `_traded_skip_logged` |

Left alone: the 'Added to qualified symbols' INFO and every decision line.
Test changed: `tests/test_orb_parity_20260930.py` last assertion (a repeat GAP_GATE tick used to emit DEBUG, now silent, per spec C).
Tests added: `tests/test_journal_flood_20261102.py` (10 tests: predicate boundaries, DEBUG line, transition-only qualified, cap, logger levels, gap-gate once).
Suite: 5497 passed, 6 skipped, 0 failed (tests/ minus integration, 348 s). New file: 10 tests.
