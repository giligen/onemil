# HOD: `shares` not updated when a PARTIAL fill completes — spec 2026-11-04

## Evidence (XPEL, HOD paper, 2026-11-04)
- 17:05:47 UTC `[HOD] XPEL: LIVE PARTIALLY_FILLED 42.43 cum 21/34` → `Saved trade: XPEL buy 21 shares` (trades.db id 706, shares=21).
- 17:16:45 UTC `[HOD] XPEL: LIVE FILLED 42.45 cum 34/34` → new OCO for 34, StopMonitor `shares=34`, but the DB row still
  has `shares=21` (`filled_qty` was updated to 34, `shares` was not). Broker position 34 vs DB 21.
- Cause: `trading/hod_break_engine.py` ~l.2125 (the `if lo.get('trade_id'):` branch of the fill handler) updates
  `order_status/fill_price/filled_at/filled_qty/pattern_data` but NOT `shares`. The new-row branch (`_save_pending_trade`)
  gets the right shares because it is called with `filled_qty`. The fill-completion path is the only one that re-enters
  with an existing trade_id and a larger qty.

## Fix (root cause, one place)
In that `update_trade` call add `'shares': filled_qty` (the cumulative broker fill is the position size). Keep the
existing behaviour otherwise. No other code path changes.

## Tests (TDD — both required)
1. Unit, `tests/test_hod_break_engine.py` (or the file that already covers `_on_live_fill`/the partial-fill path —
   grep `PARTIAL fill` in tests/): drive the handler with `partially_filled` cum 21/34 then `filled` cum 34/34 on the
   same `lo` and assert `db.update_trade` (or the real row) ends with `shares == 34` AND `filled_qty == 34`.
2. Integration with a REAL `Database` on a tempdir (pattern: `tests/test_orb_kill_rails.py::tmp_db`): after the two
   events, `SELECT shares, filled_qty FROM trades WHERE symbol='XPEL'` → (34, 34), one row only.

## Rules
- Do NOT touch data/trades.db, config files, services, orders. Do not git commit. `--ignore=tests/integration`.
- Run only the two HOD test files you touched plus `tests/test_hod_break_engine*.py`; never the full suite.
- Budget ≤ 25 tool calls. Write RESULT to `docs/hod_partial_fill_shares_spec_20261104_RESULT.md` (≤ 120 words:
  diff summary, test names, pass counts) and return ≤ 100 words.
