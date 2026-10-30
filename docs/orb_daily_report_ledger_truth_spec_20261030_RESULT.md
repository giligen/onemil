# RESULT — ORB daily report ledger truth (2026-10-30)

## Diff (trading/orb_engine.py only; no trading-rule change)
* `send_daily_report` (l.7672–7699, 7703): headline now `P&L=$<ledger> (<n> exits)` from `self.db.get_trades_by_date(today ET)`
  filtered to `strategy=='orb'`, `pnl is not None`. If `|ledger − daily_pnl| > 1.0` appends ` | in-memory $…` and logs WARNING
  `ORB DAILY: ledger/in-memory P&L drift …`. Ledger read failure -> WARNING + fallback to in-memory text (message still sent).
  Existing accessor used (no raw SQL). Note: the real DB returns `trade_date` as a `date`, so the filter compares `str(...)[:10]`
  (caught by the integration test).
* `_sync_db_after_fc` (l.8843–8844): explicit `pnl is not None -> skip` guard added next to the existing `exit_price` guard
  (the exit_price guard already existed; this adds the pnl one the spec asks for).
* `_sync_db_after_fc` (l.8916–8924): after a successful `update_trade`, `self.daily_pnl += pnl` and INFO
  `ORB FC SYNC: {sym} pnl=$… booked → daily_pnl=$…`. A failed DB update does not book.

## Tests — tests/test_orb_daily_report_ledger_20261030.py: 9 tests
* unit (report): 10/29 rows -> `P&L=$-163`, `3 exits`, drift suffix + WARNING; in-sync -> no suffix; other strategy/open/other-day
  rows ignored; DB raises -> fallback + WARNING + message sent.
* unit (sync): books once into daily_pnl with log line; pnl-set row adds 0; failed update does not book.
* integration (real `Database` on tmp path): 10/29 rows + force-closed AXTQ via stubbed Alpaca sell -> sync books once (second call
  adds 0), report prints −163 / 3 exits with no drift; drift case prints `P&L=$-163 (3 exits) | in-memory $-228`.
* Neighbouring `test_orb_force_close_chain.py` / `_fix.py`: pass (40 passed together).

## Suite
`timeout 900 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`: 5393 passed, 6 skipped, 0 failed (610 s).

## Deviations
None to the spec. Not rehearsed (Fable runs the boot rehearsal).
