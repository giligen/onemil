# Spec — ORB daily Telegram report must state ledger truth (force-close P&L), 2026-10-30 (Fable). Ships TODAY before the 12:30 UTC boot.

## Defect (10/29)
`trading/orb_engine.py force_close_all → _force_close_all_locked` books the force-closed exits through `_sync_db_after_fc`
(~l.8850–8895: `pnl = (exit_price − entry_price) * shares + scale_pnl` → `db.update_trade`) and then calls
`send_daily_report()` (~l.7651–7690), which prints `self.daily_pnl`. `_sync_db_after_fc` never adds its pnl to
`self.daily_pnl` (only the StopMonitor exit path does, `ORB EXIT … daily_pnl=`), so the Telegram said
`[ORB-LIVE] DAILY: placed=5 filled=3 canceled=0 fill_rate=60% P&L=$-228` while the ledger closed the day at −$163
(AXTQ force_close +$65 missing). `self.daily_pnl` also feeds the day-kill tripwire, so it must be right too.

## Change (Sonnet; one agent; ≤ 40 calls; no commit/stash, no restart, no orders, no writes to data/*.db, config, .env,
## crontab, logs state files; one pytest process at a time; never read a file > 300 lines in full)
1. `_sync_db_after_fc`: when it books a pnl for a trade that had no `pnl` yet, add that pnl to `self.daily_pnl` and log
   `ORB FC SYNC: {sym} pnl=$… booked → daily_pnl=$…` at INFO. Never double-count: skip rows whose `pnl` is already set
   (read the row state before writing; if the code already has that guard, keep it and say so).
2. `send_daily_report`: compute `ledger_pnl` and `ledger_exits` from the ledger — closed ORB rows for today
   (`strategy == 'orb'`, `trade_date == today ET`, `pnl is not None`; find the existing `self.db` accessor, e.g. the one
   `_sync_db_after_fc` or the EOD sync uses, do not add a raw SQL path if an accessor exists). Report
   `P&L=$<ledger_pnl> (<ledger_exits> exits)`; if `abs(ledger_pnl − self.daily_pnl) > 1.0` append
   ` | in-memory $<daily_pnl>` and log WARNING `ORB DAILY: ledger/in-memory P&L drift …`. If the ledger read raises,
   log WARNING and fall back to `self.daily_pnl` (fallback paths log why). Keep the rest of the line unchanged.
3. Tests — `tests/test_orb_daily_report_ledger_20261030.py` (fixtures from `tests/conftest.py`; `MagicMock(spec=…)` for
   domain classes; build the engine the way the nearest existing orb_engine test does — grep `tests/test_orb_*force*`,
   `tests/test_orb_*daily*`, `tests/test_orb_*fc*` for the pattern):
   * unit: db double returns two closed rows (−237, +9) and one force-closed (+65) → message contains `P&L=$-163` and
     `3 exits`; `self.daily_pnl = -228` → the drift suffix and a WARNING are present.
   * unit: db raises → fallback to in-memory, WARNING logged, message still sent.
   * unit: `_sync_db_after_fc` booking adds to `self.daily_pnl` once; a second call on the same (now pnl-set) row adds 0.
   * integration: a real sqlite trades DB (the project's DB class on a tmp path) with the 10/29 rows → `send_daily_report`
     prints −163 / 3 exits; `_sync_db_after_fc` on a row with no exit against a stubbed Alpaca sell order books once.
4. Suite once: `timeout 900 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero failures.
5. Write `docs/orb_daily_report_ledger_truth_spec_20261030_RESULT.md`: diff summary, test counts, deviations. No rehearsal
   (Fable runs the boot rehearsal).

## Not in scope
Any ORB trading rule, selection, sizing, exits (frozen under PREREG_1709); the `placed=` count semantics.
