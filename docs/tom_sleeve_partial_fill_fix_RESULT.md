# TOM sleeve partial-fill fix: RESULT (2026-10-07)

Spec: `docs/tom_sleeve_partial_fill_fix_20261007.md`. Code: `scripts/tom_sleeve.py`. Tests: `tests/test_tom_sleeve.py`.

Changes
1. `resolve_fill()` reads `filled_qty`: terminal order with 0 < filled < qty -> ledger status `partial_fill` (qty = filled part,
   ref_price = avg price), remainder stays owed; filled == qty -> `filled`; 0 -> unfilled as before. Telegram/log line states
   filled/total and price (`QQQ exit order PARTIAL FILL 25/26 @ $759.76 ... remaining 1 stays owed`). Same path in `chain_state`.
2. Owed qty = broker position, capped by `ledger_open_qty()` (entry qty minus partial_fill rows, one per coid); WARNING names both.
3. `run()` fetches the broker's open orders and `owed_exits()` before the "not a turn-of-month" no-op, and counts a live
   exit order of the chain as owed -> the 15:57-15:59 ET cancel+market fallback runs on catch-up days. Sessions count from the
   chain's FIRST exit (so 10/7 = `late_exit_2_sessions`, id `-r2`).
4. `--reconcile` (with `--dry-run` = print only): appends a missing `partial_fill` row from the orders API; ledger write only,
   never an order; idempotent; refuses (ERROR) if a later order row exists for the symbol.

Tests: `python3 -m pytest -q tests/test_tom_sleeve*.py -x --ignore=tests/integration` -> 49 passed (35 old + 14 new, incl. a
CSV round-trip flow and the `--reconcile` CLI wiring). New tests failed before the change.

Ledger backfill (the one logs write, by `--reconcile`; second run appended nothing):
`2026-10-06,exit_fillcheck,QQQ,25,759.7600,fcd8b4f2-...,partial_fill,tom-202610-QQQ-out-r1,2026-10-07T14:33:59+00:00,false,`

Dry-run proof (real read-only broker state, no orders):
```
$ tom_sleeve.py --dry-run --now-et "2026-10-07 15:45"
DRY-RUN would SELL MOC 1 QQQ id=tom-202610-QQQ-out-r2 deviation=late_exit_2_sessions
$ run(..., now=2026-10-07 15:58 ET, dry_run=True) with get_open_orders mocked to a live tom-202610-QQQ-out-r2
DRY-RUN would CANCEL tom-202610-QQQ-out-r2 (new) then SELL MARKET 1 QQQ id=tom-202610-QQQ-out-mkt
```
Not deployed by this task: no commit, no service/cron change. Cron picks the script up on its next tick.
