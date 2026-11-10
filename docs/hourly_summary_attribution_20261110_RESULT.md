# hourly_summary attribution by order origin — RESULT 2026-11-10

Rule (`scripts/hourly_summary.py`): `classify_order` — `tom-` client id = TOM; `orb-` id or order_id in the trades.db
ORB ledger (strategy='orb', since Monday; plus leg columns if present — only `exit_order_id` style columns are
checked via PRAGMA, none exist today) = ORB; else EXT. `classify_position`: open ORB row / ORB fill today = ORB,
`tom-` fill = TOM, else EXT. Ledger read-only (`load_orb_ledger`, ERROR + all-EXT if unreadable), `split_books`,
`format_ext_line`, cross-check now ORB+TOM+EXT, docstring updated.

Tests: tests/test_hourly_summary.py 33 pass (+13 new incl. 11/10-shape integration), test_holdings_pulse 7 pass.

Dry run 10:44 ET: `ORB $0 (0.0%) | 0 fills | flat   wk +$34` / `EXT (micha) −$27 | 2 fills | open 2: JPM MRK`.
Remaining cross-check WARNING (−26.81 vs account −52.43): unmatched sell with a prior-day buy (known caveat).
