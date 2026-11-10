# hourly_summary: attribute the shared ORB paper account by ORDER ORIGIN, not by symbol — spec 2026-11-10

Owner saw at 10:05 ET: "ORB −$16 (−0.2%) | 2 fills | open 2: MRK −$14 JPM −$3 | wk +$66". ORB placed NOTHING today
(BEZ/RKLZ provisional orders were cancelled at the reconcile); JPM/MRK/AAPL are third-party `micha-*` orders that
share the ORB paper account (`ALPACA_ORB_*`), and the TOM sleeve (`tom-*`, Nov cycle SPY/IWM/QQQ, sold 11/4) shares
it too. `scripts/hourly_summary.py` splits the account by symbol only (`lambda s: s != TOM_SYMBOL`, lines ~392–403,
`TOM_SYMBOL = "QQQ"`), so every external order is reported as ORB's and TOM's SPY/IWM legs were reported as ORB's.
`scripts/holdings_pulse.py` (being retired) tagged these "(external)" from trades.db — reuse that idea.

Rules: no git, no service action, no orders, never write data/*.db / config / .env / crontab / logs state files;
read trades.db with `sqlite3.connect('file:...?mode=ro', uri=True)`. pytest only on the listed files,
`--ignore=tests/integration`, one run at a time, never piped through head/tail. Never Read a file > 300 lines in
full. This task IS the owner's request; do not pivot. Budget ≤ 35 calls.

## Attribution rule (one helper, unit-tested)
`classify_order(client_order_id, order_id, orb_order_ids) -> 'ORB' | 'TOM' | 'EXT'`:
- `client_order_id` starts with `tom-` → TOM.
- `client_order_id` starts with `orb-` (add-on / unwind ids) OR `order_id in orb_order_ids` → ORB, where
  `orb_order_ids` = the set of `order_id` from trades.db rows with `strategy='orb'` and `trade_date >= <Monday of this
  week>` (ORB entry orders carry plain-UUID client ids, so the ledger is the authority; include the bracket legs if
  the row stores them — check the columns `sl_leg_id` / `tp_leg_id` / `exit_order_id` via `PRAGMA table_info` and
  add whichever exist).
- everything else → EXT (log the distinct prefixes seen today at INFO, e.g. "EXT prefixes: micha-hf, micha").
Positions: `classify_position(symbol, fills_today_by_book, open_orb_symbols)` — ORB iff the symbol has an open ORB
row in trades.db (`order_status='filled' and exit_price is null`) or an ORB fill today; TOM iff a `tom-` fill today or
this week on the symbol; else EXT.

## Report changes
- ORB line: only ORB fills/positions/realized; week-to-date likewise (the week fills are already fetched since Monday).
- TOM line: `tom-` fills and their symbols (SPY/IWM/QQQ), shown when active as today (any TOM position or fill).
- New line, only when non-zero: `EXT (micha) −$16 | 2 fills | open 2: MRK JPM` — report-only, never a book.
- Cross-check: |ORB + TOM + EXT − account day P&L| > $5 → WARNING as today (now it should reconcile).
- Docstring at the top updated (lines ~12–19 describe the symbol split).

## Tests
`tests/test_hourly_summary.py` (extend; do not delete tests — adjust the ones that asserted the symbol split):
- unit: `classify_order` on the four shapes (tom-, orb-add-, UUID in ledger, micha-hf-…, unknown prefix → EXT).
- unit: positions classification (open ORB row; TOM by tom- fill; EXT otherwise).
- integration: a temp sqlite trades.db with one open ORB row (BEX) and real `build_message`/line builders fed today's
  11/10 shape (JPM micha buy, AAPL micha sell, MRK external position, BEX ORB position) → ORB line shows 0 fills and
  only BEX; EXT line shows 2 fills / open 2; TOM hidden; cross-check passes.
- Run: `python3 -m pytest tests/test_hourly_summary.py tests/test_holdings_pulse.py -q --ignore=tests/integration`
  → 0 failures. Then a REAL dry run: `python3 scripts/hourly_summary.py --dry-run` if the script has such a flag
  (grep argparse); if not, do not run it (it sends Telegram).

## Output
`docs/hourly_summary_attribution_20261110_RESULT.md` ≤ 150 words: the rule, file:line, test counts, the dry-run
line if produced. Return ≤ 100 words.
