# HOD entry-quote / L2 capture fix — spec 2026-11-13 (owner: "fix now")

## Defect
All 325 `trades` rows with `strategy='hod_break'` (2026-09-25 → 2026-11-12, `data/trades.db`) have NULL
`entry_quote_bid/ask/bid_size/ask_size/spread`, `entry_quote_ofi` and `entry_l2_depth`, although the columns exist
and the engine has `_snapshot_cross_quote` (hod_break_engine.py ~2845) and `_write_entry_quote` (~2866). The journal
since 11/10 shows ONE "entry quote missing" WARNING (WOLF, 11/12 14:40) against ~300 fill lines — so the write
is not even being attempted on most fills. Known: `_write_entry_quote` has a single call site (~2136, one booking
branch); `_snapshot_cross_quote` is only fed from the resting-entry print path (~935) and the bar-close fallback;
`get_print_quote` returning None makes it return silently (~930). The BF path captures L2 with
`data_sources.l2_depth.log_l2_async(symbol, filled_at, trade_id, self.db.update_trade, column='entry_l2_depth')`
(trading_engine.py ~1963); the HOD path never calls it.

## Required behaviour (every HOD fill, paper and dry, every booking branch)
1. At the instant the entry fill is booked (`save_trade` / `_save_pending_trade` / the existing-trade_id branch /
   the dry-fill record if it has a trades row), persist the NBBO at the cross if a snapshot exists, else a fresh
   `self.alpaca.get_latest_quote(sym)` taken right then (flagged `entry_quote_source='rest_at_fill'` in
   `pattern_data` so the two are distinguishable), with bid/ask sizes and `entry_quote_spread = ask − bid`.
2. Launch `log_l2_async` for `entry_l2_depth` exactly as the BF path does.
3. Capture failure = ONE WARNING per symbol-day with the reason, never an exception out of the booking path, never
   a change to any order or decision. No new ERROR class on Telegram.
4. Root cause first: find and state in the commit message WHY the existing write never fired (which branch books
   the fills that reach the DB, and why the cross snapshot was empty on the one branch that writes). Fix the cause,
   not a second parallel path — ONE helper `_persist_entry_quote(sym, trade_id, lo, filled_at)` called from every
   booking branch.

## Tests (TDD; all must pass; zero failing tests in the repo)
* Unit: each booking branch calls the helper once; snapshot present → columns written from it; snapshot absent →
  REST quote used and `entry_quote_source='rest_at_fill'`; REST failure → columns NULL, one WARNING, fill still
  booked; `log_l2_async` launched with the trade id; a partial top-up does not rewrite.
* Integration (real sqlite in tmp, real `TradeDatabase`): book a fill through the engine's public path with a
  `MagicMock(spec=AlpacaClient)` quote, read the row back, all five quote columns populated and `entry_l2_depth`
  updated when the async callback fires.
* Run: `pytest tests/test_hod_break*.py tests/test_hod_*.py tests/test_stop_monitor*.py -q -p no:cacheprovider
  --ignore=tests/integration` and then the full pre-boot command used by the 11:27 UTC cron (find it in
  `docs/active_crons.md`); both green before you stop. Never run two pytest suites at once; never
  `ONEMIL_RUN_INTEGRATION=1`.

## Rules
No git commit, no service restart (the 12:30 UTC weekday start ships it after the 11:27 pre-boot suite), no orders,
no config/DB/.env/crontab writes, never print keys. Write only `trading/hod_break_engine.py`, new/changed tests under
`tests/`, and `docs/hod_entry_quote_capture_fix_20261113_RESULT.md` (≤ 250 words: root cause, diff summary, test
counts, the exact pytest commands and their last line). Return ≤ 120 words.
