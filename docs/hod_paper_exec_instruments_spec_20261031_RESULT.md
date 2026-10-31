# RESULT — HOD paper exec instruments (spec hod_paper_exec_instruments_spec_20261031.md)

Files: `trading/hod_break_engine.py` (only production file touched), `tests/test_hod_exec_instruments_20261031.py` (new, 18 tests).
Not touched: config.yaml, ORB, arm/entry rules, stop routing, dry ledger format, StopMonitor, AlpacaClient.

## Diff summary (trading/hod_break_engine.py, line anchors)
* l.287–293 config + state: `squeeze_enabled` (default False), `squeeze_minute` (840), `squeeze_max_r` (0.5); counters
  `squeeze_placed/squeeze_filled`, sets `_squeeze_tried/_squeeze_warned/_squeeze_filled_syms/_entry_quote_warned`; all reset in `_roll_session` (l.~433).
* l.605 `process_tick`: `self._run_squeeze()` right after `check_exits()` (no-op unless enabled).
* l.931 `_on_trade_print` (the cross, print-watch quote = the dry ledger's source): `_snapshot_cross_quote(cand.live_order, ...)` sets
  `cross_bid/cross_ask/cross_bid_size/cross_ask_size/cross_ts` on `live_order` (sizes from `StopMonitor.get_quote_watch_snapshot` when a watch exists; stale > `max_quote_age_s` is not stored, reason kept in `cross_missing_why`).
* l.2129 `_on_live_fill`: `_write_entry_quote` writes `entry_quote_bid/ask/bid_size/ask_size/spread` (spread = ask - bid in $, ORB convention) once per order; missing -> NULL + one WARNING `[HOD] entry quote missing {sym}: {why}` per symbol-day.
* l.2838–2955 new methods `_snapshot_cross_quote`, `_write_entry_quote`, `_squeeze_nbbo`, `_squeeze_warn_once`, `_run_squeeze`, `_squeeze_one`.
* l.2957 `_tp_reason(pos)` -> 'squeeze' once `pattern_data['squeeze_done']`; used by `_exit_legs`, `_settle_exit_legs`; `_leg_status` follows replaced legs for 'target' and 'squeeze'.
* l.2982/2999 `_book_leg_fill`: counts `squeeze_filled` once per position; reason `squeeze` -> `pricing_method='squeeze_ask'`, trigger = squeeze price, exit quote = the bid/ask captured at squeeze time.
* l.3355 flat message gets `squeeze=placed/filled`.

## Config keys (under `hod_break`, absent = OFF as today)
`squeeze_enabled: false`, `squeeze_minute: 840`, `squeeze_max_r: 0.5`. config.yaml NOT edited.

## Replace helper
Existing `AlpacaClient.replace_order_limit_price(order_id, price)` (data_sources/alpaca_client.py l.1989; also used by `_anchor_target_to_fill`). Nothing added. The new leg id is tracked on `pos.tp_leg_id` + pattern_data (`tp_leg_replaced`, `squeeze_*` fields persisted, so a restart still books reason `squeeze`).

## Tests
`tests/test_hod_exec_instruments_20261031.py`: 18 passed (4 entry-quote, 9 squeeze rule, 3 leg-fill booking, 1 integration; spec cases a–d plus: no-quote fallback, replace failure, tp-less position, before/after window, stale quote, partial top-up).
Suite: `5426 passed, 6 skipped, 4 warnings in 647.87s` (timeout 2700, --ignore=tests/integration, -p no:randomly), zero failures.

## Deviations / notes
* StopMonitor's own watch keeps the OLD tp_leg_id after a replace (same as the existing `_anchor_target_to_fill` precedent); the engine's post-exit belt-and-braces cancels `pos.tp_leg_id` (the new id), so a StopMonitor stop exit still cancels the squeezed leg. A `mark_target_resting`-style sync was not added (it flips ORB `target_resting` semantics).
* Extra guard: skip (WARNING) when the ask is not below the existing take-profit; squeeze also requires minute < flat_minute.
* "No quote -> last bar close" supplies only the unrealised-R mid; without a usable ask nothing is placed (WARNING).
* Cross-quote sizes are NULL when no StopMonitor quote watch exists for the symbol (bid/ask/spread still come from the print-watch quote).
* Exit quote on a squeeze fill = the NBBO at squeeze time (other bracket legs carry none).
