# ORB preplace-cancel DB status fix — RESULT (2026-10-20)
- trading/orb_engine.py: new `_cancel_preplaced_order(sym, pos)`; `_reconcile_preplaced` OUTSIDE-final-top-N branch now
  calls it. After the broker cancel it reads `get_order(...).filled_qty`; 0 -> `db.update_trade(trade_id,
  {order_status:'preplace_canceled', exit_reason:'preplace_reconcile'})`; >0 -> WARNING, row NOT marked, position kept
  (fill path owns it). Unreadable filled_qty -> WARNING, assumed 0.
- Reader sites: trading/trading_engine.py restart rebuild of `_pending_orders` (local set lifted to module constant
  `TERMINAL_PENDING_STATUSES`, now includes `preplace_canceled`). persistence/database.py `get_open_trades` is an
  ALLOWLIST (`_ACTIVE_ORDER_STATUSES`), so the new status is terminal automatically (asserted in test).
- NOT changed (they do not name time_stop_canceled; they count it as "ordered"): scripts/eod_sections.py l.850/1042
  `traded` sets, scripts/eod_report.py l.104 `filled`, scripts/report_common.py fill_parity (keys on fill_price).
  hod_break_engine has its own _DEAD set (HOD-only, untouched).
- Tests (tests/test_orb_preplace_cancel_db_status_20261020.py): outside_top_n_cancels_and_marks_row,
  inside_top_n_row_untouched, cancel_with_partial_fill_not_marked_and_warns,
  reader_sites_treat_preplace_canceled_like_time_stop_canceled.
- Suite (preplace*, submit_timing, eod_sections, hourly_summary, report_common_*, orb_fixes, trading_engine): 341 passed.
  (tests/test_report_common.py does not exist; report_common_bplus/scale used.)
- Live DB row (IREX 547) untouched; service not restarted — takes effect at next boot.
