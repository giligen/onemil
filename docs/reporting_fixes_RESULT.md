# Reporting fixes RESULT (2026-10-17)
Tests: touched files 129 green; `tests/test_report*.py test_eod*.py test_daily_green*.py test_hourly_summary.py` 194 green.

## A. Composite drift is hard only when a decision changes (scripts/report_common.py)
* New `composite_drift_decision(live, bt, threshold, live_in_top, bt_in_top) -> (sev, reason)`; `classify_composite_drift`
  wraps it (same name, extra optional args). NEAR_THRESHOLD_BAND removed.
* hard: (1) pass/fail vs the filter threshold differs, (2) top-N membership differs, (3) delta > 0.15. Reason starts `decision_changed: ...`.
* `decision_parity` now passes orb.yaml `filter.threshold` (0.01208) and top-N (`sizing.max_concurrent`=8), ranked among the
  compared names passing the filter, per side (top-N IS available at that site).
* Tests: RGC pair soft, sign flip hard, top-N difference hard, delta 0.2 hard, threshold-straddle hard.
* Dry run `daily_green_check.py --date 2026-10-16 --dry-run`: RGC is now a soft warning; verdict GREEN, ramp freeze "would CLEAR".
* Owner command (state untouched by me; streak file currently 0): `python3 scripts/daily_green_check.py --date 2026-10-16`
  (add `--no-telegram` to stay quiet; a green re-score of an already-recorded RED day may need `--force-downgrade` only for
  green->red, so not needed here). `--dry-run` previews it.

## B. MOM P&L from the broker (scripts/eod_sections.py, scripts/hourly_summary.py)
* `sleeve_pnl_lines` -> `broker_sleeve_pnl_line` from `get_portfolio_history(1M,1D)`; reuses `hourly_summary.fetch_history`
  (new `period` arg) and new `session_date` (week_to_date refactored onto it). Day/week/since-start/DD on the slice ($20K start);
  since-start base = close of the last session before SLEEVE_START_DATE; day not yet in history uses live account equity.
* Ledger mark kept as cross-check line (flags |diff| > $20). API down: ledger-mark fallback with WARNING, else `NO-DATA (<why>)`.
* Real dry run (read-only broker call), 10/16: `MOM P&L: day $-703.04 | week $-2,045.70 | since 2026-09-29 $-2,577.55 (slice
  equity $17,422, start $20,000) | DD from peak 14.5 % [broker history]`. 10/12: day $-413.42 (matches broker).
* FLAG: the ledger mark and broker slice differ by $262 on 10/12 (ledger 19,316.93 vs 19,054.73): the ledger's equity basis
  is not the account delta; cross-check line will show DIFF until reconciled.
