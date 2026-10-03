# EOD monitor result 2026-10-03

Real run: `bash scripts/research_run.sh -m 1500M python3 scripts/eod_report.py --date 2026-10-02 --no-telegram --no-llm` (nothing sent). Files: scripts/eod_sections.py, scripts/eod_report.py (wiring, --no-telegram alias, PROMOTION block appended verbatim after the Haiku rewrite), tests/test_eod_sections.py (28 tests; 49 with test_eod_report.py).

```
MOM:
  MOM P&L: day $-59 | week $-59 | since 2026-09-29 $-59 (equity $19,941, start $20,000) | DD from peak 0.3 %
  MOM forced run (not counted): 2026-10-02 28 names filled (not a Monday)
  MOM rotation 2026-10-02: picks 20/20 = BT top-20 | fills 28/28 | slip mean -6.0 bp (max 457.5) | size n/a | gate n/a
  MOM reconcile: broker 20 names $20,066 vs state 20 names $20,066 | OK
  MOM completeness: LOST 0.0 %, liquid not in log | OK
ORB PAPER PARITY:
  ORB picks: NO DECISION (no SCORED line in the 09:34–09:40 ET window; restart 14:04 UTC)
  ORB late scoring, not a decision: ORCU
  ORB defects: Engine tick TIMEOUT 0 | GAP_GATE WARN 0 | ERROR 0
PROMOTION:
  Sleeve: HOLD 0/2 (no scheduled rotation yet; first 2026-10-05) [2 clean scheduled Monday rotations, slip <= 20 bp, picks = BT, reconcile OK]
  ORB: HOLD 0/5 (no 09:35 decision) [5 sessions: picks = BT, fills within 30 bp, 0 tick TIMEOUT, 0 ERROR]
  Ramp ORB: HOLD (no live stage started - paper) [realized since last step >= 0 and >= 20 trading days]
  Ramp MOM: HOLD (no live stage started - paper) [realized since last step >= 0 and >= 20 trading days]
  HOD: dry-run only - no live gate (closed as a money book 9/26)
```

## Rules after the coordinator's corrections
- Sleeve: only a scheduled rotation counts (Monday, no `-f<HHMMSS>` order id, a sleeve log line 13:40-15:00 UTC). 10/2 was a Thursday forced run: listed as `forced run (not counted)`; its 457 bp slip is vs the 09:30 open hours earlier.
- ORB: only `ORB SCORED` stamped 09:34-09:40 ET is the 09:35 decision; later SCORED (ORCU 13:45 UTC, after the 14:04 UTC reboot) is `late scoring, not a decision`.

## Why the BT book ends 9/30 (not a stale producer)
Producer: `onemil-orb-backtest.timer` (20:30 UTC weekdays) -> `orb_backtest.py` -> `study_orb_pipeline_static_lock.py`, writes `analysis_results/orb_bplus_book.csv` (orb.yaml backtest.nightly_book_csv). The 10/2 run FIRED and finished (book mtime 10/2 21:18, features regen 1177 s, `orb_features_20261002_2057.csv` has rows for 10/1 (12) and 10/2 (36)). The book lists SELECTED picks only, so a day with no rows = zero BT picks (9/24, 9/26, 9/29 are also absent). No cron/path/crash defect. Consequence: absence is indistinguishable from staleness by the book alone, so `load_bt_rows` now returns zero BT picks when the newest features CSV covers the day, else NO-DATA. Last log lines: "Saved: analysis_results/orb_monthly_static_lock.csv + analysis_results/orb_bplus_book.csv ... Finished onemil-orb-backtest.service". Open question for the owner: 10/2 had 36 feature rows and 0 BT picks (vetoes/threshold) - plausible, not independently checked.

## Other caveats
- BT book has no tilt-mult / add-on column -> NO-DATA. `Engine tick TIMEOUT` count is a floor (archive is grep-filtered).
- Completeness liquid share not in the sleeve logs -> OK judged on LOST only. No shadow-gate csv -> `gate n/a`.
- logs/promotion_state.json holds ORB 10/2 = not clean; sleeve history empty.
