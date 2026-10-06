# EOD instrument fixes 2026-10-06 - RESULT
Tests: 138 pass (`test_eod_sections/report/daily_green_check*/report_common*`, -x, no integration); 384 pass on `-k "freeze or ramp or green or promotion or eod"`.
Code: `scripts/eod_sections.py` (journal-first `engine_log_text`, `book_day_rows`/`bt_marker`, `why_not_ordered`, `bt_ranked` returns comps),
`scripts/report_common.py` (`green_verdict` loads `strategy='orb'`), `scripts/daily_green_check.py` (`reconcile_freeze`),
`scripts/eod_report.py` (`freeze_summary`: GATE-1 printed the BF block of the raw JSON, never ORB's state).
Also fixed: `test_load_bt_rows...` was not hermetic (read the live features CSV; failed once the 10/5 night covered the day).
Log (INFO): `engine log 2026-10-05 served by journal+archive: journal 74292 lines, archive 75112 lines, used 75797`.

## Acceptance (eod_report.py --date 2026-10-05 --no-telegram --no-llm)
ORB PAPER PARITY
- ORB ACTION: 2 entry submit(s) FAILED -- JAGX: AlpacaClient.submit_stop_bracket_order() got an unexpected keyword argument 'client_order_id'; CRCG: (same)
- ORB ranked: engine 8 vs BT 8 | match 8 | engine-only: - | BT-only: -
- BT: rows 51 -> top-8 (8) -> vetoed 7 (PDR 7, G1 0, range 0, dedup 0) -> picks 1 | ORB picks: engine 0 vs BT 1 | BT-only: DFDV
- ORB why not ordered: DFDV -- engine SCORED comp=0.3216 Q4 at 13:35:25 UTC vs BT comp 0.3216 Q4 (same); not in the provisional top-2 @ 09:34:57.727 ET (JAGX, CRCG): range completed 13:35:02 UTC, 5 s after the snapshot; every provisional submit FAILED, no refill
- ORB defects: Engine tick TIMEOUT 0 | GAP_GATE WARN 27 | ERROR 2
- P1 ranked: engine 10 vs BT 11 | match 10 | BT-only: PUSA
- P1 picks/fills: engine 0 vs BT book: NO-DATA (no marker for 2026-10-05 P1 (book orb_bplus_book_P1.csv has no row for it - not computed))
PROMOTION: `ORB: HOLD 0/5 (entry submit FAILED x2) [5 sessions: ...]`
GATE-1: `ORB green check: RED DAY 2026-10-05` | `freeze: ORB frozen since 2026-09-21: BT picks never ordered live: ['DFDV'] | BF frozen since 2026-09-24: JAGX: LIVE_ONLY`

## DFDV (a true BT pick never ordered - a live engine defect, not an instrument gap)
The engine SCORED DFDV at 13:35:25 UTC with the BT's own comp (0.3216 Q4); the ranked sets match 8/8. The PREPLACE provisional top-2 is taken
at 09:34:57.727 ET, DFDV's range completed at 09:35:02, so the provisional set was JAGX, CRCG (both submits failed on the client_order_id
TypeError, no refill). The preplace snapshot precedes the last range completions: fix candidate = rank the preplace set after range
completion or refill on a failed submit (NOT done here; engine code untouched).

## Freeze (logs/ramp_freeze.json, ORB), before -> after the daily_green_check --date 2026-10-05 --no-telegram re-run
before: frozen since 2026-09-21, reason `unattributed exits: [SPAL, IOT, LOFF, TEMT, SPCM, TSLR, GWRE, TEM, MRNX, MRNA]; BT picks never ordered live: ['DFDV']`
after:  frozen since 2026-09-21, reason `BT picks never ordered live: ['DFDV']` (HOD attribution reason retired; DFDV survives, so the freeze STAYS).
Rule coded: breaches -> freeze reason = surviving breaches only; no breaches AND the freeze was raised by that same day's check -> cleared by code
(frozen_dates kept); a freeze raised by another day is never auto-cleared. green_verdict now: exits OK, pending OK, bt_parity DFDV; streak stays 0.
Side effects of the re-run (state files, no Telegram): green_streak.json 10/5 reasons rewritten, orb_p1_parity_state.json 10/5 = not clean (PUSA),
promotion_state ORB 10/5 = not clean (failed submits).

## Owner line (crontab NOT edited)
Move the archive cron `58 21` to `50 21` and extend its grep to `\[ORB|ORB SCORED|IGNITION|VETO|Q1 filter|WOULD BUY|ENTRY SUBMITTED|FILLED|LOCK|kill|\[HOD|ERROR|TIMEOUT`.
Journal-first makes the report independent of it, but the archive is the only source once journald rotates.
Open: P1 markers have no P1 row for 10/5 (only `production,51,1`), so P1 is NO-DATA, not 0 picks; the nightly should write the P1 marker.
