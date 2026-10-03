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
  ORB: HOLD 0/5 (no 09:35 decision (neutral)) [5 sessions: picks = BT, fills within 30 bp, 0 tick TIMEOUT, 0 ERROR]
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

## ORB ranked-set amendment (coordinator, 10/3)
- Clean is now tri-state: ranked set (engine `ORB SCORED` 09:34-09:40 ET vs the BT's ranked top-N recomputed from the newest features CSV by `bt_ranked`, a replica of the static-lock pipeline; matches the dive's stage table on 9/22, 9/29, 10/1, 10/2) must match. Match + 0 picks = clean; mismatch = not clean (resets); NO DECISION or BT NO-DATA = neutral (counter untouched). 10/2 = NO DECISION -> neutral, ORB HOLD 0/5.
- 10/1 (real run): under the 09:34-09:40 window it is also NO DECISION - the engine scored 13:44-13:48 UTC (09:44-09:48 ET), never at 09:35, and no restart (single boot 11:57 UTC). The window therefore never matches a normal day; the coordinator should confirm the intended window (`DECISION_WINDOW_ET`).
- Forced comparison for 10/1 (all 15 scored names vs BT): BT ranked 5 (WVE, CBRX, LPA, IBX, RKLX) all in the engine's 15; engine-only 10 (ADBG APPS BRZE CNXC CRMG DXC EFXT KD MEDS TDAY); funnel `BT: rows 12 -> top-8 (5) -> vetoed 5 (PDR 4, G1 1, range 0, dedup 0) -> picks 0`. The engine scores its whole candidate list, not only a top-8, so strict equality will likely never hold; BT-ranked subset-of-engine, or comparing the engine's own top-8 by composite, may be the right test - needs a decision.

## Final ORB rules (coordinator decisions, 10/3)
- Decision window 09:34-10:00 ET; a first scoring after 09:40 is still a decision, flagged `late decision (HH:MM ET)`. NO DECISION only when nothing scored before 10:00 ET.
- Ranked test: the engine's top-8 by its LOGGED comp + quintile (Q4,Q5,Q3,Q2, then comp; Q1 out; family dedup) must equal the BT top-8; if a SCORED line has no comp/quintile, fallback `BT top-8 subset of engine scored set` with `(subset test - engine scores not logged)`.
- Real runs (`--no-telegram --no-llm`):
```
10/1  ORB ranked: engine 8 vs BT 5 | match 4 | engine-only: APPS CRMG KD TDAY | BT-only: RKLX | late decision (09:44 ET)
      BT: rows 12 -> top-8 (5) -> vetoed 5 (PDR 4, G1 1, range 0, dedup 0) -> picks 0      => NOT clean
10/2  ORB ranked: engine 1 vs BT 8 | match 1 | engine-only: - | BT-only: BMNU CRCG DFDV FWDI ORCX RGTX SOC | late decision (09:45 ET)
      BT: rows 36 -> top-8 (8) -> vetoed 8 (PDR 8, G1 0, range 0, dedup 0) -> picks 0      => NOT clean
```
- 10/2 is no longer NO DECISION (ORCU scored 09:45 ET, after the 14:04 UTC reboot): it is a late decision with the engine having scored 1 name vs the BT's 8, so ORB counter 0/5 (reset). promotion_state.json: orb 10/1 False, 10/2 False. 55 tests pass.
