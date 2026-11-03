# RESULT — ORB parity defects (spec 2026-11-03)

No commit, no restart, no writes to db/config/crontab/logs/analysis_results. Every module stayed importable.

## 1. Observer (`scripts/orb_selection_observer.py`)
* Waits are now America/New_York (`et_target`, `wait_until_et`: 09:29:30 / 09:31:00 / 09:35:03 / 09:36:30 ET). The journal `--since` is the 09:25 ET instant written as explicit UTC (`journal_since_arg`). The `_classify_drop` bars window is ET-derived too.
* The journal is read ONCE (`load_journal`, timeout 180 s, returncode checked) instead of ~6 calls with a 30 s timeout each.
* `decide_verdict`: an empty live set or a failed journal sends ONE ops line `observer: engine log not parsed — no verdict` (WARNING logged); CAUGHT is only possible with a non-empty live set.
* **10/30 parse cause (reproduced on `logs/session_archive/2026-10-30.log`)**: not the journalctl timeout. The archive has ZERO `range complete` lines (0 on every archived day 10/26-11/2) and ZERO `ORB SCORED` lines on 10/30. The engine's preplace path ranked and vetoed that day and logged only `[ORB PREPLACE] provisional top / provisional veto / reconcile final` lines, plus `ENTRY SUBMITTED IREZ`. The parser now also reads those lines (`preplace_symbols`). The archive fixture parses to {IREZ, SNK, SSPC, RMNI, MICC, UVXY, OIS, BTCZ}.
* Cron: `26 13 * * 1-5` UTC is 09:26 ET in EDT and 08:26 ET in EST. The script now sleeps until the ET target, so no crontab edit is required. Optional winter-only tidy: `26 14`.
* Tests: `tests/test_orb_selection_observer.py`, 14 tests (EST and EDT targets, `--since`, 10/30 archive, verdicts, journal timeout and caching).

## 2. EOD parity counting (`scripts/eod_sections.py`)
* `engine_traded()` replaces both `traded` sets (production and P1). It excludes cancelled / canceled / rejected / preplace_canceled, and any row with no fill_price and no open-order status.
* `parse_orb_log` now returns `final_ranked` (the reconcile-final list) and `preplace_failed` (`submit failed — no refill`). `preplace_dropped()` removes preplaced names that failed or were `preplace_canceled` and are not in the final list.
* When a reconcile-final list exists it IS the engine's ranked set. The comp-sorted SCORED top-N used to add RAM and XPOF.
* Re-rendered 11/2 from the archive plus read-only trades.db:
  `ORB ranked: engine 8 vs BT 8 | match 7 | engine-only: SKHX | BT-only: AEYE`
  `ORB picks: engine 3 vs BT 2 | match 1 | engine-only: SKHX SKUU | BT-only: AEYE`
  (before: engine 12 vs 8 and engine 4 vs 2.)
* Tests: 3 new in `tests/test_eod_sections.py` on the 11/2 log lines; the file has 54 passing.

## 3. P1 ordering (cron change proposed, not made; code unchanged)
* Writer found: **systemd timer `onemil-orb-backtest.timer`** (`OnCalendar=Mon..Fri 16:30 America/New_York`), not cron. It started 11/2 21:30:46 UTC and finished 22:26:46 UTC. Stages: features regen 21:38 to 21:56 (the `orb_features_..._2156.csv`), production book 22:10, P1 book and marker 22:26. That is a 56 min run.
* Cause: the timer follows ET, but `57 21 * * 1-5 eod_report.py` is a fixed UTC time. In EDT the run ends about 21:26 UTC, before the EOD. In EST (since 11/1) it ends about 22:26 UTC, after it. So P1 NO-DATA recurs every night until March.
* Proposed crontab edit (owner, `crontab -e`), DST-proof via cronie `CRON_TZ`; replace the `57 21` line with:
  `CRON_TZ=America/New_York`
  `40 17 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/eod_report.py >> logs/eod_report.log 2>&1`
  `CRON_TZ=UTC`
  17:40 ET is 22:40 UTC in EST and 21:40 UTC in EDT. Both are after the BT, and the 21:50 UTC archive cron is still before the EOD in winter only if that cron is also moved to 17:35 ET. The `50 21` journal-archive line must move to `35 17` under the same `CRON_TZ`, or `engine_log_text` falls back to the journal directly, which works while the journal is not rotated.
  Check `CRON_TZ` support on this cronie first (`man 5 crontab`). Winter-only fallback with no TZ risk: `40 22` for eod_report and `35 22` for the archive; both must be flipped back in March.
* The second option (read the latest marker <= today and print `(BT of <date>)`) was not implemented, because the budget ran out. It is a safe follow-up if the cron move is declined.

## 4. BT gap gate (no rebuild, no panel or book write)
* Cause: the admission (`orb_backtest._qualifying_pairs_for_dates`, l.150-190, `MIN_GAP_PCT = 5.0` at l.55) tests the DAILY bar open vs the prior daily close (SQL on `daily_bars`). The stored `gap_pct` (`study_orb_features.py` l.305) is the FIRST 1-MIN bar open (`rb.open.iloc[0]`, l.264) vs the prior daily close. The engine's phantom-gap reject tests the stored definition (`feats['gap_pct']` vs `gap_floor`, `orb_engine.py` ~l.3700-3716). Daily open (auction/consolidated) and first 1-min bar open differ, so rows pass the BT gate and then carry gap_pct < 5, which the engine rejects. The spec's line numbers (study_orb_features l.55 / 165-168) are stale; the admission lives in `orb_backtest.py`.
* Panel `orb_features_20261102_2156.csv` (14,031 rows): **99 rows with gap_pct < 5.0 (min 0.0, max 4.994), 60 entered, +$20,942 panel P&L.** Rows are in `docs/orb_parity_defects_20261103_gaprows.csv` (date, symbol, gap_pct, entered, pnl).
* Entered-row P&L by month (n, $):
  2025-01 1 -1364 | 03 1 +3262 | 04 5 -3195 | 05 3 +4747 | 08 2 +2465 | 09 4 +13292 | 10 5 -4092 | 11 3 +2614
  2026-01 3 -3551 | 03 4 -6879 | 04 4 +3192 | 05 6 +9253 | 06 3 -4699 | 07 2 +7649 | 08 7 -5003 | 09 2 +730 | 10 4 +4323 | 11 1 -1802 (AEYE)
  Panel P&L is not the capped book P&L; the book delta needs the rebuild.
* Added `trading/orb_gap_gate.real_gap_violations(df, min_gap_pct)` and `tests/test_orb_gap_parity.py` (3 tests). The helper is NOT wired into the admission yet. Wiring it (drop rows with gap_pct < the pool's min gap) changes the panel, the book and the ramp band, so it belongs to the PREREG-governed rebuild. The invariant "no production row with gap_pct < min_gap_pct" cannot be asserted on the current panel (it would fail), so the real-panel parity test lands with the rebuild.
* Open choice for the PREREG: align the admission to the real first-bar gap (engine-faithful), or accept the daily open and drop rows afterwards. The spec asks for the former.

## Tests and suite
New or changed test files: test_orb_selection_observer (14), test_eod_sections (54), test_orb_gap_parity (3), test_orb_dry_run (observer import still passes). Full suite: see the last line below.

Full suite (`-n 2`, excluding tests/integration): **5562 passed, 6 skipped in 174 s**, 0 failures.
