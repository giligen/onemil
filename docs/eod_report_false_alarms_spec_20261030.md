# Spec — EOD / green-check false alarms on the ORB paper book, 2026-10-30 morning (Fable)

## Defects (seen on Telegram 10/29 21:30–21:58 UTC; all REPORT-side, the trader does not import these modules)
1. **False ramp FREEZE.** `scripts/report_common.py` fill-parity (~l.283–296) flags a BT-filled symbol as "live order never
   filled" when ANY ledger row for it has `fill_price is None` and a status outside `('filled','closed')`. The ORB preplace
   path writes a provisional row per symbol (`pending_new` → `preplace_canceled` / `stale_closed`) and then a SECOND row
   for the re-placed order, which fills. Result: 10/27 (NEBX, CRDU, OKLL — all filled, +$1,079) and 10/29 (CORD, IREZ —
   filled) froze the ORB ramp clock on a false reason; `logs/ramp_freeze.json` history shows it. Rule: a (symbol, day,
   strategy) is FILLED if any of its rows has `fill_price` or status in `('filled','closed')`; only symbols with rows and
   NO filled row are "unfilled". Keep the 7/30 IREX case flagged (one row, `time_stop_canceled`, no fill).
2. **Selection parity blind to preplace.** `scripts/report_common.py` ranked/"why not ordered" logic (~l.655–720 and the
   `ORB ranked: engine 6 vs BT 8 … BT-only: CORD IREZ` / `not scored by the engine: … no SCORED line` output) only reads
   `ORB SCORED:` journal lines. Preplaced names are ranked via `[ORB PREPLACE] provisional top-N @ …: SYM(rh=…,Qn), …` and
   `[ORB PREPLACE] reconcile final top-8 of 8 ranked: [...]` and ordered via `ORB ENTRY SUBMITTED: SYM`. Treat a symbol
   in those lines as engine-ranked / engine-ordered; the "why not ordered" explanation must never name a symbol that has
   an `ORB ENTRY SUBMITTED:` line that day.
3. **Hygiene "NOT ACTIVE: onemil-trader"** (`scripts/eod_report.py hygiene_section`, ~l.192–201) at 21:57 UTC: the service
   exits at the 16:00 ET close by design (cron stop 10:25 / start 12:30 UTC). Flag the trader NOT ACTIVE only when the
   check runs between 12:35 and 19:55 UTC on a weekday; otherwise print `onemil-trader=exited-at-close (expected)`.
   `onemil-macd-wave` has no unit file on this node (decision 9/24): print `onemil-macd-wave=not-installed` and never
   count it in NOT ACTIVE unless `systemctl cat onemil-macd-wave` succeeds.
4. **Report-only:** the engine's `ORB DAILY REPORT … P&L=$-228` fired at 19:45:08 in the same second as the AXTQ force-close
   (+$65) → Telegram P&L −$228 vs ledger −$163. This is `trading/orb_engine.py` (frozen, boots at 12:30): do NOT change it;
   write the one-paragraph fix proposal (emit the daily report after the force-close exits are booked, or compute from
   closed rows) into the RESULT file for a weekend rehearsal.

## Change (Sonnet; one agent; ≤ 40 calls; no commit/stash, no restart, no orders, no writes to logs/ state files, data/*.db,
## config, .env, crontab; never two pytest runs at once; never read a file > 300 lines in full)
* Fix 1–3 in `scripts/report_common.py` and `scripts/eod_report.py` (root cause, no work-arounds; docstrings; every
  fallback logs WARNING).
* Tests (`tests/test_eod_report_false_alarms_20261030.py`; fixtures from `tests/conftest.py`; `MagicMock(spec=…)`):
  unit — fill-parity with rows [provisional `pending_new` no fill, `stale_closed` no fill, `filled` with fill_price] →
  OK; with one `time_stop_canceled` row and no fill → flagged; selection parity with a `[ORB PREPLACE] provisional top-2`
  + `ORB ENTRY SUBMITTED` journal stub → symbol is engine-ranked and not "BT-only"; hygiene at 21:57 UTC Friday with the
  trader inactive → no NOT ACTIVE, at 15:00 UTC → NOT ACTIVE; macd-wave never NOT ACTIVE without a unit file.
  Integration — a temp sqlite copy with the 10/29 row shape (two CORD rows: `stale_closed` 1053 sh no fill; `filled`
  697 sh fill 4.78) through the real function → `checks['fill_parity'] == 'OK'`.
* Re-run the checks read-only and record the output: `python3 scripts/daily_green_check.py --date 2026-10-29 --dry-run
  --no-telegram` and `--date 2026-10-27 …` — FIRST confirm from the code that `--dry-run` writes neither
  `logs/green_streak.json` nor `logs/ramp_freeze.json`; if it does, do not run it and say so.
* Journal (read-only): `journalctl -u onemil-trader --since "2026-10-23 13:30" --until "2026-10-23 14:05" --no-pager
  -g SAIQ | head -12` — report why the 10/23 freeze reason "BT picks never ordered live: ['SAIQ']" arose (real skip or
  the same preplace blind spot).
* Suite once: `timeout 900 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero failures.
* Write `docs/eod_report_false_alarms_spec_20261030_RESULT.md`: files changed, test counts, the two dry-run verdicts,
  the SAIQ lines, the defect-4 proposal, deviations.
