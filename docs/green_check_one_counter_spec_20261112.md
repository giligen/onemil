# ORB green check: ONE counter from day one, BT parity never "skipped" — spec 2026-11-12

Owner 2026-11-12: "Telegram msg about BT parity skipped stale data. WTF. Fix it once and for all, and we said we
reset the counter."

Defects (verified):
1. `scripts/daily_green_check.py` runs at 21:30 UTC (crontab), the nightly ORB BT builds 21:30–22:05 UTC
   (`onemil-orb-backtest.timer` 21:30 + `nightly_bt_update.sh` 22:30). `report_common.py:249-251`
   (`bt_max < day → bt_stale`) therefore sees yesterday's BT every night, prints "⚠ BT parity SKIPPED (nightly BT
   data stale)" and STILL marks the day green (`daily_green_check.py:84-88`). A check that cannot check counted
   12 green sessions. The log shows SKIPPED on every recent night.
2. `report_common.streak_update` (887) counts consecutive greens over the whole record (August onward);
   `PREREG_1709_AMENDMENT_2.md` restarts the count at the 2026-11-12 boot (`ORB_DAY_ONE`, `trading/ramp_stage.py`),
   and the 10 sessions it requires are PICK-parity sessions (`scripts/orb_pick_parity.py`), not the green check's
   own fill-parity re-derivation. Two counters, two parity definitions = accidental behaviour.

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/* (incl. green_streak.json, ramp_freeze.json — tests use tmp paths; the real script runs
only with `--dry-run`). pytest only on files that exist, `--ignore=tests/integration`, one run at a time, never
piped through head/tail. Never Read a file > 300 lines in full. Budget ≤ 40 calls. This task IS the owner's request;
do not pivot.

## Rule (frozen)
1. **ONE counter.** `green_streak.json` is the Amendment-2 counter. A session is GREEN iff (a) exits attributed and
   nothing stuck (existing checks), AND (b) the pick-parity verdict for the day (`orb_pick_parity.run(day)` or its
   `--json` output — import the function, do not shell out) has zero `unexplained` and no live fill outside the
   final set, AND (c) the nightly BT for the day exists (`bt_data_max_date() >= day`). The old "BT picks never
   ordered live" / composite-drift re-derivation is deleted from the verdict (pick parity carries it with named
   reasons); keep the exit-attribution checks.
2. **Never skipped.** BT absent for the day → the check WAITS (poll every 60 s up to `--wait-bt-min` 45) for
   `bt_data_max_date() >= day`; still absent → the day is RED with the reason "nightly BT missing for {day}" and
   ONE `logger.error` (Telegram) naming the build timer/cron to look at. No `bt_stale` branch, no "SKIPPED" string
   anywhere in the message (grep it out of the tests too). A no-range vendor day (pick parity reports every live
   pick as `range_missing` and the engine logged the sweep ERROR) is NOT_COUNTED: recorded with
   `"counted": false`, neither green nor red, streak unchanged (Amendment 2 §1).
3. **Count from day one.** `streak_update` / `read_streak`: the streak = consecutive counted-green sessions with
   `day >= ORB_DAY_ONE`; earlier records stay in the file untouched and never count. Message label
   `[GREEN n/10 since 11/12, A2]` via `ORB_DAY_ONE_SUFFIX`; the RED message keeps its block. `orb_ramp_check.py`
   reads the same streak (it already reads the file; make sure it uses the day-one-filtered count).
4. **Order of operations.** The check must run AFTER the BT build and the pick-parity run. Add `--after-bt`
   semantics by default (the wait in §2); the crontab move (21:30 → 23:20 UTC) is done by the session, not you —
   put the exact line in the RESULT.
5. The EOD `parity_section` (`scripts/eod_report.py:153`) shows the LAST green line, which after the move is the
   previous session's: prefix it with that line's day (already in the text) and the words "prev session" when the
   day < today, never implying today's verdict.

## Tests (`tests/test_daily_green_check.py` — extend; delete only assertions on the removed bt_stale/SKIPPED path)
Unit: BT absent after the wait → RED with the reason + exactly one ERROR record; pick parity with 1 unexplained →
RED; clean → GREEN; vendor no-range day → `counted: false`, streak unchanged; streak counts only days ≥
ORB_DAY_ONE (fixture file with 12 green August–11/11 records + 11/12 green → streak 1); label text. Integration:
tmp streak file + tmp trades.db + a tmp BT book, the whole `main(['--dry-run', …])` path on a fixture day,
message asserted end to end. Run `tests/test_daily_green_check.py tests/test_report_common*.py
tests/test_orb_ramp_check*.py tests/test_orb_pick_parity_20261111.py tests/test_eod_*.py` (files that exist) →
0 failures. Then `python3 scripts/daily_green_check.py --dry-run --day 2026-11-11` (read-only; paste the message).

## Output
`docs/green_check_one_counter_20261112_RESULT.md` ≤ 200 words: file:line per change, the dry-run message, test
counts, the crontab line to change. Return ≤ 100 words.
