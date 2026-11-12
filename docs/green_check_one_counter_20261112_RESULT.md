# Green check: ONE counter — RESULT 2026-11-12

Changes (file:line):
- `scripts/report_common.py:248-275` green_verdict: bt_stale/"SKIPPED" and "BT picks never ordered" removed; BT absent = RED reason `nightly BT missing for {day}` (names timer/script); fill-parity and exit checks kept. `:855` counted_streak (days >= ORB_DAY_ONE, `counted:false` neutral), `:871` green_label `[GREEN n/10 since 11/12, A2]`, `:876` streak_update(counted=), `:921` read_streak recomputes.
- `scripts/daily_green_check.py:86` wait_for_bt (60 s poll, `--wait-bt-min 45`), `:102` is_no_range_day, `:115` apply_pick_parity (imports `orb_pick_parity.run_parity`), `:152` main(argv); composite-drift re-derivation deleted; one `logger.error` on missing BT.
- `scripts/orb_ramp_check.py:65,236` markers + day-one floor. `scripts/eod_report.py:157` "prev session" label.

Tests: 253 passed, 0 failed (daily_green_check 67, report_common, orb_ramp_check, pick_parity, eod_*).

Dry run 11/11 (real data, RED): `pick parity: 1 unexplained ['UMAC'] ... matched 7/8` (genuine finding, not a defect of the check).

Caveats: no-range rule = empty live final + every BT pick `range_missing` (the sweep ERROR string was not checkable).

Crontab (session): `20 23 * * 1-5 /usr/bin/python3 /home/ec2-user/onemil/scripts/daily_green_check.py >> /home/ec2-user/onemil/logs/daily_green_check.log 2>&1` (was `30 21`).
