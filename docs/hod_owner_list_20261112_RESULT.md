# HOD owner list - RESULT 2026-11-12
Built per `docs/hod_owner_list_spec_20261112.md`. No git, service, order or config action.

* `scripts/hod_owner_list.py`: parse_arms :50, rvol :81, spread_bps :89, trades/adv20/parity readers :114/:134/:148, snapshot :169, build_rows :188, message :211, publish :235, ledger :250/:262, gather_rows :281, main :297 (run_guarded :319).
* `scripts/hod_owner_list_score.py`: OUR_PREFIXES :33, orders/fills fetch :56/:69, classify_pick :102, fifo_pnl :115, machine R :157, build_report :174, collect_picks :209, main :233.
* `docs/hod_owner_list_cron.txt`: the four cron lines (CRON_TZ=America/New_York block; Friday line uses `$(date -d 'last monday' +\%F)`). NOT installed.
* `tests/test_hod_owner_list.py`: 20 tests pass; with `tests/test_cron_fixes_20261112.py` 41 pass, 0 fail (no `test_hod_break_deadman*` file exists).

Notes: LIVE ARMED lines carry no level/stop (fallback only); scorer prefix list widened beyond the spec (add one when a new book ships); snapshot = SIP feed, HOD paper key.

## Dry run (`--at 1400 --dry-run --date 2026-11-12`, real journal + real SIP snapshot), first lines
```
[HOD-LIST 14:00 ET] 93 armed · 9 filled · 8 closed -$61
ARM lvl 199.92 last 199.94 dist +0.0% r 2.4% day -2% rvol 0.6x sp 13bp re0 - STOPPED R -0.4
AAOI lvl 53.88 last 53.90 dist +0.0% r 2.4% day -4% rvol 0.7x sp 17bp re56 - ARMED
HUBS lvl 217.44 last 217.52 dist +0.0% r 1.6% day +7% rvol 1.5x sp 17bp re4 - ARMED
... (14 rows, then "(+79 further armed symbols not shown)")
pick by symbol on the live account; the ledger scores you Friday
```
Scorer `--since 2026-11-09 --dry-run` ran read-only against the live key: 0 owner picks, "n < 5, no verdict".
Not done: crontab install (session's job), Telegram real send (not exercised; unit-tested with spec'd mock).
