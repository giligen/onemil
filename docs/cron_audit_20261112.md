# Cron audit 2026-11-12 (spec: docs/cron_audit_spec_20261112.md)

Read-only. Evidence = log mtimes/tails, artifacts, argparse/docstrings, `systemctl`, `crontab -l` (ec2-user and root).
Now = 2026-11-12 07:10 UTC, **EST** (DST ended 11/1). Line numbers = `crontab_20261112.txt` snapshot (identical to live `crontab -l`).
Proposed file: `docs/crontab_proposed_20261112.txt` (NOT installed).

## Table

| line | schedule UTC | purpose | book ON? | last success | verdict | fix |
|---|---|---|---|---|---|---|
| 3 | 10:30 Mon-Fri | `main.py --batch` universe rebuild | BF/ORB/HOD (all consume) | 11/11 10:58 | **BROKEN (intermittent)** | no batch-complete 10/20-10/29 (TXSE), 11/6 finished 12:40 (after the 12:30 boot), **11/10 crashed** on `invalid symbol: 037ESC019`; add 12:20 completion alert (new line), code: skip invalid symbols, cron_guard |
| 7 | 06:00 Sat | weekly verbose rebuild | same | 11/7 06:29 | OK | log 24 MB, no rotation |
| 11 | 22:30 Mon-Fri | BF Stage-1 cache append (`nightly_bt_update.sh`) | BF PAUSED (feeds BF parity + BT panel) | 11/11 22:33 | **REORDER** | ORB BT (21:30 timer) ended 22:41 on 11/10 and 22:27 on 11/11 -> overlap on 2 CPUs; move to 22:45. Not redundant with the timer (BF cache vs ORB features) |
| 12 | 13:26 Mon-Fri | ORB selection observer | ORB paper | 11/11 14:36 | OK | waits on ET clocks internally (DST safe), cron_guarded |
| 17 | 23:20 Mon-Fri | ORB green check (Amendment 2) | ORB paper | 11/11 21:30 (old slot) | OK (unproven) | new slot/rules first run tonight; comment above it is stale (says 20:30 BT); the 12 old "GREEN n/10" lines said "BT parity SKIPPED" and are void |
| 28 | 12:40 Mon-Fri | trader startup watchdog | all | healthy = silent (log 9/7) | OK | no heartbeat, so a dead watchdog is invisible; not guarded |
| 32 | 21:04 | stupid-money eod_check | other project | - | OUT-OF-SCOPE | unchanged |
| 34 | 22:50 Mon-Fri | BF decision parity | BF PAUSED | 11/11 22:50 | **NOISE** | BF has 0 live rows (paused) so every BT trade is "live_missed": ERROR + `RAMP FREEZE set on BF` 11/6, 11/9, 11/11; code: skip live comparison when `trading.enabled` false. Move to 22:58 after line 11 |
| 38 | Sun 20:00 | ORB weekly selection refit | ORB paper | 11/8 20:00 WRITTEN | OK | `|| telegram` binds correctly |
| 39 | 23:00 Mon-Fri | HOD volume profile | HOD paper | 11/11 23:05 | OK | |
| 40 | 19:57 Mon-Fri | HOD dead-man flat | HOD paper | 11/11 19:57 ("trader active") | **BROKEN in EST** | script acts only in [close-5, close+30) ET = 20:55-21:30 UTC in winter; 19:57 UTC = 14:57 ET is outside -> never fires since 11/1. Docstring says "every 5 min"; cron is one tick. Fix: ET-based ticks |
| 41 | 11:27 Mon-Fri | pre-boot pytest | all | 11/11 11:32, 5658 passed | OK | ends 11:32, before 11:50 MOM prefetch and 12:30 boot; `\%F` escaped |
| 43 | 20:10 Mon-Fri | guardrail ledger + auto-pause | ORB/HOD/BF | 11/11 20:10 | **REORDER (EST)** | 20:10 UTC = 15:10 ET, 50 min before close and before HOD closes: the day's last trades land a day late. Move to 16:10 ET. Unguarded; comment above it belongs to the EOD line |
| 44 | 19:45,58 / 20:45,58 | TOM sleeve (both DST hours) | TOM live | 11/11 20:58 | OK | 2 of 4 ticks are no-op each day (design, documented); log 100 KB |
| 46 | @reboot +20s | claude tmux prod-onemil | infra | never run (no log; boot 10/2 14:03 predates script 14:18) | OK (unproven) | |
| 48 | @reboot +30s | stupid-money attach | other project | - | OUT-OF-SCOPE | unchanged |
| 50 | 11:50 Mon-Wed | MOM paper prefetch/dry run | MOM paper | 11/11 | OK | |
| 51 | 13:45 Mon-Wed | MOM paper submit (EDT tick) | MOM paper | no-op in EST (08:45 ET, refuses) | OK | by design |
| 52 | 14:45 Mon-Wed | MOM paper submit (EST tick) | MOM paper | 11/9 rotation; 11/11 14:45 no-op | OK | |
| 53 | 21:10 | stupid-money forward_shadow | other project | - | OUT-OF-SCOPE | unchanged |
| 55 | 15-21:05 Mon-Fri | hourly multi-book Telegram | all | 11/11 21:05 | OK (DST-fragile) | hand-switched for EST; use ET clock (10-16) so March needs no edit; unguarded |
| 57 | 13:46 Mon-Wed | MOM LIVE slice (EDT tick) | MOM live | no-op EST | OK | |
| 58 | 14:46 Mon-Wed | MOM LIVE slice (EST tick) | MOM live | 11/11 14:46 | OK | |
| 60 | 13:00 Sat | MOM panel refresh | MOM | 10/31 OK; **11/7 GATE FAILED** | **BROKEN** | 11/7: `liquid 663/825 with an 2026-11-06 bar (80.4%, min 98%)`, extension untouched; panel_ext.parquet was rewritten 20:28 (manual), max bar 2026-11-06. Cause of the 20 % hole not established (vendor lag at 13:00?). Add retry 14:00 |
| 61 | 13:20 Sat | MOM BT-vs-paper parity | MOM | 11/7 13:20 ERROR | **REORDER** | read a stale extension ("last bar 2026-10-30 is 8 days old"); move to 14:20 after the retry |
| 66 | 17:35 ET | session archive | all | 11/11 22:35 (4.1 MB) | OK | `\%` escaped, runs before journald rotation |
| 67 | 17:40 ET | EOD report | all | 11/11 22:41 | **REORDER** | producers run later: ORB BT ended 22:41 on 11/10 (after the 22:40 start), BF parity JSON 22:50, pick parity 23:15, green check 23:20. `parity_section` always prints "BF parity: not run yet for today" and the previous session's green. Move to 23:25 UTC |
| 69 | 23:15 Mon-Fri | ORB pick parity | ORB paper | 11/11 23:15 | OK | after BT (latest end 22:41); result 11/11 "unexplained 1" is the pre-fix engine, not a cron defect |
| root | 10:25 stop / 12:30 start Mon-Fri | `systemctl stop|start onemil-trader` (not in the ec2-user file) | all | 11/11 12:30:01 | OK | runs on market holidays too; engine stops itself 21:00 |
| timer | 16:30 ET Mon-Fri | `onemil-orb-backtest.timer` -> `orb_backtest.py` | ORB | 11/11 22:27 (57-70 min wall) | OK | TimeoutStartSec 5400 s |

Counts (28 active ec2-user lines + 2 root lines + 1 timer = 31 jobs): OK 20 (3 of them unproven), BROKEN 3 (lines 3, 40, 60), REORDER 4 (11, 43, 61, 67), NOISE 1 (34), STALE-OUTPUT 0, REDUNDANT 0, DEAD-REMOVE 0 jobs (7 orphan comment blocks), OUT-OF-SCOPE 3 (32, 48, 53).

## Ranked defects

1. **Dead-man flat is dead in winter (line 40).** `hod_break_deadman_flat.py` docstring: "runs every 5 minutes from 19:30 UTC; the SCRIPT decides... close - 5 min". Code `return close_m - 5 <= m < close_m + 30` in ET; cron is a single 19:57 UTC tick = 14:57 ET since 11/1, so a trader that is down at the close is never flattened (HOD paper book, orders held overnight). Fix in proposed file: `CRON_TZ=America/New_York`, ticks 12:55-16:30 ET every 5 min (also covers early closes); the script exits at `systemctl is-active` first so it is cheap.
2. **Universe build fails silently and has no completion alert (line 3).** `grep "Batch complete" logs/universe_build.log` has no entry for 10/20-10/29, 11/10; 11/6 finished 12:40 (after the 12:30 boot). 11/10: `ERROR | alpaca_client:653 | Failed to get daily bars: {"message":"invalid symbol: 037ESC019"}` then an uncaught `Traceback ... main.py line 1128`; the trader booted that day on 11/9's universe. `main.py` has no `run_guarded`. Added a 12:20 UTC check line that Telegrams if today's "Batch complete" is missing; code fix needed (drop invalid symbols, guard the entry point).
3. **Gate ordering (lines 11, 34, 67, 17).** ORB BT ends 22:27-22:41 UTC in EST, `nightly_bt_update.sh` starts 22:30 (overlap on 2 CPUs), EOD starts 22:40 (before BF parity 22:50, pick parity 23:15, green 23:20, and on 11/10 before the BT finished). `eod_report.py:150-165` prints previous-session green and "BF parity: not run yet for today". Proposed chain: BT (timer) -> 22:45 BF cache -> 22:58 BF parity -> 23:00 HOD profile -> 23:15 pick parity -> 23:20 green -> 23:25 EOD (session archive stays 17:35 ET).
4. **BF parity cries wolf on a paused book (line 34).** `logs/bf_parity/bf_parity_2026-11-11.json`: DIVERGE, bt 1, live 0; `trading.ramp_freeze` ERROR "RAMP FREEZE set on BF" three times in the last 4 sessions. A paused book has no live rows by design; the freeze and ERROR are false alarms and train the owner to ignore ERRORs.
5. **MOM panel refresh gate failed 11/7 and the parity ERROR followed (lines 60, 61).** Fixed by hand at 20:28; the 11/14 run is the first unattended test. Retry + later parity added.
6. **Eight cron entry points have no `run_guarded`/excepthook (the 10/29 class):** `trader_watchdog`, `bf_decision_parity`, `hourly_summary`, `guardrail` (the auto-pause!), `hod_break_deadman_flat`, `eod_report`, `orb_pick_parity`, `daily_green_check`; plus `main.py --batch`. An uncaught exception is a traceback in a log nobody reads. Code fix, not a crontab change.
7. **DST-fragile fixed-UTC lines:** guardrail 20:10 (15:10 ET in EST), hourly_summary hand-switched ("Switch to `5 15-21` after Nov 1"), dead-man (item 1). Proposed ET clocks.
8. **No log rotation for onemil logs** (`/etc/logrotate.d` has none): `universe_build_weekly.log` 24.8 MB, `nightly_bt_update.log` 10 MB, `universe_build.log` 7.5 MB, `momentum*`. Disk 98 GB free, so low urgency.
9. **Orphan comments / stale manifest:** crontab lines 19, 21-24 (weekly owner report, hourly holdings pulse: both jobs gone; `logs/holdings_pulse.log` last 11/10), 35, 36, 42 describe jobs that do not exist. `docs/active_crons.md` still lists holdings_pulse, 20:30 BT, 21:30 green, 21:58 archive (synced 9/19).
10. **Unproven:** line 17's first run under Amendment 2 is tonight 23:20; `@reboot` claude_tmux has never fired; trader_watchdog has no heartbeat (log empty since 9/7 by design).

Not inspected beyond logs: the Telegram side of ERRORs in cron logs (e.g. yfinance 404 ERRORs in `universe_build.log` on 11/6 from a root logger with the Telegram handler attached would page the owner).
