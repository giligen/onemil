# Active cron manifest — rewritten 2026-09-19 (owner: "keep only what we want for daily and weekly reports")

Two layers. **Session crons** live only in the Claude session's memory, expire 7 days after arming, and die with the
session — on resuming a session, read this file and re-arm every job in §1 VERBATIM via `CronCreate` before doing
anything else, then update the ids here. **Durable crontab / systemd** jobs survive everything and are listed in §2
so the weekly review can check that each one actually ran (log mtime).

Last synced: 2026-09-19 10:10 UTC. Removed today: the morning owner brief (its pre-boot test duty moved to a durable
alert-only crontab line), the HOD-break EOD check and the HOD live-watch monitor + its arming cron (folded into the
daily report; the monitor was relaying every routine skip line into chat), the superseded weekly (e0655167), the
Friday `weekly_report.py` telegram (duplicate of the Saturday review), `touchgo_daily_debug.py` (a June bug-hunt),
`go_backstop.py` (a one-shot dated 8/17). Principle: **reports are daily + weekly, once each; everything else that
survives is alert-only (silent when clean) or infrastructure.**

## 1. Session crons — exactly two

### DAILY EOD REPORT — RETIRED as a session cron 2026-09-21 (was id `989c1d20`), now durable crontab
`57 21 * * 1-5 scripts/eod_report.py` (owner: "move to haiku if no thinking is needed"). Every number is computed
by the script (trades DB per strategy, green-check verdict, BF parity JSON, both ramp checkers, services, journal
errors, pre-boot tests, disk, next-boot flags, today's research commits) and written to `logs/eod/<date>.md`; a
headless `claude -p --model haiku` call only rephrases it into the `[EOD]` telegram and falls back to the raw
summary (WARNING logged) if it fails. Code fixes are no longer part of the report — the report FLAGS (RED, NOT
ACTIVE, failed, frozen first line) and the owner asks the interactive session. Tests: `tests/test_eod_report.py`.

### WEEKLY REVIEW v2 — id `69f28d67` — `23 9 * * 6`
The one weekly telegram, prefix `[WEEKLY wk N]`. STEP 0 self-renews BOTH session crons and updates the ids here.
Runs `scripts/orb_ramp_check.py` and `scripts/bf_ramp_check.py` (Gate 2 incl. ex-best-trade and the BT band), the
week ledger vs the two-quarter BT path, HOD dry green weeks, research cells and dropped leads, hygiene incl. the
durable-cron mtimes, and the GREEN/YELLOW/RED trajectory against the scaling plan's clock.

## 2. Durable crontab (user `ec2-user`) + systemd — infrastructure and alert-only
| when (UTC) | job | role | telegram? |
|---|---|---|---|
| 10:30 Mon–Fri / 06:00 Sat | `main.py --batch` | universe build (BF + ORB daily_bars) | on failure only |
| **11:27 Mon–Fri** | **pre-boot full test suite → `logs/preboot_tests.log`** | **NEW 9/19: the 12:30 boot must not load a tree with failing tests** | **on failure only** |
| 12:40 Mon–Fri | `trader_watchdog.py` | did the 12:30 boot come up | on failure only |
| 13:26 Mon–Fri | `orb_selection_observer.py` | live-vs-BT selection watch (owner 7/23) | on real drops only |
| 14:05–21:05 hourly Mon–Fri | `holdings_pulse.py` | positions vs DB / orphans; **silent when flat** | only with open positions |
| 19:57 Mon–Fri | `hod_break_deadman_flat.py` | HOD dead-man flat (no-op in dry mode) | on action only |
| 20:30 daily | `onemil-orb-backtest.timer` → `orb_backtest.py` | ORB nightly features/book refresh (feeds the green check) | on failure |
| 21:04 Mon–Fri | `stupid-money/scripts/eod_check.sh` | **a different project — not ours, untouched** | — |
| 21:30 Mon–Fri | `daily_green_check.py` | ORB Gate-1 parity; sets the ramp FREEZE on a hard fail | on RED |
| 21:58 Mon–Fri | journalctl session archive | log archival | no |
| 22:30 Mon–Fri | `nightly_bt_update.sh` | BF cache append (Stage-1) | on failure |
| 22:50 Mon–Fri | `bf_decision_parity.py` | BF Gate-1 parity; sets the ramp FREEZE on disagreement | on disagreement |
| 23:00 Mon–Fri | `build_hod_volume_profile.py` | HOD rv_profile checkpoints (dry run needs it) | no |
| Sun 20:00 | `orb_weekly_refit.py` | ORB selection refit, 26-week window (owner 9/8) | on failure |

Gate-1 freeze state (set by the two parity jobs above, read by both ramp checkers):
`logs/ramp_freeze.json` — `{frozen, since, reason, by, frozen_dates}` per book. A freeze stops the stage clock and
blocks ADVANCE; clearing is MANUAL and logged:
`python3 scripts/bf_ramp_check.py --clear-freeze bf "<reason>"` (or `orb_ramp_check.py --clear-freeze orb ...`).
No cron invocation changed for this — the jobs' own commands are unchanged.

Backup of the crontab before today's edit: the session scratchpad `crontab.bak.20260919`; `crontab -l` is the
truth. Retired scripts stay in the repo (`weekly_report.py`, `touchgo_daily_debug.py`, `go_backstop.py`,
`hod_break_eod_check.py`, `hod_break_miss_audit.py` — the last two are still CALLED by the daily report, just not
scheduled on their own).
