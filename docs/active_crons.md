# Active cron manifest — regenerated 2026-11-12 from `docs/crontab_proposed_20261112.txt` (== `crontab -l`)

`crontab -l` is the truth; this file is its index (when, TZ, purpose, log, producer it depends on). One row per
active line. Reports are daily + weekly, once each; everything else is alert-only (silent when clean) or
infrastructure. Every Python entry point below is wrapped in `monitoring.cron_guard.run_guarded` (11/12): an uncaught
exception = one ERROR naming the script -> Telegram (send awaited) -> exit 1. `main.py --batch` has its own wrapper
(`run_batch`: ERROR with traceback + exit 1, plus the LOST-symbol completeness gate).

## 1. Session crons
Weekly review `69f28d67` (`23 9 * * 6`, `[WEEKLY wk N]`) lives in the Claude session only (7-day expiry): on resume,
re-arm it verbatim and update the id here. The daily EOD report is a durable cron since 9/21 (row below).

## 2. User crontab (`ec2-user`), TZ = UTC unless the block says otherwise
| schedule | TZ | purpose | log | depends on |
|---|---|---|---|---|
| `30 10 * * 1-5` | UTC | `main.py --batch` universe build (BF + ORB `daily_bars`) | `logs/universe_build.log` | Alpaca assets + daily bars; must finish before the 12:30 boot |
| `20 12 * * 1-5` | UTC | **universe-build completion alert**: Telegram if today has no `Batch complete` line | Telegram | the 10:30 build |
| `0 6 * * 6` | UTC | weekly verbose universe build | `logs/universe_build_weekly.log` | — |
| `45 22 * * 1-5` | UTC | `nightly_bt_update.sh` BF cache append | script's own | after the ORB BT timer (ends <= 22:41) |
| `26 13 * * 1-5` | UTC | `orb_selection_observer.py` live-vs-BT selection watch | `logs/orb_observer_cron.log` | the morning universe/boot |
| `20 23 * * 1-5` | UTC | `daily_green_check.py` ORB Amendment-2 green streak (waits <= 45 min for the BT) | `logs/daily_green_check.log` | ORB BT (timer), 23:15 pick parity |
| `40 12 * * 1-5` | UTC | `trader_watchdog.py` (service up / crash-loop) | `logs/trader_watchdog.log` | root 12:30 start |
| `58 22 * * 1-5` | UTC | `bf_decision_parity.py` (BT-only record while BF is paused, no freeze) | `logs/bf_parity/cron.log` | 22:45 BF cache |
| `0 20 * * 0` | UTC | `orb_weekly_refit.py` (alerts on failure) | `logs/orb_weekly_refit.log` | before Monday's boot |
| `0 23 * * 1-5` | UTC | `build_hod_volume_profile.py` | `logs/hod_volume_profile.log` | nightly bar refresh |
| `27 11 * * 1-5` | UTC | pre-boot full pytest, alert on failure | `logs/preboot_tests.log` | clean tree before 12:30 |
| `45,58 19,20 * * 1-5` | UTC | `tom_sleeve.py` | `logs/tom_sleeve_cron.log` | — |
| `@reboot` (sleep 20 / 30) | UTC | tmux Claude sessions `prod-onemil` / stupid-money `attach.sh` | `logs/claude_tmux.log` | — |
| `50 11 * * 1-3` | UTC | `momentum_sleeve.py` prefetch + dry run | `logs/momentum_sleeve.log` | — |
| `45 13`, `45 14 * * 1-3` | UTC | `momentum_sleeve.py --submit --skip-fetch` (EDT / EST) | `logs/momentum_sleeve.log` | the 11:50 prefetch |
| `46 13`, `46 14 * * 1-3` | UTC | `momentum_sleeve.py --live --submit` ($10K slice) | `logs/momentum_sleeve_live.log` | the paper submit's bars cache |
| `0 13`, `0 14 * * 6` | UTC | `mom_panel_refresh.py` (14:00 = retry, idempotent) | `logs/mom_panel_refresh.log` | Alpaca bars |
| `20 14 * * 6` | UTC | `mom_bt_parity.py` | `logs/mom_bt_parity.log` | the panel refresh |
| `4 21`, `10 21 * * 1-5` | UTC | stupid-money `eod_check.sh`, `forward_shadow.sh` — **other project, untouched** | its own | — |
| `35 17 * * 1-5` | ET | session archive (`journalctl` grep -> `logs/session_archive/<date>.log`) | the archive | the 16:30 ET BT timer |
| `5 10-16 * * 1-5` | ET | `hourly_summary.py` (10:05-16:05 ET) | `logs/hourly_summary.log` | — |
| `10 16 * * 1-5` | ET | `guardrail.py --check` (live ledger + auto-pause) | `logs/guardrail_cron.log` | after the HOD flats |
| `55 12,15 * * 1-5`, `0-30/5 13,16 * * 1-5` | ET | `hod_break_deadman_flat.py` ticks (gates on [close-5, close+30) from the calendar; no-op when the trader is active) | `logs/hod_deadman.log` | Alpaca calendar |
| `15 23 * * 1-5` | UTC | `orb_pick_parity.py --date <today UTC>` | `logs/orb_pick_parity.log` | the ORB BT |
| `25 23 * * 1-5` | UTC | `eod_report.py` -> `logs/eod/<date>.md` + `[EOD]` Telegram | `logs/eod_report.log` | every producer above |

## 3. systemd / root
* `onemil-orb-backtest.timer`: `Mon..Fri 16:30 America/New_York` -> `orb_backtest.py` (ends 22:27-22:41 UTC in EST, ~21:41 in EDT).
* Root crontab: `25 10 * * 1-5 systemctl stop onemil-trader`, `30 12 * * 1-5 systemctl start onemil-trader`.

## 4. Chains and windows
* **Nightly chain (UTC):** ORB BT (<= 22:41) -> 22:45 BF cache -> 22:58 BF parity -> **23:15 pick parity -> 23:20 green
  check -> 23:25 EOD report**; the green check waits up to 45 min for the BT, so it is never judged before it exists.
* **Morning chain:** 10:25 root stop -> 10:30 universe build -> **12:20 completion alert** (no `Batch complete` today = stale
  universe warning) -> 12:30 root start -> 12:40 watchdog.
* **Dead-man windows:** the HOD dead-man acts only in [close-5, close+30) ET (EDT 19:55-20:30 UTC, EST 20:55-21:30); the
  ET ticks 12:55 + 13:00-13:30 and 15:55 + 16:00-16:30 cover normal and early closes.
* Gate-1 freeze state: `logs/ramp_freeze.json`; clearing is manual (`bf_ramp_check.py --clear-freeze`) except the BF
  freeze left by the pre-11/12 paused-book false path, which `bf_decision_parity.py` clears with a logged reason.
* Removed 11/12 (do not re-add): holdings_pulse (hourly), 20:30 UTC ORB BT line, 21:30 green check, 21:58 archive,
  19:57 UTC dead-man, 20:10 UTC guardrail, 17:40 ET EOD.

## HOD owner-in-the-loop list (added 2026-11-12, ET block)
| 40 9 / 30 10 / 0 14 * * 1-5 ET | `scripts/hod_owner_list.py --at HHMM` | Telegram list of armed HOD names with context; ledger logs/hod_owner_list.csv | producer: the trader journal + SIP snapshot |
| 0 17 * * 5 ET | `scripts/hod_owner_list_score.py --since <Monday>` | owner picks (live manual orders) vs the machine on the same names | producer: the ledger above |
