# Independent pre-Monday code review — 2026-10-03 (owner's ask)

Question for every track: **what would crash us on Monday 10/5 (paper) or silently waste the day**, and **where does the
live rule differ from the backtest rule** (parity by construction is the project's standard: one helper shared by BT and
live, enforced by a parity test). Report defects, not style. Rank by "loses the day" first, then "wrong vs BT", then rest.

Rules for the reviewer: read code with grep/offset (never a file > 300 lines in one read); you may run ONE unit test
file at a time with `python -m pytest -q <file> -x --ignore=tests/integration` (never tests/integration: it trades a
real paper account); never start/stop/restart any service, never submit orders, never edit code, never git commit,
never read `.env` contents, never write under data/ or logs/. Write your report to the file named in your track section
(≤ 120 lines, each finding: file:line, what happens, trigger on Monday, severity, one-line fix). Return ≤ 150 words.

Service: systemd `onemil-trader` = `python main.py --scan --trade --verbose` + flags (see `systemctl cat onemil-trader`
read-only and the crontab via `crontab -l`). Timeline UTC: 10:25 stop, 11:50 sleeve prefetch, 12:30 start, 13:30 US open,
13:45 sleeve submit, 19:45 TOM QQQ sell, ~20:05 exits. Monday checklist: `docs/monday_boot_checklist_20261005.md`.

## Track A — ORB paper (report: docs/review_20261003/A_orb.md)
Code: `trading/orb_engine.py` (open tick, `universe_build_due`, `_fetch_today_open_bars`, gap gate call, add-on, tilt,
last_entry_submit_time cutoff, shutdown handling), `trading/orb_gap_gate.py` (`gap_input_needs_today_open`,
`resolve_gap_input`), `scanner/realtime_scanner.py` `_orb_tick`, `data_sources/alpaca_client.py`
(`get_1min_bars_range_multi`, `_qty_num`), `scripts/orb_open_tick_replay.py`. Config `orb.yaml` (read-only).
BT rule: `research/orb_machine_rules.md`, `study_orb_pipeline_static_lock.py`, parity notes
`research/orb_freq/gap_input_parity_20261002.md`, `docs/orb_allday_rebuild_20261002.md`. Tests
`tests/test_orb_gap_input_parity_20261002.py`. Friday's defects for context (already fixed, verify the fix is complete):
universe rebuild ran all day (209,937 WARNINGs, 279 tick TIMEOUTs); the gap-input hook read a table nothing writes.
Look especially at: exceptions escaping the open tick, API timeouts inside the open budget, empty/None snapshots, the
10:00 ET cutoff, RVOL tilt edges vs BT, add-on at +1 R vs BT, spread gate ≥ 150 bps, skip_q1, Q5 1.5× cap, no refill after veto.

## Track B — HOD paper dry-run (report: docs/review_20261003/B_hod.md)
Code: `trading/hod_break_engine.py` (focus: `_enforce_kill_rails_on_resting`, `_reconcile_positions_to_broker`,
`_sync_registry_qty_to_broker`, `_adopt_unregistered_positions_on_boot`, `_on_live_fill`, 15:55 flatten, caps, kill rails),
`trading/stop_monitor.py` (routing), `scripts/hod_dry_ledger.py`. Spec of Friday's fix: `docs/hod_killrail_restart_qty_20261002.md`;
tests `tests/test_hod_killrail_restart_qty.py`. BT rule: `research/hod_entry/PREREG_1466.md`, `RESULT_1466.md` (stop-limit
exit 20 bps below the stop), `research/hod_entry/LADDER.md`. Friday's incidents: registry qty 21 vs broker 44 after a
mid-session restart; a fill after the daily kill sat with no stop; after-hours manual flatten needed.
Look especially at: the 60 s reconcile loop's failure modes (API error → exception → loop dies?), foreign positions on the
same paper account, qty as float, flatten completeness, caps counted on fills not resting, kill-rail idempotence.

## Track C — momentum sleeve paper (report: docs/review_20261003/C_sleeve.md)
Code: `scripts/momentum_sleeve.py` (prefetch, --submit, `fetch_cboe_close`, `shadow_gate_info`, `gate_scale`, ledger/CSV
writers, REJECTED handling, broker resync), `trading/momentum_sleeve.py` (universe, guard, ranking, `term_structure_gate`,
`target_dollars`), tests `tests/test_momentum_sleeve.py` (parity fixtures under `research/momentum_weekly/recon/`).
Spec: `docs/momentum_sleeve_guard_spec_20261002.md`. BT engine: `research/momentum_weekly/1700s_lowvix.py` (daily engine,
guard), `1700u_gate_guarded.py` (half-size gate cell `VIXratio|w252|p20|half`), PREREGs 1,700t/1,700u. Cron lines for
11:50 prefetch and 13:45 submit (`crontab -l`). Account = paper (ALPACA_MOM_*), $20K, 20 names, Monday 09:45 ET market orders.
Look especially at: what happens if the CBOE file is missing/stale (gate n/a → full size? is that the safe direction?),
duplicate submission if the cron fires twice or the run is retried, partial fills / REJECTED, fractional vs whole shares,
the 273-bar boundary and the hygiene guard vs the BT, the percentile convention vs the BT (strict-less + half ties),
Monday-open timing vs BT (BT trades Monday open; live trades 09:45 ET), what the 11:50 run writes that the 13:45 run reads.

## Track D — platform (report: docs/review_20261003/D_platform.md)
`main.py` startup flags and the systemd unit, `scripts/research_run.sh` (cage), crontab (every line: escaped %, paths,
the 10:25/12:30 service stop/start, pre-boot test cron), Telegram sender timeouts, journald rotation, disk (`df -h`), the
memory reaper, second-writer risk (`ps -eo pid,etimes,args | grep claude`), DB locks (`data/trades.db` writers), any
uncommitted engine edit (`git status --short`), anything in `git log --since=2026-10-02` touching trading/ scanner/
data_sources/ scripts/ that lacks a test. What could keep the service from booting at 12:30 or kill it mid-session?
