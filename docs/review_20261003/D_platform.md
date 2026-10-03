# Track D platform review, 2026-10-03 (read-only; 12 tool calls)

Verified OK
- Service stop/start is in ROOT's crontab (10:25 stop, 12:30 start), not ec2-user's; unit enabled, Restart=on-failure, MemoryMax=2G, flags `--scan --trade --flag --orb --hod --r2g --verbose`. Service exited status 0 at 20:05:27 on 10/2 (clean).
- Engine dirs (trading/ scanner/ data_sources/ scripts/ main.py config.yaml) have NO uncommitted tracked edits; untracked items are only .bak files and loose research scripts at repo root (not imported by main).
- Crontab: every `%` in the pre-boot line and the session_archive line is escaped. Paths absolute. Pre-boot pytest 11:27, watchdog 12:40, deadman 19:57, guardrail 20:10.
- df: 49% used (104G free). RAM 7.8G, 5.6G available, swap 3.8G free. journald 1.9G of 3G cap. trades.db is WAL.
- Last day's journal at priority err: 1 line (the interpreter-shutdown message at 20:05:23, already downgraded to WARNING in c010fdd).

Findings (severity order)
1. MEDIUM, "kills the boot gate": 11:27 UTC full `pytest tests/ -x` (nice 10, not caged by research_run.sh, ~minutes) overlaps the 11:50 Mon/Tue sleeve prefetch (13.2K-symbol fetch, ~7 min) and any active research agent on an 8 GB / 2 vCPU node (froze once 10/2 under this load). A frozen box or OOM at 11:27-12:30 stops the 12:30 start. Fix: run pytest inside `scripts/research_run.sh -m 3G`, or move the sleeve prefetch to 11:57+ after pytest ends; keep research agents off 11:20-12:40.
2. MEDIUM, second-writer risk: 3 Claude sessions alive (tmux prod-onemil resume of this session, a `--fork-session --resume` of the same jsonl with --allow-dangerously-skip-permissions, a `claude --continue` in bash), plus @reboot cron relaunching sessions. A fork on a live-trading repo can commit/edit engine files concurrently (399771b was swept in by a forked session; HEAD was inconsistent between two commits). Fix: before 12:00 Monday, `git status` + `git log -3` clean-check and kill the forked/--continue sessions.
3. LOW, silent failure: the 11:27 failure alert and the watchdog both rely on Telegram; no timeout was audited in this pass (not read). If the sender hangs, an alert stalls but the trader does not (alert-only). Fix: confirm `send_telegram_alert.py` uses requests timeout<=10.
4. LOW: `Description=... (LIVE)` and `.bak.pre_orb` unit file in /etc/systemd/system are inert; the cron tail comments (Aug 17 backstop) are stale clutter. No action needed Monday.

Not checked (budget): Telegram timeouts in code, memory-reaper internals, DB writer enumeration under live load.
