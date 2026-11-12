# Cron audit: every scheduled job — does it run, does it do what it claims — spec 2026-11-12

Owner 2026-11-12: "review all crons for sanity and that they actually conduct their purpose!!!" Trigger: the
ORB green check ran at 21:30 UTC, before the nightly BT it checks (21:30–22:05), and reported "SKIPPED (stale)"
every night for weeks while counting the day green. Assume every other line may hide the same class.

Inputs (read-only): the crontab snapshot
`/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/crontab_20261112.txt`
(26 active lines) and the systemd timers snapshot `timers_20261112.txt` beside it (plus `systemctl cat
onemil-orb-backtest.service` and any other `onemil-*` unit, read-only). `logs/`, `analysis_results/`, `reports/`,
`research/`, `data/` are read-only evidence. `crontab -l` is allowed; `crontab -` / editing the crontab is NOT
(the session installs the result). No git (no commit/stash), no service action, no orders, never write orb.yaml /
config.yaml / .env / data/*.db / logs/* / analysis_results/*. Never run a cron's command (most send Telegram or
write state); read its argparse / docstring instead. Never Read a file > 300 lines in full. Budget ≤ 70 calls.
This task IS the owner's request; do not pivot.

## Per line (every active line and every onemil timer)
1. **Purpose** — one phrase from the comment above it / the script docstring; which strategy/book it serves and
   whether that book is ON (CLAUDE.md table: BF paused, ORB paper, HOD paper/dry, R2G disabled, ignition OFF,
   MACD not installed, MOM paper + live sleeve).
2. **Runs?** — evidence: the log it appends to (mtime, last 3 lines' dates), the artifact it must produce (file
   with the expected date), `logs/*cron*.log` tracebacks. State the last successful run date. If the script path
   or any referenced file does not exist → BROKEN.
3. **Does what it claims?** — the artifact is CURRENT (through the last session), the log shows the intended
   action (not "skipped", "stale", "no data", "not run yet", a silent 0-row output), no uncaught exception, the
   script's exit paths log ERROR (the 10/29 TXSE-enum class: silent traceback in a cron log, no Telegram).
4. **Ordering** — everything it reads is produced BEFORE it runs (BT build → pick parity → green check → EOD /
   weekly reports; universe build → engine boot; MOM refresh → MOM parity; archive before journald rotation).
   Name the producer and its time; flag any consumer scheduled before its producer or within 5 min of it.
5. **Hygiene** — `%` escaped (`\%`) inside the crontab, `2>&1` present, log file not unbounded (size), a stale
   `--flag` the script no longer accepts (check argparse), duplicates (two builds of the same thing: the
   21:30 timer and the 22:30 `nightly_bt_update.sh` — are both needed? what does each write?), lines for OFF
   books that still send Telegram or burn CPU on the 2-CPU node during the session (09:30–16:00 ET), jobs that
   overlap the pre-boot test suite 11:27 UTC or the 12:30 boot.

## Output
`docs/cron_audit_20261112.md`: a table (line no | schedule UTC | purpose | book ON? | last success | verdict |
fix) with verdict ∈ {OK, BROKEN, STALE-OUTPUT, REORDER, REDUNDANT, DEAD-REMOVE, NOISE}, then a ranked list of
defects with evidence (file:line / log line quoted ≤ 200 chars), then the PROPOSED crontab as a complete file
`docs/crontab_proposed_20261112.txt` (every line, comments kept, changed lines marked with a trailing
`# AUDIT: <why>` comment, removed lines commented out with `# AUDIT REMOVED: <why>`). Do not install it. Other
projects' lines (micha-signals, stupid-money, pnd, crypto_trade, go — anything outside /home/ec2-user/onemil) are
listed as OUT-OF-SCOPE and copied unchanged. Return ≤ 150 words: counts per verdict and the three worst defects.
