# Session-scheduled jobs (CronCreate, in-memory) — re-arm after any session restart

These jobs live only inside the interactive Claude session (not in crontab, not on disk). They auto-expire 7 days after
creation. After a session restart or on Saturdays, re-create every job below with CronCreate using the exact cron and
prompt, then run CronList and update the "created / expires" lines here. Owner rules: 2026-11-14 "from here, instead of
a cron" (nightly maintenance); 2026-11-16 "monitor telegram during trade hours every 15 min for errors, fix if needed and
report" (trading-hours watch) and "dump all existing monitors to an md file if session restarts" (this file).

Box clock = UTC. ET = UTC−5 (EST, until 2027-03-14).

## 1. GAPSHORT nightly maintenance — id 2b543d5c
* cron: `7 22 * * 1-5` (22:07 UTC ≈ 17:07 ET, after the 16:35 ET parity and the 16:15 ET score)
* created 2026-11-14 (Sat) → expires 2026-11-21; re-arm every Saturday
* prompt: the text in `docs/autofix/gapshort_autofix_prompt.md` (read today's `[GAPSHORT PARITY]` row and the
  gap_short logs; OK line to `logs/gap_short_autofix.log` or dispatch ONE Sonnet fixer under the allow-list / frozen
  constants in `trading/gap_short_rules.py`; full suite; commit + push; `[GAPSHORT FIX]` Telegram; strategy questions
  go to the owner, never changed)
* fallback when no session is alive: `scripts/gapshort_autofix.sh` (headless `claude -p`), not in crontab by default

## 2. Trading-hours error watch — id ae4a75e5
* cron: `3,18,33,48 14-21 * * 1-5` (every 15 min 14:03–21:48 UTC = 09:03–16:48 ET; covers the 09:20 pick, 09:25
  locate ledger, 10:30 cover, 10:40 flatten, 16:15 score, 16:35 parity)
* created 2026-11-16 (Mon) → expires 2026-11-23; re-arm every Saturday together with job 1
* prompt (verbatim):
```
Trading-hours error watch (owner 2026-11-16: "monitor telegram during trade hours every 15 min for errors, fix if
needed and report"). Working dir /home/ec2-user/onemil. Check, for the last 16 minutes only: (1) `journalctl -u
onemil-trader --since "16 minutes ago" --no-pager | grep -E "ERROR|CRITICAL|Traceback"`; (2) every file in logs/*.log
modified in the last 16 minutes (`find logs -name "*.log" -mmin -16`) grepped for "ERROR|CRITICAL" with lines
timestamped inside the window; (3) logs/gap_short_paper.log, logs/gap_short_parity.log, logs/locate_ledger.log,
logs/orb_observer_cron.log for WARNING lines that mean a lost trade (rejected, expired, not submitted, stale,
refusing). If nothing: append one line "<UTC time> OK" to logs/telegram_watch.log and say nothing to the owner (no
message). If something: diagnose from the journal/log context first (read the lines around it, check the broker
state read-only via GET if an order/position is involved; never submit/cancel orders on the LIVE account; cancelling
one of OUR dangling orders on a PAPER account is allowed when the position is flat), then: trivial and safe fix
(classifier/parser/guard, a path, a test) → do it inline with a test, run that test file only, commit + push; engine
or order-path change → write the spec and dispatch one Sonnet agent, review its diff, run the touched test files,
commit + push; strategy or money question → do not change anything, report. Never restart the service during trading
hours. Then report to the owner in ≤ 5 lines: what the error was, root cause, what was changed (commit) or why
nothing was changed, and what to watch next. Also append the summary line to logs/telegram_watch.log. Keep model use
per the user-level cascade policy and end with the Models: line.
```

## Re-arm checklist (Saturday or after a restart)
1. `CronList` — note what is still armed and its expiry.
2. `CronCreate` each missing job with the cron + prompt above (`recurring: true`).
3. Update the "created / expires" lines here, commit.
4. Memory pointer: `project_gapshort_session_maintenance.md` (nightly) and `project_session_jobs_nov2026.md` (this file).
