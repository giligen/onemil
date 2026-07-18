# Deploy watch runbook — Mon 2026-07-20 + Tue 2026-07-21

Owner is AWAY both days (phone/Telegram only, no laptop). Claude operates
the watch autonomously. This document is the pre-committed decision
matrix — checkpoint sessions follow it EXACTLY; no improvisation beyond
it without a 🔴 emergency.

## Deploy surface (first live day Monday)
- **Catalyst-required veto** (`orb.yaml::filter.catalyst_veto`, ON):
  newsless-and-alone ORB picks vetoed post-ranking, slot consumed, no
  refill. Money-safe by construction: can only SKIP trades.
- **Ignition shadow** (`config.yaml::ignition_shadow`, ON): journal-only,
  zero orders by construction, worker-thread isolated.
- Everything validated: 2,814 tests + Saturday dress rehearsal (real
  boot, real-API probes, real report + Telegram).

## Watch layers
1. **Mechanical** (crontab, marked `# DEPLOY-WATCH-JUL20`):
   `scripts/deploy_watch.py` at 12:38 / 13:44 / 16:33 / 21:52 UTC —
   ALWAYS telegrams, green or not. Plus the pre-existing
   `trader_watchdog` (12:40, silent-when-healthy) and the 21:30 green
   check / 21:40 shadow report.
2. **Claude checkpoints** (session cron): a few minutes after each
   mechanical run. Read the mechanical output, investigate non-greens,
   act per the playbook below, telegram a signed `[CLAUDE WATCH]`
   verdict. If the user sees mechanical messages but no Claude
   sign-off, the Claude session died — layer 1 still detects.

## Timeline (UTC)
Claude sessions trimmed to 3/day (2026-07-18, owner credit-cost ask):
T+5 boot triage, post-burst open audit, EOD wrap. All mechanical
telegrams + the 2-min sentinel (zero credit cost while quiet)
unchanged — detection coverage identical, Claude only wakes where
judgment is needed or when the sentinel fires.
- 12:30 service auto-starts (deploys current tree)
- 12:35 Claude T+5 boot triage (P1 clock starts here if red)
- 12:38 mechanical boot check; 12:40 trader_watchdog
- 13:30 market open; 13:31 news prefetch; 13:33 lag pass; 13:35 ORB burst
- 13:44 mechanical open check → 13:47 Claude open checkpoint (veto audit)
- 16:33 mechanical midday check (Telegram only, no Claude)
- hourly :07 (13:07-20:07) mechanical error-storm backstop
  (silent-when-healthy; thresholds vs Fri 7/17 baseline: 0 ERROR,
  0 Traceback, 2-11 WARNING/hr)
- 20:00 service exits (market close)
- 21:30 green check, 21:40 shadow report (pre-existing crons)
- 21:52 mechanical eod → 21:56 Claude eod checkpoint
- Wed 06:52 Claude two-day digest for the owner's return

## Error-storm sentinel (continuous)
A persistent journal monitor runs in the Claude session polling every
2 minutes during Mon/Tue 12:00-20:00: wakes Claude within ~2min on ANY
traceback, >=3 ERROR lines in 2min, or service death after 12:35.
Detection latency for a storm is minutes, not checkpoint gaps. If the
Claude session dies the sentinel dies with it — the hourly mechanical
storm backstop (crontab, Claude-independent) still telegrams within
the hour.

## Rollback ladder (in order — never skip a rung)
1. **Feature flags (designed, zero-state)** — keeps all fixed code:
   - veto off: `orb.yaml` → `filter.catalyst_veto.enabled: false`
   - shadow off: `config.yaml` → `ignition_shadow.enabled: false`
   - Both need a service restart to take effect → only apply AFTER
     20:00 UTC (for the next day) unless P2 emergency.
2. **Nuclear (boot-broken only)**: `git reset --hard cbb0115` (the tree
   that ran live Friday 7/17 and exited clean) + restart. All weekend
   work is pushed to origin — nothing is lost by resetting; restore
   later with `git pull`/`git reset --hard origin/fix/spy-regime-shared-helper`.

## Playbook
- **P1 — boot failure (12:41 checkpoint, service not active/crash at
  init)**: diagnose from `journalctl -u onemil-trader --since '12:25'`.
  Budget 20 minutes for a minimal fix-forward (one-line class of fix
  only). If not certain-fixed by **13:10 UTC** → rollback rung 2 +
  `sudo systemctl restart onemil-trader` + telegram what/why. A late
  boot by 13:25 still catches the 13:35 burst.
- **P2 — crash-loop mid-session (>2 restarts)**: systemd already
  auto-restarts. Diagnose; if the traceback implicates veto/shadow code
  paths → flag-flip that feature (rung 1) + restart (this is the ONE
  authorized mid-session restart — the service is already bouncing).
  If implicating pre-existing code → rollback rung 2 + restart.
  Telegram immediately either way.
- **P3 — veto anomalies (wrong vetoes / vetoes everything)**: money-safe
  (skip-only). **NO mid-session action.** Spot-check each vetoed
  symbol: the log line carries anchor+cohort; cross-check news via
  `logs/orb_news_flags_<day>.json`. If wrong-veto confirmed → after
  20:00 flip veto off for Tuesday + telegram evidence. If correct →
  report the vetoed symbols' would-have P&L context at EoD.
- **P4 — shadow anomalies (drops, worker errors, wedge)**: journal-only,
  zero market risk. NEVER restart for the shadow alone. If (and only
  if) scanner cycle overruns are BOTH elevated vs Friday's baseline AND
  correlated with shadow warnings → after 20:00 flip shadow off for
  Tuesday + telegram.
- **P5 — 21:40 report missing**: run
  `python3 scripts/ignition_shadow_report.py` manually; if it sends,
  investigate the cron env; telegram result.
- **P6 — anything not covered**: do NOT invent mid-session
  interventions. Document, telegram with 🔴 and a clear ask, wait for
  the next checkpoint. The owner reads Telegram.

## Standing constraints (unchanged by away-mode)
- No mid-session restarts except P2.
- Never touch caches (`cache.db`, CSV caches, `--build-cache`).
- No refactors on watch days: fix-forward diffs only, each committed +
  pushed with a clear message and telegrammed.
- Every action taken must appear in a Telegram message — the owner must
  never come back to a surprise.

## EoD checkpoint duties (Mon and Tue)
1. Confirm 21:30/21:40 crons fired (green check + shadow report).
2. Veto audit: list vetoed symbols, spot-check 2-3 against news
   snapshot + anchor/cohort (P3 criteria).
3. Shadow S1 data quality: journal records have `latency_s`,
  `spread_bps`, verdicts distributed sanely (not 100% one bucket).
4. Telegram a day-wrap: trades, P&L, vetoes, shadow triggers, S1
   scoreboard verdicts, any actions taken, tomorrow's plan.
5. Tuesday only: leave the Wednesday digest everything it needs
   (nothing to do — journals + this doc persist).

## Cleanup (Wednesday digest session)
- Remove the `# DEPLOY-WATCH-JUL20` crontab lines.
- Compile the two-day digest telegram (S1 pass-bar tracking vs the
  week-1 gate, veto cost/benefit tally, incidents, recommendation for
  the S3 micro-live go/no-go conversation).
