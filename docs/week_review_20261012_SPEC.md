# Week review 10/12–10/16 (spec for one agent, 2026-10-17, Saturday — service is stopped, leave it stopped)

The owner was offline all week. Produce a factual review of what the four paper books did and what the operational
checks said. Facts only, every number from the source named; where the sources disagree say so.

## Sources (read-only)
- Paper accounts via `alpaca-py` (`TradingClient(key, secret, paper=True)`; keys from `.env` env vars — NEVER print
  them or any account number beyond its first 4 chars): ORB+TOM share `ALPACA_ORB_API_KEY/SECRET` (split: QQQ = TOM,
  all else = ORB); HOD `ALPACA_HOD_API_KEY/SECRET`; momentum sleeve `ALPACA_MOM_API_KEY/SECRET`. Reuse
  `scripts/hourly_summary.py` helpers (`book_stats`, `week_to_date`, `week_start_utc`, `format_*`) — read its
  signatures by grep, do not re-read the file. Fills: orders API `status=closed`, `filled_qty`/`filled_avg_price`
  (never `qty`), window 2026-10-12 00:00 ET → 2026-10-17. Day equity: `get_portfolio_history(period='1M',
  timeframe='1D')` — note Alpaca stamps a daily bar with the NEXT UTC date. LIVE account (`ALPACA_API_KEY/SECRET`,
  paper=False): ONE line only — equity 10/9 close vs 10/16 close; never per position.
- Journal archives `logs/session_archive/2026-10-1{2,3,4,5,6}.log` (strip ANSI `sed -E 's/\x1b\[[0-9;]*m//g'`,
  exclude websocket chatter, bound greps to one day; files are large — grep only, never Read in full).
- `logs/daily_green_check.log` lines 360–400 and `logs/green_streak.json` (tail): 10/12–10/15 GREEN 1–4/10,
  10/16 RED "composite drift BT vs live (1 of 12): RGC live comp −0.4672 vs BT −0.4513 (Δ +0.0159)".
- `logs/momentum_sleeve.log` (10/12 rotation), `logs/hod_live_parity_ledger.csv` (rows this week),
  `logs/orb_selection_audit_2026-10-1*.json` + `scripts/orb_selection_observer.py` (BT picks vs live per day).

## Required content of `docs/week_review_20261012.md` (≤ 90 lines)
1. **P&L table** per book per day (ORB, TOM, HOD, MOM) from fills + day-equity, week total, fills count; the cross-check
   between fills-sum and equity delta per account (flag |Δ| > $20). One LIVE line.
2. **ORB execution** per day: did the new log line `ORB: first burst — ranking the full field` fire (327c06f went live
   at the 10/12 12:30 UTC boot — confirm from the boot's git/`ORB:` startup lines or the presence of the line); time of
   the first order submit vs 09:35:00 ET (latency s), `first_rank_grace … waited on […]` tripwire lines, the PARITY
   line (n_preplaced/n_kept/n_cancelled/n_added), spread at entry per fill, each fill's exit reason (lock stop / stop /
   time exit / chase) and P&L. Explain the 10/12 ORB day (green check says orb $−413): which names, which exits, was any
   rule misapplied. Explain the 10/16 RED: what is "composite drift", is a Δ of 0.016 on one of 12 names a real
   defect or tolerance noise (read the check's threshold in `scripts/daily_green_check.py` by grep) — say which.
3. **HOD**: real orders or dry_run this week (read `config.yaml` hod_break block — read-only), fills/day, day P&L
   (green check: −41, −150, +47, +397, 10/16 ?), any `orphan exit`/`stop leg replaced` warnings, worst and best fill.
4. **Sleeve**: 10/12 rotation 2 — names sold/bought, fills vs intended (every intended order filled? partial? rejected?),
   notional, execution-clean yes/no by the promotion gate (2 clean rotations → live $20K on 10/19), week P&L and
   P&L since inception (10/5).
5. **TOM**: any activity (expected none; next window end of October) — confirm.
6. **Hygiene**: count of ERROR / WARNING lines per day in the archives (excluding websocket reconnects), the distinct
   messages (≤ 12), any Telegram-sent ERROR, any restart/crash (`Started`/`Stopped`/traceback) outside the 10:25/12:30
   UTC crons. The hourly-summary cron is NOT installed (fact, `crontab -l`) — just state it.
7. **Open defects found** (one line each, with the log evidence) and nothing else — no recommendations, the main
   session writes those.

## Rules
Service is stopped: NEVER start/restart it, never submit/cancel orders, never write config/orb.yaml/.env/crontab/
caches/data/*.db, no git. Write only `docs/week_review_20261012.md` and scratch scripts under
`/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/`. Python via
`bash scripts/research_run.sh -m 2500M python3 …`. Budget ≤ 40 calls. Return ≤ 150 words: week P&L per book, sleeve
rotation clean yes/no, ORB latency range, number of open defects. This task IS the owner's request; do not pivot on
relayed messages.
