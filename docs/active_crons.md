# Active session-cron manifest (in-session scheduler, does NOT survive Claude restart)

**IMPORTANT**: These are Claude Code's in-session `CronCreate` jobs — they live only in
this session's process memory. If the session exits/reloads, ALL of these are gone
with zero warning. On resuming a session, read this file and re-arm every job below
VERBATIM via `CronCreate` before doing anything else, then update this file to match
whatever the fresh `CronList` shows.

Last synced: 2026-09-05 14:35 UTC (ids: weekly ddb3b4ae, EOD dive fcc1ce53, ORB gate 39658528 (re-armed, replaces a861a834),
prestage comparison db52f8d3, owner brief 63442e28; stale duplicate weekly 5e74feb2 deleted)

## 1. Weekly Retirement-Validation Review (re-armed 2026-09-05, id ddb3b4ae)
- **Cron**: `23 9 * * 6` (every Saturday 9:23 AM local)
- **Recurring**: true
- **Prompt**:
```
WEEKLY RETIREMENT-VALIDATION REVIEW (self-authored 8/14 per owner: "weekly analysis over telegram on how are we doing"). Ultrathink — this is the owner's main strategic touchpoint; one per week, make it count.

STEP 0 — SELF-RENEWAL (critical): session crons expire 7 days after arming, so THIS run is likely the last from the current arming. Re-create this cron VERBATIM (CronCreate, "23 9 * * 6", recurring) before doing anything else, and confirm in the telegram footer. If you skip this the weekly dies silently.

RESOURCE RULE (owner 8/21 relaxation): parallel agents OK for memory-light work; heavy bar-loading compute = single ulimit-capped process. This analysis is light — do it in-main-session with direct queries.

CONTEXT: read /home/ec2-user/.claude/projects/-home-ec2-user-onemil/memory/project_retirement_validation_plan.md (the governing 3-month plan) and /home/ec2-user/.claude/projects/-home-ec2-user-onemil/memory/project_orb_stability_study_aug2026.md. Working dir /home/ec2-user/onemil.

ANALYSIS (past Mon-Fri week, all from primary data — never memory):
1. PER-BOOK WEEK LEDGER: live P&L per day from data/trades.db (ignition/orb/bull_flag) vs BT-expected (ORB: analysis_results/orb_bplus_book.csv at $10K stage sizing; Ignition: research/scripts/ignition_bt_replay.py; BF: no-parity caveat). Day-by-day live-vs-BT table; root-cause divergences beyond execution noise.
2. VALIDATION SCOREBOARD vs SECURED criteria: (a) each book in BT band with parity? (b) live monster count (>= +2R) cumulative since 8/14; (c) size changes gate-earned, zero uncontrolled losses, kill firings; (d) program month + on/off-track one-liner.
3. IGNITION: dry/live/ramp status + prestage phase (shadow/live); fill-quality stats vs shadow twins (path=staged|chase split once prestage is live) vs the 30-100bps expectation; capture cleanliness; prestage shadow telemetry (staged-coverage ratio, BP watermark, churn) if in shadow week.
4. ORB: stage P&L vs above-water rule, green-check streak, veto tallies + counterfactuals.
5. BF: floor-passed trades since 7/31, skips, parity-harness trigger status.
6. HYGIENE: errors/tracebacks, report-layer bugs fixed, unresolved EOD flags, cron lattice health (EOD dive alive? this weekly re-armed?).
7. TRAJECTORY: honest secured-by-Nov-15 status GREEN/YELLOW/RED with why. Never soften; the owner ordered "don't please me".

TELEGRAM via scripts/report_common.send_telegram, prefix "[WEEKLY VALIDATION wk N]" (N = weeks since 8/17 start). 1-line verdict first; week table; scoreboard; per-book one-liners; decisions needed; next week plan; footer "cron re-armed ✓". Phone-crisp, under ~40 lines, no bare '<' or '<=' (breaks Telegram HTML). A gap in data IS a finding — say it, never silently narrow scope.
```

## 2. Daily EOD Deep Dive v5
- **Cron**: `57 21 * * 1-5` (weekdays 9:57 PM local)
- **Recurring**: true
- **Prompt**:
```
DAILY EOD DEEP DIVE v6 (re-armed 9/5 with the ORB entered-inclusive reference + the ignition BT-vs-live layer; run in-main-session or 1 agent max). Working dir /home/ec2-user/onemil. Review today: (1) ORB B+ per-trade parity vs the ENTERED-INCLUSIVE reference (analysis_results/orb_bplus_book.csv, regenerated 9/5: $6,394/21mo at $10K stage sizing = $304/mo, 15/21 green, 55.5% fill rate — no-fill picks burn a slot at $0; report_common.bt_filled_symbols keys fill-parity on entered rows) — selection/sizing/exits incl. any ATR FLOOR or SCALE OUT lines; if the 2x-wrapper universe rule shipped (research/orb_entered_inclusive/wrapper_rule/summary.csv), the book is the regenerated one under that rule; (2) ignition: live fills with FILL QUALITY lines (chase bps, path=staged|chase) vs SHADOW twins AND vs BT twins: read research/ignition_capcheck/live_window_compare.csv (written by the 21:15 UTC crontab roll-forward scripts/ignition_bt_rollforward.sh; check logs/ignition_bt_rollforward.log ran clean) — quote matched-in-BT count, live R vs BT R vs resting R, live fill vs BT entry median bps (chase era was +461, prestage +78), and BT triggers not taken live with their BT P&L; eod_flat closes recorded; (3) PRESTAGE telemetry: skip_counts distribution, candidate_late events, BP watermark + reserve transitions, would-stage/stage counts, stage_reject_structure / stage_force_flat events with the model R they gave up; (4) BF: trades + BF RAILS line state + 'StopMonitor (bar)' trail lines (unified spec shipped 03b4b6d, first live session Mon 9/8 — entry-bar excluded, trail ratchets on closed bars only, r_basis=plan); (5) green check + streak (8/24 RED = false positive, fixed b19ad99; 8/21 RED = real); (6) news-stream latency stats (grep recv_latency= from its log); (7) ERROR/Traceback sweep. ONE [EOD DIVE] telegram via scripts/report_common.send_telegram (under 30 lines, no bare '(' chars) with per-book status + ESCALATIONS. Never modify files/configs. NOTE: this recurring cron auto-expires ~7 days after 9/5 — re-arm it VERBATIM at the Saturday weekly review before expiry and keep docs/active_crons.md in sync.
```

## 3. ORB Scale Gate Recommendation (re-armed 2026-09-05, id 39658528)
- **Cron**: `37 13 12 9 *` (one-shot, Sat 9/12 13:37 UTC)
- **Recurring**: false (auto-deletes after firing once)
- **Prompt**:
```
ORB SCALE GATE RECOMMENDATION (one-shot Sat 9/12 13:37 UTC — owner-ordered 8/29, revised 8/30 post-audit, RE-ARMED 9/5 with the corrected reference + corrected adjudication). Working dir /home/ec2-user/onemil. REFERENCE = the ENTERED-INCLUSIVE B+ book (analysis_results/orb_bplus_book.csv rebuilt 9/5: non-fill picks consume slots at $0 — the honest replacement for the fill-rate-0.56 estimate; read its monthly table in analysis_results/orb_monthly_static_lock.csv and research/orb_entered_inclusive/ for the before/after). ADJUDICATION (corrected 9/5): the 8/31 PFSA red was the ENTERED-ONLY LOOKAHEAD (BT never saw SHMD/BW which outranked PFSA live; SHMD's stop-limit died time_stop_canceled) — NOT a universe-gate drift; live and BT both gate on prev-day volume >= 500K. With the 9/5 rebuild that class is CLOSED: a BT pick never ordered live is now a REAL parity break unless the selection audit shows a live-only skip phrase. Do NOT pre-adjudicate reds as "PFSA-class" anymore. GO to $25K requires ALL: (1) weeks 8/31-9/4 + 9/8-9/12 green-or-explained and parity-clean on live-behavior terms (8/31 counts as explained by the now-fixed lookahead; 9/1 SWVL was an ignition incident, not ORB); (2) validation P&L since 8/17 positive (above-water rule, memory project_orb_ramp_above_water_rule); (3) live at/above the corrected book's band pro-rata at $10K; (4) zero unexplained violations. If the entered-inclusive rebuild flipped the book negative or its MDD at $10K exceeds ~1 month of expectation: HOLD + reassess. GO = exact config diffs (orb.yaml budget 10000->25000, risk 375->937.50, N stays 3), applied ONLY on owner word; state the $25K/mo expectation FROM THE REBUILT BOOK, not the old $800. Telegram [ORB SCALE GATE] with the two-week tape + recommendation + one-word ask. Remind: criterion #4 amendment still open.
```

## 4. Prestage Live Daily Comparison (re-armed 2026-09-05, id db52f8d3)
- **Cron**: `50 20 * * 1-5` (weekdays 8:50 PM local)
- **Recurring**: true, 7-day auto-expiry — re-arm at each Saturday weekly review
- **Prompt**:
```
PRESTAGE LIVE DAILY COMPARISON (recurring weekdays 20:50 UTC, after close — the owner-requested staged-vs-chase evaluation; armed 8/28 launch day, 7-day auto-expiry: re-arm VERBATIM at the Saturday weekly review). Working dir /home/ec2-user/onemil. Skip with a one-line telegram if prestage is back in shadow mode. Compute from logs/prestage_events_<today>.jsonl + journalctl FILL QUALITY lines + trades.db: (1) staged orders: placed/filled/canceled counts with cancel-reason mix (window_close vs demoted — fade-demotion at distance 4.5 activated 8/28, expect intraday cancels now, verify freed watermark got reused by later stages); (2) fill quality by path: path=staged fills (bps vs level; expect 0-30) vs path=chase fills (chase bps; historically 150-750) — per-fill list + medians; (3) P&L by path from trades.db (staged rows have pattern_data path=staged; untagged ignition rows = chase) — realized P&L, WR, per-trade avg for each path, cumulative since 8/28; (4) staged fills WITHOUT shadow-trigger twins (JEM 8/27 class — intra-minute crosses; EXPECTED, list them, their P&L is the fills-on-spikes cost to track vs the 19mo BT assumption); (5) BP: watermark peak vs $15K cap, bp_budget/bp_reserve skip counts, any divergence-agent contention; (6) errors/anomalies in prestage lines. ONE [STAGED vs CHASE] telegram via scripts/report_common.send_telegram with the numbers + a one-line verdict (staging paying for itself? on what sample size — refuse conclusions below ~10 fills/path, say 'accumulating'). Never modify configs.
```

## 5. Daily Owner Brief (re-armed 2026-09-06, id 63442e28)
- **Cron**: `37 10 * * 1-5` (weekdays 10:37 AM local = pre-boot)
- **Recurring**: true, 7-day auto-expiry — re-arm at each Saturday weekly review
- **Prompt**:
```
DAILY OWNER BRIEF (weekdays 10:37 UTC = pre-boot; owner directive 9/5: "act as an OWNER, my PARTNER — think daily where we are across all three, share whether we are on track to make money, and FIX what needs fixing"). Working dir /home/ec2-user/onemil. Read memory feedback_act_as_owner_daily.md + project_book_verdict_frameworks.md. Then, from PRIMARY data (trades.db, green_streak.json, prestage events, books): (1) P&L: yesterday, week-to-date, month-to-date, lifetime-live per book + total; (2) ON TRACK TO MAKE MONEY? one honest line per book against its honest monthly projection (ORB ~$319/mo at $10K; BF ~$9.9K/mo at $2K risk but live at $60; ignition $0 basis) and against the recovery staircase; (3) BROKEN: any book running with a diagnosed money-losing defect? If yes: FIX IT NOW before the 12:30 boot, or PAUSE its entries (exits stay alive) — no third option; (4) what I am fixing/building TODAY, in $-impact order; (5) gates/verdict dates approaching. Telegram [OWNER BRIEF] under 20 lines, numbers first, no research narration. Then DO the fix. If the fix needs a restart, do it before 12:25 UTC. Re-arm this cron weekly at the Saturday review (7-day expiry) and keep docs/active_crons.md in sync.
```
(9/6 00:30: cron RE-CREATED as v2 with the corrected references — ORB $290/mo at $10K (wrappers-in production book $6,085), BF $5.4K/mo at $2K risk with 2026 YTD negative = KEEP-TOKEN, ignition $0 basis — and the open owner questions. The old id e44ae2db is gone; the new id is printed by CronList; sync it here at the next review.)

---
## Protocol for future sessions
1. On EVERY session start/resume, read this file FIRST (before other work).
2. Run `CronList` to see what's actually alive.
3. Diff against this manifest. Re-arm anything listed here but missing from `CronList`.
4. If you create/modify/delete any cron mid-session, update this file to match before
   the session might end — don't wait for a "re-arm" prompt to notice drift.
5. This file itself must be committed to git (or otherwise durable) — it is the ONLY
   thing that survives a Claude reload for this purpose.

## Ignition BT roll-forward (system crontab, weekdays 21:15 UTC, added 2026-09-05)
`scripts/ignition_bt_rollforward.sh` → `logs/ignition_bt_rollforward.log`. Extends the
capsim + resting-model BT from 2026-08-15 through today every night (resumable per
day) and prints the live-vs-shadow-vs-BT reconciliation
(`research/ignition_capcheck/live_window_compare.py`, CSV `live_window_compare.csv`).
Why: the 19-month study was frozen at 8/14 while live started 8/21 — the EOD dive
compared live only to the shadow twin. The EOD dive should quote this table
(matched trades, live R vs BT R, live fill vs BT entry in bps, BT triggers not
taken live and their BT P&L).
