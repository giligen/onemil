# Weekend program state — 2026-07-03/04 (CONTINUATION FILE)

Purpose: survive context compaction. Everything needed to continue the
weekend money-machine program is here. Read this + task list, then continue.

## Mandate (user, 2026-07-03)
Treat as MY OWN money machine. Full autonomy, creative license, BT-proven
ships only. Market CLOSED until Monday (July-4 full holiday — Fri 7/3 was
NOT a half-day; scanner correctly skips both holidays and short days).
Everything shipped lands at Monday 12:30 UTC auto-start (root crontab).

## Methodology guardrails (non-negotiable)
- Walk-forward: fit on 2025-H1 TRAIN (or 2025), validate 2026 OOS.
- MONSTERS-KEPT check on every ORB selection change: top-20 winners must
  survive (top-5 = 101% of defended P&L — lottery-with-edge structure).
- Per-rank / small-bucket P&L is NOISE at this concentration — bootstrap
  before believing (resample trades, check stability).
- Exits are settled (April 50-variant sweep, V0 Pareto). Refits are
  settled (frozen H1-25 fit beats refits by $34-47K OOS). ETF exclusion
  and own-P&L regime gating: REFUTED (see money_machine_audit_jul2026.md).
- Ships: config-flagged, tests, full suite green, commit+push.

## Key tools/data (all exist, validated)
- /tmp/orb_candidates_resim.csv — 6,208 candidate-level resims (static-lock
  exits, $50K flat pnl col). REGENERATE via /tmp/dump_orb_candidates.py
  (PYTHONPATH=/home/ec2-user/onemil, ~2min warm) if /tmp was cleared.
- /tmp/orb_replica.py + research/scripts/orb_pipeline_replica.py — exact
  defended-pipeline replica (validated: 1,193 trades / $154,892 == real
  pipeline). run_pipeline(df, train_start/end, exclude, skip_q1, n_slots).
- /tmp/orb_base_sel.csv — baseline defended selection w/ _sized_pnl.
- research/orb_symbol_class.json — ETF/single classifier (Alpaca names).
- data/bull_flag_cache_e50_x30.csv — BF Stage-1 raw cache (through 7/2,
  nightly-maintained). Stage-2: `python3 batch_backtest.py --start X --end Y`
  (reads config.yaml; use Config.set_config_path or --config for variants;
  NEVER --build-cache).
- backtest_results_march_2026.csv — latest Stage-2 output (floor 1.8:
  74 tr, +$31,865, 2026 +$6,674).

## This week's shipped state (all live Monday)
- Touchgo keying fix (range_end_ts; REKEY validation cron through 7/10).
- Selection-race fix (sweep retry 4s + first-rank grace 25s + selection
  audit JSONL). Commit 913e3f7. NOT yet live-validated (holiday) —
  Monday's observer run (cron 9:26 ET) is the validation.
- BF conviction floor min_threshold 1.4→1.8 (2026 BT +74%).
- Refit mandate cancelled (CLAUDE.md).
- Half-day skip policy pinned w/ BT evidence (tests/test_scanner_half_day_policy.py).
- exit_reason taxonomy + FABC/GLXG fixes + orphan-reconciler merge (origin).
- Branch fix/spy-regime-shared-helper @ 4fb6e00 pushed.

## Weekend findings so far
- W1 CLOSED (slot structure): all REJECTED. N=3/5 noise-or-worse. Rank-wt
  sizing bootstrap: P(delta>0)=86% (below bar), era-INCONSISTENT (2025
  +$28K / 2026 −$5K), top-20 monsters scattered across ALL ranks (1/6/6/7)
  — weighting damps 13/20 lottery tickets. Slots stay 4, dedup stays.
- W3 CLOSED: BF intraday threshold is a DEAD KNOB — Stage-2 byte-identical
  at 10/12/13/15% (TTF+conviction stack rejects the whole 10-15% band).
  BF supply cannot be expanded there. Price bands folded into W8.
- W2 ★★ SHIP CANDIDATE — PDR VETO (prev-day-range): veto ORB picks whose
  PREV day range_pct <= 8.0 ("trade day-2 of fireworks, not day-1"),
  NO-REFILL form (rank normally, skip vetoed picks, slot stays empty).
  Evidence: monotone thr 6-10; at 8.0: TOT $210K vs $155K (+35%), 25H1
  +$53K/25H2 +$51K/2026 +$106K all >> base eras, MDD −$29.3K→−$20.1K,
  WR 35.8→40.2%, trades/day 3.3→1.6, ALL top-10 giants kept (top-20: 3
  mid-size hit, aggregate still wins incl. their loss). 18/19 months
  positive delta at post-hoc thr. Search honesty: 1 of 418 rules but
  monotone dose-response + era-consistent + economic mechanism.
  ⚠ REFILL FORM IS TOXIC (2025H2 →$0, MDD −$50K — ETF-exclusion failure
  mode). Live impl = post-ranking skip WITHOUT backfill, ONLY.
  Search artifacts: /tmp/orb_veto_search.csv.

## STATUS 2026-07-04 EOD: PROGRAM COMPLETE
All W1-W8 done. PDR veto SHIPPED (commit d52e80b, +35% BT / MDD −31%,
dollar-exact in integrated pipeline). Code-review fixes SHIPPED (commit
aeefedf — grace gate was blind on WS-drain path, DST regression, orphan
reconciler cancel-first, partial-fill escapes, taxonomy). Full suite
2,622/0. Report: research/weekend_program_jul2026.md. Pending user
sign-off: ramp policy (docs/ramp_policy_proposal_jul2026.md), gap_and_go
deletion, $30-60 band build. Monday: auto-start 12:30 UTC + observer +
touchgo crons validate; watch "PDR VETO" journal lines.

## Task queue (IDs in task list) — ALL COMPLETE
- W1 #130: finish with bootstrap of rank-weighting. IN PROGRESS.
- W2 #131: loser veto-rule stump search, walk-forward, monsters-kept.
- W3 #132: BF threshold 15→12/13 under floor 1.8 (supply expansion;
  Stage-2 via /tmp config copies + Config path override or --config).
- W4 #133: ORB stop autopsy (descriptive; time-to-death needs bars —
  join resim exit_reason + bar paths only if screaming signal).
- W5 #134: /code-review high on the week's diff (money paths).
- W6 #135: gap_and_go WIP assessment (root: gap_and_go_detector.py,
  gap_and_go_backtest.py, gap_and_go_viability.py — untracked) + draft
  ramp-policy doc change (advance on absolute-loss + operational gates,
  NOT %-cushion) for Monday sign-off.
- W7 #136: weekend report (research/weekend_program_jul2026.md), ship,
  full suite, push, Telegram digest.
- W8 #137: winner expansion — price bands $30-60 & $1-3 daily-bar
  feasibility read; gap 4% vs 5% on existing resim; optionally 2026-only
  expansion resim in background.

## Owner-level creative candidates (evaluate if time permits)
- Cross-strategy capital allocator: BF capital idles on no-setup days
  (most 2026 days); flex ORB budget up on BF-dry days. Approximate test:
  scale ORB sized_pnl by budget ratio on days BF Stage-2 has 0 trades.
- Add-to-winners revisit (April parked at ~$10K/yr; touchgo fix changes
  the failed-breakout cost side).
- Short-side gap-down ORB: needs new resim w/ short mechanics + borrow
  reality — note only, don't build this weekend.

## Live validation Monday (automatic)
- Observer cron 9:26 ET: BT-parity field vs live scored (expect 0 drops).
- Touchgo debug cron 21:00 UTC (window through 7/10): expect REKEY=0.
- ORB selection audit JSONL on every placement burst.
- First-rank GRACE / sweep-retry journal lines if stragglers occur.
