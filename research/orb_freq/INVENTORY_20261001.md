# ORB frequency/profitability lever inventory (2026-10-01)

Read-only sweep of every ORB lever tested to date. Purpose: stop re-testing rejected knobs. Grep this
before proposing a new ORB frequency or profitability idea. Verdict key: SHIPPED (live today) /
OFF (shipped, later disabled) / PAPER (shipped to paper only) / DRY (research-flag, zero orders) /
FAIL (tested, rejected) / NEVER (hard do-not) / PARK (promising, deferred) / LEAD (not a claim yet).

| # | Type | Lever (cells) | Verdict + key number | Rule now | Evidence |
|---|---|---|---|---|---|
| 1 | FREQ | Add-on pools / union rung: gap4-5% $3-30 + gap3-5% $30-50 (cell 1,328) | DRY — added cohort +0.09R(n77)/+0.15R(n68), TRAIN $9,190 vs $6,561 prod; fails TRAIN weekly-MDD≤1.25x (2.05x) | exploration tier only; per-pool selection, never shared, never refit on addon-only data; orb.yaml `addon_pools.enabled:true, dry_run:true` | orb_seed_wide/PREREG_LIVE_UNION.md, REPORT_S1_CREATIVE.md §2, INDEPENDENT_1328.md |
| 2 | FREQ | Multi-window entry, W=15/30 added to W=5 | FAIL — nothing cleared G1 on TRAIN, TEST never scored | none shipped | orb_multiwindow/PREREG.md, REPORT.md |
| 3 | FREQ | Stack W=5+W=10 / W=5+W=15, added picks only (cells 1,274-1,277, F2-10/F2-15) | FAIL — TRAIN −0.329R t−2.67 (F2-10), −0.254R t−1.87 (F2-15); 2nd window buys +6-11% more picks, not +30% (ceiling = post-veto pool, not slots) | none; frequency work must move the TRIGGER or the POOL | orb_frequency/REPORT.md |
| 4 | FREQ | Widen chase cap 30→60bps on gap-through picks (F1a) | FAIL — n=0 addable in TRAIN | none | orb_frequency/REPORT.md §2 |
| 5 | FREQ | Passive re-arm limit at range_high after gap-through (F1b) | FAIL — underpowered, n=1, MDE 6x the ship bar | none | orb_frequency/REPORT.md §3 |
| 6 | FREQ | Slot count 3→8→12→16 | 3→8 = whole green-week gain; 8→12→16 ≈0 to −1.9pp green, 2-3x drawdown, account-capped | SHIPPED: 8 slots (`sizing.max_concurrent`) | fuckup_audit/D1_orb/REPORT.md; orb_gates2/PREREG.md §0 |
| 7 | FREQ | No-fill slot recycling, release at T=15/30/45min, max/day 1 or 3 | REJECT all 6 variants — releasing cancels late fills before any gain; ~1 fill recycled per 50+ released slots (vetoes already stripped the field) | none | orb_slot_recycle/REPORT.md |
| 8 | FREQ | Refill a slot after a post-ranking veto | NEVER — toxic: 2025H2→~$0, MDD −$29K→−$50K | no-refill invariant (hard rule, standing doctrine 3) | orb_machine_rules.md L2/doctrines; CLAUDE_HISTORY PDR veto |
| 9 | FREQ | Index ORB: QQQ/SPY, W∈{5,15,30}, long+short, 2016-2026 (cells 1,329-1,346) | CLOSED — 0/18 cells pass; gross ≈+0.1R tail-carried, dies at 1-3bp cost | no re-run without a new mechanism | index_orb/PREREG.md, REPORT.md |
| 10 | FREQ+PROFIT | Whole-market "stocks in play" ORB (Zarattini/Barbon/Aziz replication + Cell D exit variant + Cell E liquid sub-universe) (cells 1,280-1,282) | FAIL decisively every side/combined/variant — net R −1.0 to −1.8R; half-spread cost is the driver, not signal | none | orb_inplay/PREREG.md, REPORT.md, REPORT_D.md, REPORT_E.md |
| 11 | FREQ+PROFIT | ORB mirrored to the short side, incl. chase-tolerant fill (cells 1,271-1,273) | NO SHIP — raw edge ≈0, selected book negative on VAL even after loosening fill rate | none | orb_short/PREREG.md, REPORT_C.md |
| 12 | FREQ+PROFIT | Retest bid 1 tick under range_high instead of chasing (HOD learning transfer) (cells 1,562-1,563) | FAIL both — pooled t 1.3-1.4 < 2.5 bar; VAL ex-top-5% negative both cells; judge rebuild found paired ΔR NEGATIVE in consistent units | ORB keeps the chase entry; only the stop-limit EXIT idea transfers (#40) | orb_retest/PREREG_1562.md, RESULT_1562.md |
| 13 | FREQ+PROFIT | Entry delay / "first five seconds" (preplace_submit_delay_s 0 vs 5s) (cells 1,655-1,657) | PASS — 0-5s bucket weak-negative OOS (−0.057R t−1.78); 5s delay costs ≈0 (+0.002R) | SHIPPED to paper: `entry.preplace_submit_delay_s: 5` | orb_deadzone/PREREG_1655.md, RESULT_1655.md |
| 14 | PROFIT | Weekly selection refit window: mults vs selection-only, 8/13/20/26/39/52w/expanding | refit MULTS = whipsaw (one ANNA fill sized 3x drove the "gain"); refit SELECTION-only 26w best ($7,588 vs $5,669 frozen) | SHIPPED: `scripts/orb_weekly_refit.py` Sun 20:00 UTC, selection only, NEVER mults | orb_refit_walkforward/REPORT.md |
| 15 | PROFIT | Anchor dedup: 1 pick per underlying/day beyond family+super-group dedup | NOT SHIPPED — N=8 slots: −11.9% total, worst month worse; N=3(live): +0.9% but MDD/worst month worse | do not re-litigate from the CIFG/CIFU anecdote; shared-risk-budget form untested | orb_anchor_dedup/PREREG.md, REPORT.md |
| 16 | PROFIT | Failed/held-break signal declared at 10:30 ET (short failed break, VWAP-target short, avoid held-break long) (cells 1,564-1,566) | FAIL all 3 — VAL net negative every cell (−0.33/−0.65/−0.05R); real DO-NOT-BUY veto, not a trade | none shipped; needs independent rebuild + full NBBO before any claim | orb_failure/PREREG_1564.md, RESULT_1564.md |
| 17 | PROFIT | Entry micro-filters: RVOL veto/rank, red/green candle pre/post gate, mid-trade kill, re-arm (C1a,b; C2x4; C3; C4) | REJECT all 8; C1c (RVOL_MAX>2.0, post-hoc) EXPLORATORY-PARK | none shipped; C1c needs a fresh-era PREREG before it can be proposed | orb_signal_study/REPORT.md |
| 18 | PROFIT | Gate removal, 10 cells scored on green weeks vs count-matched null | FAIL — 0/20 measured cells (both splits) beat own null; null predicts green-week share r=0.966 | nothing ships from this stage | orb_gates2/PREREG.md, REPORT.md |
| 19 | PROFIT | 50+ exit variants: trail-after-arm, late-arm+trail, partial-then-runner, MFE-conditional, quintile-aware, classifiers, bull-flag-gated add | V0 `static_lock_1R` = Pareto frontier; bull-flag add PARKED (~$10K/yr, n=5 HOQ1+ anecdotal) | add-to-winners NOT shipped (Q1 filter + touchgo shipped separately, see #29/#32) | docs/orb_research_apr_2026.md |
| 20 | PROFIT | Real-ledger give-back anatomy, R-floored (cell 1,679) | report-only — 21% of fills reach +1R then close ≤0, give back 1.88R avg; reconstruction 80% exact-minute vs real ledger | no rule yet; feeds cell 1,680 | orb_exit/PREREG_1679.md, RESULT_1679.md |
| 21 | PROFIT | 50% partial at +1R for consistency (cell 1,680) | FAIL — weekly-P10 and gap-median clauses fail 2025, null-percentile clause fails both years (ALL must pass BOTH years) | not shipped | orb_exit/PREREG_1680.md, RESULT_1680.md |
| 22 | PROFIT | S1-stratum exit knobs: touchgo off (E1) / lock-arm 1.0R (E2) / scale 50%@+2R (E3) (cells 1,319-1,321) | FAIL all — S1 stratum flat under every exit incl. baseline | none | orb_seed_wide/PREREG_S1_EXIT.md, REPORT_S1_EXIT.md |
| 23 | PROFIT | Breakout thermometer: hot/cold sizing by trailing breakout success rate (cell 1,420) | FAIL — causal hot-cold −0.0077R t−0.14; only the look-ahead oracle version separates | none; thermometer sizing FAILED | thermo/REPORT.md (PREREG cells 1,420-1,422) |
| 24 | PROFIT | Spread gate threshold 100/150/300bps | 150bps was a monster-killer (BKKT +$20.7K, XNDU +$11.9K skipped); 100-150bps is the richest per-trade bucket | SHIPPED 150→300bps (7/4); NEVER tighten below 150bps | orb_spread_gate_verdict.md |
| 25 | PROFIT | V1 veto study: range_size / adr20 / spy_3d / bar_range / retvol20 thresholds | V1a,V1b,V1e KEEP; V1c,V1d,V1all REJECT; V1ab (a+b combo) ships | SHIPPED: range-size veto (`filter.range_size_veto`) | orb_veto_study/REPORT.md |
| 26 | PROFIT | PDR veto threshold sweep 6-10% (prev-day range) | +35% TOT, MDD −$29K→−$20K, monotone, all 3 eras positive | SHIPPED: `filter.prev_day_range_veto.min_prev_day_range_pct: 8.0`, no-refill | orb_machine_rules.md L2; CLAUDE_HISTORY |
| 27 | PROFIT | Catalyst-required veto: own-ticker news OR complex-confirmation | book $293K→$253-257K (−$36K, owner-approved); newsless-alone cohort negative all eras | SHIPPED 7/18 → turned OFF 9/19 (owner GO) | CLAUDE_HISTORY; orb.yaml `filter.catalyst_veto.enabled: false` |
| 28 | PROFIT | G1 short-history veto (`return_volatility_20d==0`) | shipped 9/8 on 4-fill evidence; FAILS its own 2025(n30,−0.26R)→2026(n29,+0.12R) check | DROPPED 9/13, fail-open restored | orb.yaml `g1_veto.short_history_veto: false`; books_deep_dive_20260913 |
| 29 | PROFIT | Q1 (bottom composite quintile) filter | +$8,556 OOS, no DD increase, never refills a slot | SHIPPED: `filter.skip_q1: true` | orb_research_apr_2026.md |
| 30 | PROFIT | Q5 adaptive-mult cap (1.5x ceiling) | guards the TRAIN-only Q4/Q5 tail artifact | NEVER remove | orb_machine_rules.md L5; CLAUDE.md do-NOTs |
| 31 | PROFIT | adaptive_mults refit (sizing multipliers) | refitting mults = whipsaw; the "expanding" gain was one ANNA fill sized 3x | NEVER refit mults (refit selection only, #14) | orb_refit_walkforward/REPORT.md |
| 32 | PROFIT | Touchgo Rule M/D early failed-breakout exit + breakout-bar re-key fix | +$27K OOS, WR 47.8%→52.1%; re-key fix +$251.8/33 trades (BT/live parity bug, 23% of fills had flipped) | SHIPPED: `filter.touchgo.enabled: true`, `breakout_bar_source: market` | CLAUDE_HISTORY lines 345-394 |
| 33 | PROFIT | News-gated PM$ sizing mult (2x if PM$>$5.82M AND news AND common-stock) | +$1,580/$1,569/$935 per trade per era; but lift is monster-concentrated (top-5=all of it) | SHIPPED 7/10 → DISABLED at B+ restart 8/15 (mult=1.0 always) | CLAUDE_HISTORY; orb.yaml `sizing.pm_dollar_vol_mult.enabled: false` |
| 34 | PROFIT | LLM/keyword catalyst-quality filter for longs | REFUTED — recap-only articles perform equal to real catalysts (AMCI +$23K / BNAI +$13.6K on recaps) | NEVER add | orb_machine_rules.md L6; CLAUDE.md do-NOTs |
| 35 | PROFIT | Map wrapper news to underlyings | REFUTED — negative all 3 eras (−$324/−$125/−$27) | NEVER | orb_machine_rules.md L6; CLAUDE.md do-NOTs |
| 36 | FREQ+PROFIT | 2x-leveraged-wrapper universe treatment: IN vs OUT vs as-is | as-is(accident) $6,394 > IN $6,085 > OUT $4,998 (negative era); wrappers = 42% of picks | SHIPPED: IN (pre-committed rule honored though 2nd-best) — honest ref $6,085/21mo | CLAUDE_HISTORY "2x-wrapper universe rule" |
| 37 | PROFIT (methodology) | Entered-inclusive book construction (no-fill rows win a slot at $0) | fixed a selection look-ahead; $6,394(lookahead)→$6,085 honest after wrapper correction, 55.5% fill rate | SHIPPED: `FEATURES_CODE_VERSION=2026-09-05.entered_inclusive` | CLAUDE_HISTORY "Entered-inclusive book" |
| 38 | EXEC/FREQ | 09:35 entry-eval on its own thread (entry-drain) | root cause of the 9/18 48.9s-late order: 15.3s `daily_bars` window-function scan + scanner-cycle ownership, NOT news prefetch | SHIPPED: `ORBEngine.start_entry_drain_thread`, 0.25s poll | CLAUDE_HISTORY line 400 |
| 39 | EXEC/PROFIT | Order-latency sensitivity, 0-60s replay on XNAS tick data (cell 1,426) | +0.028R/fill OOS at zero latency (~$11/trade); only 21% of fills match the BT's 30bps entry model; ~72bps adverse ask-to-fill drift live | book stays CLOSED regardless of the latency fix; no ramp/rehearsal for it | orb_latency_bt/PREREG.md, REPORT.md |
| 40 | EXEC/PROFIT | Stop-limit/touchgo TP-leg repricing to a resting limit at the target | 39 live target exits (tag_bb) filled 68.8bps worse than the target mean, no fill ever beat it | SHIPPED to PAPER only (9/29): `exit.target_resting_limit: true` | exec_quality/REPORT_20260928.md §4; orb.yaml |
| 41 | PROFIT | Out-of-regime re-verification under the LIVE exit rule: 2023-01..2024-06, 2024H2 (cells 1,415,1,416,1,418,1,419) | FLAT — +0.015 to +0.031R/fill (t 0.3-0.4), pooled +0.020±0.043 (t 0.48); regime changes FREQUENCY (1.4-2.3 vs 4.3-7.3 fills/wk), not per-fill edge | per-fill edge ≈+0.1R in and out of regime; no sizing lever follows from regime | orb_2023/REPORT.md+REPORT_liveexit.md; orb_2024/REPORT.md+REPORT_liveexit.md |
| 42 | PROFIT (methodology) | Adversarial verification of the "+0.08R OOS edge" claim | REFUTED as stated — true point estimate +0.03 to +0.09R, tail-carried (top ~4% of fills carry the whole book), in-sample for selection | fits the live-exploration tier only; not proven, not a scale-up | orb_verify/REPORT.md |
| 43 | FREQ+PROFIT | S1 wide-seed stratum (gap 3-5%, $3-30): standalone + loser filters + creative features (cells 1,300-1,327) | standalone flat/closed; EVERY loser filter (F-A/F-B/F-AB) and creative feature (gap-rank/PM$/RVOL/HMM-calm/SPY-gap) fails or sign-flips 2025→2026 | closed standalone; feeds the union rung (#1) as a LEAD only | orb_seed_wide/REPORT_S1_FILTERS.md, REPORT_S1_CREATIVE.md |
| 44 | FREQ+PROFIT | Wide-seed forward quarter test: production vs pool union, Jun-Sep 2026 | union = more green weeks (8/16 vs 7/14) but lower mean R (+0.036 vs +0.098); addon_gap4 NEGATIVE (−$1,732); addon_p30(S3) positive, small n (+$1,654) | informs #1 only, not a ship decision | orb_seed_wide/REPORT_QUARTER.md |
| 45 | OPS | Ramp advance gate (stage sizing) | advance requires ≥40 live fills AND realized stage P&L ≥ 0 | standing rule: never advance underwater, never below the fill floor | orb_ramp_check.py (`min_fills_advance`); project_orb_ramp_above_water_rule |
| 46 | FREQ | Corpse gate: universe bar freshness (stale-snapshot gate) | 2-month LIVE DEFECT found 9/21: killed ≈2,950 symbols/day from the candidate universe | FIXED 9/21: corpse gate = bar age < 4 days | project_orb_corpse_gate_defect_sep2026; CLAUDE_HISTORY |

## The five hardest NEVER rules (frequency/profitability knobs, do not re-test)
1. Never refit `adaptive_mults` (sizing quintile multipliers) — walk-forward proved it whipsaws (#14, #31).
2. Never remove or lower the Q5 1.5x adaptive-mult cap — anti-overfit guard (#30).
3. Never refill a slot after a post-ranking veto — tested TOXIC twice, MDD blows out (#8).
4. Never tighten the spread gate below 150bps — it was a monster-killer at 150, shipped to 300 (#24).
5. Never disable `filter.skip_q1` without re-reading `docs/orb_research_apr_2026.md` first (#29).
   (Also hard: never add an LLM catalyst-quality filter, never map wrapper news to underlyings — both
   REFUTED, #34/#35; never re-litigate anchor dedup from the CIFG/CIFU anecdote without a new form, #15.)

## Current live orb.yaml essentials (read-only, 2026-10-01)
- `strategy.enabled: true`, `strategy.dry_run: false` — running on ORB's OWN paper account
  (`ALPACA_ORB_PAPER=true`) as a mechanism test (latency ≤1.5s, picks vs BT); live $ only after owner
  GO post a clean paper session. Per git log (10/1 boot parity work), parity does NOT yet hold — no
  live GO as of this writing.
- Universe: `min_price 3.0`, `max_price 30.0`, `min_gap_pct 5.0`, `min_prev_volume 500000`.
- Add-on pools: `enabled: true`, `dry_run: true` — addon_gap4 (gap 4-5%, $3-30), addon_p30 (gap 3-5%,
  $30-50). Zero orders; `[ORB+ DRY]` telegrams only.
- Entry: `range_minutes: 5` (09:30-09:35 window); stop-limit buy at range_high, `max_spread_bps: 300`;
  `preplace_submit_delay_s: 5.0`.
- Slots: `max_concurrent: 8`, `account_budget_usd: 26666.67`, `risk_per_trade_usd: 375`.
- Filters/vetoes: `skip_q1: true`; `prev_day_range_veto.enabled: true`; `range_size_veto.enabled: true`;
  `g1_veto.short_history_veto: false` (fail-open); `catalyst_veto.enabled: false`; `touchgo.enabled: true`
  (market-bar re-key).
- Exit: `stop_mode: range_low`, `lock_arm_at_r: 1.75`, `lock_stop_r: 0.5`, `target_resting_limit: true`
  (paper only).
- Sizing: `pm_dollar_vol_mult.enabled: false` (disabled at the 8/15 B+ restart).
- Ranking: Q4 first, then Q5/Q3/Q2/Q1. Dedup: by_family + by_super_group (NOT by_anchor, #15).

## Current frequency
≈6.1 fills/week at the live config, 2025-07-01..2026-09-23 (n=389 fills / 201 days) —
`research/orb_verify/REPORT.md`. Out of that regime (2023-24, point-in-time universe, same live exit
rule) frequency drops to 1.4-2.3 fills/week (`orb_2023/REPORT_liveexit.md`, `orb_2024/REPORT_liveexit.md`)
— the regime changes FREQUENCY, the per-fill edge (≈+0.1R) stays roughly constant (#41).
