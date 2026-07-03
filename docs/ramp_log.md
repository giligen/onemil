# ORB Live Ramp Log

Audit trail for cushion-gated ORB ramp transitions on this node.
Format: `YYYY-MM-DD  <event>: <details>`.

Per `docs/orb_rollout_plan.md`, `scripts/orb_pre0_daily.py` and
`scripts/orb_ramp_check.py` rely on `ORB ramp:` commit messages to compute
days-in-stage. `orb.yaml` itself is gitignored on this node, so the YAML
diff is recorded inline here.

---

2026-05-19  Pre-Stage-0 LIVE launched: $15K budget / $500 risk / −$750 daily.
            Telegram prefix `[ORB-LIVE-PRE0]`. First trades fired 9:35 ET
            after StopMonitor cascade incident the same day (B1+B2 fixes
            shipped within the day; see commits f0fd573, 2dfbfb7).

2026-06-04  Pre-Stage-0 → Stage 0 promotion.
            13 days at Pre-Stage-0. Cumulative cushion **+$2,057** (gate ≥+$1K).
            Entry slippage **16.5 bps mean** (gate ≤45). Exit slippage
            **−62.4 bps mean** (favorable; gate ≤25 trivially passes).
            Round-trip **−45.9 bps** (gate ≤60). No demotion triggers fired.
            `orb_pre0_daily.py` reports ✓✓ ELIGIBLE.

            Caveats accepted:
              * The 5-19 StopMonitor cascade was the only P0 in the window;
                10 consecutive sessions clean post-fixes.
              * Cumulative cushion concentrated in top-3 winners (IREG,
                TDOC, BOIL) — net +$181 without them. Broadening (Q4 mults
                + force-close holds dominating; lock_stop fired 2x).

            YAML diff applied to orb.yaml:
              account_budget_usd:   15000 → 30000
              risk_per_trade_usd:     500 → 1000
              daily_loss_limit_usd: -750  → -1500
              telegram.prefix: "[ORB-LIVE-PRE0]" → "[ORB-LIVE]"
              max_concurrent: 4 (unchanged); per-pos cap auto-rescales
                $3,750 → $7,500.

            Backup at `orb.yaml.bak.pre_stage_0` (gitignored). Trader was
            inactive at time of change (06:31 ET); cron-driven start at
            12:30 UTC picks up new config.

2026-06-12  HELD Stage 0 through a −20.1% demotion flag (override, no config
            change). `orb_ramp_check.py` fired DEMOTE: realized cushion
            $3,553.52 vs peak $4,446.66 (−$893.14, −20.1% ≥ 20% threshold).

            Override justification (advance/hold on cushion, not vibes):
              * BT parity confirms regime, not defect — this week ORB BT took
                the same 7/7 names/direction and lost MORE: BT −$4,861 vs
                live −$893. Regime sizing dampened the down week by ~$4K.
              * Absolute DD −$893 << Stage-0 expected max DD $5,400 (~16% of
                budget). Nothing stressed.
              * −20.1% is a small-denominator artifact: 20% of a $4,446 peak,
                not a real risk event. The % trigger is hair-trigger this
                early in the ramp before the cushion base is large.
              * The flag was tripped BY today's Alpaca reconciliation (FABC
                #196 −$560→−$1,243; GLXG #210 $0→+$114). Pre-reconcile DD was
                −7.3% (under both the 8% health ceiling and 20% demote line).
                The reconciliation made the drawdown honest — the FABC
                partial-fill bug had been masking it. Books now match broker.

            NOT advancing to Stage 1 either. Stage 1 gate when ALL hold:
              * cushion ≥ +$5,000 (need +$1,446; new peak auto-clears the
                demote flag + 8% health check)
              * 5 clean trading days since the 6/12 reconcile/vol-guard/race
                fixes (operational-incident gate, ≈ through 6/19)
              * days-in-stage already satisfied (20)
            Target: flip the three Stage-1 numbers next week once
            `orb_ramp_check.py` prints ELIGIBLE with no demote flag.

2026-06-19  HELD Stage 0 through a REAL −69.8% demote trigger (override, no
            config change). Cushion $3,553 → $1,344; peak $4,446; 3 red days
            6/15–6/17; week −$2,141 (ORB −$1,694, BF −$447, MACD $0).

            NOT an artifact this time — genuine regime drawdown. Override is
            an explicit acceptance-of-variance decision, weaker-justified than
            the 6/12 hold. Basis:
              * Confirmed regime, not defect: ORB BT (static_lock) this week
                −$13,302 @ 19% WR vs live −$1,694 @ 20% WR — same direction,
                same WR, every day aligned. Ramp size turned a 5-figure BT
                loss into a 4-figure live scratch (de-risking worked).
              * Whole gap-up universe failed: 18% breakout WR (vs 34% prior),
                Mon 6/15 13% across 113 candidates, Tue 6/16 0%. No filter
                saves you at 82–87% fade rates — regime, not picks.
              * Still net +$1,344 since launch; ~$13K of room to the −$12K
                hard stop (non-negotiable floor, unchanged).
              * Universe WR recovering intraweek: 13→0→43→50%. Thu 6/18 live
                +$202 while BT −$3,244 (filter dodged BT losers). Possible
                regime turn — demoting into a turn locks in small size for the
                recovery.

            Watch items (not blocking, revisit post-regime):
              * MACD wave $0 contribution all weeks — verify it's actually
                running / has setups.
              * No cooldown on freshly-stopped symbols (ARQQ stopped 6/16 +
                6/17; RGTZ up-sized into a fade −$556). Cheap filter to weigh
                AFTER the regime, not during.

            Re-advance to S1 on standard gates ($5K cushion + clean days +
            no demote flag). Hard stop remains the only override-proof line.

## 2026-07-06 — RAMP POLICY REVISION (owner-approved 2026-07-04)

Cushion (profit-target) advancement gates RETIRED. New gates:
operational-green x10 sessions + loss floor (-1x weekly loss budget) +
slippage parity + min days. Demotion: operational failure OR -2x weekly
loss budget; BT-consistent drawdown explicitly NOT a trigger (codifies
the 6/12 and 6/19 overrides). Full rationale:
docs/ramp_policy_proposal_jul2026.md. Trigger for revisiting the policy:
PDR veto (2026-07-04) halves trade count — cushion accrual slows while
per-trade quality doubles; a profit gate would read improvement as
"ramp slower".

Status at revision: Stage 0, day 35, stage P&L -$2,587 (above -$7.5K
floor). Path to Stage 1 = 10 clean sessions from Monday 7/6 with all
July fixes live.
