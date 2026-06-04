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
