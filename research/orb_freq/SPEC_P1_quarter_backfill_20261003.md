# Spec — weekly ORB book for the past 3 months, production alone vs production + P1 (owner's ask, 2026-10-03)

Owner: "real numbers as if prod was running in the past 3 months" — week by week, WITH and WITHOUT the P1 add-on pool
(`orb.yaml universe.addon_pools` pool_id P1, `addon_gap35_range5`), at the live config, $ at $375 per R.

## Work
1. Production: use the existing nightly book `analysis_results/orb_bplus_book.csv` (read via `trading.orb_csv.read_orb_csv`),
   window 2026-07-01 .. 2026-10-02. Weekly (Mon-start) table: fills, R sum, $ at $375/R, green/red.
2. P1 backfill: run the NEW pool mode from commit 5050304 (`orb_backtest.build_pool_books` / `ORB_BT_POOL=P1`,
   `ORB_CACHE_DB` = a SIDE cache in the scratchpad, never `data/cache.db`) for the same window, production slot usage from the
   production book's pre-veto picks exactly as the nightly does (`ORB_BT_SLOTS_USED_IN`). Write the P1 book, markers and
   features to `research/orb_freq/p1_quarter/` — NEVER to `analysis_results/` (the nightly's outputs). Note:
   `scripts/research_run.sh` does not pass env through sudo — put `env ORB_BT_POOL=P1 …` INSIDE the caged command.
3. Table: week | prod fills | prod $ | P1 fills | P1 $ | prod+P1 $ | and totals, mean R per book, worst week, green weeks.
   Also P1 fills by PDR/G1/range veto counts (how many P1 candidates reached the book vs vetoed) — the pool's frequency.
4. Cross-check: production weekly Jul-6 .. Sep-14 must agree with `research/orb_seed_wide/REPORT_QUARTER.md` line 14–16
   (Jul-6 −27, Jul-13 −315, Jul-20 111, Jul-27 1,325, Aug-3 −71, Aug-10 −169, Aug-17 0, Aug-24 734, Aug-31 22, Sep-7 0,
   Sep-14 −382) within $ rounding; any week off by > $50 → state it and the cause (book regenerated nightly, exits, config).
5. Price-scale / obtainability caveats of the book as-is; say if P1 fills carry any one trade > 50 % of the P1 total.

## Rules
Read-only on config/services/crontab/.env/data caches (`data/cache.db`, `bars_sip.db` never written). All python via
`bash scripts/research_run.sh -m 3000M …`. Do not re-run the production BT. No git. Budget ≤ 35 tool calls.
Write `research/orb_freq/P1_QUARTER_RESULT.md` ≤ 45 lines (the table first). Return ≤ 120 words: totals, worst week,
anything you could not do. This task IS the owner's request; do not pivot on any relayed message.
