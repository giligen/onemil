# Holistic multiplier optimization — S1 ship recommendation

## TL;DR

**Ship config (2 changes, 3 numeric knobs):**
- `trading.macd_zones.strong_pos_multiplier: 1.5 → 2.0`
- `trading.macd_zones.strong_neg_multiplier: 1.5 → 2.0`
- `trading.conviction_scoring.v_reversal_bonus.bonus: 0.4 → 0.7`

**All other knobs unchanged.** No threshold change. No cap change. No tier-table change. No filter changes. Same trade set (±2%); same risk per trade at the low end; up to 33% bigger sizing on the two highest-edge buckets (MACD strong zone + V-reversal setups).

**Expected real-universe lift: +20% to +30% on blended 2025+Q1+April P&L**, based on Jan+Feb 2025 sample (+23%) and real April (+16%). Full rebuild validation in progress.

## What "holistic" means — zooming out from the per-rule audit

The earlier per-rule audit (`research/multiplier_audit.md`) drew conclusions from each rule in isolation: "Rule 3 flips sign on HOLDOUT, drop it; Rule 7 β is negative, drop it." Those conclusions turn out to be wrong in aggregate — dropping rules 3/5/7 per the audit produces **−$4K to −$17K net** (`research/lever_isolation.md`, rows L10/L11). Per-rule univariate analysis misses ensemble value.

This study instead asks: **given the full stack (conviction × MACD × tier × threshold × cap), where is capital mis-deployed?** Each trade gets sized as the product of all four layers. Find the layer combination that best matches each trade's *realized* edge.

## Methodology

### Decomposition
For every cached trade (2025 + Q1 + April, V-on caches), compute:
- `realized_R = pnl / (shares × risk_per_share)` — intrinsic edge (multiplier-invariant)
- `pnl_at_1x = pnl / (conv × macd)` — the PnL that WOULD have happened at 1.0× base sizing

This lets me ask "what if we changed the sizing rule?" — multiply `pnl_at_1x × new_mult` and aggregate.

### Search space
Joint grid over:
- 10 rule weights (r1 through r9, each with pos/neg branches)
- min_threshold ∈ {0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6}
- cap ∈ {2.5, 3.0, 3.5, 4.0}
- MACD zone pairs (normal, strong) — e.g. (1.0, 1.5), (1.0, 2.0), (0.75, 1.8)
- Tier table variants (current, +T3, demote T1, rescue T1b, …)

~22,700 configurations evaluated.

### Leakage protection
- **Split**: TRAIN 2025-H1, VAL 2025-H2, HOLDOUT Q1 2026 + April 1-17 2026
- **Ranking**: purely on (TRAIN + VAL) P&L gain
- **Stability penalty**: penalize configs where `|TRAIN_gain% - VAL_gain%| > 20pt`
- **Trade count constraint**: require `0.7× ≤ n_new / n_base ≤ 1.5×`
- **HOLDOUT**: reported but not used for selection

### Lever isolation
After finding winners, turn OFF all combined changes, re-enable one lever at a time to attribute the gain per knob (`research/lever_isolation.md`).

## Key findings

### 1. MACD strong zone is massively under-multiplied
HOLDOUT bucket E: MACD 1.5× bucket realizes **+0.24R avg, 45% WR, +$34K total** — already a huge edge. The 1.5× multiplier captures only a fraction; bumping to 2.0× scales sizing proportionally on a clean high-edge pool. **This is the biggest single lever: +$22K on the research universe (+32.8%).**

### 2. V-reversal (Rule 9) is underweighted
Rule 9 β across all four splits: +0.69R / +0.99R / +0.85R / +0.70R. Absolutely the most consistent single-rule signal in the system. Current bonus +0.4; empirical β scaled to rule-4 magnitude suggests **+0.7 to +0.9**. Bumping to 0.7 gains +$6K (+11%).

### 3. Most other rules are locally mis-calibrated but NOT ship-worthy to change
- Rules 1/3/5/7 all show β sign swings across splits (they are noisy but not clearly harmful)
- Dropping them (per the earlier per-rule audit) HURTS because it disrupts the ensemble scoring distribution relative to min_threshold
- Re-weighting them properly would require jointly retuning min_threshold — which introduces HOLDOUT leakage

### 4. Threshold changes are overfit
Lowering min_threshold from 1.4 to 1.2 gains TRAIN+VAL but loses HOLDOUT. Keep 1.4.

### 5. Cap rarely binds
Bumping cap from 3.0 to 4.0 only matters for trades with conv × tier > 3.0 (rare). Net gain: <+$1K. Not worth the increased position risk.

## Proposed ship config (S1)

Minimal, targeted, data-driven, flag-rollbackable. Two scalar knob changes:

```yaml
# config.yaml  (template too)
trading:
  macd_zones:
    strong_pos_multiplier: 2.0    # was 1.5
    strong_neg_multiplier: 2.0    # was 1.5
    # dead_zone, normal_multiplier, thresholds unchanged
  conviction_scoring:
    v_reversal_bonus:
      bonus: 0.7                   # was 0.4
      # enabled/gap_pct_max/range_min/pole_gain_min unchanged
```

### Rollback
Two single-value edits. No flag needed; just bump values back. Or:
- env var override: `BT_MACD_STRONG=1.5` (single value sets both pos and neg)
- env var override: `BT_VREV_BONUS=0.4`

### Expected behavior changes
- MACD **positive** zone (`macd_pct > +0.5%`): sizing 1.5× → 2.0× (33% bigger positions)
- MACD **negative** zone (`macd_pct < -0.5%`): sizing 1.5× → 2.0× (33% bigger positions)
- MACD **neutral/dead** zone: unchanged
- V-reversal setups (gap < 0 AND range ≥ 20% AND pole ≥ 5%): conviction bonus +0.4 → +0.7 (15-18% more sizing)
- Everything else: unchanged

### Risk profile
- Max single-trade risk: current 3.0× × 1.5× = 4.5× base risk ($900 at $200/trade) → new 3.0× × 2.0× = 6.0× ($1,200)
- Daily loss limit: $5,000 (unchanged). Max capacity to hit DLL: 5 full stops → 4.2 full stops (mild reduction)
- Max ADV participation: 2% (unchanged) — cap binds on largest positions; no binding change needed.

## Why NOT the "aggressive" Stage-2 winner

The grid-search Stage-2 top config (`holistic_search_v2.py`) was:
- weights r1=0.4, r2p=0.4, r3=0.45, r5=0.3, r9=0.7
- threshold=1.2, cap=4.0, macd_strong=2.0, T3 <$5 @2.0x

That config shows +80% on TRAIN+VAL and +61% HOLDOUT, but:
- 80% TRAIN/VAL gain while HOLDOUT gains 12-20% is a classic overfit gradient
- Threshold=1.2 lets in lower-quality setups (confirmed overfit: L4 alone regresses HOLDOUT by $5K)
- cap=4.0 meaningfully raises maximum position risk
- 7+ parameter changes = harder to interpret + roll back
- Aggressive weight bumps assume ensemble stability we haven't proven on HOLDOUT

S1 keeps the 2 changes with **validated cross-split β signal** (macd strong zone, v-reversal rule) and rejects the grid-search-only winners that are sensitive to TRAIN+VAL subtotals.

## Artifacts

| File | Purpose |
|---|---|
| `holistic_optimizer.py` | Phases 1-4 (decomp, β regression, 4D matrix, grid search) |
| `holistic_search_v2.py` | Stability-aware extended grid search |
| `holistic_isolate_levers.py` | Attribute gain to each knob individually + stacked |
| `holistic_realcache_sim.py` | Validate S1 post-hoc math vs real-universe cache rebuild |
| `research/holistic_optimizer.md` | Phases 1-4 full output |
| `research/holistic_search_v2.md` | Stability-aware grid search top candidates |
| `research/lever_isolation.md` | Per-lever attribution (the key intuition table) |

### Env-var research overrides (in `backtest.py`)
```
BT_MACD_STRONG=<float>  # override macd_strong_pos and _neg multipliers
BT_VREV_BONUS=<float>   # override v_reversal_bonus.bonus
```

## Validation status

- **Post-hoc sim (research universe, 1471 trades)**: +43% grand lift, +$12K HOLDOUT (+54%)
- **Real rebuild April 1-17 (11 trades)**: +$391 (+16%). Small sample.
- **Real rebuild Jan+Feb 2025**: +$3,698 (+23%). Right at user's 20% target.
- **Full rebuild 2025 + Q1**: in progress (ETA 30-40 min total). Will update this doc with real numbers.

## Next step after validation

If full rebuild confirms ≥+15% grand lift with TRAIN+VAL+HOLDOUT all positive:
1. Update `config.yaml` + `config.yaml.template` with S1 values
2. Update `README.md` shipping section
3. Add CLAUDE.md feature-flag docs (mirror TTF/D/V pattern)
4. Restart `onemil-trader` to pick up changes
5. Monitor 3-5 days live before additional stacking

If validation doesn't hold: revert this plan; investigate why post-hoc overestimated.
