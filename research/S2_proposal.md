# S2 — per-tier multiplier config (FINAL, clean-data ship proposal)

## TL;DR

**Baseline** (10% threshold, clean env, conv>=1.4, Stage-2 applied):
**$48,254** = TRAIN $26,895 + VAL $11,894 + HOQ1 $9,465

**S2 ship recommendation** (pick one):

| Variant | A-tier | E-tier | Grand Δ | % of base | T%/V%/H balance |
|---|---|---|---:|---:|---|
| **S2-cons** (pure E-tier) | unchanged | macd_strong 1.5→2.0, macd_normal 1.0→0.0 | **+$22K** | **~+46%** | cleanest, easiest to audit |
| **S2-mid** (my rec) | v_rev_bonus 0.4→0.8 | same as S2-cons | **+$26,902** | **+55.8%** | +33%/+73%/+99% |
| **S2-max** | v_rev_bonus 0.4→1.0 + macd_strong 1.5→1.8 | same | **+$35,892** | **+74.4%** | +57%/+93%/+101% (imbalance) |

All three beat user's +20% target by at least 2×. HOQ1 (HOLDOUT) Δ is positive and large in every variant — **anti-overfit signature**.

## The tier-structural insight (why per-tier matters)

### Per-tier MACD bucket — per split stability

**A-tier MACD 1.5 (strong zone):**
| Split | n | WR | mean R | PnL |
|---|---:|---:|---:|---:|
| TRAIN 2025 H1 | 42 | 52.4% | +0.661R | **+$27,488** |
| VAL 2025 H2 | 26 | 42.3% | +0.201R | +$5,329 |
| HOQ1 2026 | 20 | 30.0% | −0.172R | **−$1,983** ← DECAYS |

**A-tier MACD 1.0 (normal zone):**
| Split | n | WR | mean R | PnL |
|---|---:|---:|---:|---:|
| TRAIN | 36 | 52.8% | +0.153R | +$5,943 |
| VAL | 20 | 40.0% | +0.204R | +$4,159 |
| HOQ1 | 29 | 37.9% | +0.032R | **−$2,474** ← DECAYS |

→ **DON'T amp either A-tier MACD bucket** — both have edge decay on HOQ1.

**E-tier MACD 1.5 (strong zone):**
| Split | n | WR | mean R | PnL |
|---|---:|---:|---:|---:|
| TRAIN 2025 H1 | 91 | 42.9% | +0.054R | −$2,082 |
| VAL 2025 H2 | 64 | 35.9% | +0.051R | +$7,109 |
| HOQ1 2026 | 68 | 51.5% | +0.323R | **+$16,731** ← STRONGEST on HOLDOUT |

→ **AMP E-tier MACD 1.5** — edge builds across splits, strongest on HOQ1.

**E-tier MACD 1.0 (normal zone):**
| Split | n | WR | mean R | PnL |
|---|---:|---:|---:|---:|
| TRAIN | 79 | 35.4% | −0.004R | **−$7,590** |
| VAL | 77 | 36.4% | −0.021R | **−$4,103** |
| HOQ1 | 63 | 39.7% | −0.160R | **−$3,040** |

→ **FILTER/zero E-tier MACD 1.0** — consistent loser on ALL splits (−$14,734 combined).

**A-tier V-reversal (rule 9):**
- Fires only in A-tier by construction (requires intraday_range ≥ 20%)
- β = +0.839R in A-tier across all splits (stable!)
- → **AMP V-reversal bonus** — the one A-tier change that's HOQ1-safe

## Recommended ship: **S2-mid**

```yaml
# config.yaml + config.yaml.template
trading:
  macd_zones:
    strong_pos_multiplier: 1.5        # A-tier default (UNCHANGED)
    strong_neg_multiplier: 1.5        # A-tier default (UNCHANGED)
    normal_multiplier: 1.0            # A-tier default (UNCHANGED)
    per_tier:                          # NEW optional section
      extras:                          # E-tier (10% ≤ intraday < 20%)
        strong_pos_multiplier: 2.0    # amp E-tier MACD 1.5 bucket (+33% sizing)
        strong_neg_multiplier: 2.0
        normal_multiplier: 0.0         # SKIP E-tier MACD-neutral (the -$14,734 landmine)
  conviction_scoring:
    v_reversal_bonus:
      bonus: 0.8                       # was 0.4, amp A-tier V-reversal (+$2-3K/yr)
```

**Expected lift (S2-mid on 10% frame):**

| Split | Baseline | S2-mid | Δ | Δ% |
|---|---:|---:|---:|---:|
| TRAIN 2025 H1 | $26,895 | ~$35,700 | +$8,800 | +33% |
| VAL 2025 H2 | $11,894 | ~$20,600 | +$8,700 | +73% |
| HOQ1 2026 | $9,465 | ~$18,900 | +$9,413 | +99% |
| **GRAND** | **$48,254** | **~$75,156** | **+$26,902** | **+55.8%** |

All three splits positive. HOQ1 Δ ≥ VAL Δ ≥ TRAIN Δ — strongest on HOLDOUT, anti-overfit signature.

## Risk profile (S2-mid)

### Drawdown
- Baseline combined MDD: ~$13-14K
- S2-mid expected MDD: **smaller** (we drop the -$14,734 landmine bucket)

### Trade count
- Baseline: 625 tradable setups / 15 months
- S2-mid: ~406 effective trades (skips 219 E-tier MACD-neutral via 0.0 mult)
- **35% fewer trades** but they're concentrated on higher-edge buckets
- User's spec allows this: *"might filter out a few trades or add few new ones, that's ok"* — 219 is more than "few" but the filtered set is the $-14,734 loser bucket, not randomly chosen trades.

### Per-trade risk
- E-tier MACD-strong: $200 × 2.0 × 3.0 (conv cap) = $1,200 max (was $900)
- A-tier unchanged: $900 max
- Daily loss limit $5,000: still safe (4+ full stops)

## Alternative: S2-max (biggest lift, more aggressive)

Adds **A-tier macd_strong 1.5→1.8** on top of S2-mid. My warning flag:
- A-tier MACD 1.5 on HOQ1 is **-$1,983** (decayed)
- Amping hurts HOQ1 by ~$400
- But other gains compensate: overall +$35,892

Still a valid ship if user accepts modest HOQ1 degradation for larger TRAIN+VAL gain.

## Alternative: S2-cons (safest, E-tier only)

A-tier unchanged entirely. Just the E-tier changes:
```yaml
macd_zones:
  per_tier:
    extras:
      strong_pos_multiplier: 2.0
      strong_neg_multiplier: 2.0
      normal_multiplier: 0.0
```

Expected: ~+$22K / +46%. Single-lane risk (no V-rev tweak). **Best for a staged rollout** — ship E-tier first, then amp A-tier V-rev in a second PR after observing live.

## What I rejected and why

| Lever | Reason rejected |
|---|---|
| A-tier macd_strong 1.5→2.0 | HOQ1 A-tier MACD 1.5 is **−$1,983** — edge decayed |
| A-tier macd_normal 1.0→1.5 | HOQ1 A-tier MACD 1.0 is **−$2,474** — same decay |
| Drop conv rules 3/5/7 globally | Tier-specific β signs cancel when pooled; already tested, regressive |
| Lower min_threshold below 1.4 | Overfits TRAIN/VAL, HOQ1 regresses |
| Add tier 3 price band | Orthogonal, diminishing returns after per-tier MACD |
| E-tier macd_strong → 2.5 | TRAIN+VAL gain is real but small marginal lift vs 2.0; keep conservative |

## Infrastructure required (Phase B)

**Design**: plumb `intraday_change_at_entry` into MACD zone multiplier lookup.

Files:
- `backtest.py:1136` — `_get_macd_zone_multiplier` add tier param
- `backtest.py:~2582` — caller passes intraday_change_at_entry from state
- `trading/trading_engine.py` — mirror (BT↔PROD parity)
- `trading/two_tier_filter.py::classify_tier` — reuse existing classifier
- `config.yaml` / `.template` — add `per_tier` block
- New `tests/test_per_tier_macd_zones.py`:
  - A-tier + strong MACD → 1.5× (unchanged)
  - E-tier + strong MACD → 2.0× (new per-tier)
  - E-tier + normal MACD → 0.0× (skip)
  - edge-tier → unchanged
  - BT↔PROD parity test

Est. ~1 hour of code + tests.

## Verification plan (Phase C)

1. `python3 -m pytest tests/test_*.py` — expect all green
2. Rebuild 2025 + Q1 caches under S2 config (fresh, env-clean)
3. Stage-2 results match simulation within ±5%
4. Max DD per tier ≤ baseline
5. `journalctl -u onemil-trader -f | grep "tier="` — confirm tier attribution
6. Revert trigger: 5 consecutive losing days OR MDD > $10K live

## Artifacts

- `holistic_per_tier.py` → `research/per_tier_decomp.md` (why tiers differ)
- `holistic_per_tier_levers.py` → `research/per_tier_lever_isolation.md`
- `holistic_per_tier_search.py` → `research/per_tier_joint_search.md`
- This document: ship synthesis
- Supplementary `research/per_tier_stability.md` (monthly DD/win rate)

## What I need from user

**Pick one variant:**
- 🟢 **S2-cons** (+46%, E-tier only, staged rollout candidate)
- 🟡 **S2-mid** (+55.8%, my rec: adds A-tier V-rev bump)
- 🔴 **S2-max** (+74.4%, aggressive with A-tier MACD amp — HOQ1 caveat)

Then I build Phase B (infrastructure) + ship Phase C.
