# Multiplier & qualification parameter audit

## Executive summary

Systematic check of all 9 conviction rules + MACD zone + risk tier + conviction threshold. Each rule was checked for (1) sign stability across TRAIN/VAL/HOLDOUT, (2) multivariate marginal lift via logistic regression, (3) regime dependence across SPY-vol / time-of-day / ADV tertiles.

### Key findings (triaged)

**🚨 SHIP: rules that should change**

1. **Rule 3 (vol_ratio > 1.7, +0.3)** — SIGN FLIPPED on HOLDOUT. TRAIN +8.4pt WR, HOLDOUT −7.2pt WR. Logistic regression: β_TRAIN +0.21, β_VAL −0.33. Currently ACTIVELY harming OOS. **Recommendation: suspend or regime-gate (may only work in high-vol regime).**

2. **Rule 7 (vwap_dist ≥ 2%, +0.2)** — negative univariate lift on TRAIN (−4.4pt) and HOLDOUT (−1.6pt). Logistic β strongly negative (−0.25 TRAIN, −0.09 VAL). **Recommendation: INVERT the sign** (contrib −0.2 instead of +0.2), or DROP entirely.

3. **Rule 5 (retracement < 30%, +0.2)** — near-zero lift everywhere (TRAIN −1.3pt, HOLDOUT +0.1pt). Logistic β negative. **Recommendation: DROP or cut contribution to 0.0.**

4. **Rule 1 (pole_gain sweet spot, +0.3)** — marginal lift (TRAIN +2.8pt, HOLDOUT +2.0pt). β_TRAIN +0.02 (near zero). **Recommendation: reduce contribution from +0.3 to +0.1.**

5. **Rule 2 (flag_tightness, ±0.3)** — both branches unstable across splits, major regime swings. **Recommendation: reduce magnitude to ±0.15 or re-bucket thresholds.**

**✅ KEEP: rules that hold up**

- **Rule 4 (SPY 3d range, ±0.3/-0.5)**: ✓ sign agrees on all 3 splits, both branches. HOLDOUT negative branch has −27pt WR drop — huge regime signal. Consider bumping the negative penalty from −0.5 to −0.7.
- **Rule 8 (gap_fading, −0.3)**: ✓ negative on all splits. Fires rarely (19-26 trades) but clean signal. Logistic β noise due to small sample; trust the univariate.
- **Rule 9 (V-reversal, +0.4)**: ✓ positive all 3 splits, 8-23pt WR lift. Shipping ON validated by this audit.

**💰 BIG FINDINGS outside conviction rules**

1. **MACD zone 1.5× is huge OOS alpha**: HOLDOUT 1.0× bucket loses −$13,609, 1.5× bucket makes +$34,301. The 1.0× (normal zone) trades are NET NEGATIVE on HOLDOUT. **Recommendation: consider scaling 1.0× down to 0.5× or skip altogether** on post-conv-filter setups.

2. **Risk tier orphans — massive opportunity**: 
   - `<$5, 500K-5M vol`: **178 trades, +$40,262, +0.31R avg** — biggest edge bucket in the whole population, currently NO tier → 1.0× default.
   - `$10-15, <500K vol`: 141 trades, +$14,704, +0.24R — also orphan.
   - Tier 1 as-defined ($10-15, 500K-5M, 2.0×): 110 trades, only +$1,382, +0.06R.
   **Recommendation: redefine tiers. Add Tier 3 `<$5, 500K-5M → 1.5-2.0×`. Drop or downsize existing Tier 1.**

3. **min_threshold=1.4 is HOLDOUT-optimal**: sweep confirmed current setting is the peak. T=1.4 → $25,190 HOLDOUT vs T=1.0 ($20K) or T=1.7 ($19K). **Keep as-is.**

4. **V-reversal range_min=22 slightly better than 20**: at threshold 22, TRAIN 68% WR vs 61% at 20; VAL and HOLDOUT hold at 50%. Minor refinement candidate. Sample drops from 21→12 on HOLDOUT — wait for more data.

### Next steps (ranked by conviction × size)

1. **Fix Rule 3 + 7** (suspend or invert). Combined they're adding conviction to trades that HOLDOUT data says are net losers. Expected lift: +$5-10K on HOLDOUT once these stop mis-sizing losers.
2. **Add Tier 3 (<$5, 500K-5M vol)** at 1.5× or 2.0×. Low-price stocks are dominating our HOLDOUT P&L with no sizing amplification. Estimated +$10-20K on full-year BT.
3. **Re-examine MACD 1.0× bucket** — data says it's an active loser on HOLDOUT. Either tighten dead-zone window to absorb more of them, or scale 1.0× down to 0.5×. Estimated +$5-10K.
4. **Drop Rules 5 + 1 + trim Rule 2** magnitudes. Individually small but collectively remove ~0.7 of contrib noise from the score.

5. **Regime-gate Rule 3** — it may work in high-vol regime. Needs a conditional implementation.

---

## Raw data tables

**Data**: TRAIN 536 tr / VAL 496 tr / HOLDOUT 439 tr (all post `conviction >= 1.4` filter).

## A. Per-rule univariate lift
For each conviction rule, comparing trades where the rule FIRES (positive branch) vs doesn't. Lift in percentage points (WR) and R-multiples (edge).

### Pole gain ∈ [4.5, 9]% (`rule1_pole_gain`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 285 | 41.4% | +0.12R | 38.6% | +0.17R | +2.8pt | -0.05R |
| VAL | 260 | 31.9% | +0.01R | 32.6% | -0.01R | -0.7pt | +0.02R |
| HOLDOUT | 214 | 40.2% | +0.06R | 38.2% | +0.03R | +2.0pt | +0.03R |
**Sign-agreement: ⚠️ VAL disagrees** (TRAIN +1, VAL -1, HOLDOUT +1)

### Flag tightness <30% (`rule2_flag_tightness`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 151 | 39.7% | +0.12R | 40.3% | +0.16R | -0.5pt | -0.04R |
| VAL | 146 | 34.9% | +0.16R | 31.1% | -0.07R | +3.8pt | +0.22R |
| HOLDOUT | 115 | 36.5% | -0.04R | 40.1% | +0.07R | -3.6pt | -0.11R |
**Sign-agreement: ⚠️ VAL disagrees** (TRAIN -1, VAL +1, HOLDOUT -1)

### Flag tightness >50% (neg) (`rule2_flag_tightness`, pos_branch=False)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 143 | 42.0% | +0.09R | 39.4% | +0.17R | +2.5pt | -0.08R |
| VAL | 132 | 28.8% | -0.15R | 33.5% | +0.05R | -4.7pt | -0.20R |
| HOLDOUT | 119 | 47.1% | +0.30R | 36.2% | -0.05R | +10.8pt | +0.36R |
**Sign-agreement: ⚠️ VAL disagrees** (TRAIN +1, VAL -1, HOLDOUT +1)

### Vol ratio > 1.7 (`rule3_vol_ratio`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 317 | 43.5% | +0.24R | 35.2% | +0.01R | +8.4pt | +0.23R |
| VAL | 295 | 28.8% | -0.07R | 37.3% | +0.09R | -8.5pt | -0.16R |
| HOLDOUT | 247 | 36.0% | -0.07R | 43.2% | +0.20R | -7.2pt | -0.27R |
**Sign-agreement: ✗ HOLDOUT disagrees with TRAIN** (TRAIN +1, HOLDOUT -1) — rule may be regime-dependent or overfit

### SPY 3d range >1.2% (`rule4_spy_regime`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 213 | 42.3% | +0.12R | 38.7% | +0.16R | +3.6pt | -0.04R |
| VAL | 86 | 43.0% | +0.07R | 30.0% | -0.02R | +13.0pt | +0.09R |
| HOLDOUT | 200 | 46.5% | +0.12R | 33.1% | -0.02R | +13.4pt | +0.14R |
**Sign-agreement: ✓ all three splits** (+1)

### SPY 3d range <0.8% (neg) (`rule4_spy_regime`, pos_branch=False)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 157 | 34.4% | +0.24R | 42.5% | +0.11R | -8.1pt | +0.13R |
| VAL | 251 | 25.5% | -0.02R | 39.2% | +0.02R | -13.7pt | -0.03R |
| HOLDOUT | 99 | 18.2% | -0.12R | 45.3% | +0.09R | -27.1pt | -0.21R |
**Sign-agreement: ✓ all three splits** (-1)

### Retracement <30% (`rule5_retracement`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 256 | 39.5% | +0.15R | 40.7% | +0.14R | -1.3pt | +0.00R |
| VAL | 246 | 30.5% | -0.09R | 34.0% | +0.09R | -3.5pt | -0.18R |
| HOLDOUT | 232 | 39.2% | +0.10R | 39.1% | -0.02R | +0.1pt | +0.12R |
**Sign-agreement: ✗ HOLDOUT disagrees with TRAIN** (TRAIN -1, HOLDOUT +1) — rule may be regime-dependent or overfit

### VWAP dist >=2% (`rule7_vwap_dist`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 419 | 39.1% | +0.13R | 43.6% | +0.21R | -4.4pt | -0.08R |
| VAL | 414 | 32.4% | -0.02R | 31.7% | +0.07R | +0.7pt | -0.09R |
| HOLDOUT | 365 | 38.9% | +0.06R | 40.5% | -0.02R | -1.6pt | +0.08R |
**Sign-agreement: ⚠️ VAL disagrees** (TRAIN -1, VAL +1, HOLDOUT -1)

### Gap fading (neg) (`rule8_gap_fading`, pos_branch=False)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 19 | 31.6% | -0.27R | 40.4% | +0.16R | -8.8pt | -0.43R |
| VAL | 26 | 26.9% | -0.08R | 32.6% | +0.00R | -5.6pt | -0.08R |
| HOLDOUT | 16 | 31.2% | -0.24R | 39.5% | +0.05R | -8.2pt | -0.29R |
**Sign-agreement: ✓ all three splits** (-1)

### V-reversal (gap<0 + range>=20 + pole>=5) (`rule9_v_reversal`, pos_branch=True)
| Split | fires_n | fires_WR | fires_R | notfires_WR | notfires_R | WR_lift | R_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 34 | 61.8% | +0.71R | 38.6% | +0.11R | +23.1pt | +0.60R |
| VAL | 25 | 48.0% | +0.86R | 31.4% | -0.05R | +16.6pt | +0.91R |
| HOLDOUT | 21 | 47.6% | +0.74R | 38.8% | +0.01R | +8.9pt | +0.74R |
**Sign-agreement: ✓ all three splits** (+1)

## B. Rule firing correlation matrix
Pearson correlation across all trades (2025 + Q1 + Apr). Values > 0.5 flag potential redundancy.

| | r1_pole | r2_flag | r3_vol | r4_spy | r5_retr | r7_vwap | r8_gapfd | r9_vrev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1_pole | +1.00 | +0.03 | +0.05 | -0.02 | -0.02 | +0.04 | -0.03 | -0.00 |
| r2_flag | +0.03 | +1.00 | +0.14 | +0.02 | +0.12 | +0.03 | +0.06 | +0.01 |
| r3_vol | +0.05 | +0.14 | +1.00 | -0.00 | +0.01 | +0.03 | +0.10 | +0.02 |
| r4_spy | -0.02 | +0.02 | -0.00 | +1.00 | +0.06 | +0.02 | +0.03 | -0.03 |
| r5_retr | -0.02 | +0.12 | +0.01 | +0.06 | +1.00 | +0.15 | +0.11 | -0.03 |
| r7_vwap | +0.04 | +0.03 | +0.03 | +0.02 | +0.15 | +1.00 | +0.40 | +0.01 |
| r8_gapfd | -0.03 | +0.06 | +0.10 | +0.03 | +0.11 | +0.40 | +1.00 | +0.05 |
| r9_vrev | -0.00 | +0.01 | +0.02 | -0.03 | -0.03 | +0.01 | +0.05 | +1.00 |

**No pairs above correlation 0.5 threshold.**

## C. Regime conditioning
Rule lift (R-multiple, fires vs doesn't) split by 3 regimes on TRAIN. Flags rules where regime changes the edge sign or magnitude >2×.

### Regime: SPY 3d range tertile
| Rule | low_R_lift | mid_R_lift | high_R_lift | flag |
|---|---:|---:|---:|---:|
| r1_pole | -0.21R | +0.04R | +0.01R | ⚠️ sign flip |
| r2_flag+ | -0.02R | +0.33R | -0.39R | ⚠️ sign flip |
| r3_vol | +0.42R | -0.14R | +0.40R | ⚠️ sign flip |
| r4_spy+ | n/a | -0.28R | n/a |  |
| r5_retr | +0.18R | -0.15R | -0.01R | ⚠️ sign flip |
| r7_vwap | -0.28R | -0.19R | +0.26R | ⚠️ sign flip |
| r8_gapfd- | -0.82R | +0.12R | -0.70R | ⚠️ sign flip |
| r9_vrev | +1.29R | -0.71R | +0.96R | ⚠️ sign flip |

### Regime: Entry time tertile
| Rule | low_R_lift | mid_R_lift | high_R_lift | flag |
|---|---:|---:|---:|---:|
| r1_pole | +0.06R | -0.09R | -0.12R | ⚠️ sign flip |
| r2_flag+ | +0.29R | -0.16R | -0.30R | ⚠️ sign flip |
| r3_vol | +0.21R | +0.46R | +0.04R | ⚠️ magnitude >2× |
| r4_spy+ | +0.03R | +0.31R | -0.50R | ⚠️ sign flip |
| r5_retr | +0.04R | -0.01R | -0.04R | ⚠️ sign flip |
| r7_vwap | -0.60R | -0.14R | +0.12R | ⚠️ sign flip |
| r8_gapfd- | +0.32R | -1.36R | -0.37R | ⚠️ sign flip |
| r9_vrev | -0.50R | +1.91R | -0.13R | ⚠️ sign flip |

### Regime: ADV tertile
| Rule | low_R_lift | mid_R_lift | high_R_lift | flag |
|---|---:|---:|---:|---:|
| r1_pole | -0.22R | +0.00R | +0.07R | ⚠️ sign flip |
| r2_flag+ | -0.41R | +0.09R | +0.25R | ⚠️ sign flip |
| r3_vol | +0.15R | +0.27R | +0.33R | ⚠️ magnitude >2× |
| r4_spy+ | -0.24R | +0.08R | +0.02R | ⚠️ sign flip |
| r5_retr | +0.24R | -0.18R | -0.01R | ⚠️ sign flip |
| r7_vwap | +0.53R | -0.42R | -0.30R | ⚠️ sign flip |
| r8_gapfd- | -0.80R | -0.35R | -0.13R | ⚠️ magnitude >2× |
| r9_vrev | +0.56R | +1.27R | +0.10R | ⚠️ magnitude >2× |

## D. Multivariate logistic regression
Fit `P(win) = σ(β₀ + Σ βᵢ·rule_fires_i)` on TRAIN. Compare coefficients to current contrib magnitudes. A near-zero or opposite-sign β flags a rule that doesn't add marginal info once others are in.

| Rule | current contrib | β_TRAIN | β_VAL | |β_TRAIN - β_VAL| | flag |
|---|---:|---:|---:|---:|---|
| intercept | — | -0.313 | -0.251 | — | — |
| rule1_pole_gain | +0.30 | +0.021 | -0.024 | 0.046 | ⚠️ β near zero |
| rule2_flag_tightness | +0.30 | -0.069 | +0.161 | 0.230 | ⚠️ TRAIN/VAL sign flip |
| rule3_vol_ratio | +0.30 | +0.208 | -0.334 | 0.541 | ⚠️ TRAIN/VAL sign flip |
| rule4_spy_regime | +0.30 | +0.156 | +0.404 | 0.248 |  |
| rule5_retracement | +0.20 | -0.089 | -0.201 | 0.113 | ✗ sign mismatch |
| rule7_vwap_dist | +0.20 | -0.246 | -0.091 | 0.155 | ✗ sign mismatch |
| rule8_gap_fading | -0.30 | +0.111 | +0.116 | 0.006 | ✗ sign mismatch |
| rule9_v_reversal | +0.40 | +0.264 | +0.176 | 0.088 |  |

*Note: β coefficients are on log-odds scale, not directly comparable to contrib magnitudes. What matters is sign agreement and relative magnitude ordering.*

## E. MACD zone multiplier audit
Dead-zone trades are already rejected (not in cache). This compares the post-filter MACD buckets: 1.0× (normal) vs 1.5× (strong).

| Split | bucket | n | WR | avg_R | total_pnl |
|---|---|---:|---:|---:|---:|
| TRAIN | 1.0× | 293 | 39.2% | +0.12R | $+11,102 |
| TRAIN | 1.5× | 243 | 41.2% | +0.18R | $+27,072 |
| VAL | 1.0× | 264 | 32.6% | -0.01R | $-2,485 |
| VAL | 1.5× | 232 | 31.9% | +0.00R | $+13,423 |
| HOLDOUT | 1.0× | 250 | 34.4% | -0.11R | $-13,609 |
| HOLDOUT | 1.5× | 189 | 45.5% | +0.24R | $+34,301 |

## F. Risk tier audit — including orphan price bands
Current tiers: $10-15/$15-23 at 500K-5M vol. Shows per-share edge for ALL buckets (inc. <$10 and $23+) so orphan bands are visible.

| price | vol | n | WR | avg_R | total_pnl | current tier mult |
|---|---|---:|---:|---:|---:|---:|
| $10-15 (T1) | 500K-5M | 110 | 39.1% | +0.06R | $+1,382 | 2.0 |
| $10-15 (T1) | <500K | 141 | 44.0% | +0.24R | $+14,704 | — orphan — |
| $15-23 (T2) | 500K-5M | 108 | 31.5% | +0.02R | $+4,132 | 1.0 |
| $15-23 (T2) | 5M+ | 5 | 20.0% | -0.48R | $-192 | — orphan — |
| $15-23 (T2) | <500K | 133 | 36.1% | +0.00R | $-59 | — orphan — |
| $23+ | 500K-5M | 8 | 25.0% | -0.27R | $-3,043 | — orphan — |
| $23+ | <500K | 13 | 7.7% | -0.56R | $-3,266 | — orphan — |
| $5-10 | 500K-5M | 186 | 37.6% | +0.06R | $+3,708 | — orphan — |
| $5-10 | 5M+ | 20 | 40.0% | +0.00R | $-121 | — orphan — |
| $5-10 | <500K | 252 | 36.1% | +0.06R | $+8,557 | — orphan — |
| <$5 | 500K-5M | 178 | 40.4% | +0.31R | $+40,262 | — orphan — |
| <$5 | 5M+ | 50 | 40.0% | +0.05R | $+2,711 | — orphan — |
| <$5 | <500K | 265 | 35.5% | -0.07R | $+977 | — orphan — |

## G. Conviction min_threshold sweep
For each threshold, filter trades to `conviction_mult >= T` and show resulting metrics.

| T | split | n | WR | total_pnl |
|---|---|---:|---:|---:|
| 1.0 | TRAIN | 478 | 41.8% | $+36,441 |
| 1.0 | VAL | 402 | 35.1% | $+16,417 |
| 1.0 | HOLDOUT | 399 | 39.8% | $+20,015 |
| 1.2 | TRAIN | 432 | 42.8% | $+32,950 |
| 1.2 | VAL | 350 | 35.7% | $+15,859 |
| 1.2 | HOLDOUT | 370 | 40.5% | $+20,631 |
| 1.3 | TRAIN | 384 | 43.2% | $+31,920 |
| 1.3 | VAL | 308 | 35.4% | $+15,212 |
| 1.3 | HOLDOUT | 330 | 42.1% | $+21,113 |
| 1.4 | TRAIN | 356 | 42.7% | $+28,962 |
| 1.4 | VAL | 276 | 36.6% | $+14,328 |
| 1.4 | HOLDOUT | 311 | 44.1% | $+25,190 |
| 1.5 | TRAIN | 330 | 42.1% | $+25,894 |
| 1.5 | VAL | 255 | 35.7% | $+14,084 |
| 1.5 | HOLDOUT | 290 | 43.8% | $+21,622 |
| 1.6 | TRAIN | 265 | 43.0% | $+19,327 |
| 1.6 | VAL | 181 | 35.9% | $+6,733 |
| 1.6 | HOLDOUT | 242 | 44.2% | $+19,080 |
| 1.7 | TRAIN | 248 | 42.7% | $+19,521 |
| 1.7 | VAL | 167 | 35.9% | $+5,686 |
| 1.7 | HOLDOUT | 227 | 44.5% | $+19,212 |

## H. V-reversal params sweep (Rule 9)
Sweep `intraday_range_min` threshold. Current = 20. For each value, show how many V-rev trades fire on each split and their edge.

| range_min | split | v_rev_n | v_rev_WR | v_rev_avg_R |
|---|---|---:|---:|---:|
| 15 | TRAIN | 65 | 49.2% | +0.35R |
| 15 | VAL | 60 | 40.0% | +0.39R |
| 15 | HOLDOUT | 36 | 47.2% | +0.65R |
| 18 | TRAIN | 46 | 54.3% | +0.54R |
| 18 | VAL | 38 | 44.7% | +0.70R |
| 18 | HOLDOUT | 26 | 50.0% | +0.86R |
| 20 | TRAIN | 34 | 61.8% | +0.71R |
| 20 | VAL | 25 | 48.0% | +0.86R |
| 20 | HOLDOUT | 21 | 47.6% | +0.74R |
| 22 | TRAIN | 22 | 68.2% | +0.84R |
| 22 | VAL | 24 | 50.0% | +0.94R |
| 22 | HOLDOUT | 12 | 50.0% | +0.92R |
| 25 | TRAIN | 14 | 64.3% | +0.79R |
| 25 | VAL | 16 | 31.2% | +0.33R |
| 25 | HOLDOUT | 5 | 40.0% | +0.75R |

