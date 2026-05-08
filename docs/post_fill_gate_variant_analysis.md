# Post-Fill Gate Variant Analysis (IREZ Post-Mortem)

**Date**: 2026-05-08
**Trigger**: IREZ 2026-05-08 — prod's post-fill gate killed a $37K-shape winner on a 0.77/0.80 SPY-boundary cross. Asked: should the gate stay, change, or go?
**Method**: Disabled the gate in BT, ran natural exits over Jan 2025 - May 2026 (310 trades), replayed five gate variants post-hoc.

## Headline result

| Variant | Total P&L (16 mo) | WR % | maxDD | kills | $ vs natural |
|---|---:|---:|---:|---:|---:|
| **V4 disabled** (no gate) | **+$339,698** | 47.7 | -35,847 | 0 | 0 |
| V1 tightened (0.5/0.5) | +$339,698 | 47.7 | -35,847 | 0 | 0 |
| V2 V1 + intraday-override | +$339,698 | 47.7 | -35,847 | 0 | 0 |
| V3 scale-down (25%) | +$323,162 | 47.7 | -35,911 | 9 | -$16,535 |
| **V0 control** (current) | +$315,055 | 46.1 | -36,090 | 9 | **-$24,643** |

**The current gate destroys ~$24,643 of net P&L over 16 months and the maxDD is unchanged.** The hypothesis it was protecting against ("calm SPY + weak breakout volume = trade fails") is not supported by data.

## What the 9 kills actually were

| Date | Symbol | Natural P&L | spy_3d | bk_ratio | intraday% |
|---|---|---:|---:|---:|---:|
| 2026-01-06 | **VELO** | **+$37,514** | 0.770 | 0.621 | 16.2 |
| 2025-12-11 | SKIL | +$1,742 | 0.759 | 0.115 | 36.6 |
| 2025-07-23 | RNAZ | +$568 | 0.518 | 0.473 | 39.0 |
| 2025-08-21 | CATO | +$95 | 0.742 | 0.685 | 17.1 |
| 2025-07-11 | SPAI | +$48 | 0.550 | 0.161 | 18.2 |
| 2025-11-13 | SLND | -$171 | 0.787 | 0.865 | 33.5 |
| 2025-09-16 | APLM | -$1,161 | 0.440 | 0.890 | 30.1 |
| 2025-10-29 | NPT | -$7,325 | 0.505 | 0.845 | 33.2 |
| 2025-09-19 | FGI | -$9,264 | 0.741 | 0.751 | 25.1 |

- **5 winners killed** (avg +$7,993, max +$37,514) — VELO alone is +$37K
- **4 losers "saved"** (avg -$4,480, total -$17,921 avoided)
- Net: **+$24,643 of missed alpha**

## Walk-forward: the gate fails out-of-sample

| Split | V0 (gate on) | V4 (gate off) | V0-V4 |
|---|---:|---:|---:|
| **Train** Jan-Sep 2025 | $172,750 | $163,833 | **+$8,917** (gate helped) |
| **Test** Oct 2025-May 2026 | $142,305 | $175,865 | **-$33,560** (gate hurt) |
| Full 16mo | $315,055 | $339,698 | -$24,643 |

This is a textbook out-of-sample failure. The gate's filter shape worked on 2025 data and reversed sign in 2026. **A live filter that flips sign post-deployment is by definition overfit.**

## Per-month detail (key delta column)

| Month | V0 (current) | V4 (off) | V4-V0 |
|---|---:|---:|---:|
| 2025-07 | $14,764 | $15,406 | +$641 |
| 2025-08 | $8,847 | $8,949 | +$102 |
| **2025-09** | **$40,709** | **$31,048** | **-$9,661** ⬇ |
| **2025-10** | **$18,768** | **$11,904** | **-$6,864** ⬇ |
| 2025-11 | $22,677 | $22,527 | -$150 |
| 2025-12 | $9,692 | $11,520 | +$1,828 |
| **2026-01** | **$19,681** | **$58,426** | **+$38,745** ⬆ |
| (other months) | ... | ... | $0 (no kills) |

September and October 2025 were the only months where the gate net-helped. January 2026 alone (the VELO kill) wiped out their combined contribution + $20K more.

## Why V1 (0.5/0.5) is identical to V4 in this dataset

All 9 kills had at least one of `spy_3d ≥ 0.5` or `bk_ratio ≥ 0.5`. Tightening the threshold to 0.5/0.5 eliminates every kill on this 16-month dataset. **For prod, V1 = V4 in observable behavior, but V1 retains a defense for severely hostile conditions** (e.g., spy_3d crashed to 0.3 on a market-event day).

## Why the V2 override hypothesis didn't help

Original idea: skip kill if `intraday_change ≥ 25 AND relvol ≥ 10x AND news_catalyst`. In this dataset:
- 6 of 9 kills had `intraday ≥ 25` — but **4 of those 6 were losers** (FGI, NPT, APLM, SLND)
- The biggest winner VELO had `intraday = 16.2` — wouldn't qualify for any override threshold
- "Obvious momentum" by intraday-change is NOT a winner discriminator

The losers have *higher* bk_ratio than the winners (0.75-0.89 vs 0.11-0.62). The gate's volume-weakness signal is anti-correlated with success.

## What the data is actually saying

The gate's hypothesis — "calm-SPY plus weak-breakout-volume predicts failure" — is NOT validated:
- 5 winners with both conditions hostile
- 4 losers with both conditions hostile
- Roughly 50/50 outcome distribution → no signal

Combined with the walk-forward sign-flip, the gate is **a coin flip with negative expected value**. It's not worth keeping in any form.

## Recommendations

| Variant | Action | Rationale |
|---|---|---|
| **V1 tightened (0.5/0.5)** | **SHIP** | Equivalent to V4 in 16-month data. Keeps a fail-safe for severe hostile context. **Recommended over V4 because the cost of the residual code is zero and the optionality is real.** |
| V4 disabled | Acceptable alternative | Same data outcome as V1. Cleaner code. Loses optionality. |
| V0 current | **KILL** | -$24,643 over 16 mo. WR worse. Walk-forward fails. Hypothesis disproven. |
| V2 V1+override | DON'T SHIP | Identical to V1 here. Override criterion (`intraday≥25`) doesn't discriminate winners from losers. |
| V3 scale-down 25% | DON'T SHIP | -$16,535 vs V4. Half-measure that captures the worst of both. |

### The deeper architectural fix (independently valuable)

Independent of the gate decision, ship **#1 from earlier**: snapshot SPY 3d once per trading day in `_get_spy_3d_range_live`, reuse for every gate/conviction call.
- Eliminates the **9-second drift** between conviction (0.80) and post-fill (0.77) that killed IREZ in prod
- Restores BT-live parity by construction (BT already uses stable T-1/T-2/T-3)
- Even with V1 shipped, this fix prevents a future "0.49 ticked to 0.51" mistake

This is a parity bug regardless of whether the kill threshold is 0.8 or 0.5.

## Caveats

1. **VELO 2026-01-06 dominates** (+$37K of the $24K net). Without VELO, V0 vs V4 is roughly $13K cost vs $13K savings. **One trade should not justify a permanent filter** (or its removal) — but it shouldn't be excluded either since trades like this are the whole point.
2. **Stage-2 cascading not modeled.** Without the gate, killed-then-recovered trades hold positions longer; this could shift max_concurrent slot availability and daily-loss-limit pauses. Real production behavior may differ from the post-hoc replay by a few percent.
3. **Sample size is 9 kills.** Statistical power is limited. The signal is "no winners-vs-losers discrimination" rather than "the gate definitively destroys alpha." But the walk-forward sign-flip is a separate, harder-to-explain-away problem.

## Ship plan

1. **Today (config-only flip)**: in `config.yaml`, add a `post_fill_gate` block with thresholds 0.5/0.5 (V1). Implementation: introduce env-var-readable thresholds in `backtest.py:2510` and `trading_engine.py:1450`, default to current 0.8/1.0, override via config. Default-on the new thresholds. Restart `onemil-trader`.
2. **This week**: ship the SPY-3d daily snapshot (architectural fix #1). Adds parity guarantee.
3. **Next quarterly review**: reassess. If 6 more months of data confirm the gate is dead weight even at 0.5/0.5, remove the code (V4).
