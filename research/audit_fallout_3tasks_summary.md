# Audit fallout — 3-task research summary

Date: 2026-04-18. Source: research/multiplier_audit.md (commit 54974a5).

## TL;DR — all three audit recommendations produce smaller lift than the report estimated, once layered on top of the shipping stack with `min_conv>=1.4` already active. None clear the "ship" bar as-is. The audit's estimates of "+$20-40K" were extrapolated from pre-filter per-rule numbers; the real conv filter + MACD zone + 3.0× cap interactions compound down to small single-digit-thousand deltas.

---

## Task 1 — Tier redesign

Post-hoc sim (`tier_redesign_sim.py`): evaluated 5 variants against baseline
on TRAIN / VAL / HOLDOUT Q1 / HOLDOUT April.

**HOLDOUT (Q1 + April) deltas vs baseline:**

| Variant | HOLDOUT Δ | TRAIN Δ | VAL Δ | Notes |
|---|---:|---:|---:|---|
| A: add T3 <$5 @1.5× | +$928 | +$813 | +$408 | cleanest, all positive |
| B: demote T1 to 1.0×, <$5 @2.0× | +$1,118 | +$2,035 | −$1,206 | VAL regresses |
| C: <$5 any-vol @1.5× | +$254 | −$817 | +$1,009 | TRAIN regresses |
| D: B + rescue $10-15 <500K @1.5× | +$1,413 | +$2,662 | **−$2,339** | VAL fails guard |
| E: A + rescue $10-15 <500K @1.5× | +$1,224 | +$1,441 | −$725 | marginal |

**Decision**: **DO NOT SHIP**. No variant hits the plan's $5K HOLDOUT gate.
Why so small: the `<$5, 500K-5M` bucket's $38,771 all-splits edge comes
from trades that MOSTLY already have `conviction*macd >= 3.0` (capped);
adding tier × 2.0 hits the cap and only boosts the non-capped subset by
~11-33%. The audit's "+$10-20K" extrapolation didn't account for the cap.

**Cleanest (variant A) ships +$928 HOLDOUT, clean TRAIN/VAL.** Arguably
defensible as risk-free polish. But not the "$20-40K" promised.

## Task 3 — MACD 1.0× bucket treatment

Post-hoc sim (`macd_1x_sim.py`): baseline vs skip-1.0× vs 0.5×/0.75× scale.

**HOLDOUT-combined (Q1 + April):**

| Variant | HOLDOUT Δ | TRAIN Δ | VAL Δ | April Δ |
|---|---:|---:|---:|---:|
| **S** (skip) | +$6,270 | **−$6,292** | +$444 | **−$3,632** |
| H5 (0.5×) | +$3,135 | −$3,146 | +$222 | −$1,816 |
| H75 (0.75×) | +$1,568 | −$1,573 | +$111 | −$908 |

**Decision**: **MIXED SIGNAL**. 1.0× bucket is actively negative in Q1
2026 (−$9,902) but **positive in April 2026** (+$3,632, 53% WR, 30 tr).
Dropping 1.0× gives Q1 win but April loss. TRAIN also loses (because
2025 1.0× is weakly positive).

**Recommend**: do not skip outright. If we ship anything, `H75` is the
most conservative (shrinks losers, keeps upside on decent 1.0× trades).
But even H75 fails the TRAIN guard, so: **do not ship**.

## Task 2 — Conviction audit-fix (drop rules 3/5/7)

Post-hoc sim (`audit_fix_posthoc.py`): recomputes conviction score by
subtracting cached contributions of rules 3/5/7 and re-clamping.

**Threshold sweep, TRAIN+VAL only (leakage-clean):**

| threshold | T+V PnL | Δ vs base (T+V) |
|---|---:|---:|
| 0.8 | $+38,717 | −$4,574 |
| 1.0 | $+37,898 | −$5,392 |
| 1.1 | $+38,764 | **−$4,526** (best) |
| 1.4 | $+25,428 | −$17,863 |

**Best TRAIN+VAL threshold: 1.1 → TRAIN+VAL Δ −$4,526**

**One-shot HOLDOUT @ T=1.1, audit_fix ON:**
- HOLDOUT baseline: +$25,190
- HOLDOUT audit_fix: +$18,458
- **HOLDOUT Δ: −$6,732**

**Decision**: **DO NOT SHIP**. The audit's recommendation to drop the 3
sign-unstable rules (Rule 3: vol_ratio; Rule 5: retracement; Rule 7:
vwap_dist) produces NEGATIVE P&L delta at every threshold level.

Why: per-rule analysis (audit sections A, D) identifies univariate sign
flips but misses that these rules contribute **ensemble value** when
combined with the other 6 rules in the scoring function. The logistic
β-coefficients flagged them correctly as "maybe noisy", but the actual
downstream filter behavior at `conv>=1.4` treats the noise as signal
for separating winners from losers.

**This is a known pitfall of per-feature audits in ensemble classifiers.**

## Impact on claimed "+$20-40K" ship opportunity

The audit's estimated lift was based on pre-filter analysis (e.g., audit
Section F shows `<$5, 500K-5M = 178 trades / +$40,262 / +0.31R avg` —
the entire 2025+Q1+April pool). Once the existing `conv>=1.4` filter +
MACD zone + 3.0× cap are applied:

- The relevant bucket is `<$5 500K-5M with conv≥1.4` = 98 trades, not 178
- Most of those have MACD 1.5× already boosting them
- Tier × conviction hits the 3.0× cap for most "good" setups
- Marginal lift per trade drops from "$395 avg" to "~$50" on the head
  that isn't capped

**Real achievable lift from these 3 tasks: +$1-6K on HOLDOUT**, best case
from variant A tier addition. Not the $20-40K promised.

## Recommendation

1. **Ship nothing from these 3 tasks** — none clear the regression gates.
2. **Keep the `audit_fix` flag infrastructure** (default OFF) as a harness
   for future research that jointly retunes rule weights + min_threshold.
3. **Close the 3 task tickets** as "researched — no ship".
4. **Update CLAUDE.md / README**: note that the current multiplier stack
   is at a local optimum; per-rule audits alone are insufficient to
   propose changes because of filter/cap interactions.

## Where the real edge might still be (out-of-scope for today)

- **Skip mid-cap stale setups**: the `$15-23, <500K` bucket is 90 trades
  / -$3,285 / 36.7% WR — a clean filter-out candidate.
- **Skip $23+ entirely**: 12 trades total, -$5,933 combined. Already ~0%
  of our volume but removes a consistent leak.
- **Filter by daily_range_pct < X**: orthogonal to conviction — worth a
  dedicated audit pass.

These would be **new research tasks**, not revisions of the three run today.

## Artifacts (this session)

Code (shipped, all default-OFF, parity-tested):
- `tier_redesign_sim.py` + `research/tier_redesign_sim.md`
- `macd_1x_sim.py` + `research/macd_1x_sim.md`
- `audit_fix_posthoc.py` + `research/audit_fix_posthoc.md`
- Audit-fix plumbing: `backtest.py`, `trading/trading_engine.py`,
  `config.yaml(.template)`, `batch_backtest.py`, `batch/monthly_runner.py`
- `tests/test_audit_fix_conviction.py` (15 tests, all pass)
- Env var: `BT_CONVICTION_AUDIT_FIX=1` enables the flag in batch runs

No prod config changes. No shipping default flips. A_f6 + TTF + D + V
shipping stack is unchanged.
