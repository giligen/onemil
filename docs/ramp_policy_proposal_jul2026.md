# ORB ramp-policy change proposal — for Monday sign-off (2026-07-06)

## Problem with the current %-cushion gate

The rollout plan advances stages on **cushion** (cumulative live P&L ≥ +$5K
at Stage 0). Two incidents show the metric fights the strategy's own math:

1. **6/12 and 6/19 demotion flags** — both overridden by owner. The June
   drawdown was **BT-consistent** (defended BT June on Stage-0 sizing:
   +$3,766 with intra-month DD of comparable depth; live −$3,187 gap was
   traced to the selection race + touchgo keying bugs, both now fixed).
   The cushion gate cannot distinguish "strategy broken" from "lottery
   variance while under-scaled" — at Stage-0 sizing, one monster pays for
   ~2 months of stops. Requiring +$5K before scaling means waiting for a
   monster at the size where monsters are smallest.
2. **The gate punishes exactly what the BT says to expect.** 18-month BT
   (post-PDR-veto): 60% of trades lose; P&L lives in ~130 EOD holds/18mo
   (~1.7/week at full slots). Sub-scale stages can easily run 4-6 weeks
   flat-to-negative while behaving exactly as validated.

## Proposed gates (replace %-cushion)

Advance from stage N to N+1 when ALL of:

1. **Operational green** — 10 consecutive sessions with:
   - 0 unexplained BT↔live selection diffs (observer + selection-audit JSONL)
   - 0 touchgo REKEY/negative-age tripwires
   - order-fill mismatch rate 0 (DB vs broker, reconciler)
   - all exits attributed (no `unknown_exit` rows)
2. **Loss floor, not profit target** — cumulative live P&L over the stage
   ≥ **−1.0 × (stage daily loss limit × 5)** (i.e. worse than a full
   losing week = hold). Rationale: we scale unless the strategy is
   performing WORSE than its own validated loss distribution, instead of
   demanding it prove profitability at a size where edge is smallest.
3. **Slippage parity** — median entry slippage ≤ BT model +10bps over the
   stage (analyze_orb_slippage.py).
4. **Minimum 10 trading days in stage** (unchanged).

Demotion (unchanged in spirit, tightened in trigger): only on
**operational failures** (unexplained selection diff, fill mismatch,
unattributed exit) or cumulative stage P&L < −2 × (daily limit × 5).
Pure P&L drawdown consistent with BT percentile bands is NOT a demotion
trigger (this codifies the two June overrides).

## Why now
- All four July fixes (race, keying, partial-fill, exit pricing) land
  Monday; the operational-green gate becomes measurable from day 1.
- PDR veto (shipping this weekend) cuts trade count ~50% and gross losses
  ~50% in BT — cushion accrual under the old gate would slow even though
  risk-adjusted quality rose (Calmar roughly doubles). A profit-target
  gate would perversely read this improvement as "slower ramp".

## Not changing
- Stage budget/risk ladder ($30K→$50K→…→$174K) unchanged.
- Daily loss limits per stage unchanged.
- Pre-Stage-0 requirement for any NEW strategy (e.g. $30-60 band ORB)
  unchanged.
