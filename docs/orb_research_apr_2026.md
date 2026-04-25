# ORB Exit & Add-to-Winners Research — April 2026

Reference for future ORB strategy work. Captures the negative findings (so
we don't re-test) and the one parked positive (so we know where to resume).

## TL;DR

After 50+ variants tested:

- **V0 `static_lock_1R` is at the Pareto frontier for ORB exits.** No rule-based,
  MFE-conditional, classifier-based, or quintile-gated exit beats it OOS with
  tail-robustness.
- **Q1 filter shipped** (commit `ccd990b`): +$2.5K FULL / +$8.5K OOS combined,
  tail-robust, no DD impact. Default ON via `orb.yaml::filter.skip_q1`.
- **Add-to-winners parked** as a research artifact: bull-flag-gated add to Q4/Q5
  + $5-$20 stocks shows +$10K/16-month lift on FULL, but HOQ1+ has only n=5
  fires (anecdotal-strength). Opportunity cost vs L2 work too high to ship.

---

## Negative findings — DON'T repeat these

| Variant family | Best variant | Why it failed |
|---|---|---|
| Trail-after-arm (V1, V1b, V1c) | trail_1.5R | V0's "ride to EOD if no +1R pullback" beats every trail. Tight trails exit on intra-bar volatility, lose the EOD tail. |
| Late-arm + trail (V4a, V4b) | late_arm_4R_trail_0.5R | Calmar 20.76x (best) but P&L -$31K vs V0. Worse than V0 in absolute dollars. |
| Partial-then-runner (V2, V6) | partial_50_runner_2.0R | -$60K vs V0. Static half captures less, runner half exits on flag-low pullbacks before EOD ride. |
| MFE-conditional remove-stop (V11) | remove_after_MFE5R | +$1,226 marginal win on FULL, in noise range. |
| Quintile-aware exits (V3, V4Q5) | Q4/Q5 stay-rule | Failed HOQ1+ (-$21K to -$27K). TRAIN-overfit. |
| Pre-trade runner classifier | linear_a1.5 | TRAIN +$63K, OOS −$30K. Classic overfit. |
| Stay-Breakeven on +1R pullback | SB_Q5 (Q5 only) | TRAIN +$19K, VAL +$14K, **HOQ1+ -$11K**. Failed OOS. |
| Post-+1R classifier (in-trade features) | GB at p>=0.40 | Marginal positive Δ but worse than always-stay. Mult-fit issue + small N. |
| Always-add at +3R (oracle) | n/a | All slices positive but **fails top-3 tail removal on TRAIN**. Hero-trade-driven. |
| Bull-flag-gated add (loose detector) | +2.5R / loose | All slices positive headline, but tail check fails on every slice except FULL. |
| Strict bull-flag detector | n/a | Only 2 fires across 16 months. Textbook flags too rare in small-cap ORB. |

**Common failure mode:** small-cap ORB's edge is structurally fat-tailed.
Any add/exit variant that depends on capturing a few hero trades (3-10 per
slice) cannot survive the tail-removal test demanded by the user's robustness
criterion. The bull flag detector improved the SIGNAL but couldn't change
the underlying distribution.

---

## The one positive parked finding — bull-flag-gated add (Q4/Q5 + $5-$20)

### Configuration

```python
# Trigger
ADD_TRIGGER_R = 2.5  # add when MFE first reaches +2.5R

# Detector (loose params)
detector = BullFlagDetector(
    min_pole_candles=2,
    min_pole_gain_pct=2.0,
    max_retracement_pct=60.0,
    max_pullback_candles=8,
    min_breakout_volume_ratio=1.2,
    max_green_in_flag=2,
    max_pole_bars=0,
)

# Segment filter (CRITICAL — without this, tail-dependence kills it)
QUINTILE_GATE = {'Q4', 'Q5'}
PRICE_GATE = (5.0, 20.0)  # entry_price in [5, 20)

# Add mechanics
ADD_PCT = 0.5                  # 50% of original _rp_position
SAFETY_STOP_R = 2.0            # stop = max(flag_low, entry + 2R)
# Exit: stop or EOD, whichever first
```

### Walk-forward results (study_orb_add_bullflag_filtered.py)

```
                Fires    WR      Sum       rm3       rm5
TRAIN              16   50.0%   +$7,458   +$66      -$1,924
VAL                11   54.5%   +$2,459   -$63      -$942
HOQ1+               5   60.0%   +$586     -$381     +$0
FULL               32   53.1%  +$10,503   +$3,111   +$477   ★
```

### Why it works mechanically

- **Q4/Q5 = high-conviction setups** — flags reflect real institutional patterns
- **$5-$20 = institutional liquidity zone** — cleaner technicals, less squeeze risk
- **Loose detector** — strict was too rare (2 fires in 16 mo); loose at 31% WR
  baseline jumps to 53% after segment filter
- **Excludes the noise**: <$5 stocks (13% WR) and Q3/Q2 (14-18% WR) are
  pump-and-dumps and weak setups — segment filter removes them

### Why it's parked, not shipped

- **HOQ1+ sample is anecdotal (n=5)** — could be 0-5 wins in another sample
- **Annualized lift is ~$8K/year** — small absolute dollars
- **Engineering cost** to add the production add-path (state machine, parity
  tests, monitoring) is 1-2 days for $8K/year ROI
- **L2 microstructure is the parked $200K/year opportunity** per
  `memory/project_l2_entry_system.md` — order-of-magnitude better target

### To resume this work later

1. Load `study_orb_add_bullflag_segmented.py` — re-runs the segmented BT
2. Or `study_orb_add_bullflag_filtered.py` — applies filter to saved CSV
3. The detector + filter constants above are the validated config
4. Production wiring would go in `trading/orb_engine.py::check_open_positions`
   — monitor each open position for +2.5R MFE, run detector, evaluate filter,
   submit add order. Coordinate stop with V0's static_lock state machine.
5. Ship behind feature flag (`add_winners_enabled: false` default) and
   paper-validate for 6-8 weeks before live.

---

## Q1 filter — what shipped (commit ccd990b)

### Diagnostic
Q1 was TRAIN-positive (+$6,052) but OOS-negative on both VAL (-$5,151) and
HOQ1+ (-$3,405). TRAIN-fit 0.5x adaptive mult was sized for a TRAIN-mean
that didn't generalize. Classic overfit signature.

### Filter mechanics
At candidate ranking time in `orb_engine.check_entries()`, drop any
candidate where `quintile == 'Q1'`. Remaining candidates ranked normally.

Slot mechanics: 60% max-slot days → 49% max-slot days. Slots do **not**
refill with higher-quintile candidates because Q1 is last-priority in
`ranking_order` — removing it never unblocks higher-Q via dedup
(verified by `check_q1_refill_potential.py`).

### Live monitoring

```bash
journalctl -u onemil-trader -f | grep -E "Q1 filter"
```

Expected: ~0.4 fires per day (131 Q1-rejections/year over ~250 trading days).
If no Q1 filter lines fire across a full day, something's wrong with ranking.

### Rollback

In `orb.yaml`: set `filter.skip_q1: false`. Restart `onemil-trader`. No
state to unwind — pure config flip.

---

## Files map

### Production (Q1 filter)
- `trading/orb_engine.py` — loads `self.skip_q1`, applies pre-ranking
- `orb.yaml` — local instance config, gitignored
- `orb.yaml.template` — repo-tracked template, ships `skip_q1: true`
- `study_orb_pipeline_static_lock.py` — BT mirrors filter (`ORB_SKIP_Q1=0`
  to disable)
- `tests/test_orb_engine_q1_filter.py` — 7 unit tests

### Research scripts (committed in 0430d39 — for reference only)
- `study_orb_pullback_oracle.py` — diagnostic for stay-vs-exit at +1R
- `study_orb_3r_differentiation.py` — feature ranking at +3R MFE
- `study_orb_add_oracle.py` — upper-bound add lift (always-add)
- `study_orb_add_bullflag.py` — first bull-flag-gated add variant
- `study_orb_add_bullflag_segmented.py` — adds rich metadata + 3 detector configs
- `study_orb_add_bullflag_filtered.py` — applies segment filter to saved CSV
- `study_orb_exit_runners.py` ... `_v4.py` — 4 rounds of exit variant studies
- `study_orb_pullback_classifier.py` — post-+1R ML classifier
- `study_orb_runner_classifier.py` — pre-trade runner classifier
- `study_orb_stay_rule_oos.py` — stay-rule OOS validation
- `study_orb_exit_classifier.py` — classifier-based exit
- `study_orb_q1q2_filter.py` — Q1/Q2/Q3+ filter walk-forward (validated Q1 filter)
- `check_q1_refill_potential.py` — verifies Q1 slots can't be refilled

---

## What to do next (recommended priority)

1. **Live-monitor Q1 filter for 1 week** after Monday's deployment. Verify
   `Q1 filter dropped` log lines fire daily and BT-vs-prod composite scores
   stay aligned (continue the existing `ORB SCORED` parity audit).
2. **Pivot to L2 microstructure research** — see
   `memory/project_l2_entry_system.md` for the parked plan. The $200K/year
   opportunity is order-of-magnitude bigger than anything on the exit side.
3. **Defer add-to-winners** indefinitely. Only revisit if L2 work doesn't
   pan out AND we have ≥30 paper-fires of bull-flag-gated adds to
   statistically validate the n=5 HOQ1+ finding.
