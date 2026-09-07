# ORB budget ramp on POSITIVE realized P&L — proposal (2026-09-07, for the owner)

**Why this exists.** At the $10K stage ORB is notional-sized: the per-position
cap (budget ÷ slots = $3,333) binds on every trade, so `risk_per_trade_usd`
is inert (verified 9/6: $375/$500/$600/$750 give identical P&L). The only
levers on ORB's 2026 dollars are budget, fill rate, slots and margin. This
proposal is the budget lever, gated the same way BF's P1 ramp is gated:
on realized live P&L and fill count, in units that hold at every stage.

**What we have.** Honest B+ backtest (wrappers-in, entered-inclusive, two
new vetoes): $6,531 / 21 months at $10K = ~$310 a month ≈ 37% a year on the
budget, MDD −$551 (5.5%), worst month −$185 (1.9%), 6 red months of 21.
Live under this book: 2 fills, +$113, both the backtest's picks. That is
agreement, not a rate. Nothing below assumes the rate; the gates test it.

## The unit
u = one per-position cap = budget ÷ slots (at $10K / 3: u = $3,333). A fill's
P&L in u is its percent move on the capped notional. Backtest per fill:
mean +$97 = +2.9% of u; worst month −5.5% of u ×3 slots.

## Stages (budget, 3 slots)
| stage | budget | per-position cap | daily loss limit | month pause |
|---|---|---|---|---|
| **S0** (now) | $10K | $3,333 | −$750 (current yaml) | — |
| S1 | $30K | $10,000 | −$1,500 | −$3,000 |
| S2 | $60K | $20,000 | −$3,000 | −$6,000 |
| S3 | $100K | $33,333 | −$5,000 | −$10,000 |

## Advance — ALL must hold (the BF P1 form)
1. Stage realized P&L > 0 (the above-water rule, owner 7/23).
2. ≥ 8 fills in the stage (backtest: 8-fill windows positive far more often
   than a coin flip — to be measured on the honest book before S0→S1; if the
   honest book's 8-fill positivity is < 90%, raise to 12).
3. ≥ 15 sessions in the stage.
4. Fill parity clean: every backtest pick got a live order (filled or
   canceled) and every live fill was a backtest pick (the green check's two
   lists), zero exceptions in the stage.
5. Slippage within the backtest's 30/10 bps model on the stage's fills
   (`scripts/analyze_orb_slippage.py`) — this is the capacity test that the
   backtest cannot run; a stage that fills worse than the model does not
   advance regardless of P&L.
6. No daily-loss-limit hit in the last 10 sessions.

## Demote one stage — ANY
Stage P&L ≤ −6% of budget; 5 consecutive losing fills; a daily limit hit
twice in a stage; slippage > 2× the model on ≥ 3 fills.

## Pause and review — ANY
Stage P&L ≤ −8% of budget; a fill-parity defect (BT and live disagree on a
pick's existence) that is not explained the same day; two demotions in a row.

## Calendar honesty
At the live pace so far (2 fills in 3 weeks), 8 fills is ~3 months per
stage. The gate is on fills, not dates. Raising the fill rate (the no-fill
slot-recycling study, `research/orb_slot_recycle/DESIGN.md`) is the only
way to make the ladder faster without lowering the bar.

## What this replaces
`docs/orb_rollout_plan.md`'s cushion-gated stages ($15K / $30K / $50K–$174K
with cushion + days-in-stage). Cushion in dollars at $10K is too small to
ever clear a meaningful bar; fills and parity are the evidence that exists.
`scripts/orb_ramp_check.py` would be re-pointed at these stages.

**Decision needed:** adopt this ladder (S0→S1 at 8 positive fills with clean
parity and slippage), or keep the cushion ladder. My recommendation: adopt.
No budget moves until the checker says ADVANCE.
