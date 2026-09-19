# BF P1 live ramp — gates on POSITIVE realized P&L (owner decision 2026-09-06)

Owner: "P1 is the way to go. No shadow, go live on Monday. Run at low risk
till we prove LIVE that this works. Define the gates based on positive P&L."

P1 = above-VWAP gate + pole gain ≥ 5% + entry price ≤ $20 + 50% off at +2R
(stop to fill, remainder trails) + regime sizing OFF. Live from Monday
2026-09-07 at base risk $150. Evidence: `research/bf_consistency/README.md` §6.

## The unit
Everything below is in **u = base risk per trade** (`trading.risk_per_trade`),
so the gates are the same at every stage. P1's backtest, in u (55 trades,
Jan-25 → Aug-26, $2K base):

| statistic | value |
|---|---|
| mean per trade | +1.19u, WR 67%, median win +2.1u, median loss −2.2u |
| worst trade / day / week / month | −3.7u / −4.6u / −5.8u / −3.8u |
| trade-level max drawdown | −7.6u |
| longest losing streak | 4 |
| rolling 8-trade sum positive | **98% of windows** (a coin-flip edge: 50%) |
| rolling 6-trade sum positive | 88% (p10 −0.9u, median +6.8u) |
| trades per month | 2.8 |

## Stages (base risk)
| stage | base risk | daily rail | weekly rail | month pause | daily loss limit |
|---|---|---|---|---|---|
| **L0** (Mon 9/7) | **$150** | −$750 | −$1,050 | −$1,200 | −$750 |
| L1 | $400 | −$2,000 | −$2,800 | −$3,200 | −$2,000 |
| L2 | $1,000 | −$5,000 | −$7,000 | −$8,000 | −$5,000 |
| L3 | $2,000 | −$10,000 | −$14,000 | −$16,000 | −$10,000 |

### Stage START dates (the window every gate is measured over)
Source of truth: `trading/ramp_stage.py` (`STAGE_STARTS['bf']`), read by
`scripts/bf_ramp_check.py`. Append a row there and here in the same commit;
never delete one — an old start is history, not a mistake.

| stage start | why |
|---|---|
| 2026-09-07 | P1 live launch — L0 $150 |
| **2026-09-21** | **ADV-gate-off stage (`min_daily_volume 0`)** — BF has ZERO live trades under the config that boots Monday, so its stage clock starts Monday (docs/scaling_plan_2026.md). Trades, sessions and parity flags from the 9/7 stage are OUT of this window. |

`--stage-start YYYY-MM-DD` overrides the table for a one-off look back.

Rails are −5u / −7u / −8u: each sits past the backtest's worst observed
day (−4.6u), week (−5.8u) and drawdown (−7.6u), so a rail hit means live is
doing something the backtest never did. They are code-enforced
(`trading.bull_flag.kill_rails`, `trading.daily_loss_limit`) and move with
the stage — change all five numbers together.

## Advance to the next stage — ALL must hold
1. **Positive**: realized stage P&L > 0 (the ORB above-water rule).
2. **Enough trades**: ≥ 8 filled trades in the stage — OR ≥ 6 trades with
   stage P&L ≥ +4u (a strong early read; the null edge clears +4u on 6
   trades ~20% of the time, P1 does it most of the time).
3. **Enough sessions**: ≥ 15 sessions in the stage (two same-day fills like
   BNRG/BRNX 1/27 are one observation, not two).
4. **Parity clean**: every stage trade's rule decisions and exit type agree
   with the EOD BT-vs-live check (no gate/partial disagreement, no
   `exit_pending_verification`). One disagreement = fix first, then count.
5. **No rail hit** in the last 10 sessions.
6. **Above water EX-MONSTER** (2026-09-19, scaling plan Gate 2.3): stage P&L
   > 0 with the single best trade removed. One +8R trade is not a stage.
7. **Consistent with the backtest** (Gate 2.4): realized stage R/trade
   (pnl ÷ the stage base risk) inside the bootstrap [p10, p90] band of the
   BT's own per-trade R for that n — reference
   `research/bf_frequency/runs/VOL_OFF.csv` (R = pnl/$2,000), the config that
   boots Monday. Below p10 → no advance. ABOVE p90 → also no advance until n
   grows: beating your own backtest is a leak or a bug before it is luck.
8. **No open FREEZE** (Gate 1): `scripts/bf_decision_parity.py` freezes the BF
   ramp on a decision/exit-type disagreement (`logs/ramp_freeze.json`). While
   frozen the stage clock does not accrue — those sessions are excluded from
   the 15-session minimum permanently — and no ADVANCE is emitted regardless
   of P&L. Clearing is MANUAL:
   `python scripts/bf_ramp_check.py --clear-freeze bf "<reason>"`.

Items 6-8 are computed by `scripts/bf_ramp_check.py` (shared modules
`trading/ramp_bt_band.py` and `trading/ramp_freeze.py`).

## Demote one stage — ANY, immediately
- stage P&L ≤ −6u, or
- 5 consecutive losers (BT max streak 4), or
- a weekly rail hit, or
- realized R/trade below the BT's p5 band for that n after ≥ 8 trades
  (2026-09-19 — the edge is not there live).

## Pause and review together — ANY
- stage P&L ≤ −8u (the month-pause rail; the backtest's whole-history drawdown), or
- two demotions in a row, or
- a parity defect that changes a trade's outcome (the CWVX class).

## What "prove LIVE" means in calendar terms
At the backtest's 2.8 trades/month, 8 trades is ~3 months per stage; the
6-trade early read is ~2 months. The ladder is gated on trades, not on
dates — if live produces more P1 setups than the backtest's capital- and
slot-constrained book, it moves faster; if fewer, slower. Check daily:
`python scripts/bf_ramp_check.py` (decision aid — it changes no config).

## Not gates
P&L targets, calendar dates, and the 11/15 window are not gates. A stage
that is positive but under 8 trades waits. A stage that is negative waits
until it is positive again or demotes.
