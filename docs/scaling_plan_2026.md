# Scaling plan — parity converts backtest evidence into live evidence (2026-09-19)

Owner: "what is the right pattern for scaling? should include both BT parity and live proof on positive pnl."

## The principle
Two gates on two clocks. **Parity** answers "is the engine running the backtest?" and resolves in DAYS — every
session produces a miss rate, a fill drift, a selection match. **Proof** answers "is the edge real live?" and
resolves in WEEKS — it needs a P&L sample. The pattern that makes scaling fast without being reckless:

> **You are not re-proving the edge live. You are proving the engine runs the backtest. Once parity is clean, the
> backtest's 21 months COUNT as evidence about the live book, and the live P&L bar can be "consistent with the
> backtest" instead of "significant from scratch".**

Without parity, live P&L is the only evidence and at 1–5 trades a week it takes years. With parity, the BT sample
does the statistical work and live only has to confirm it is not contradicted. This is why parity is the PRIMARY
gate, why a parity breach FREEZES size rather than demoting it (a breach means we do not know what we are running —
it is not a P&L event), and why this week's parity finds (49 s entry latency, the fill model, the cost model) were
worth more than any P&L number.

## Gate 1 — PARITY (daily, per book, zero-tolerance on the hard items)
| item | ORB (daily_green_check.py) | BF (bf_decision_parity.py) | breach → |
|---|---|---|---|
| selection | every BT pick was ORDERED live (filled or canceled); no live order the BT would not place | every live entry has a matching BT detection + Stage-2 decision | **FREEZE** |
| fill | miss rate 0; fill-vs-cap within the Stage-Q band; no-fill rate within band | fill vs planned entry within the live slippage band | soft flag; 2 days → FREEZE |
| sizing | recorded mult == recomputed mult (HARD fail on drift) | recorded shares == planner recompute | **FREEZE** |
| exit | exit reason and exit path recorded; zero `exit_pending_verification`; slippage within band | trail/partial/exhaustion decisions agree with the shared spec on the same bars | **FREEZE** |
| engine | 0 tracebacks; 0 "tick raised"; latency ≤ 5 s after the bar close | same | **FREEZE** |
**FREEZE** = size stays exactly where it is, entries continue, the owner is told the same day, and the stage clock
STOPS until the breach is explained and fixed. A freeze is never a demotion: it is "unknown", not "bad".

## Gate 2 — PROOF (weekly, per book, sequential — not a fixed n)
Advance to the next stage when ALL hold, checked every Saturday:
1. **Parity clean** for the whole stage (Gate 1, no open freeze).
2. **Above water**: realized stage P&L > 0 (owner rule 2026-07-23).
3. **Above water ex-monster**: realized stage P&L > 0 with the single best trade removed. Encodes "green weeks
   over monsters" — one +8R trade is not a stage.
4. **Consistent with the backtest**: realized stage R/trade inside the BT's [p10, p90] band for that n. Below p10
   → not advancing (and see demote). ABOVE p90 → also not advancing until n grows — a live book beating its own
   backtest is a leak or a bug before it is luck.
5. **Green weeks ≥ 50%** of the stage's weeks with a trade.
6. **Minimums**: ≥ 15 sessions AND ≥ 8 trades in the stage (or ≥ 6 trades at ≥ +4u — the BF rule).
7. **No rail hit** in the last 10 sessions.
Steps are ≤ 2× (ORB) / the existing L-ladder (BF). Never skip a stage. Move every rail with the size.

Demote ONE stage, immediately, on ANY: stage P&L ≤ −6u; 5 losers in a row; a weekly rail; 4 consecutive red
weeks; or realized R/trade below the BT's p5 band for that n after ≥ 8 trades (the edge is not there live).
Pause (size to the floor, entries on, owner decides) on stage P&L ≤ −8u or a monthly rail.

## The ladders (rails move with size, always all together)
**ORB** — 8 slots fixed (D1_orb: edge gone by rank 9; 12/16 slots buy zero green weeks). The per-position cap is
the real knob (it binds on every pick; `risk_per_trade_usd` is inert). Budget = 8 × cap.
| stage | per-position cap | budget | daily / weekly rail | account share at $66K |
|---|---|---|---|---|
| **S0 (now)** | **$3,333** | **$26,667** | −$750 / −$1,050 | 40% |
| S1 | $6,667 | $53,333 | −$1,500 / −$2,100 | 81% |
| S2 | $13,333 | $106,667 | −$3,000 / −$4,200 | 162% (intraday margin) |
| S3 (full DTBP) | $21,750 | $174,000 | −$4,900 / −$6,900 | 264% |
Expected clock at ~5 picks/week: the 15-session minimum binds → ~1 month per stage if every gate holds.
Watch at S2+: the 1%-of-ADV participation cap and impact on the thin names — a binding cap is a WARNING and a
reason not to advance, not a reason to raise the cap.

**BF** — `docs/bf_p1_ramp.md` stands: L0 $150 → L1 $400 → L2 $1,000 → L3 $2,000, five rail numbers per stage.
Expected clock at ~5 trades/month (ADV gate off): ~2 months per stage. **BF has ZERO live trades under the config
that boots Monday** — its stage clock starts Monday, and its parity harness must be re-pointed at the new config
(`min_daily_volume 0`) before its first Saturday check or Gate 1 cannot be evaluated.

**Cross-book cap**: total simultaneous open risk across all books ≤ 5% of account equity (today: 15 slots × ~$150 =
$2,250 = 3.4%). Ladders are otherwise independent — one book's freeze never blocks the other's advance.

## The honest clock
If every gate holds every stage: ORB S3 in ~4 months, BF L3 in ~6 months → both full around March 2027. At full
size the two-quarter backtest path scales to roughly $17K/month IF linear — and linearity is the thing the S2/S3
participation checks exist to test. The November-15 milestone is not reachable on evidence; this plan is what
reaching it on evidence would have required.

## What this changes on Monday
Nothing in size. What changes is that a parity breach now FREEZES the stage clock (it used to only turn a streak
red), and the Saturday review scores Gate 2 items 3 and 4 (ex-monster and BT-band) which no ramp checker computes
yet — `scripts/bf_ramp_check.py` and `scripts/orb_ramp_check.py` need those two columns added.
