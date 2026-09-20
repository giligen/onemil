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

**Pooled reading (ADVISORY, added 2026-09-20 — `trading/ramp_pool.py`, both checkers print it).**
Gate-2 item 4 asks "is this book consistent with its backtest?" at each book's own frequency, and at
ORB's 6.5 trades a week a 0.2 R question needs **1.66 years** at 80 % power (BF alone: 21.8 years;
the two pooled: 1.55 — BF is 9 % of the trades). Pooling the live books with the **HOD-break dry
run's ~27 paper trades a week** takes the same question to **10 weeks**, for free, because a dry book
earns nothing and therefore risks nothing (`research/mature_method/frames9/REPORT.md` F30).

The pooled statistic is `z_i = R_i / SD_b` — each book's realized R standardised by the per-trade SD
of its OWN frozen BT reference — averaged with a **day-clustered** SE (one cluster per session across
all books: they share the session, the account and the market factor), classified against a bootstrap
of the same pooled statistic from the reference books (BELOW-p5 / BELOW-p10 / IN-BAND / ABOVE-p90).

Pre-committed constraints, asserted in `tests/test_ramp_pool.py`:
* The pooled z is a **precision** gate, never a P&L gate: it can only ever BLOCK. **The above-water
  rule (owner 2026-07-23) is inviolable** — the pool can never lift a losing book on a winning
  sibling's evidence, and a hot dry stream can never advance anything.
* The **dry stream counts toward n and the band ONLY**; the printed line names its share so a reader
  can see how much of the precision is paper.
* Intended ADVANCE form (not yet in force): the book's existing gate **AND** the pooled z is not
  BELOW-p10 — i.e. a stage that cleared on 8 trades while the portfolio runs a standard error below
  its backtest is HELD. Intended DEMOTE form: pooled z below p5 with n ≥ 30 demotes every live book
  one stage.
* **Until the owner approves switching it on, the per-book verdict remains the decision** and the
  pooled line is printed for information only.

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
Nothing in size. What changed is that a parity breach now FREEZES the stage clock (it used to only turn a streak
red), and Gate-2 items 3 and 4 are now computed rather than eyeballed. **Built and shipped 2026-09-19** (tests
first, suite green; no config, order, service or cron invocation touched):

* **Both ramp checkers print the two new columns**, read-only over `data/trades.db` (`mode=ro`):
  * `ex-monster:` stage P&L with the single best trade/fill removed, in u (BF) or % of budget (ORB), and whether it
    is above water. ADVANCE now requires it.
  * `BT band:` live mean R on n vs the bootstrap `[p5, p10, p90]` of the backtest's own per-trade R for that n
    (2,000 draws, fixed seed — reproducible), with the classification BELOW-p5 / BELOW-p10 / IN-BAND / ABOVE-p90 and
    the rule printed next to it. ADVANCE requires IN-BAND; DEMOTE now includes BELOW-p5 after ≥ 8 trades.
    R is `pnl ÷ base risk` (BF, vs `research/bf_frequency/runs/VOL_OFF.csv` at its $2K normalization) and
    `pnl ÷ total_risk` (ORB, vs `pnl_pct ÷ range_size_pct` in the book). The ORB reference FOLLOWS the running
    config and is named in the output: `research/orb_gates2/book_G3_meas.csv` while the catalyst veto is OFF (the
    variant that boots Monday), `research/fuckup_audit/Q_fill/book_measured_n8.csv` when it is ON.
    A missing/unreadable reference reads NO-DATA and blocks ADVANCE — an unscored gate is never a passed gate.
  * Shared modules: `trading/ramp_bt_band.py`, `trading/ramp_freeze.py` (one spec, both checkers).
* **Parity FREEZE is state, not a memo**: `logs/ramp_freeze.json`, keyed by book, `{frozen, since, reason, by,
  frozen_dates, history}`. `scripts/daily_green_check.py` freezes ORB on any HARD parity fail it already detects
  (recorded-vs-recomputed mult drift, BT pick never ordered, fill-parity, unattributed exit,
  `exit_pending_verification`, composite drift, floored-stop drift); `scripts/bf_decision_parity.py` freezes BF on a
  decision/exit-type disagreement (BT_ONLY / LIVE_ONLY / exit_reason mismatch / pnl sign flip — fill drift stays the
  soft flag the table above makes it, and a BT_STALE day never freezes). A `[RAMP FREEZE]` Telegram goes out on the
  first freeze via `scripts/send_telegram_alert.py`. Both ramp checkers print `FROZEN since <date>: <reason>`,
  refuse to emit ADVANCE regardless of P&L, and EXCLUDE frozen sessions from the stage clock — permanently, so a
  cleared freeze never retro-credits the 15-session minimum. Clearing is MANUAL and logged (who/why):
  `python scripts/{bf,orb}_ramp_check.py --clear-freeze {bf|orb} "<reason>"`.
* **BF parity harness re-pointed at the config that boots**: `bf_decision_parity.py` read its Stage-2 sizing from a
  hardcoded `--risk 60` while live had ramped to $150; it now reads `config.yaml` at RUN TIME (capital, risk,
  max_shares) and prints the gate values it judged against (`trading.enabled`, `scanner.min_daily_volume`,
  `conviction_scoring.min_threshold`) so a config change is visible in the report. The Stage-2 subprocess already
  re-reads the gates itself, so the whole harness now follows the live config with nothing cached.
