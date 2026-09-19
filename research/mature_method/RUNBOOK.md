# The mature method — one runbook, applied to every closed book (written once, 2026-09-19)

Owner 9/19: "the fact that orb and bf are live now with positive numbers is ONLY because I pushed and insisted …
if you go deep on the other strategies you tested but really deep with current goals, you'll be able to find
others that actually survive."

The structural point is right: ORB and BF were judged by the method BELOW; every other book was judged by the
method we had BEFORE this week (R/trade or total P&L, the band cost table, touch fills, no gate map, no week shape).
That is an unfair comparison and this runbook removes it. What it does NOT assume is the conclusion — a book that
is gross-negative with adequate power after this pass is dead, and saying so is the deliverable.

## The ten steps, in order, every one mandatory
1. **Reproduction gate.** Reproduce the book's last honest reference to the cent before quoting anything.
2. **Gross before net.** The gross R/trade on the honest population with NO cost. If gross ≤ 0 with MDE below the
   effect the book would need, stop: no cost or fill model revives a gross-negative book (audit 9/18: 0 of 30 did).
3. **Measured cost, never the band.** Per-trade NBBO at the decision minute (Alpaca SIP quotes, or the trades.db
   telemetry for live books). The band table was wrong in BOTH directions (too wide for liquid names, too narrow
   for halted microcaps) — it is a hypothesis, not a cost. Auction fills pay no quoted spread.
4. **The engine's real fill model.** A limit at OUR price: fills iff the market comes to it, non-fill = 0 P&L, never
   a loss. Then MEASURE the unfilled counterfactual: is a fill "a better price on the same setup" (a chase guard —
   ORB, good) or "the setup broke and that is why it filled" (a dip-buy — halt-resume, −2.03R on fills vs +0.63R
   on non-fills)? Report both populations.
5. **Gate-separation map.** Every gate, at its cascade position: kept-minus-rejected R/trade, n each side, t, per
   year and pooled. A gate wrong-side or insignificant in BOTH years is a candidate for removal (BF: 2 of 6; ORB:
   2 of 7). Report median position risk kept vs rejected — the sizer can hide a wrong-side gate.
6. **Frequency frontier.** Single-gate ladders, then declared combined points, from the shipped rate up to the
   structural ceiling (all gates off). State the ceiling and the raw detector's gross R at it.
7. **Rank on the owner's metric.** PRIMARY = % green weeks over every market week (no-trade = flat, in the
   denominator). Then longest red streak, worst week, % green months, MDD. Total P&L is TERTIARY. Monsters are
   permitted if green weeks dominate; ex-top-1%/5% are reported diagnostics, never rejection reasons.
8. **Week-by-week at LIVE sizing, dollars beside every ratio.** F7's lesson (9/19): identical green-week share,
   +$1,254 vs −$667 — a ratio without the dollar path lies. Print the weekly table for the last two quarters.
9. **Count-matched permutation null.** Shuffle each cell's own P&L across its own weeks (2,000 draws), pick count
   held fixed. Green weeks that sit inside that band are pick COUNT, not skill — say which.
10. **Both bars, then the adequacy review.** Claim bar: G1 t ≥ 2 TRAIN, G2 VAL sign + ≥ 55% green weeks, TEST once
    behind FREEZE.md. Live-exploration bar: positive point estimate on green weeks AND dollars at live size,
    a stated mechanism, bounded downside with a pre-committed stop, resolution inside a quarter at the book's own
    frequency. Before any closure answer in writing: did we test what the book actually IS, is the cost and fill
    model right for its venue, does any caveat in our own report explain the headline, and what is the MDE.

## Splits, rails, deliverable (identical for every candidate)
Splits as the book already uses; TEST sealed until a recommendation is committed. Availability audit on every
field; test tickers and non-`daily_bars` names out via `research/scripts/pit_listings.py`; independent rebuild
from prose before any engine is touched. Cells declared in PREREG.md BEFORE scoring, counted, permutation across
them. One-page REPORT.md with: the gross-vs-net line, the gate map, the frontier on green weeks with the weekly
dollar table, the null band, both bars, and ONE of: SHIP TO DRY / SHIP TO LIVE-SMALL / STAY DEAD with the adequacy
review answered. A survivor goes to a DRY RUN first (free forward data), then live-small under the scaling plan.

## Candidates, in the order they run (one heavy job at a time on this node)
| # | book | infrastructure | trades/wk | judged so far by | why this order |
|---|---|---|---|---|---|
| 1 | **HOD-break** | engine, dry run LIVE, parity 0/37, 4 sessions of forward data | 25–30 | R/trade + a 12-cell causal filter; NEVER green weeks, NEVER a gate map (28% r_min rejects flagged WATCH, never measured) | most infrastructure, highest frequency, forward data accruing daily |
| 2 | **QQQ noise band (M6)** | none; index ETF | ~5 | "marginal ≈$5K/yr, SR 0.99" — dismissed on P&L; never green weeks | the ONLY July survivor; no borrow, no capacity limit, auction execution, additive to both live books |
| 3 | **Red-to-green (F6-PDR)** | engine (`HodBreakEngine(book='red_to_green')`), spec, tests | ? (PDR-gated) | killed on ZVZZT + a level bug + the scan rule; first-break was +0.06/+0.21/−0.04; never green weeks, no gate map | engine exists; the kill was two bugs and one rule choice |
| 4 | **MACD wave** | engine, 18 live trades (−$17K) | ? | filters tuned in-sample in 2026-03; never re-examined | has an engine and live data; weakest prior |
| 5 | bf_zero2 families | none; populations DELETED (SIP tape kept, ~3 h rebuild) | 1–3K/config | best +0.007R net on the band cost; 12 of 52 positive with the entry leg zeroed, best t 1.68 | only if 1–4 are dry: rebuild cost is real |
Not re-opened without new evidence: halt-resume (77 looks, gross ≈ 0 under the honest limit, short side
capacity-dead), the multi-day families (the prize is $25–65/month at our size — structural), ignition (day-cohort
look-ahead; 19 live trades).

## Division of labour (owner 9/16, 9/19: conserve Fable)
Fable wrote this file and reads only each candidate's final table. Opus executes the ten steps per candidate from
this runbook and the candidate's own prior reports, and reports ≤ 18 lines. No candidate ships anything; a
SHIP-TO-DRY verdict is a config flip the owner approves.
