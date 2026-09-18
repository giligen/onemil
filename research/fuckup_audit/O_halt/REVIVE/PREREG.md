# Stage O / S1-REVIVE — pre-registration

Written **before any scoring**, 2026-09-18. Frozen. Gates and phrasing per `research/fuckup_audit/PLAN.md` §1.
Supersedes nothing: `O_halt/REPORT.md` (12 cells) and `O_halt/PASSIVE/REPORT.md` (6 cells) stand as written.

## 0. Why this exists

PASSIVE established that the LULD reopening print **overshoots** — gross +0.319 / +1.156 / +1.072 R
(TRAIN/VAL/TEST) at b=0/touch — and that the book dies **net**, because the measured NBBO at the resume
is a median 1.9% of price against a 1.27% TRAIN breakeven. Owner 9/18: *"smells like there can be more
options to revive halt-resume, play with levels Rs etc."* He is right that the closure was premature:
PASSIVE varied ONE thing (the short limit offset) and left the R scale, the exit construction, the
spread as a **selection** variable, and **the entire long side** untested at measured cost.

## 1. THE FILL MODEL (owner correction, 9/18 — this is the contract for every cell)

**We control the entry price.** The entry is a LIMIT order at a price we choose. A crossing/quoted
spread is therefore **NEVER** charged on entry.

```
entry:      a resting LIMIT at a price we declare before the resume prints
fill:       iff the market trades at or through OUR limit inside the declared window
no fill:    NO TRADE. Exactly 0 P&L. Counted and reported, NEVER booked as a loser.
entry cost: ZERO spread. The honest entry-side cost is ADVERSE SELECTION + OPPORTUNITY COST,
            and both are MEASURED, per rung:
              (a) fill rate
              (b) mean R of filled trades vs the full signal population
              (c) what the NON-FILLS would have done (the missed-winner cost), scored on the
                  same exit construction from the reopen print
```

The 1.9% median NBBO is therefore **a fact about the limit-placement problem** (how far from the print
our limit must sit to get filled), **not a per-trade fee**.

**Audit of the prior stages under this model, stated here so the decomposition is re-derived, not
re-used:** PASSIVE's `score.py` already charged `0.0 * half_entry` — its entry was already free, and
its −0.840 R is **entirely the marketable COVER** (`1.0 * half_exit`). O_halt's published +0.530 charged
`0.25*half_entry + 0.875*half_exit` at a 0.40% band constant, i.e. it did charge a (small) entry
spread. The A→B step ("−1.05 R from measuring the spread") is therefore **not** an entry-cost step; it
is 0.25 of an entry half-spread plus 0.875 of an exit half-spread, and the exit term is ~93% of it.
REVIVE re-derives the decomposition with the entry term forced to zero everywhere.

**Exit contract** (L3 — passive first, marketable only as fallback):

```
bracket on the fill:  stop = 1R adverse,  target = 2R favourable,  else horizon exit
target  = a resting LIMIT   -> charged 0.0 half-spread
stop    = marketable        -> charged 1.875 half-spreads (1.0 crossing + the score4 0.875 stop excess)
horizon = marketable        -> charged 1.412 half-spreads (1.0 crossing + the score4 0.412 eod excess)
half-spread in R units      = 0.5 * spread_pct / R_pct
tie inside one bar (stop and target both touched): STOP first, always (conservative)
```

**Spread source.** Cost uses the **MEASURED** Alpaca SIP mean NBBO over the resume minute
(`entry_nbbo.csv::mean_spread`), as PASSIVE declared. It is used as the proxy for the exit instant;
PASSIVE §6 measured entry spreads slightly WIDER than cover spreads, so the proxy is conservative, and
its bias is re-validated on the `cover_nbbo.csv` overlap and reported. The **GATE** (L1) uses
`entry_nbbo.csv::spread` — the last quote at or BEFORE `entry_t` — because the minute-mean is computed
over `[entry_t, entry_t+60s)` and is **not** observable at the decision instant. This distinction is
load-bearing and is asserted in code.

## 2. Population (inherited, unchanged)

`O_halt/trades.parquet` — 2,550 resumed LULD halts (Nasdaq-listed, prev close ≥ $5, ADV20 ≥ 100K, test
tickers and non-`daily_bars` symbols removed, causal daily screens). Splits fixed: TRAIN 2025-01-01..
2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-17.

* **SHORT book** (inherited): `reopen >= ref * 0.994` — 1,445 events (TRAIN 826 / VAL 355 / TEST 264).
* **LONG book** (NEW, never scored at measured cost): `reopen <= ref * 1.006` — 1,290 events
  (TRAIN 731 / VAL 318 / TEST 241). The two overlap on 185 events inside the ±0.6% band, by
  construction; they are **separate books**, never summed.

Side (up-halt / down-halt) is a **descriptive cut**, never a cell — O_halt's four "cells" per side were
one rule each, as its §5 established.

## 3. Entry limit rungs

```
SHORT:  limit = max(NBB + 0.01, reopen * (1 + b)),  b = 0      (PASSIVE settled the b-ladder:
        fill iff a bar HIGH >= limit in [entry_t, resume_ts+5m]  net R falls monotonically in b on
                                                                 all 3 splits, both arms — b=0 is it)
LONG:   limit = min(NBO - 0.01, reopen * (1 + d)),  d in {0, -0.005, -0.010, -0.020}
        fill iff a bar LOW <= limit in [entry_t, resume_ts+5m]
```

Fill price = the limit in both directions (never the better print). The window is the same 5 minutes
PASSIVE used. Unfilled = 0 R, counted, and their counterfactual scored (§1(c)).

## 4. R and the levels (L2)

`R_pct` ∈ {2%, 6%} of the FILL price in the gated cells; {2, 4, 6, 8} in the TRAIN-only lever table.
Stop and target are both scaled to it (§1). **Declared in advance, because it disciplines the reading:
under a pure horizon exit with no stop/target, re-scaling R is SIGN-NEUTRAL** — gross and cost both
scale as 1/R_pct, so net R scales as 1/R_pct and cannot change sign. R can only bite through the
stop/target levels and through the fact that a target fill is passive (0 half-spread). `spread/R` is
reported per cell; that ratio is the binding quantity of `cost_curve.md`'s contract.

## 5. Horizon (L4)

{+5m, EOD} in the gated cells; {+5m, +30m, EOD} in the lever table. PASSIVE's `netR_eod` swung
+0.129 (TRAIN) vs +2.33 (TEST) against +5m's −0.84 — **that swing must be EXPLAINED, not reported**:
the per-month and top-name decomposition of the EOD arm is a required output.

## 6. The 24 pre-declared cells

| group | side | entry rung | spread gate (L1) | R_pct | horizon | cells |
|---|---|---|---|---|---|---|
| A | SHORT | b = 0 | {none, ≤1.25%} | {2%, 6%} | {+5m, EOD} | 8 |
| B | LONG | d = 0 | {none, ≤1.25%} | {2%, 6%} | {+5m, EOD} | 8 |
| C | LONG | d ∈ {−1%, −2%} | none | {2%, 6%} | {+5m, EOD} | 8 |

**24 cells. Nothing else is gated.** The 1.25% gate is the primary threshold because it is PASSIVE's
measured TRAIN breakeven (1.27%), rounded down — it is not chosen from the data.

**Declared sensitivity arms, on the best-by-TRAIN cell of each side ONLY (2 cells), TRAIN+VAL,
reported and counted as looks, never gated:**
* **L3**: exit target passive (the contract) vs everything marketable — fill rate of the target leg.
* **L5**: price ≥ $10 · price ≥ $20 · ADV20 ≥ 500K · ADV20 ≥ 1M.

**TRAIN-only lever table (descriptive, one lever at a time from the base cell):** L1 gate
{1.25, 0.75, 0.50, 0.25}% · L2 R {2,4,6,8}% · L3 {passive, marketable} · L4 {+5m,+30m,EOD} · L5 the
five floors · L6 the long rung d = −0.5%. These are **looks and are counted**; they are not cells and
never promote anything.

## 7. Gates

* **G1 TRAIN**: mean net R > 0, **t ≥ 2.0**, **≥ 3 trades/week**, **mean net R ≥ +0.15**.
* **G2 VAL**: same sign, **≥ 55% of weeks green**, trades/week holds (≥ 3).
* **G3 TEST**: read **ONCE**, behind `REVIVE/FREEZE.md`, **for G2 survivors only**. TEST has already
  been read on this population twice (O_halt, PASSIVE). It is **confirmatory only and never promotes**.
* Every cell: **MDE = 2.8 × SE** per split · **ex-top-5%** · **winners capped at +3R** · per-month table.
* **Permutation**: symmetric sign-flip null, B = 2,000, max |t| across MY 24 cells, on TRAIN.
* **Bonferroni / BH** across the **cumulative** cell count for this population.
* **Availability audit** on every gating field, per split.
* **Borrow** share reported per surviving SHORT cell (Alpaca `shortable AND easy_to_borrow`, today's
  flag — survivorship caveat as in PASSIVE).
* **Capacity**: shares at 1% of the resume-minute volume; $/month at $100 and $375 risk.

## 8. Cells looked at

24 gated in this stage + 18 TRAIN-only lever looks + 8 sensitivity-arm looks = **50 looks**.
Cumulative for this population: 12 (O_halt) + 6 (PASSIVE) + 50 = **68**.

## 9. Capacity asymmetry — declared before the result

A SHORT survivor is **capacity-dead**: PASSIVE measured Alpaca `shortable AND easy_to_borrow` on only
**8.1 / 6.6 / 3.3%** of the booked short trades, and median capacity of $4,843 notional already binds
at $100 risk. A LONG survivor is **instantly 100% tradeable** — no borrow, no Reg SHO, no locate. This
asymmetry is stated in the report regardless of which side wins.

## 9b. Amendments, 2026-09-18, folded in BEFORE any cell was scored

**(i) Stale-quote quarantine** (from the parallel `AUDIT_COST_REVIVAL` finding that PASSIVE's mean
cover spread of 4.63% is outlier-driven — max 474% of price, 16.7% of cover quotes older than 30 s).
A quote is QUARANTINED, at either the entry or the exit instant, if it is **older than 30 s** at the
instant it is used, or if its **spread exceeds 50% of price**. That is a broken print, not a cost.
The dropped share is reported per split, and every number in this stage is scored after the drop.

**(ii) Three charges, always reported side by side.** `pt` = each trade charged its own measured
exit-instant spread · `w95` = the same, winsorised at the split's p95 spread · `med` = every trade
charged the split's MEDIAN spread. Stage P made the mean-vs-median argument in the OPPOSITE direction
for ORB; the program has now been bitten by that choice twice, so the sensitivity is shown rather
than a winner picked. Any decision that flips between the three is named.

**(iii) Bar B — the live-exploration bar** (owner 9/18: *"we can always run if there's an edge"*). A
research gate decides whether we may CLAIM an effect; it must not silently decide whether to risk
bounded money. Every cell is therefore scored against BOTH bars:
* **Bar A** = G1/G2/G3 above — the claim bar.
* **Bar B**, all four required: (1) positive point estimate on TRAIN **and** VAL under the measured
  cost/fill model (sign, not significance); (2) a stated mechanism, with the cell declared ex ante or
  labelled a grid maximum; (3) bounded downside — $100 risk, the rails in §9c, a written demote/kill
  rule; (4) **resolution within one quarter**: `weeks = (SD/mean)² / trades_per_week ≤ 13`, computed
  at the cell's OWN frequency, and for a SHORT cell at its **borrow-constrained** frequency.

## 9c. Bounded-downside spec for any Bar-B cell (declared, not proposed)

Risk $100/trade · daily kill −$300 · weekly kill −$600 · demote to paper on −$600 in a week or 6
consecutive losers · kill on −$1,000 cumulative or on a parity break between the live fill and the
spec. No BF/ORB slot is taken (both are paused); the halt book would need its own account isolation.

## 10. What a PASS would and would not mean

A pass promotes nothing to production. The engine gaps from `O_halt/REPORT.md` §7 are untouched: no
halt ingestion exists in the repo, `subscribe_trading_statuses` is unused, and **the off-hours Alpaca
`statuses` websocket check is STILL UNRUN**. An independent rebuild from prose, trade by trade on
(day, symbol), remains required before a line of engine code.
