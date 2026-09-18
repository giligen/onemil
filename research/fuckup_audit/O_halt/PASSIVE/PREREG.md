# Stage O / S1-PASSIVE — pre-registration

Written **before any run**, 2026-09-18. Frozen. Gates and phrasing per `research/fuckup_audit/PLAN.md` §1.

## 0. Why this exists

`O_halt/REPORT.md` §5 found ONE rule — *short the first post-resume bar when the reopening print is at
or above the pre-halt price* — reading **+0.530 / +0.996 / +0.904 net R** (TRAIN/VAL/TEST) at ~16
trades/week, 20 of 21 months green. §6 killed it: the tested fill convention is a **marketable sell at
the next bar's open**, and `is_short_sell_restricted = Y` on **76%** of the booked trades. Reg SHO
rule 201 forbids a short sale *at or below* the national best bid while SSR is in force.

Rule 201 does **not** forbid a short **limit** order priced **above** the current NBB. That order is
real and executable on Alpaca. It was never tested. This stage tests it.

**Status of the population and of TEST.** Same events, same universe screens, same splits as O_halt.
TEST was already read once in O_halt for these two cells. This is a **NEW FILL MODEL on the same
population**, not a new population. Therefore: **TRAIN and VAL decide. TEST is reported as
confirmatory only, and only if G2 passes.** No FREEZE.md exists at the `fuckup_audit` root; this file
is the freeze.

## 1. Population (inherited, unchanged)

`O_halt/trades.parquet` — 2,550 resumed LULD halts (Nasdaq-listed, prev close ≥ $5, ADV20 ≥ 100K,
test tickers and non-`daily_bars` symbols removed, causal daily screens). Splits fixed:
TRAIN 2025-01-01..2025-12-31 (n 1,449), VAL 2026-01-01..2026-05-31 (625), TEST 2026-06-01..2026-09-17 (476).

**Admission (the rule, unchanged from the survivors):** an event is a candidate iff the **reopen print**
— the OPEN of the first 1-minute bar starting strictly after the resume message, `fill` in the parquet —
satisfies `reopen >= ref * 0.994`, where `ref` is the last trade before the halt. 1,445 of 2,550 events
(TRAIN 826 / VAL 355 / TEST 264). This is the union book of cells A and B; both sides are SHORT, so
the 0.6% no-chase cap is a floor. Side is retained for reporting only — it is one book.

## 2. The fill model under test — "passive short limit"

At the reopen (decision instant = the entry bar's open timestamp `entry_t`, the first instant the reopen
print exists):

```
tick  = 0.01                                  (every name here is >= $5)
NBB   = the national best bid from the LAST Alpaca SIP quote at or before entry_t
limit = max(NBB + tick, reopen * (1 + b))     for b in {0.000, 0.005, 0.010}
```

Three b cells. **No other threshold, band, side split, sizing or horizon variant is added.**

Two fill arms, both reported, both counted as cells (3 x 2 = **6 cells** for the permutation):

* **touch** (primary): filled iff some 1-minute bar with `high >= limit` exists in the window
  `[entry_t, resume_ts + 5 min]`. The entry bar itself is included — its open is the decision input and
  its high is later in time than its open. Fill price = `limit`. This is the conservative convention for
  a **resting limit on the offer side**: at the touch, our order is at the top of the book queue only if
  it was already resting, which it is.
* **strict** (the harsher arm): filled iff some bar in the same window has `open >= limit`; fill price
  = `limit`. A bar that opens through our limit fills it for certain.

Unfilled candidates are **no trades** (0 R), not losses. Fill rate is reported per cell per split.

**SSR is not a filter.** Under SSR the order is legal because it rests above the NBB; under no-SSR the
same order is placed. ONE rule. The SSR = Y / N split is reported as a descriptive cut, never as a cell.

## 3. Exit

Horizon **+5 minutes** — the surviving cells' primary; `+30m` and `close` are reported as secondary
descriptives and are **not** counted as cells or gated.

**Buy-to-cover at the OPEN of the bar following the horizon bar** (marketable, legal for a cover —
Rule 201 restricts short sales, not purchases). Horizon bar = `fill_bar + 5 min`, truncated at 15:55;
if no bar follows, the last available close is used and the event is flagged. The O_halt convention
(cover at the horizon bar's CLOSE) is also computed and reported side by side.

**No stop.** As in O_halt, R is declared flat at **2.0% of the fill price**; it is the scale only. Raw %
is reported next to net R so the two are separable. A live short needs a stop; its absence is a
disclosed gap, identical to O_halt.

## 4. Cost — MEASURED, not the band table

Per trade, from **Alpaca SIP** quotes (the P_cost path, `P_cost/fetch_spreads.py` conventions reused,
not rewritten): the NBBO at the **resume minute** (last quote at or before `entry_t`) and at the
**cover instant**. Spread summarised as the **MEAN** over the minute, per the brief, not the median.

Cost contract (declared, conservative):

```
half_entry = 0.5 * entry_spread_pct  / 2.0     (R = 2.0% of fill)
half_exit  = 0.5 * exit_spread_pct   / 2.0
net_R = raw_pct / 2.0  -  0.0 * half_entry  -  1.0 * half_exit
```

* entry charge **0**: the entry is a resting limit ABOVE the bid; it is hit, it does not cross. Any
  price improvement it earns is **not** credited either.
* exit charge **1.0 half-spread**: the cover is marketable and lifts the offer.

Also reported for comparability: the O_halt charge (`0.25 * half_entry + 0.875 * half_exit`).

**Missing quotes.** If the entry-instant NBBO is unavailable the event is **excluded and counted**
(it also has no limit price, so it cannot be simulated). If the cover-instant quote is unavailable the
entry spread is used for the exit charge and the event is flagged.

## 5. Borrow

An event is `tradeable` iff Alpaca's asset record for the symbol says **`shortable` AND
`easy_to_borrow`** (read-only `get_all_assets`). **Survivorship caveat, stated up front: this is
TODAY's flag, 2026-09-18, not the flag on the halt date.** Borrow status for a $5-10 Nasdaq microcap
minutes after a volatility halt is the single most volatile executability field in the whole study, and
today's snapshot is a weak proxy for it. Reported: the share of events that pass, and the book scored
on **both** the tradeable subset and the full set.

## 6. Gates (PLAN §1)

* **G1 TRAIN**: mean net R > 0 and **t >= 2.0**.
* **G2 VAL**: mean net R > 0, t >= 1.0, and **>= 55% of weeks green**.
* **G3 TEST**: read once, reported as it comes — **confirmatory only**, because TEST has already been
  read once on this population. A TEST number never promotes a cell here.
* Economic bar (reported, not gated): expected R/week and $/month at $100 and $375 risk.
* Every cell: **MDE = 2.8 x SE** per split; **ex-top-5%**; **winners capped at +3R**; per-month table.
* **Search-adjusted permutation** on TRAIN: symmetric sign-flip null, B = 2,000, max |t| across all
  **6** cells.
* **Availability audit**: the limit price uses only `reopen` (entry bar open) and the NBB at or before
  `entry_t`; both are shown to be computable at or before the decision instant, with a missingness
  table per split.

## 7. Capacity

Shares = 1% of the **resume-minute volume** (the entry bar's volume). $/month at $100 and $375 risk
computed from the per-trade net R x risk x trades/month, capped by the share limit x fill price where
that binds.

## 8. Cells looked at

**6** in this stage (3 b x 2 fill arms). Cumulative for the S-series after this stage: 12 (O_halt) + 6 = **18**
of the 26 pre-declared in `lit_review_2026/DATABENTO_OPPORTUNITIES.md` §B. The +30m / close horizons and
the SSR / borrow cuts are descriptives, not cells, and are not gated.

## 9. What a PASS would and would not mean

A pass would mean: the measured edge survives a fill convention Reg SHO permits. It would **not** mean
shippable. The blockers that remain regardless of this stage's result, from `O_halt/REPORT.md` §7:
no halt ingestion exists in the repo, `subscribe_trading_statuses` is unused, no short-sell path exists
in `trading_engine.py`, and the **off-hours Alpaca `statuses` websocket check is still unrun**. Those are
a BUILD, and this stage does not start it.
