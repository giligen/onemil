# orb_inplay — PREREG

Pre-registered replication of **Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy
for the U.S. Equity Market"** ("stocks in play" opening-range book) on our own data, our own cost
model, our own splits. Written BEFORE any scoring. Cell count for the programme: **1,280–1,282**
(three cells added here: long side, short side, combined book).

## 1. The rule (verbatim spec, no discretion)

Each trading day, at **09:35 ET**:

1. **Universe** — every symbol with a `daily_bars` row for the prior session, prior close **≥ $5**,
   and **ADV20 ≥ 1,000,000 shares** (mean `volume` over the 20 prior daily bars). This is the WHOLE
   market, NOT the ORB gap-up seed. 2x/inverse wrappers are IN (the 9/5 universe rule) and flagged.
   Universe size per day is reported.
2. **Relative volume** — `RVOL = volume(09:30–09:34 inclusive, 5 one-minute bars) /
   mean(volume of the same five minutes over the prior 14 sessions)`. A name needs ≥ 10 of those 14
   prior sessions present, else it is not rankable (counted).
3. **Stocks in play** — rank by RVOL descending, keep the **top 20** with **RVOL ≥ 1.0** (the paper's
   own floor; stated here because the paper is not explicit — we use 1.0).
4. **Direction** — from the 09:30–09:34 five-minute candle (open of 09:30, close of 09:34):
   close > open → **LONG**; close < open → **SHORT**; exactly equal → **skip** (doji).
5. **Entry** — the **09:35 bar open** (the first price after the decision is knowable; no touch fills,
   no look-ahead into the 09:35 bar). If the name has no 09:35 bar, it is a missing fill (availability
   rail), not a skip.
6. **Stop** — entry ∓ `0.10 × ATR14`, ATR14 = Wilder-free simple mean of the 14 prior **daily** true
   ranges (all data ≤ prior close).
7. **Target** — entry ± `10 R`, R = the stop distance.
8. **Intrabar resolution** — walking 09:35 → 15:55 one-minute bars in order. If a bar's low ≤ stop
   (long) the stop fills AT the stop price; if high ≥ target the target fills AT the target price.
   If BOTH are inside the same bar, the **STOP** is taken (conservative). Gap-through: fill at the
   bar's open if the open is already beyond the level.
9. **Time exit** — the **15:55 close** if neither level is hit (the paper uses the close).
10. **Sizing** — risk **1 % of equity** per trade: `shares = floor(0.01 × equity / R)`. Equity is the
    **static $66,000** book (no compounding within a split; compounding is reported separately only
    as the annualised figure). Notional across all open positions capped at **4 × equity**
    ($264,000); when the day's 20 candidates would exceed it, candidates are admitted in RVOL rank
    order until the cap binds and the rest are dropped (counted). The **unlevered 1 × book**
    ($66,000 gross cap) is reported alongside as the primary dollar number.
11. **Reg SHO 201** — a short candidate whose 09:35 entry price is **≥ 10 % below the prior daily
    close** is EXCLUDED (the circuit-breaker uptick rule makes a market short unfillable at the
    open). Their share of short candidates is reported.
12. **Borrow** — assumed available for every name at price ≥ $5 / ADV ≥ 1M. This is an assumption,
    not a measurement; it is listed in the caveats.

## 2. Costs (charged on BOTH legs)

- **$0.0035 / share** commission, both legs (the paper's assumption).
- **Half-spread**, both legs, in dollars per share:
  - **Measured** where we have NBBO: Alpaca **SIP** consolidated quotes over the entry minute and the
    exit minute, half-spread = `mean(ask − bid) / 2` (the `frames16/nbbo.py` / `frames17/nbbo17.py`
    method). EQUS.MINI quote schemas are never used.
  - Measuring every leg of a ~7,000-trade book is out of budget. **Pre-committed sampling rule:** a
    **stratified random sample of ≥ 600 legs** (strata = price band × leg clock) is measured from
    TRAIN+VAL; from it we fit a **price-band × leg-clock median half-spread-as-%-of-price table on
    OUR OWN population**, and impute every unmeasured leg from it. The **imputed share** is reported.
  - The `frames14/f45_minute_table.csv` minute-of-day table is NOT used as the primary source: its
    population is thin bull-flag movers, its declared validity window is 09:37–14:01, and our
    universe (ADV ≥ 1M) is far more liquid — using it would be a known-biased over-charge. It IS
    reported as a **sensitivity upper bound** (book re-scored with its 09:35 / 15:55 rows).
- No slip is double-charged: entries and exits are bar prices, not already-slipped fills.
- Auction prints are not charged a quoted spread — the 09:35 open and the 15:55 close are continuous
  session bars, not auctions, so the charge applies.

## 3. Splits (TEST SEALED)

| split | window | note |
|---|---|---|
| TRAIN | 2025-01-01 .. 2025-12-31 | both halves (H1/H2) reported separately |
| VAL   | 2026-01-01 .. 2026-05-31 | the pass bar is scored HERE |
| TEST  | ≥ 2026-06-01 | **SEALED — never queried in this run** |

## 4. Pass bar (pre-committed, scored on VAL, combined book)

1. Net **≥ +0.10 R / trade** after all costs.
2. **Day-clustered t ≥ 2.0** on net R/trade.
3. **Tail** — see Amendment 1 below.
4. **TRAIN halves same-signed** net R/trade.
5. **Green-week share** above the count-matched null (null = weeks green under a random-sign
   bootstrap of the same trade counts, 2,000 draws; report the null's mean and the p-value).
6. **Unlevered (1×) annualised return > 0 on BOTH splits**, with MDD reported.

Each side (long, short) is reported separately **against the same bar**; the combined book is what
the bar formally judges.

### Amendment 1 (owner, pre-scoring)
Item 3 of the pass bar was "ex-top-5 % ≥ 0 on both splits". The owner replaces it, before any
scoring, with a **winner-frequency** rule — a fat right tail is acceptable provided the big winners
are not once-a-quarter events:

- **(a)** the book is still **net positive with the top 1 % of trades removed**, on both splits;
- **(b)** a winner **≥ +3 R occurs at least once in every rolling 4-week window** of each split
  (report the **longest gap in weeks** between ≥ +3R winners);
- **(c)** the **top 5 trades carry < 50 %** of the split's P&L.

ex-top-5 % and the winner-capped book are still REPORTED, as diagnostics only — no longer pass/fail.
Everything else in this PREREG stands unchanged.

## 5. Availability rail (VOID conditions)

- 1-min bar coverage of the selected top-20 must be **≥ 80 %** per split, else the split is VOID.
- The **winner/loser missingness gap** must be ≤ 5 pp (a name-day whose bars are missing cannot be
  scored; if missing name-days are systematically the movers the book is unscoreable). Reported.
- A name-day that qualifies for the top 20 but has no `intraday_bars_1min` rows counts against
  coverage; it is NOT silently replaced by rank 21.

## 6. Reported alongside every number

MDE (the smallest net R/trade this sample could have detected at 80 % power, two-sided α = 0.05,
from the realised per-trade SD), **iid AND day-clustered SE**, fills/day, win rate, avg win / avg
loss, the top-5 trades' share of P&L, and the dollar book at **$66,000 equity at 1 × and at 4 ×**
(total, per week, MDD).

## 7. Phrasing

A null here is a claim about THIS test: this universe, this horizon, this book size, this window,
this cost model. The MDE is printed beside it. TEST stays sealed unless VAL clears the bar.
