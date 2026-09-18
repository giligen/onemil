# Stage O / S1-PASSIVE — the passive short limit, and the measurement O_halt never made

Pre-registered in `PREREG.md`, frozen before any run. Gates and phrasing per `research/fuckup_audit/PLAN.md` §1.

**STATUS: 0 of 6 cells clears G1. G2 was never reached. The passive short limit is NOT the problem —
it is a small improvement over the marketable fill. What kills the book is the cost measurement
`O_halt/REPORT.md` §5.3 flagged and never made: the measured NBBO spread at the LULD reopening is a
median 1.4–2.0% of price, 4–5x the 0.40% band constant the original result was scored on, and the
book breaks even at 1.27% on TRAIN.**

---

## 1. What was tested

Reg SHO rule 201 forbids a short sale at or below the NBB; it does **not** forbid a short **limit**
priced above it. O_halt's killed result used a marketable sell at the next bar's open, unobtainable on
the 76% of its trades that were SSR-restricted. This stage replaces that single convention:

```
at the reopen:  limit = max(NBB + $0.01, reopen_print x (1 + b))     b in {0, 0.5%, 1.0%}
fill window:    [entry bar, resume + 5 min]
  touch  arm:   a bar's HIGH reaches the limit    (primary)
  strict arm:   a bar's OPEN is at/through it     (harsher)
fill price:     the limit, in both arms (never the better print)
cover:          the OPEN of the bar after the +5m horizon bar — marketable, legal for a cover
cost:           MEASURED Alpaca SIP NBBO, mean over the minute; 0 half-spreads on the passive entry,
                1.0 on the marketable cover
```

Same population, same universe screens, same splits, same admission rule (`reopen >= ref x 0.994`) as
O_halt. **1,445 candidates**; 12 (0.83%) excluded for a missing entry NBBO and counted in
`excluded_no_quote.csv`; **1,433 scored**.

## 2. Fill rate — the passive limit fills

| b | touch | strict |
|---|---|---|
| 0.0% | **94.0%** | 83.5% |
| 0.5% | 81.2% | 47.7% |
| 1.0% | 76.7% | 44.4% |

A resting short limit at the reopening print gets hit 94% of the time within five minutes. Executability
was never the binding constraint it was assumed to be.

## 3. The 6 cells (net R per trade; primary contract)

| b / arm | TRAIN mean (t) | MDE | VAL mean (t) | TEST mean (t) | TRAIN ex-top5% | TRAIN cap+3R | G1 |
|---|---|---|---|---|---|---|---|
| 0.0 touch | **−0.840** (−3.31) | 0.711 | +0.013 (0.04) | +0.456 (1.63) | −1.358 | −1.419 | FAIL |
| 0.0 strict | −0.992 (−3.58) | 0.776 | −0.216 (−0.71) | +0.164 (0.51) | −1.528 | −1.564 | FAIL |
| 0.5 touch | −1.210 (−4.33) | 0.782 | −0.204 (−0.58) | +0.367 (1.29) | −1.725 | −1.674 | FAIL |
| 0.5 strict | −2.689 (−5.46) | 1.378 | −1.858 (−4.44) | −1.812 (−4.08) | −3.200 | −2.951 | FAIL |
| 1.0 touch | −1.189 (−4.09) | 0.813 | −0.239 (−0.64) | +0.411 (1.44) | −1.700 | −1.655 | FAIL |
| 1.0 strict | −2.472 (−6.24) | 1.109 | −1.945 (−4.35) | −1.819 (−3.83) | −2.954 | −2.701 | FAIL |

**Every cell is negative on TRAIN. G1 requires mean > 0 and t >= 2.0, so nothing is promoted and G2 is
not reached.** TEST is shown only because it had already been read once in O_halt for this population;
it promotes nothing, and under the gate order it is not evidence.

Weeks green on TRAIN: 23–29% for the touch arms, 8–23% for the strict arms. Monthly on the best cell
(b=0/touch): **7 of 21 months green** — against 20 of 21 in O_halt.

**Search-adjusted permutation, TRAIN, symmetric sign-flip null, B = 2,000, max |t| over the 6 cells:
observed 6.24, p = 0.0005.** That significance belongs to a **losing** cell (b=1.0/strict). It says the
b-ladder is real, not that an edge is: raising the limit above the reopen selects only the events that
kept running *up* after the reopen — adverse selection against a short — and net R falls monotonically
in b in both arms, on all three splits.

**Cells looked at: 6.** Cumulative for the S-series: 12 (O_halt) + 6 = **18**.

## 4. Why it fails — the decomposition (b=0/touch, the best cell)

| arm | TRAIN | VAL | TEST |
|---|---|---|---|
| A O_halt fill, O_halt cover, **band** cost (= the published result) | **+0.375** (t 2.43) | +0.966 (3.97) | +0.858 (3.44) |
| B O_halt fill, O_halt cover, **MEASURED** cost | **−0.671** (−2.84) | −0.065 (−0.21) | +0.355 (1.42) |
| C **passive** fill, O_halt cover, band cost | **+0.405** (2.63) | +0.991 (4.08) | +0.882 (3.54) |
| D **passive** fill, O_halt cover, MEASURED cost | −0.641 (−2.71) | −0.040 (−0.13) | +0.378 (1.52) |
| E passive fill, next-open cover, MEASURED cost (**the cell**) | −0.840 (−3.31) | +0.013 (0.04) | +0.456 (1.63) |

Read A → C: **the passive short limit is worth +0.03 R more than the marketable sell** (gross +0.517 vs
+0.487 on TRAIN, and it is better on every split). Reg SHO is not what kills this book. Read A → B:
**replacing the band-table spread with the measured one costs −1.05 R per trade on TRAIN** and turns the
published +0.375 into −0.671. Read D → E: the honest next-bar-open cover costs a further −0.20 R on
TRAIN (and *helps* on VAL/TEST — the same timing instability `O_halt/REPORT.md` §5 robustness already
found).

### The measurement itself

Measured NBBO at the cover instant, as % of price:

| split | mean | median | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| TRAIN | 4.63 | **1.92** | 3.91 | 7.81 | 13.14 | 474.5 |
| VAL | 4.57 | **1.98** | 4.12 | 10.10 | 15.42 | 228.0 |
| TEST | 2.46 | **1.43** | 3.29 | 6.02 | 8.32 | 15.1 |

`cost_curve.md`'s band constant, which O_halt charged: **0.40%**. Entry-instant spreads are the same
order (mean 2.37–2.75%, median 1.65–1.89%). The mean is heavy-tailed and partly stale, so the verdict
is reported under three charges, and **TRAIN is negative under all of them**:

| charge on the b=0/touch cell | TRAIN | VAL | TEST |
|---|---|---|---|
| mean spread (the declared contract) | −0.840 (t −3.31) | +0.013 | +0.456 |
| winsorised at the split's p95 | −0.458 (t −2.55) | +0.286 | +0.492 |
| **median** spread | **−0.160** (t −0.90) | +0.662 | +0.714 |
| O_halt's 0.40% band constant | +0.405 | +0.991 | +0.882 |

Breakeven cover spread for the cell's gross edge: **TRAIN 1.27%**, VAL 4.63%, TEST 4.29% — against a
measured TRAIN median of 1.92%. O_halt §5.3 predicted breakeven "at a 2.85% quoted spread" on the full
book and called this the one unmeasured number that could erase the result. **It did.**

## 5. Borrow — the second, independent blocker

`shortable AND easy_to_borrow` on Alpaca's asset record **today (2026-09-18)**: 35.8% of 14,355 active
US equities, but only **8.1% / 6.6% / 3.3%** (TRAIN/VAL/TEST) of the booked trades. **Survivorship
caveat, as declared: this is today's flag, not the flag on the halt date**, and for a $5–10 Nasdaq
microcap minutes after a volatility halt it is a weak proxy in both directions. On the tradeable subset
the best cell is still negative on TRAIN (−0.423, t −0.93, n 62) and the subset is ~1 trade/week, so
nothing survives there either. A 15-trades-per-week book becomes a 1-trade-per-week book before a
single spread is paid.

## 6. Availability audit and power

* Every decision field is computable at or before the decision instant: the limit uses only `reopen`
  (the entry bar's open) and the NBB from the last SIP quote **at or before** `entry_t`. Missing entry
  NBBO 12/1,445 = 0.83%, excluded and listed.
* Cover spread measured on **91.7 / 93.3 / 94.1%** of trades (TRAIN/VAL/TEST); the remainder falls back
  to the entry-minute mean spread and is flagged in `scored_trades.parquet::cov_spread_src`. The
  fallback is mildly conservative (entry spreads run slightly wider than cover spreads on TRAIN).
* Quote staleness at the cover instant: median 2.2 s, 16.7% older than 30 s, 300 s lookback cap.
* **MDE** (2.8 x SE) on the best cell: TRAIN 0.711, VAL 0.905, TEST 0.784 R. O_halt's union book was
  +0.530 / +0.996 / +0.904. **TRAIN is therefore a powered rejection**: an effect of the published size
  would have been visible there, and what is seen instead is −0.840. VAL and TEST are not powered to
  resolve anything below ~0.8 R, and under the gate order they were never in play.

## 7. Capacity

Shares at 1% of the fill-bar volume: median **568 shares / $4,843 notional** (b=0/touch). At R = 2% of
price, $100 of risk needs $5,000 of notional — the median trade is right at the capacity limit. **$375 of
risk needs $18,750, about 3.9x the median capacity, so the owner's larger size is not attainable on this
book at all.** No $/month is claimed: the gate failed, and the only positive weekly figures
(VAL +$86/mo, TEST +$3,035/mo at $100 risk) sit behind a TRAIN of −$5,369/mo.

## 8. Verdict (PLAN §1 phrasing)

> **No edge was detectable in THIS universe** (Nasdaq-listed LULD halt-resumes with the reopening print
> at or above the pre-halt price, prev close >= $5, ADV20 >= 100K), **at THIS horizon** (+5 minutes),
> **at THIS book size** (every event, no slot competition), **over THIS window** (2025-01 → 2026-09),
> **at THIS cost** (the per-trade MEASURED Alpaca SIP NBBO at the resume and cover instants), under a
> **passive short limit that Reg SHO permits**. 0 of 6 pre-declared cells clears G1; the best is
> −0.840 R at t −3.31 on TRAIN, against an MDE of 0.711 R — a powered rejection of an effect of the
> published size, not a shrug. The verdict is unchanged under a p95-winsorised spread (−0.458) and
> under the median spread (−0.160), and it reverses only under the out-of-population 0.40% band
> constant the original result was scored on (+0.405).

**The positive finding of this stage is a correction, not a book.** O_halt's `+0.530 / +0.996 / +0.904`
is reproduced exactly (arms A/C) and is **an artefact of charging a 0.40% spread to a cohort that quotes
a median 1.9%.** `O_halt/REPORT.md` §8's claim that "the LULD reopening print overshoots" survives only
in the gross: the overshoot is real (+0.49 R gross on TRAIN, +1.08 / +0.97 on VAL/TEST) and is **smaller
than the spread you must cross to collect it**. Rule 201 was never the binding constraint; the second
blocker (borrow: 3–8% of trades) would have bound before it.

## 9. What is NOT done, and what a revival would need

Nothing was enabled, proposed, built or touched: no config, order, service, cron or cache. The engine
gaps from `O_halt/REPORT.md` §7 are untouched and remain real if this line is ever reopened — **no halt
ingestion exists in the repo, `subscribe_trading_statuses` is unused, there is no short-sell path in
`trading_engine.py`, and the Alpaca `statuses` websocket check is STILL UNRUN** (it must be done
off-hours or on a separate key: the live `onemil-trader` holds the single market-data websocket for this
key through the session, and a second connection would knock the live stream off).

One honest observation, explicitly **not** a result and **not** pre-registered here: the entire cost in
this stage is the **marketable cover** (1.0 half-spread, 0.48–0.58 R at the median). A passive cover — a
resting buy limit below the NBO — is the symmetric idea to the one this stage tested, and it is a
**different fill model needing its own pre-registration, its own fill-rate measurement and its own
unfilled-position handling** (an unfilled cover is an open short, not a zero). It must not be scored by
editing this stage's numbers.

**Independent rebuild status:** none of this supersedes `O_halt/REPORT.md` §7's standing requirement that
the base trade set be independently reimplemented from prose before a line of engine code is written.
That requirement is now moot for shipping purposes and remains on the record.

Artefacts: `PREREG.md`, `cells.csv`, `monthly.csv`, `capacity.csv`, `scored_trades.parquet`,
`sim_rows.parquet`, `entry_nbbo.csv`, `cover_nbbo.csv`, `borrow_flags.csv`, `excluded_no_quote.csv`,
`score_summary.json`, `decompose.log`; scripts `fetch_entry_nbbo.py`, `sim.py`, `fetch_cover_nbbo.py`,
`fetch_borrow.py`, `score.py`, `decompose.py`. Data spend: **$0** (Alpaca SIP is on the existing
subscription; no Databento byte was pulled).
