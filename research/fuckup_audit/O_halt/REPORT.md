# Stage O — S1: LULD halt-resume

Pre-registered in `research/lit_review_2026/DATABENTO_OPPORTUNITIES.md` §S1, frozen in the docstring
of `score_cells.py` before any split was read. Gates and phrasing per `research/fuckup_audit/PLAN.md`
§1; cost contract per `lit_review_2026/cost_curve.md` + `cost_by_outcome.md`.

**STATUS: complete. 2 of 12 cells clear G1, G2 and G3. Both are SHORTS on Nasdaq-listed names, and
both carry an unresolved execution blocker (Reg SHO SSR + borrow). Nothing proposed for the engine;
the next step is an independent rebuild, not code.**

---

## 0. What was bought, and what was not — $12.47 of the approved $100

| item | `metadata.get_cost` | bought | $ |
|---|---|---|---|
| `status` ALL_SYMBOLS 2026-09-15 (step-0 probe day) | 0.0319 | yes | 0.03 |
| `status` ALL_SYMBOLS 2025-01-01→2026-09-18, pulled month by month | 12.437 | yes | 12.44 |
| `status` ALL_SYMBOLS 2018-05-01→2024-12 (the deep history) | 41.92 of the $54.39 full range | **no** | 0 |
| `definition` ALL_SYMBOLS 2024-07-01→2026-09-18 | 35.27 | **no** | 0 |

Every request was priced before the byte was pulled; `fetch_status.py` aborts the month that would
cross $100. Ledger in `spend.json`.

Two approved lines were deliberately not spent:

* **`definition` is not needed — $35.20 saved, permanently.** The brief assumed `ALL_SYMBOLS` status
  rows carry only `instrument_id` and that `definition` is the only way back to a ticker. It is not:
  `Historical.symbology.resolve(stype_in='instrument_id', stype_out='raw_symbol')` is a **free
  metadata endpoint**, and resolved **100.0000% of the 21,070 halt events with 0 `not_found`**,
  day by day (XNAS.ITCH ids are per-day locate codes; cached in `iid_symbol_map.csv`). This applies
  to every future ALL_SYMBOLS pull, not just this one.
* **The 2018–2024 deep history was not bought, because it cannot be scored.** The 1-minute outcome
  tape on this node starts **2025-01-02** (`research/bf_zero/bars_sip.db`, 307,343 symbol-days) and
  `cache.db::intraday_bars_1min` is not materially deeper. Six years of halt timestamps with no
  price to exit against buys a list, not a study. **TRAIN is therefore the program split (2025) as
  PLAN §1 fixes it, not the brief's 2018-05→2024-12** — that is the truncation, stated. The $41.92
  option stays open for the day a deep 1-minute tape exists.

---

## 1. Step 0 — the halt taxonomy, decoded from `databento_dbn`

In-session rows (ET 09:25–16:05), whole market, 2025-01 → 2026-09:

| action | reason | n | what it is |
|---|---|---|---|
| `PAUSE` (9) | `LULD_PAUSE` (50) | **21,072** | **the LULD volatility halt — the study population** |
| `TRADING` (7) | `NONE` (0) | 5,175,938 | the trading state. After a pause, the FIRST such row with `is_trading='Y'` **is the resume**. The bulk is the 09:30 open broadcast |
| `SSR_CHANGE` (14) | 0 | 176,926 | Reg SHO state change — not a trading state, excluded from the event logic (but read back in §6) |
| `HALT` (8) | `NONE` (0) | 3,111 | regulatory / other halt — counted, **never mixed into the LULD cells** |
| `QUOTING` (3) | `NEW_SECURITY_OFFERING` (110) | 841 | IPO quoting period |
| `HALT` (8) | `NEWS_PENDING` (30) | 146 | **news halt** — classified separately |
| `QUOTING` (3) | `NEWS_AND_RESUMPTION_TIMES` (32) | 128 | the news-halt resume quoting phase |
| `QUOTING` (3) | `NEW_ISSUE` (15) | 62 | new issue |
| `HALT` (8) | `ADDITIONAL_INFORMATION_REQUESTED` (70) | 47 | regulatory |
| other | | 10 | |

So: **LULD volatility halt = `PAUSE` + `LULD_PAUSE`; its resume = the next `TRADING` with
`is_trading='Y'`.** News halts are `HALT`+`NEWS_PENDING` / `QUOTING`+`NEWS_AND_RESUMPTION_TIMES`
(274 rows) and are excluded. The 07:04 UTC pre-session broadcast (~39,700 rows/day — every
instrument gets `PRE_OPEN`+`TRADING`+`SSR_CHANGE`) falls outside the session window by construction
and never enters the population.

**21,070 of 21,072 pauses resume before 16:05**; median halt duration **exactly 5.0 min** (mean 7.67;
a minority re-halt or roll into a regulatory halt).

**Availability / timestamp ordering.** `halt_ts < resume_ts` on 100% of events;
`entry_bar_open > resume_ts` on 100% of trades; resume message → entry bar open is **median 26 s,
min 0 s, max 178 s**. The resume instant is known live (the LULD halt is a fixed 5 minutes and the
status message arrives at the resume), so no decision field is back-dated. All eight decision fields
are 100.0000% non-null on the scored trades.

---

## 2. Venue coverage — the cap on the whole study

| listed venue of the halted symbol | share of the 1,702 halted symbols | share of the 21,070 events | share of `universe` |
|---|---|---|---|
| NASDAQ | **72.44%** | 65.97% | 49.54% |
| not in `universe` (delisted / test tickers / never-members) | 27.44% | 34.00% | — |
| ARCA | 0.06% | 0.02% | 16.80% |
| AMEX | 0.06% | 0.01% | 3.23% |
| **NYSE** | **0.00%** | **0.00%** | 19.86% |
| **BATS** | **0.00%** | **0.00%** | 10.58% |

**Not one NYSE- or BATS-listed name carries an LULD pause in this feed across 21,070 events.** The
control rules out a symbology problem: 20 NYSE megacaps (BAC, F, GE, PFE, T, XOM, KO, …) all resolve
on XNAS.ITCH and all carry `PRE_OPEN`/`TRADING`/`SSR_CHANGE` rows — they simply never carry a halt
action. XNAS.ITCH `status` is a **primary-listing halt feed**: Nasdaq publishes trading actions for
what it lists and nothing else. The §0(a) caveat in `DATABENTO_OPPORTUNITIES.md` ("UNVERIFIED") is
now resolved, negatively.

**What it caps.** The study sees the NASDAQ **49.5%** of `universe` and essentially none of the
NYSE/ARCA/BATS/AMEX **50.5%** — and ARCA+BATS is where the leveraged wrappers live (42% of ORB's
picks). Any rule built on this data source is Nasdaq-listed-only. Alpaca's own consolidated
`statuses` websocket channel would not have the hole — see §7.

---

## 3. Population and attrition

21,070 resumed LULD halts → **2,550 tradeable events**:

| dropped | n | rule |
|---|---|---|
| test tickers `^Z[A-Z]ZZT$` | 4,115 | the standing rule from F6's ZVZZT |
| symbol absent from `daily_bars` | 3,315 | the standing rule |
| prev close < $5 | 7,432 | causal, daily bars strictly before the halt date |
| ADV20 < 100K | 2,735 | the engine's own PIT definition |
| < 6 pre-halt 1-min bars (no `ref`) | 326 | |
| no print within 2 min of the resume | 429 | no order would have filled |
| no post-resume bar / entry past 15:55 | 83 | |
| no 1-min tape at all | 30 | |
| flat 5-min pre-halt return (side undefined) | 55 | |

| split | window | events | days | weeks | events/week |
|---|---|---|---|---|---|
| TRAIN | 2025-01-03 … 2025-12-31 | 1,449 | 221 | 53 | **27.3** |
| VAL | 2026-01-02 … 2026-05-29 | 625 | 91 | 22 | **28.4** |
| TEST | 2026-06-01 … 2026-09-17 | 476 | 72 | 16 | **29.8** |

Side mix 51.2% down-halts / 48.8% up-halts. Bar source 97% `cache.db`, 3% `bars_sip.db` — **both are
the Alpaca consolidated tape, so there is no cross-vendor price scale to reconcile** (the Databento
daily-vs-Alpaca-intraday hazard does not arise here). Every fill was asserted inside its own bar
(`low ≤ fill ≤ high`), zero failures.

---

## 4. The 12 pre-declared cells

R is declared flat at **2.0% of the fill price** for every cell — there is no stop in this stage, so
R is only the scale in which return and cost are expressed; `raw %` is the same result in percent so
the two are separable. Entry is **the open of the first 1-minute bar starting strictly after the
resume, taken only if it is within 0.6% of `ref` (the last trade before the halt) on the side being
taken** — the engine's no-chase cap, never the touch of a level. `fill rate` is the share of events
the cap accepts; the rejected events are exactly the ones a touch convention would have booked at
the cap level, so **`1 − fill rate` = the share of trades whose fill would differ from a touch:
27–68% by cell.** MDE = the smallest mean the split could resolve at 80% power (2.8 × SE).

### TRAIN 2025 — G1 is mean netR > 0 and t ≥ 2.0

| side | rule | horizon | fill rate | n | mean netR | t | MDE | raw % | wks green | ex-top5% | cap +3R | G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| up | continuation | +5m | 0.362 | 250 | +0.239 | 0.90 | 0.745 | +0.66 | 42.3% | −0.274 | −0.546 | |
| up | continuation | +30m | 0.362 | 250 | −0.394 | −0.78 | 1.417 | −0.61 | 46.2% | −1.494 | −2.118 | |
| up | continuation | close | 0.362 | 250 | −0.639 | −0.70 | 2.545 | −1.17 | 30.8% | −3.041 | −3.949 | |
| **up** | **fade** | **+5m** | 0.714 | 493 | **+0.566** | **2.90** | 0.546 | +1.31 | 67.3% | +0.138 | −0.180 | **PASS** |
| up | fade | +30m | 0.714 | 493 | +0.436 | 1.02 | 1.197 | +1.05 | 65.4% | −0.451 | −1.651 | |
| up | fade | close | 0.714 | 493 | +0.836 | 1.21 | 1.937 | +1.78 | 63.5% | −0.747 | −2.981 | |
| **down** | **continuation** | **+5m** | 0.439 | 333 | **+0.476** | **2.10** | 0.636 | +1.13 | 62.0% | +0.052 | −0.201 | **PASS** |
| down | continuation | +30m | 0.439 | 333 | +0.206 | 0.39 | 1.462 | +0.58 | 54.0% | −0.858 | −1.922 | |
| down | continuation | close | 0.439 | 333 | +0.980 | 1.14 | 2.412 | +2.06 | 54.0% | −0.961 | −3.207 | |
| down | fade | +5m | 0.634 | 481 | −0.150 | −0.75 | 0.563 | −0.12 | 50.9% | −0.620 | −0.808 | |
| down | fade | +30m | 0.634 | 481 | +0.294 | 0.65 | 1.270 | +0.77 | 58.5% | −1.009 | −2.171 | |
| down | fade | close | 0.634 | 481 | −1.128 | −1.50 | 2.100 | −2.15 | 49.1% | −3.347 | −5.180 | |

### VAL 2026-01→05 — G2 is mean > 0, t ≥ 1.0, ≥ 55% weeks green (the two G1 survivors)

| cell | n | mean netR | t | MDE | wks green | ex-top5% | cap +3R | G2 |
|---|---|---|---|---|---|---|---|---|
| up / fade / +5m | 205 | **+0.915** | 2.94 | 0.872 | **77.3%** | +0.507 | +0.051 | **PASS** |
| down / continuation / +5m | 150 | **+1.107** | 3.10 | 1.000 | **71.4%** | +0.659 | +0.122 | **PASS** |

(The other ten VAL cells are in `cells.csv`. The best of them, down/cont/+30m at +1.712 / t 2.45, had
already failed G1 at t 0.39 — it is not promoted.)

### TEST 2026-06-01 … 09-17 — read ONCE, after both gates, reported as it came

| cell | n | mean netR | t | MDE | wks green | ex-top5% | cap +3R |
|---|---|---|---|---|---|---|---|
| up / fade / +5m | 174 | **+0.883** | 2.84 | 0.870 | **87.5%** | +0.496 | +0.119 |
| down / continuation / +5m | 90 | **+0.943** | 2.64 | 0.999 | **86.7%** | +0.537 | +0.429 |

**Cells looked at in this stage: 12** — the 12 pre-declared, no variant, threshold, sizing or
price-band cell added. Cumulative for the S-series: 12 of the 26 pre-declared in
`DATABENTO_OPPORTUNITIES.md` §B.

**Search-adjusted permutation (TRAIN, symmetric sign-flip null, B = 2,000, max |t| across all 12
cells): observed max |t| = 2.90, p = 0.0535.** The winner sits *at* the multiplicity-adjusted
threshold, not comfortably through it.

---

## 5. What the survivors actually are

Both are the **same trade**, and it is not the one the cell names suggest. Both cells are **SHORT**,
so the no-chase cap is a *floor* (`fill ≥ ref × 0.994`), and it admits exactly the events where the
reopening print came back **at or above the pre-halt price**. Median resume gap on the accepted
trades: **+4.90%** (up-halts) and **+3.27%** (down-halts, i.e. a bounce); on the rejected ones
−3.97% and −6.03%. Stated as one rule:

> **After an LULD resume, when the first post-resume print is at or above the last pre-halt price,
> that print is too high: it gives back about 1.3–2.4% over the next five minutes.**

The long-side twins (up/continuation, down/fade) are the disjoint complement — they fill only when
the resume printed *below* `ref` — and neither clears anything, so there is no symmetric "buy the
cheap reopen" effect.

**Union book, both cells together** (median resume gap **+4.16%**, median ADV20 1.10M, price mix
$5-10 50% / $10-20 26% / $20-50 18% / $50+ 6%):

| split | n | trades/wk | mean netR | t | MDE | wks green | win% | median | ex-top5% | cap +3R |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | 826 | 15.9 | +0.530 | 3.58 | 0.414 | 76.9% | 54.4% | +0.388 | +0.103 | **−0.189** |
| VAL | 355 | 16.1 | +0.996 | 4.25 | 0.656 | 90.9% | 59.2% | +1.024 | +0.590 | +0.081 |
| TEST | 264 | 16.5 | +0.904 | 3.80 | 0.666 | **100.0%** | 58.3% | +1.209 | +0.507 | +0.225 |

Per month (union net R): 20 of 21 months positive; the single negative is **2025-08 at −14.7 R over
45 trades**. Monthly sums range +4.5 R (2025-04) to +140.2 R (2026-05).

### The four things wrong with it

1. **Tail dependence.** Ex-top-5% the edge survives on all three splits (+0.10 / +0.59 / +0.51). But
   **with winners capped at +3R the TRAIN book is negative (−0.189)**, turning positive only on VAL
   and TEST. That is the shape this owner has already rejected once. Not disqualifying (VAL and TEST
   hold under the cap, and the median trade is positive on every split) but it is the first thing a
   re-run must re-check.
2. **Power.** MDE is 0.414 / 0.656 / 0.666 R against observed means of 0.530 / 0.996 / 0.904 — the
   test can *just* see the effect it found and nothing much smaller. If the live version came in at
   half this size, this study could not have distinguished it from zero.
3. **The cost model is out of population.** `cost_curve.md`'s band×hour spreads were measured on
   ≥5%-range movers in normal trade, and charge this sample **≈0.40%**. The book's gross edge is
   +0.802 R, so it breaks even at a **2.85% quoted spread** — and half these trades are $5-10 Nasdaq
   microcaps in the first minutes after a volatility halt, where a 2–3% quoted spread is entirely
   ordinary. **This single unmeasured number can erase the whole result**, and measuring it needs
   quote data at the resume minute (a `bbo-1s`/`tbbo` pull that was not part of this budget).
4. **It is a short book** — see §6.

### Robustness (`robustness.py`)

| | TRAIN | VAL | TEST |
|---|---|---|---|
| up/fade/+5m, **open→open** (both legs bar opens; kills a bid-ask artefact) | +0.506 (t 2.78, n 347) | +0.782 (t 2.71) | +0.784 (t 2.50) |
| up/fade/+5m, **entry one minute later** | +0.094 (t 0.42, n 314) | +0.748 (t 2.16) | +0.759 (t 1.99) |
| down/cont/+5m, **open→open** | +0.276 (t 1.35, n 253) | +0.630 (t 2.00) | +0.605 (t 1.85) |
| down/cont/+5m, **entry one minute later** | +0.469 (t 1.91, n 223) | +0.943 (t 2.57) | +0.379 (t 0.78) |

The up-side cell is **not** a bid-ask bounce — open→open keeps it on all three splits. The down-side
cell loses ~40% of its TRAIN edge under open→open, so part of that one is quote mechanics. Delaying
the entry by one minute destroys the up-side cell on TRAIN but not on VAL/TEST, i.e. the timing
sensitivity is itself unstable. Both sub-tests run on smaller n (a print is required at exactly that
minute), which is its own selection.

---

## 6. The blocker: Reg SHO rule 201 (SSR). This is where the result dies.

Both survivors are short sales. Rule 201 forbids a short sale at or below the national best bid for
the rest of the day (and the next) once a security trades 10% below its prior close — and an LULD
halt is by construction a large move. The status record carries the state directly
(`is_short_sell_restricted`), so this is measured, not assumed. Stamped onto every event by
`merge_asof` at the resume instant (`ssr.py`):

| | SSR = Y at the resume | SSR = N |
|---|---|---|
| all 21,070 LULD events | **55.5%** | 44.5% |
| cell A filled trades (up-halt fade) | **75.6%** | 24.4% |
| cell B filled trades (down-halt cont.) | **76.8%** | 23.2% |

Per split the restricted share is 77.1 / 75.1 / 71.8% (A) and 81.1 / 71.3 / 70.0% (B) — stable, not
a regime artefact. **Our entry convention is a marketable sell at the next bar's open.** On an
SSR-restricted name that order cannot execute at or below the bid; it has to rest above the NBB and
may never fill at all. So on roughly **three of every four** booked trades the modelled fill is not
obtainable — a direct hit on PLAN §1's obtainability rule ("reachable by an order the live engine
would have had resting").

**The gates re-run on the SSR = N subset — the only trades a short-capable stack could actually place:**

| cell | TRAIN | VAL | TEST | verdict |
|---|---|---|---|---|
| A up-fade | n 113, +0.999, t **2.67**, MDE 1.048 | n 51, **+0.111, t 0.24** | n 49, +0.576, t 0.83 | **fails G2** (VAL t < 1.0) |
| B down-cont | n 63, +0.871, t **1.95**, MDE 1.250 | n 43, +1.630, t 2.45 | n 27, +1.896, t 3.36 | **fails G1** (TRAIN t < 2.0) |

**Neither survivor clears its own pre-registered gate once the unobtainable trades are removed.**
The MDEs on those subsets are 1.05 R and 1.25 R per trade — larger than the effect itself — so this
is a near-powerless test, and its failure is weak evidence, not a refutation. What it does establish
firmly is that **the measured edge lives predominantly in trades Reg SHO makes unreachable.**

Second, unmeasured, executability item: **borrow.** The filled shorts are 45–57% $5–10 Nasdaq
microcaps (median prev close $8.70–9.22, median ADV20 ~1.1M) in the minutes after a volatility halt —
the classic hard-to-borrow cohort. Nothing in this study checked locate availability or borrow fees,
and no short has ever been submitted anywhere in this repo (`AlpacaClient` has six submit sites, all
`OrderSide.BUY` entries with `SELL` only as an exit leg).

---

## 7. If it were pursued: the live rule, what would have to be built, and the rebuild bar

Written because two declared cells did clear G1, G2 and G3 as declared. **Nothing here is proposed;
§6 and §8 say why it is not shippable as it stands.**

**The rule.** On a Nasdaq-listed symbol with prev close ≥ $5 and ADV20 ≥ 100K: on an LULD halt
(`PAUSE`/`LULD_PAUSE`) record `ref` = the last trade before the halt. On the resume (`TRADING` with
`is_trading = Y`), at the open of the first full 1-minute bar after it, **sell short if that open is
≥ `ref × 0.994`**; cover at the close of the bar five minutes later. No stop in the tested spec —
which is itself a gap, since a live short needs one.

**What would have to be built (none of it exists):**

1. **A halt feed.** `grep` over the repo: **zero occurrences of `subscribe_trading_statuses` or any
   trading-status ingestion** — the only `halt` matches are a news-headline regex and a reject
   categoriser. `alpaca.data.live.stock.StockDataStream.subscribe_trading_statuses` **does exist** in
   the installed alpaca-py 0.43.2, and it is the consolidated feed, so it would not carry the
   Nasdaq-listed-only hole of §2.
2. **A websocket check that was NOT run, and why.** The S1 step-0 list included "verify Alpaca's
   `statuses` channel resolves on our key". **I did not run it.** Alpaca allows one market-data
   websocket per key, the live `onemil-trader` holds it (StopMonitor) through the session, and
   opening a second connection on the shared key would have knocked the live data stream off
   mid-session. This check must be run **off-hours, or on a separate key**, and it is a hard gate:
   without a live halt feed there is no rule. This is the one step-0 item left open.
3. **Short-side order support.** A sell-short entry, an SSR-aware price (limit above the NBB, never a
   marketable sell, on the 55% of halts that are restricted), a locate/borrow check, and a stop. None
   of that exists; the shared live account also carries the owner's manual positions, so a short book
   would need hard isolation.
4. **The missing cost measurement.** Post-resume quoted spreads on this exact cohort
   (`bbo-1s`/`tbbo` at the resume minute, priced but not bought). At a 2.85% quoted spread the book
   is exactly zero, and half of it is $5–10 microcaps seconds after a volatility halt.

**Independent-rebuild requirement, before a single line of engine code.** Per CLAUDE.md and PLAN §1,
and because this stage's result reverses direction under a rule the cell names actively obscured
(§5): a second implementation, written from a prose spec by someone who has not read `score_cells.py`,
must reproduce the trade set **trade by trade on (day, symbol)**, not in aggregate — specifically
(a) the halt/resume pairing off the raw `status` records, (b) the side classification, (c) which
events the 0.6% floor accepts, and (d) the SSR stamp. Aggregates here would hide exactly the
compensating error that made a short book look like two different directional cells.

---

## 8. Verdict

**Two of twelve pre-declared cells cleared G1, G2 and G3 as declared** — up-halt/fade/+5m and
down-halt/continuation/+5m, which are one rule: *short the first post-resume bar when the reopening
print is at or above the pre-halt price.* Union book **+0.53 / +1.00 / +0.90 net R per trade at
15.9 / 16.1 / 16.5 trades per week, 20 of 21 months positive, weeks green 77% / 91% / 100%.** That
regularity is real in this data and worth naming: **the LULD reopening print overshoots.**

It is **not a shippable edge**, and the phrasing matters:

> No shippable edge was detectable in **this universe** (Nasdaq-listed LULD halt-resumes, prev close
> ≥ $5, ADV20 ≥ 100K — 49.5% of our tradeable names, zero NYSE/BATS coverage), at **this horizon**
> (+5 minutes), at **this book size** (every event, no slot competition), over **this window**
> (2025-01 → 2026-09), at **this cost** (`cost_curve.md` band×hour spreads), because the effect is
> concentrated in short sales that Reg SHO rule 201 makes unobtainable: 76% of the booked trades are
> SSR-restricted, and on the 24% that are not, **neither cell clears its own gate** (A fails G2 at
> VAL t 0.24; B fails G1 at TRAIN t 1.95). **The smallest per-trade effect those SSR-free subsets
> could resolve at 80% power is 1.05 R and 1.25 R — larger than the effect being tested — so that
> null is weak.** Independently, the full-sample result breaks even at a 2.85% quoted spread, which
> was never measured on this cohort, and its TRAIN edge is negative with winners capped at +3R.

**Nothing is enabled, nothing is proposed, no config, service, cron, cache or order was touched.**
Artefacts: `cells.csv`, `trades.parquet`, `survivor_trades.csv`, `luld_events_raw.parquet`,
`ssr_state.parquet`, `status_taxonomy.csv`, `nonluld_halts.csv`, `spend.json`,
`iid_symbol_map.csv`; scripts `step0_cost.py`, `step0_probe.py`, `fetch_status.py`,
`build_events.py`, `score_cells.py`, `coverage.py`, `ssr.py`, `robustness.py`, `survivors.py`.
The raw `status` pulls under `raw/` are gitignored (regenerable for $12.44).

**If this line is re-opened, the cheapest decisive next step is not more cells — it is the two
measurements that would settle it: (1) the off-hours Alpaca `statuses` websocket check ($0), and
(2) a quote pull at the resume minute for these 2,550 events to replace the out-of-population spread
constant (priced, a few dollars). Either one can kill it, and both are cheaper than the $41.92 deep
history that was left unbought.**
