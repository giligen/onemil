# Stage Q — the FILL MODEL of the honest ORB book, settled against the live account

**STATUS: complete.** Run 2026-09-18. One python process at a time, `nice -n 10`, `ulimit -v 2500000`,
peak RSS < 1 GB. Everything written is under `research/fuckup_audit/Q_fill/`; `data/cache.db` and
`data/trades.db` were opened `file:...?mode=ro` and nothing else outside this directory was touched —
no config, no order, no service, no cron. **Databento spend $0.00** (Alpaca SIP quotes/trades only,
free at this account tier; total pull 1,040 order-life walks + 124 live trigger minutes, 9 minutes
of wall clock).

---

## 0. One page

Stage P flagged that on **14.4% of the 7,402 honest ORB fills the NBBO ask at the trigger instant was
ABOVE the pick's own stop-limit cap**, and that booking those as no-fills takes the 8-slot book from
$14,429 to $9,239. Stage Q walked the real quote path over the full life of each of those 1,040
orders and re-scored the book, then checked the answer against the 166 live ORB orders in
`data/trades.db`.

**The answer: the orders mostly DO fill, but not for free — and the whole $5,190 P flagged comes down
to SIX picks of 215. The honest 8-slot book is $10.3K–$12.2K, not $14.4K and not $9.2K. A one-line
config change recovers almost all of it, with the live account's own data behind it.**

| fill model, **8 slots**, 215 picks / 162 fills | P&L 21 mo | R/pick | TRAIN | VAL | TEST | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|---|---|
| **(a) as-is** — every elected stop-limit fills at the cap (= D1, to the cent) | **14,429** | +0.495 | +0.486 | +0.862 | +0.172 | -576 | -148 | 2 |
| **(b) hard no-fill** — ask > cap at the trigger => $0, slot spent | **8,158** | +0.295 | +0.404 | +0.282 | +0.106 | -537 | -218 | 3 |
| **(c) MEASURED** — the quote path decides; cap as the engine SENDS it (2 dp) | **12,197** | +0.416 | +0.448 | +0.720 | +0.075 | -709 | -148 | 4 |
| (c') MEASURED, cap unrounded (the BT's own `range_high x 1.003`) — conservative | 10,297 | +0.344 | +0.423 | +0.484 | +0.066 | -751 | -148 | 5 |

| fill model, **3 slots** (the pre-9/18 shipped book), 88 picks / 70 fills | P&L | R/pick | TRAIN | VAL | TEST | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|---|---|
| (a) as-is (= D1 $7,186.64) | **7,187** | +0.532 | +0.502 | +0.902 | +0.278 | -714 | -223 | 6 |
| (b) hard no-fill | **2,243** | +0.172 | +0.356 | **-0.090** | +0.003 | -671 | -311 | 11 |
| (c) MEASURED | **5,044** | +0.345 | +0.417 | +0.537 | +0.027 | -714 | -372 | 8 |
| (c') MEASURED, unrounded cap | 5,002 | +0.340 | +0.417 | +0.537 | +0.006 | -714 | -372 | 9 |

**The live cross-check (the only ground truth here): 166 live ORB orders, 124 with a locatable market
breakout bar. The NBBO ask at the trigger instant sits within 30 bps of `range_high` on 83.9% of them
(simulation: 85.6%), within 50 bps on 94.4% (sim 93.9%), within 100 bps on 99.2% (sim 98.4%); median
11.3 bps (sim 11.9), p90 37 bps (sim 37). Of the 121 orders whose stop elected, 116 (95.9%) filled and
5 (4.1%) never did; 19 of the 116 (16.4%) filled in a LATER minute than the breakout bar, and 15 of
those 19 filled at EXACTLY the cap. Every live fill was at or below the cap (median 13.8 bps below).**
The simulated fill model does not merely agree with the account — it reproduces its entry
microstructure to a couple of percentage points.

**Recommendation (one config value, not a code change): raise `orb.yaml::entry.stop_limit_buffer_bps`
from 30 to 50.** Measured, not assumed — the 701 orders it converts from "resting" to "immediate fill"
are re-simulated paying the ACTUAL ask, so the wider cap's cost is charged: 8-slot book **$14,048
(+0.479 R/pick, MDD -576, worst month -151, 2 red months)** vs the measured 30-bps book's $12,197 /
$10,297. It restores the drawdown and the red-month count as well as the dollars. 100 bps is worse
($13,672) — past ~50 bps you pay up on fills you were getting anyway. The live account agrees: **4 of
the 5 live orders that elected and never filled had an ask 33-40 bps over `range_high`; all four are
inside a 50-bps cap** (the fifth was 63 bps).

---

## 1. What was measured, and the conventions

### 1.1 The live order, read from the engine (not assumed)

`trading/orb_engine.py::_submit_entry_order` + `_cancel_stale_pending_orders` + `orb.yaml`:

| | value | source |
|---|---|---|
| stop (the trigger) | `round(range_high, 2)` | `stop_trigger = round(plan.range_high, 2)` |
| limit (**the cap**) | `round(range_high x (1 + 30 bps), 2)` | `limit_price = round(plan.entry_price, 2)`, `entry.stop_limit_buffer_bps: 30` |
| submitted | 09:35 ET (range end), after ranking | the 9:35 submit pass |
| cancelled | submit + `entry.time_stop_minutes: 60` -> **10:35 ET**, `order_status='time_stop_canceled'` | `_cancel_stale_pending_orders` |
| rests? | **yes** — an elected stop-limit that is not marketable stays a limit BID at the cap until 10:35 | Alpaca order semantics; confirmed live (S3) |

The BT's own entry minutes run 09:35 -> 10:34 inclusive, i.e. the 60-minute stop is already the BT's
entry window — the two clocks agree.

### 1.2 The fill convention, and why it matches the engine

* **Marketable at the trigger** (ask <= cap): the elected order is a marketable limit and **lifts the
  ask** -> fill at `min(ask, cap)`.
* **Not marketable** (ask > cap): the order **rests as a bid at the cap**. It fills when an offer comes
  down to it, and because we are the resting bid the fill is **AT THE CAP** — not at the incoming
  offer, not at the next bar's open. *This is the convention the task asked for, and the live account
  confirms it: 15 of the 19 live late fills printed at exactly the cap, to the cent.*
* **Never reached before 10:35**: `time_stop_canceled`, $0, and the slot is still spent (the ORB
  pipeline's standing NO-REFILL invariant).
* The trade is then re-simulated from the bar containing the later fill with the **SHIPPED** exit
  physics (`simulate_winner_stack` through Stage P's instrumented twin, winner stack per `orb.yaml`),
  so arms (a)/(b)/(c) differ **only** in the fill model. Entry price is the cap in every arm, so the
  arms are not confounded by a price change.
* **Touchgo on a delayed fill**: applied when the fill lands in the breakout bar itself (then the trade
  *is* the as-is trade, and the arm reproduces as-is to 1.1e-11 on all 524 such rows — asserted), and
  NOT applied when the fill lands in a later bar, because Rule M/D's decision bars closed before we
  held the position and a retroactive tag exit at that stale limit is not obtainable. The live engine's
  own 15-minute late-fill guard says the same thing. The opposite choice (re-key touchgo to the fill
  bar, arm `measured_tg`) is worth **+$313 at 8 slots and -$14 at 3** — 2% of the arm, not a driver.

### 1.3 The quote walk

`walk_quotes.py`: for each of the 1,040 flagged fills, Alpaca **SIP consolidated** quotes from the
trigger print forward to 10:35 ET, paged (10,000/page, resumed from the last timestamp, 30-page
ceiling — **0 rows truncated, 0 errors**), stopping at the first quote with `ask <= cap`.

| | raw cap (`range_high x 1.003`, unrounded — the BT's number) | **live cap** (`round(.,2)` — what the engine sends) |
|---|---|---|
| flagged (ask > cap at the trigger) | 1,040 of 7,202 = **14.4%** | 743 = **10.3%** |
| of those, **filled later** | 956 = **91.9%** | 681 = **91.7%** |
| median seconds to that fill | **20.8 s** | **27.7 s** |
| fill lands in the breakout bar anyway (no clock change) | 524 | 358 |
| fill lands in a **later bar** | 432 | 323 |
| **never filled** before 10:35 | 84 = 1.2% of all fills | 62 = 0.9% |

The gap between the two columns is pure rounding: 297 of the 1,040 have an ask above the unrounded cap
but at or below the cent the engine actually sends. **Stage P's 14.4% headline is 10.3% against the
order the engine really submits.**

---

## 2. Where the money actually is — six picks

Cost is additive through this pipeline (selection reads `_composite`, the quintiles and the vetoes,
never P&L), so every arm holds the same 215 / 88 picks and the same slot mechanics. **`asis` reproduces
D1 to the cent — 215 picks, $14,428.62, +0.486/+0.862/+0.172 R, MDD -576, worst -148, 2 red months;
3 slots $7,186.64.** Sized delta of the MEASURED arm vs as-is, by fill outcome:

| 8-slot book (215 picks) | picks | sized delta vs as-is |
|---|---|---|
| marketable at the trigger (no change by construction) | 119 | 0 |
| rested, filled inside the breakout bar (no change — asserted) | 20 | 0 |
| rested, filled in a **later bar** | 15 | **-751** |
| **never filled** | **6** | **-3,381** |
| no-fill picks (`entered=0`, $0 in every arm) + 2 uncovered | 55 | 0 |

| 3-slot book (88 picks) | picks | sized delta |
|---|---|---|
| marketable / same-bar | 57 | 0 |
| later bar | 8 | -420 |
| **never filled** | **5** | **-1,765** |
| no-fill + uncovered | 18 | 0 |

**82% of the 8-slot correction and 81% of the 3-slot correction come from six and five picks.** They
are ARQQ 2025-01-16 (+$600 as-is), BNAI 2026-02-24 (+$743), ANTX 2026-03-09 (+$1,616), ATPC 2026-06-03
(+$42), RPGL 2026-06-04 (+$218), CAST 2026-06-30 (+$162). **Two of them — ANTX and ATPC, including the
single largest — are not flagged at all against the cap the engine really sends** (ANTX: unrounded cap
3.09927, ask 3.10, sent cap 3.10 -> marketable). That one cent is the entire difference between arm
(c) at $12,197 and arm (c') at $10,297.

So the honest statement is not "the fill model costs the book 29%". It is: **the fill model is worth
-$2.2K to -$4.1K over 21 months at 8 slots, it is carried by six trades, and the largest single input
to it is the rounding of a limit price to the cent.**

### Tail dependence (PLAN S1 item 5)

| arm | slots | n | mean R | ex top 1% | ex top 5% | winners capped +3R | t |
|---|---|---|---|---|---|---|---|
| as-is | 8 | 215 | 0.495 | 0.398 | 0.165 | 0.296 | 3.93 |
| **measured** | 8 | 215 | **0.416** | 0.318 | 0.097 | 0.235 | **3.39** |
| measured (unrounded cap) | 8 | 215 | 0.344 | 0.261 | 0.063 | 0.207 | 3.16 |
| hard no-fill | 8 | 215 | 0.295 | 0.228 | 0.044 | 0.193 | 3.15 |
| as-is | 3 | 88 | 0.532 | 0.432 | 0.250 | 0.350 | 2.71 |
| **measured** | 3 | 88 | **0.345** | 0.243 | 0.091 | 0.207 | **1.90** |
| hard no-fill | 3 | 88 | 0.172 | 0.107 | **-0.019** | 0.118 | 1.34 |

Every arm stays positive on the mean and ex-top-1%. The 3-slot book's t falls 2.71 -> 1.90 under the
measured model; under the hard no-fill arm its VAL split turns negative (-0.090 R) and ex-top-5% goes
negative. **The 8-slot book is materially more robust to this question than the 3-slot book was** —
incidentally an argument for the 9/18 move to 8 slots rather than against it.

---

## 3. The live cross-check — `data/trades.db`, read-only

166 ORB orders ever sent (117 closed, 42 `time_stop_canceled`, 4 rejected, 2 `pending_new`, 1 filled),
2026-05-19 -> 2026-09-17, almost all under the retired pre-B+ config. This sample validates the **fill
mechanics**, not the book. Sample sizes are on every line because they are small.

**(1) Do elected orders fill?** A 1-min bar high cleared `range_high` inside 09:35-10:35 on **121** of
the 160 submitted-and-not-rejected orders (the other 37 never broke out and time-stopped — the BT's
`no_fill` row, already modelled). Of the 121: **116 filled (95.9%), 5 never did (4.1%)**. Simulated
never-fill on the same definition: 84 / 7,202 = **1.2%** (unrounded cap), 0.9% (live cap); **6 / 215 =
2.8%** among the picks the 8-slot book actually takes. With n=121 the live 4.1% has a 95% interval of
roughly [1.4%, 9.3%] — the simulation sits inside it, at the low end.

**(2) At what price?** Of the 116 fills, **116 (100%) filled at or below the cap** — median 13.8 bps
BELOW it, mean 14.4 below, 43 exactly at it, **zero above**. The as-is book's "fill at the cap" is
therefore a mildly pessimistic price assumption (Stage P measured the same from the other side: the
entry is $697 cheaper than assumed over 21 months).

**(3) When?** Lag from the breakout bar's open to the live fill: median **30 s**, p75 51 s, p90 220 s,
max 2,199 s. **97 of 116 (83.6%) filled inside the breakout minute**; 19 (16.4%) filled in a later
minute, and **15 of those 19 filled at exactly the cap** — the resting-bid convention, observed. The
simulation's counterpart: 85.6% marketable at the trigger instant, 6.0% resting but filling inside the
same bar, 6.0% filling in a later bar, 1.2% never.

**(4) The microstructure the cap has to cover** (`live_trigger_quotes.py`, SIP trades+quotes of the
breakout minute, all 124 live orders with a trigger-instant NBBO):

| ask over `range_high`, bps | live (n=124) | simulated (n=7,202) |
|---|---|---|
| median | 11.3 | 11.9 |
| mean | 18.2 | 17.5 |
| p75 / p90 | 25 / 37 | 23 / 37 |
| within **30 bps** (the shipped cap) | **83.9%** | **85.6%** |
| within 50 bps | 94.4% | 93.9% |
| within 100 bps | 99.2% | 98.4% |

**The simulated fill model reproduces the live account's entry microstructure to within ~2 points at
every cut.** That is the strongest validation available in this study, and it says the measured arm —
not as-is, not hard-no-fill — is the number to carry.

**(5) The five live no-fills, named**, with their trigger-instant ask over `range_high`: QUBX
2026-05-21 (33 bps; the bar ran to +1,168), GLXU 2026-06-08 (63; ran to +1,529), GLL 2026-06-10 (4),
NBIZ 2026-07-17 (40), IREX 2026-07-30 (37; ran to +1,869). **Four of the five are inside a 50-bps cap.**

---

## 4. The recommendation, and what it is worth — measured

Nothing in the book's construction, the vetoes, the ranking or the slot count should move. The one
parameter this stage indicts is the cap itself: `orb.yaml::entry.stop_limit_buffer_bps: 30`.

`cap_sweep.py` re-simulates it honestly — an order the wider cap makes marketable fills **at the actual
NBBO ask**, not at the new cap, and the trade is re-run from the breakout bar so the lock trigger, the
+3R scale level and the ATR-floored stop all move with the higher entry. Orders the wider cap still
cannot reach keep the measured (delayed / never) treatment, which is conservative.

| cap | orders converted to an immediate fill | 8-slot P&L | R/pick | TRAIN | VAL | TEST | MDD | worst mo | red mo | 3-slot P&L |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 bps (shipped), measured | — | 12,197 / 10,297 | +0.416 | +0.448 | +0.720 | +0.075 | -709 | -148 | 4 | 5,044 |
| **50 bps** | 701 of 1,040 | **14,048** | **+0.479** | +0.469 | +0.860 | +0.144 | **-576** | -151 | **2** | 6,822 |
| 100 bps | 945 of 1,040 | 13,672 | +0.463 | +0.454 | +0.825 | +0.144 | -743 | -151 | 2 | 6,644 |

* **+$1,851 over the live-cap measured book and +$3,751 over the unrounded one, at 8 slots over 21
  months**, improving **all three splits** against the measured baseline.
* It restores the shape as well as the level: MDD back to -576 (from -709/-751) and red months back to
  2 (from 4/5) — the same MDD and red-month count as the unattainable as-is book.
* It is **not free and the cost is charged**: 100 bps is $376 WORSE than 50, because past ~50 bps you
  pay up on fills you were already getting. An interior optimum, which is what a real cost/fill
  trade-off looks like rather than a free lunch.
* Live agrees from the other direction: 94.4% of live triggers had an ask within 50 bps, and 4 of the 5
  live no-fills were inside it.
* **It does not weaken the no-chase protection.** The cap still binds absolutely: QUBX/GLXU/IREX, which
  ran 1,168-1,869 bps through the level, remain no-fills at 50 bps exactly as they did live. The
  300-bps spread gate is a separate, untouched rail.

**What I am NOT recommending.** No change to the book, the slot count, the vetoes or the exits on the
strength of this stage. And the honest caveat on the cap itself: three values (30/50/100) were read on
the full sample, the improvement rests on the same six-pick tail as the problem does, and the effect
(+0.063 R/pick) is a fraction of this book's own 0.42 R minimum detectable effect at 215 picks. It is a
**mechanism-first** change — it removes a known, live-observed failure (an order that elects and cannot
fill) rather than fitting a threshold — which is why it is worth putting in front of the owner at all,
and it should be a config flip whose rollback is the single value `30`.

**What it means for the book we are trading TODAY.** ORB went to 8 slots on 2026-09-18. At $10K stage
sizing the honest 21-month reference moves from **$14,429 ($687/month)** to **$12,197 ($581/month)**
under the measured fill model, or **$10,297 ($490/month)** on the conservative unrounded cap. The book
stays positive on every split, every tail treatment and every arm; the 8-slot choice over 3 slots
survives this question comfortably (the 3-slot book's t falls to 1.90 and its VAL goes negative under
the harsher arm; the 8-slot book's does not). **Nothing about the live book needs to stop.** The
$14,429 in circulation should be restated as $12.2K, and if the cap goes to 50 bps the attainable
figure is $14.0K.

---

## 5. Cells looked at, and where this could be wrong

**Cells.** 1 population (7,202 covered fills of 7,402 entered). 8 fill/cap arms x 2 slot counts = **16
pipeline runs**, each read on 1 whole window + 3 splits = **64 config x split cells**, plus 16 tail
rows, 2 descriptive tables on the live sample and 4 threshold cuts (30/50/100/300 bps) on each of the
live and simulated ask distributions. **2 cap values were chosen and reported** (50 and 100); no
threshold was fitted; no veto, z-param, quintile cutoff, mult or slot count was moved. TEST is reported
because the picks are identical in every arm — this stage selects nothing.

**Approximations, each a place this could be wrong:**

1. **Queue priority is not modelled.** A resting bid at the cap is assumed filled by the first offer
   that reaches it; in reality we are behind the size already there. This makes the measured arm
   **optimistic** on the 432 later-bar fills; the hard-no-fill arm (b) is the bound in that direction.
2. **The walk's hit test uses the unrounded cap**, so 4 rows whose min ask lands between the unrounded
   and the rounded cap are counted as never-filled in arm (c) when the live cap would have caught them.
   None of the four is in either book.
3. **Rows that flip from flagged to marketable under the live cap keep their as-is P&L** (entry at the
   unrounded cap rather than the ask; <= 0.5c/share optimistic on 297 rows).
4. **Touchgo on a delayed fill** is a modelling choice, bounded at +/-$313 / 2% by `measured_tg` (S1.2).
5. **2.7% of entered rows have no measured quote** and are carried at as-is in every arm; they are the
   thinnest names (Stage P S2.3), so all measured arms are marginally optimistic. At book level this is
   2 of 215 picks.
6. **Alpaca SIP quote history** is the reference, not a tick-by-tick NBBO reconstruction; a quote that
   lived microseconds between two returned records is invisible.
7. **The live sample is 121 elected orders / 116 fills / 124 trigger quotes over 4 months, almost all
   under a retired config.** It validates fill mechanics — election, marketability, price, lag — and
   nothing about the book's edge. Every live percentage carries a +/-3-9 point interval.
8. **Stage P's `strict` arm was $9,239; this stage's is $8,158.** Not a contradiction: P's arm also
   re-priced every marketable row at the measured NBBO (net cheaper), while this arm changes the fill
   rule ALONE so that (a)/(b)/(c) are a clean comparison. The two stack in `meas_cost` ($11,050 at 8
   slots, $5,302 at 3).

**Stage P's two free fixes, folded in.** (i) *Price a capped limit as `(min(ask, cap), ask > cap => no
fill)`* — that pair **is** the construction of arms (b), (c) and `meas_cost` here, and it is now
measured rather than assumed: the "no fill" half is right only 8% of the time, the rest is a delay.
(ii) *Charge the cell MEAN, not the median* — it does not touch arms (a)/(b)/(c), which charge no band
constant at all (they use the per-trade measured quote); it applies to the `band` arm, which this stage
does not use. Recorded so the fix is not lost: `acore.corrected_spread_table()` still takes `.median()`
of a distribution whose mean is ~1.9x its median, and that one-line change remains outstanding.

**Phrasing.** In THIS population (7,202 ORB fills, 2025-01 -> 2026-09), under THIS order (a 30-bps
capped stop-limit resting to a 60-minute time stop), at THIS book size (3 and 8 slots, $10K stage), the
fill model is worth **-$2,232 to -$4,132 of the 21-month 8-slot book (-0.08 to -0.15 R per pick)**, it
is carried by six picks, and it is **not** the $5,190 the hard-no-fill reading suggested. The smallest
per-pick effect this book can resolve at 80% power is 0.424 R (N1, same 215 picks), so the fill-model
correction is smaller than the book's own detection threshold — but unlike a cost question it has a
named mechanism, a live counterpart, and a knob.

## 6. Files

```
research/fuckup_audit/Q_fill/
  REPORT.md                  this file
  walk_quotes.py             the NBBO walk over each flagged order's life -> walk_rows.csv
  resim_delayed.py           the delayed fills re-simulated with the SHIPPED exit physics
                             -> delayed_resim.csv (asserts same-bar rows == as-is)
  rescore_q.py               build | run | analyse -> dump_*.csv, book_*_n{8,3}.csv,
                             monthly_*, rescore_table.csv, per_trade_fill_models.csv
  cap_sweep.py               the measured value of a wider stop_limit_buffer_bps
  summarize.py               the per-trade deliverable + the book decomposition
  tails.py                   tail dependence -> tails.csv
  live_crosscheck.py         trades.db: elected / filled / never, fill vs cap -> live_orders.csv
  live_lag.py                trades.db: breakout bar -> fill lag -> live_lag.csv
  live_trigger_quotes.py     the live NBBO at the trigger instant -> live_trigger_quotes.csv
```
`dump_*.csv` (5.6 MB each) are gitignored — regenerable from `rescore_q.py build`. The per-trade
deliverable is **`per_trade_fill_models.csv`** (7,202 rows: cap raw/live, ask at the trigger, filled
later, seconds to fill, lag in bars, fill model, and the P&L under every arm).
