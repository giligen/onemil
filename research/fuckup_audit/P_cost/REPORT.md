# Stage P — the MEASURED per-trade cost of the honest ORB book

**STATUS: complete.** Run 2026-09-18. One python process at a time, `nice -n 10`, `ulimit -v 2500000`.
Everything written is under `research/fuckup_audit/P_cost/`; everything outside it was read only
(`data/cache.db` opened `file:...?mode=ro`). No config, service, cron, cache or order touched.
**Databento spend: $0.00** — see §1 for why, and why that is the right answer rather than a shortcut.

---

## 0. One page

**The band table is 1.9x too wide for THIS book at the median trade and almost exactly right at the
mean trade — and neither fact changes a single ORB conclusion, because ORB's R is 4% of price and the
whole cost question is worth 0.05 R per pick.**

Measured on 7,402 ORB fills (2025-01-02 -> 2026-09-16), consolidated Alpaca SIP NBBO at the fill
instant and at the exit instant:

| cost model, 8-slot book, 215 picks | P&L 21 mo | R / pick | MDD | worst mo | red mo | cost vs zero |
|---|---|---|---|---|---|---|
| **zero cost** (the ceiling: fill at `range_high`, exit at the level) | 16,451 | +0.565 | -470 | -108 | 1 | — |
| **MEASURED** (NBBO ask at the fill instant, NBBO bid at the exit instant) | **15,221** | **+0.520** | -583 | -143 | 1 | **0.045 R** |
| as-is, the shipped 30 bps entry buffer + 10 bps exit slip | 14,429 | +0.495 | -576 | -148 | 2 | 0.070 R |
| the BAND TABLE charged on top of the shipped physics (contract (c)) | 13,635 | +0.469 | -625 | -174 | 3 | 0.096 R |
| *(obtainability, not cost)* strict: an ask above the stop-limit cap is NO fill | 9,239 | +0.332 | -494 | -207 | 2 | — |

The four cost arms span **$2,816 over 21 months — 17% of the gross book, 0.096 R per pick.** Every
arm keeps the same 215 picks (cost is additive here; selection never reads P&L), the same sign, the
same shape, and the same ordering across slot counts. **No conclusion moves. The level moves +/-6%.**

The one number in the table that is NOT a cost finding is the last row, and it is the biggest: on
**14.4% of fills the NBBO ask at the trigger instant was above the stop-limit's own cap**
(`range_high x 1.003`). Those orders would not have lifted that ask. That is an *obtainability*
question about the ORB entry convention, worth **-$5,982 / -0.19 R per pick** at 8 slots and
-$4,673 at 3 slots — four times the entire measured cost of the book. It is flagged here and left
open; Stage P was not chartered to re-open the fill model.

---

## 1. Why no Databento data was bought

The brief allowed $50 to pull `tbbo` for the exit minutes. It was not spent, and the reason is a
measurement fact, not thrift:

* **The data already on disk is the wrong instrument.** `N1/tbbo/*.parquet` is **EQUS.MINI**, a
  *venue subset*: its `bid_px_00/ask_px_00` is one publisher's book, not the consolidated NBBO. On
  the 4,568 entry instants where both sources see the same breakout print (`crosscheck_dbn.py`),
  the venue BBO reads **median 34.2 bps against the NBBO's 24.6 bps, median ratio 1.25x, p90 ratio
  12.7x**. Charging a book the widest venue's quote is a different error from the one this stage
  exists to fix.
* **The band table it is being compared against was built from Alpaca SIP NBBO**
  (`lit_review_2026/build_cost_curve.py`). Measuring with the same source and the same definition
  makes band-vs-measured an apples-to-apples comparison instead of a source artefact.
* Alpaca SIP quotes and trades cover the whole session at **$0** and at 117 symbol-days/minute.
  Buying the exit minutes on EQUS.MINI would have cost money to produce a worse number.

So: **priced plan $0, spent $0, budget $50 untouched.** `fetch_spreads.py` pulled 7,402 entry
minutes (quotes + trades) and 7,402 exit minutes (quotes) from Alpaca SIP in 63.5 minutes.

---

## 2. What was built, and the honesty rails

### 2.1 The exit instants did not exist and had to be reconstructed

`D1_orb/candidates_dump.csv` carries `exit_reason` but no timestamp. `exit_times.py` replays the
**shipped** exit physics — `study_orb_pipeline_static_lock.simulate_winner_stack` with the
`orb.yaml` winner stack ON (static lock 1.75R->0.5R, ATR stop floor k=0.25, 40%@+3R scale-out,
touchgo M/D, 15:45 force close) — with an instrumented twin that also records bar timestamps, and
**asserts the twin reproduces the shipped function's `(exit_price, reason)` on every row**.

> 7,402 entered candidates - twin == shipped on **100%** (0 mismatches) - reason == dump **100.00%**
> - max |pnl_pct diff| **0.000000** - unresolved (no bars / no breakout) **0**.

### 2.2 The instants

| | rule | n |
|---|---|---|
| entry | the **fill instant** = the first TRADE in the breakout minute with `price > range_high` (the print that triggers the resting stop-limit). NBBO = the last quote at or before it. | 99.6% found |
| exit, stop/lock/scale_* | the first quote in the exit minute whose **BID <= the stop level** (`exit_price / (1 - 10 bps)`) — the trigger instant | 4,131 (4,098 found; 33 fell back to the bar's last quote) |
| exit, eod/tag_bb/tag_b1/scale_eod | a **closed-bar** decision, so the last quote at or before the bar's close | 3,271 |

**Rails, asserted in `spreads.py`; the run aborts on failure and all three passed:**
* **R1 causality** — every quote used for a decision is timestamped **at or before** that decision
  (entry quote <= fill print; exit quote inside the exit bar).
* **R2 obtainability of the instant** — the fill print lies inside the breakout minute.
* **R3 test tickers** — `research/scripts/pit_listings.is_test_ticker` applied; **0 rows dropped**
  (no `Z*ZZT` ever reached the ORB candidate set).

### 2.3 Availability audit (PLAN §1 standing rule)

Entry-spread coverage **97.3%**, exit **99.7%**. Missingness is flat across splits (TRAIN 2.72% /
VAL 2.70% / TEST 2.66%) — no split leak. It tilts, as expected, to the thin end: by price band
`<$5` 4.3% - `$5-10` 3.3% - `$10-20` 2.1% - `$20-50` 1.4%; by hour 09:30-09:35 3.1% - 09:35-10:00
2.7% - 10:00-11:00 1.6%. Uncovered rows keep their as-is P&L in every arm, so the measured arms are
**97.1%-coverage estimates** (7,186 of 7,402 entered rows), and the missing 2.9% are the thinnest
names — i.e. the measured cost here is if anything a touch optimistic.

---

## 3. The band constant vs the measurement, per (price band x hour band)

Entry instant, bps of price. `band_bps` = the constant `acore.corrected_spread_table()` charges
(median of the signal-minute median over ~115 sampled `bf_zero2` signals per cell). `minmed_bps` =
the same statistic recomputed on THIS population (like-for-like). `med/mean/p25/p75/p90` = the
distribution of the **fill-instant** spread on this population.

| pb | hb | n | band | minute-med | **med** | **mean** | p25 | p75 | p90 | band/minmed | band/med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $5-10 | 09:30-09:35 | 599 | 35.8 | 28.7 | 24.0 | 47.1 | 15.4 | 54.5 | 101.0 | 1.25 | **1.50** |
| $5-10 | 09:35-10:00 | 1164 | 34.9 | 28.2 | 24.6 | 48.8 | 14.9 | 55.8 | 109.6 | 1.24 | **1.42** |
| $5-10 | 10:00-11:00 | 231 | 29.8 | 18.4 | 17.2 | 51.7 | 13.4 | 43.6 | 99.7 | 1.62 | **1.73** |
| $10-20 | 09:30-09:35 | 768 | 43.8 | 28.0 | 23.1 | 39.2 | 9.4 | 46.8 | 86.4 | 1.56 | **1.89** |
| $10-20 | 09:35-10:00 | 1347 | 42.2 | 24.1 | 20.1 | 35.5 | 8.4 | 44.8 | 77.1 | 1.75 | **2.10** |
| $10-20 | 10:00-11:00 | 293 | 28.8 | 14.7 | 13.2 | 26.2 | 7.5 | 30.7 | 54.2 | 1.95 | **2.18** |
| $20-50 | 09:30-09:35 | 492 | 51.6 | 27.1 | 22.4 | 41.9 | 8.5 | 53.5 | 93.0 | 1.91 | **2.31** |
| $20-50 | 09:35-10:00 | 867 | 46.0 | 23.0 | 19.1 | 31.6 | 6.5 | 41.1 | 73.5 | 2.00 | **2.41** |
| $20-50 | 10:00-11:00 | 176 | 45.3 | 16.6 | 12.0 | 23.6 | 4.7 | 27.8 | 59.4 | 2.73 | **3.79** |
| **<$5** | 09:30-09:35 | 341 | **—** | 32.6 | 31.7 | 61.2 | 25.0 | 64.1 | 112.3 | — | — |
| **<$5** | 09:35-10:00 | 764 | **—** | 31.2 | 31.3 | 67.7 | 24.9 | 68.0 | 150.3 | — | — |
| **<$5** | 10:00-11:00 | 160 | **—** | 28.0 | 28.1 | 58.8 | 23.7 | 53.3 | 125.2 | — | — |

**Is the constant too wide or too narrow? Both, and which one depends on the statistic you owe the
book.** n-weighted over the nine cells the band can price (5,937 of 7,202 rows):

> band **40.6 bps** - measured **median 21.1** (band = **1.92x**) - measured **minute-median 25.1**
> (band = **1.62x**) - measured **mean 39.5** (band = **1.03x**).

The band constant is a **median**, and this distribution is hard right-skewed (mean ~ 1.9 x median in
every cell). A book's average net R is a linear functional of the spread, so the statistic the
contract actually owes is the **mean** — and against the mean the band table is accurate to 3% on
this population. Against the trade you are most likely to take, it is twice too wide. Both
statements are true and the second one is the one N3 hit: on a book whose R is 40 bps the tail of
that distribution is unaffordable, and the median-vs-mean gap is 100% of the argument.

**The band table cannot price 17.6% of this book at all.** Its own population was filtered
`price >= 5`; 1,265 ORB fills are under $5, and they are the widest cells in the table (median
28-32 bps, p90 112-150). Charging them the `$5-10` row understates the typical trade and
understates the tail.

### 3.1 Cross-sectional dispersion inside a cell — what "one constant" costs

The band charges one number to a population spanning an order of magnitude:

| pb | hb | n | p10 | p90 | p90/p10 | IQR/median | share of trades within 2x of the band |
|---|---|---|---|---|---|---|---|
| $5-10 | 09:35-10:00 | 1164 | 11.7 | 109.6 | 9.4 | 1.66 | **0.463** |
| $10-20 | 09:35-10:00 | 1347 | 6.0 | 77.1 | 12.8 | 1.82 | **0.393** |
| $20-50 | 09:30-09:35 | 492 | 4.2 | 93.0 | **22.2** | 2.01 | **0.358** |
| $20-50 | 10:00-11:00 | 176 | 3.9 | 59.4 | 15.3 | 1.93 | **0.267** |
| <$5 | 09:35-10:00 | 764 | 21.8 | 150.3 | 6.9 | 1.38 | — |

Full table in `dispersion.csv`. **p90/p10 runs 5.1x to 22.2x, IQR is 1.0-2.0 x the median, and only
27-49% of trades sit within a factor of two of the constant they are charged.** A per-cell constant
is not a description of any trade in the cell; it is a description of the cell's average, and it is
only usable for a book mean.

### 3.2 Entry instant vs the breakout minute's median — the band's sampling is fine

| hb | n | median instant | median minute-median | median ratio | mean ratio | share instant wider |
|---|---|---|---|---|---|---|
| 09:30-09:35 | 2200 | 26.0 | 29.4 | 1.000 | 1.049 | 0.342 |
| 09:35-10:00 | 4142 | 25.3 | 27.5 | 1.000 | 1.039 | 0.315 |
| 10:00-11:00 | 860 | 19.4 | 21.1 | 1.000 | 1.000 | 0.281 |
| **ALL** | **7202** | **24.8** | **27.3** | **1.000** | **1.037** | **0.319** |

**No, the breakout minute's median does NOT understate the fill instant.** On 68% of fills the
instant spread is at or *inside* the minute median (on a thin name the quote is one tick wide for
the whole minute), the median ratio is exactly 1.000, and the mean ratio is 1.037. The band table's
use of a minute median in place of the fill instant is worth **under 4%** — it is not where the
error is. The error is median-vs-mean and population, not timing.

### 3.3 The exit side

`exit_cell_table.csv` carries the same table keyed on the EXIT hour. Two facts: exit-instant spreads
run **1.2-1.6x the entry-instant spread in the same price band** (a stop firing is, by construction,
a moment of one-sided flow), and they decay through the day exactly as the band table's shape
predicts ($10-20: 29.6 bps at 09:30-09:35 -> 12.7 after 13:00). The band constant is 0.9-2.4x the
measured exit median, i.e. the same picture as the entry with the $5-10 cells slightly *under*-charged.

---

## 4. The re-scored book

Every arm is produced by the **same pipeline path** that produced `book_n8_q1on.csv`
(`study_orb_pipeline_static_lock.py` + `ORB_BT_RESIM_CACHE`, `ORB_BT_N`, `ORB_BT_ACCOUNT = 3333.33 x N`,
`ORB_SKIP_Q1=1`, `ORB_BT_RISK=375`). **Parity control: the `asis` arm reproduces D1's book to the
cent — 215 picks, $14,428.62, +0.486 / +0.862 / +0.172 R per pick, MDD -576, worst -148, 2 red
months.** Cost is additive in this pipeline (selection reads `_composite`, quintiles and the vetoes,
never P&L), so every arm holds the same picks and the same slot mechanics; only the dollars move.
Share counts are held at the as-is value in every arm so the arms differ **only in price**.

### 8 slots (215 picks, 162 fills, 21 months)

| arm | P&L | R/pick | TRAIN | VAL | TEST | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|---|---|
| zero cost (ceiling) | 16,451 | +0.565 | +0.559 | +0.934 | +0.232 | -470 | -108 | 1 |
| **MEASURED** | **15,221** | **+0.520** | **+0.515** | **+0.854** | **+0.216** | -583 | -143 | 1 |
| as-is (shipped 30/10) | 14,429 | +0.495 | +0.486 | +0.862 | +0.172 | -576 | -148 | 2 |
| band table | 13,635 | +0.469 | +0.459 | +0.834 | +0.146 | -625 | -174 | 3 |
| strict obtainability | 9,239 | +0.332 | +0.459 | +0.281 | +0.148 | -494 | -207 | 2 |

### 3 slots (88 picks, 70 fills) — the shipped book

| arm | P&L | R/pick | TRAIN | VAL | TEST | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|---|---|
| zero cost (ceiling) | 8,049 | +0.610 | +0.579 | +0.981 | +0.355 | -568 | -184 | 4 |
| **MEASURED** | **7,544** | **+0.558** | **+0.521** | **+0.908** | **+0.337** | -647 | -228 | 5 |
| as-is (shipped 30/10) | 7,187 | +0.532 | +0.502 | +0.902 | +0.278 | -714 | -223 | 6 |
| band table | 6,878 | +0.506 | +0.477 | +0.873 | +0.250 | -758 | -246 | 6 |
| strict obtainability | 2,871 | +0.229 | +0.435 | -0.076 | +0.053 | -566 | -295 | 10 |

### Where the measured cost sits

On the 8-slot book's 162 fills, at book sizing: the **entry is measured $697 CHEAPER** than the
shipped assumption (the NBBO ask was at or below the stop-limit cap on 73.5% of fills, so the order
gets its price or better), and the **exit is measured $502 WORSE** (the bid at the stop-trigger
instant sits below the modelled `level x (1 - 10 bps)`). Net, the shipped 30/10 model is
**$792 pessimistic** over 21 months. The 30 bps entry buffer is not a cost estimate at all — it is
the limit price, and it is a conservative one.

### Tail dependence (PLAN §1 item 5)

| arm | slots | n | mean R | ex top 1% | ex top 5% | winners capped +3R | t |
|---|---|---|---|---|---|---|---|
| as-is | 8 | 215 | 0.495 | 0.398 | 0.165 | 0.296 | 3.93 |
| measured | 8 | 215 | 0.520 | 0.422 | 0.190 | 0.319 | 4.08 |
| band | 8 | 215 | 0.469 | 0.371 | 0.139 | 0.271 | 3.71 |
| as-is | 3 | 88 | 0.532 | 0.432 | 0.250 | 0.350 | 2.71 |
| measured | 3 | 88 | 0.558 | 0.459 | 0.277 | 0.375 | 2.81 |
| band | 3 | 88 | 0.506 | 0.405 | 0.223 | 0.326 | 2.57 |

The book stays positive under every tail treatment in every arm, and the **cost correction never
changes which side of any line the book is on** — ex-top-5% it is +0.14 to +0.19 R whichever cost
model is used, and t moves 3.71 -> 4.08 across the whole range of cost assumptions. (The book's
tail-dependence is the same as D1 reported; nothing here improves or worsens it.)

---

## 5. The honest verdict for ORB

**The cost correction changes the level, not one conclusion.**

1. Over 21 months and 215 picks, every cost model from "free" to "the band table" fits inside
   **$2,816 / 0.096 R per pick / 17% of the gross book**. D1's slot dose-response, the veto
   standings, the Q1 filter, the 3-vs-8 slot choice and the "ORB is a $342-687/month book at stage
   sizing" sentence are all unchanged at every point in that range.
2. The reason is arithmetic, and it is the mirror image of N3. **ORB's median R is 4.03% of price**;
   a measured 25 bps round trip is **0.065 R**. N3's published ORB has an R of 40 bps and the same
   spread is **1.0 R**. The band table is not "wrong"; it is a number with no denominator. *Spread
   in bps is never a cost — spread / R is.*
3. The shipped 30 bps / 10 bps model is **pessimistic by $792 over 21 months (0.025 R per pick)**.
   It does not need changing, and changing it would buy 5%.
4. **The one thing here that could matter to ORB is not a cost.** On 14.4% of fills the NBBO ask at
   the trigger instant was above the stop-limit's own cap. The BT books those as fills at the cap.
   If they are instead the live outcome — the order rests, the 60-minute time stop cancels it —
   the 8-slot book is $9,239 (+0.332 R) and the 3-slot book's VAL split goes negative
   (-$211, -0.076 R). That is **four times the entire cost question** and it belongs to the fill
   model, not to the spread. It is reported, not resolved: the honest test needs the quote path over
   the whole 60-minute window (does the ask come back under the cap?), which is another Alpaca pull
   of the same size as this one and was outside this stage's charter.
5. **Phrasing.** Nothing here says a cost correction cannot matter. It says that **in THIS
   population** (7,402 ORB fills, 2025-01 -> 2026-09), **at THIS R** (median 4.0% of price), **under
   THIS entry convention** (a capped stop-limit that truncates the spread tail it pays), **at THIS
   book size** (3 and 8 slots, $10K stage), the full span of defensible cost models is 0.096 R per
   pick, and the smallest per-pick effect the 8-slot book could resolve at 80% power is **0.424 R**
   (N1's MDE on the same 215 picks). The cost question is **four times smaller than this book's own
   detection threshold**.

---

## 6. The transfer question — should B, C, E and J be re-scored?

**Recommendation: NO for a re-score, YES for one cheap, targeted change to the contract that does
not need any rebuild.** Reasoning, with the arithmetic that decides it:

**(a) The band table is already in-population for B/C/E/J, and it is not in-population here.**
`cost_curve.csv` was sampled FROM `bf_zero2/candidates3.csv` — the >=5%-range-day population those
four stages score. The 1.9x median gap measured above exists *because ORB's names are different*:
ORB candidates clear a 500K prior-day volume screen and a 15K 09:35 volume floor, so they are the
liquid end of the mover distribution. Transferring "the band is 1.9x too wide" from ORB to `bf_zero2`
is exactly the population error the band table itself commits, run backwards. **There is no measured
basis for expecting the same gap there, and this stage produced none.**

**(b) Even the maximum conceivable gain is below those stages' own MDE.** Stage C reports the whole
band charge directly: over its 107 cells, net -0.073 R vs gross -0.016 R, i.e. **the band contract
costs 0.057 R per trade on that population**. That is the ceiling on ANY cost correction there —
setting the spread to exactly zero is worth +0.057 R. The closest miss in the entire stage (`F6 {}`,
next-open fill, 2R stop-1%, entries >= 10:00) is TRAIN +0.031 R with **MDE 0.0615 R** and needs
0.044 R to clear G1; its gross is +0.049 R. So at zero cost it still sits inside its own MDE, and
Stage C's permutation null already puts the 95th percentile of max-t over 107 cells at **3.88**
against an observed best of **1.93**. A correction that can move the mean by at most +0.057 R cannot
carry a t from 1.93 past 3.88. **The re-score cannot change Stage C's verdict, and Stage C is the
scoring stage for B; J is 0 of 144 on the same gate; E was never scanned.**

**(c) What the compute would actually cost.** B's pass-1 rebuild is the binding item: its own report
records **19.8 s/day -> 138.5 min and 1.99 GB** for `candidates4.csv` (3.1M signal rows), and that is
the cheapest of the four (J's tape alone was 6.6 GB, deleted; E's causal universe builder was never
run to completion). Honest estimate to put all four back on disk and re-score them: **B ~ 2.5 h +
2 GB - C ~ 20 min (scoring is cheap once B exists) - E ~ 4-6 h + ~4 GB (a build that has never
finished) - J ~ 6-8 h + 6.6 GB.** Call it **13-17 compute hours and ~13 GB on a node with 26 GB
free and a live trader on it**, to move a set of cells whose best t is 2 units below its own noise
ceiling by at most +0.057 R.

**(d) What IS worth doing, for free.** Three changes to the cost contract, each supported by a
measurement in this report and none needing a rebuild:

1. **Charge the cell MEAN, not the cell median.** `acore.corrected_spread_table()` takes
   `.median()` of a distribution whose mean is 1.9x its median in every cell. A book's mean net R
   owes the mean. Rebuilding the constant from `cost_curve.csv` with `.mean()` is a one-line change
   and makes every past number *more* conservative, not less — which is the direction an audit
   should err.
2. **Extend the table below $5.** It has no cell there and 17.6% of ORB's fills live there; the
   `bf_zero2` population was filtered at $5 so the gap is invisible in those stages, but any future
   population that allows sub-$5 names is unpriced today.
3. **Stop charging a capped limit the full spread distribution.** The measurement that actually
   generalises from this stage is mechanical, not populational: when the entry is a limit with a cap
   (ORB's `range_high x 1.003`, `bf_zero`'s next-open-under-0.6%), the fills you get are drawn from
   the LEFT of the spread distribution — 73.5% of ORB fills were at or inside the cap — and the
   right tail shows up as **non-fills, not as cost**. Contract (c) currently charges the cell
   constant to every fill and prices no non-fills at all, which double-counts in one direction and
   ignores the other. The honest form is the pair (`entry at min(ask, cap)`, `ask > cap => no fill`),
   which is what this stage's `measured` and `strict` arms are.

**The next stage this points at is therefore not B/C/E/J.** It is the 60-minute quote path behind
§5 item 4 — the same Alpaca pull, $0, ~1 hour of wall clock — because that decides whether the ORB
entry convention is worth $6,000 of the 21-month book, which is six times everything Stage P
measured.

---

## 7. Cells looked at

* **1 population** (7,402 entered ORB candidates), **1 measurement** per trade at **2 instants**.
* Descriptive: 12 (pb x hb) entry cells + 20 exit cells + 12 dispersion rows + 4 instant-vs-minmed
  rows = **48 descriptive cells**.
* **5 cost arms x 2 slot counts = 10 pipeline runs**, each read on 1 whole window + 3 splits =
  **40 config x split cells**, plus 10 tail rows.
* **0 thresholds tuned, 0 parameters fitted, 0 selections made.** No veto, z-param, quintile cutoff,
  mult or slot count was moved from `orb.yaml`. TEST is reported because the book's picks are
  identical in every arm — this stage selects nothing, so there is nothing for a TEST read to
  contaminate.

## 8. Approximations, each one a place this could be wrong

1. **Exit instants are inferred from quotes, not from the exit print.** For a stop/lock the trigger
   is taken as the first quote whose bid <= the stop level; the true trigger is a trade. On a thin
   name the two are within one quote update; on a fast one they can differ.
2. **2.9% of entered rows have no measured spread** and keep their as-is P&L (§2.3). They are the
   thinnest names, so the measured arms are marginally optimistic.
3. **The `band` arm charges the band contract ON TOP of the shipped 30/10 physics**, because that is
   the only way to apply contract (c) to a book whose gross is not separately stored. B/C/E/J charge
   it to a true gross. The `zero` arm bounds that difference: the shipped physics are 0.025 R.
4. **Share counts are held at the as-is value in every arm.** A cheaper entry would in reality buy
   marginally more shares at a fixed risk; holding shares fixed isolates price and understates the
   measured arm by a fraction of a percent.
5. **The scale-out leg is charged nothing in the measured arm** (a resting limit at +3R does not
   cross the spread) and 10 bps in the as-is arm. If ORB's live scale leg is ever sent as a market
   order this is wrong by ~0.5 bps of the book.
6. **Alpaca SIP quote history is the reference**, not a tick-by-tick NBBO reconstruction; a quote
   that existed for microseconds between two returned records is invisible.
7. **No borrow, no fees, no impact.** The participation question is D1's (`capacity.py`), unchanged.

## 9. Files

```
research/fuckup_audit/P_cost/
  REPORT.md               this file
  exit_times.py           the instrumented replay + its parity assertion -> exit_times.csv
  fetch_spreads.py        Alpaca SIP quotes/trades at both instants -> spread_rows.csv
  spreads.py              rails + spreads.parquet + cell_table.csv, dispersion.csv,
                          exit_cell_table.csv, instant_vs_minmed.csv
  rescore.py              build | run | analyse -> dump_<arm>.csv, book_<arm>_n{8,3}.csv,
                          monthly_*, rescore_table.csv, measured_cost.csv
  decompose.py            entry-vs-exit split + tails.csv
  crosscheck_dbn.py       EQUS.MINI venue BBO vs SIP NBBO -> crosscheck_dbn.csv
```
`spreads.parquet`, `spread_rows.csv`, `exit_times.csv` and `dump_*.csv` are gitignored (bulk,
regenerable from the scripts).
