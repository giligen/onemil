# Which strategy families would benefit most from our Databento entitlements

Written 2026-09-18. Literature review + costing only — **no data was pulled**; every dollar figure below comes from
`metadata.get_cost` on the live key (calls logged in §A). No engine, config or cache was touched.

**Question answered:** ranked by expected value to a **$50K account trading US equities intraday**, where
`EV = published effect size × our ability to EXECUTE it live on Alpaca × the power of the test our data would support`.

**Phrasing rule (PLAN.md §1) applies throughout.** Nothing here says "no edge exists". Where a near-neighbour was
already run, the statement is "no edge was detectable in THAT universe / window / book / cost, and the smallest effect
that test could see was X".

---

## 0. The four constraints that decide the ranking (read before the table)

**(a) XNAS.ITCH is one venue, not the tape.** Our own N3 calibration measured **median XNAS share of consolidated
volume = 0.114** (p25 0.053, p75 0.272, 188,793 symbol-days); the XNAS close is within 0.1% of the consolidated close
on only **59.8%** of symbol-days. So any XNAS.ITCH *book* or *trade* signal (mbo, mbp-10, mbp-1, tbbo, ohlcv) measures
~11% of the flow. Two schemas are **exempt from this** and that is what makes them interesting:
- `imbalance` — the Nasdaq opening/closing **cross** is single-venue *by construction*. For a Nasdaq-listed security
  there is no other opening or closing auction. 100% of the auction, not 11% of it.
- `status` — LULD halts are **market-wide regulatory events** (Reg NMS Plan Rule VI); every venue must halt. Nasdaq's
  trading-action messages therefore describe a state that is true on all venues.
  *(Caveat: whether XNAS.ITCH `status` emits actions for NYSE-**listed** names that Nasdaq merely trades is
  UNVERIFIED — see §S1 step 0, a $0.70 check.)*

**(b) Half our universe is not Nasdaq-listed.** `data/cache.db::universe` (7,346 rows, read-only):
NASDAQ **3,639 (49.5%)** · NYSE 1,459 (19.9%) · ARCA 1,234 (16.8%) · BATS 777 (10.6%) · AMEX 237 (3.2%).
ARCA+BATS (27.4%) is overwhelmingly the ETF / leveraged-wrapper side. Note also that **46% of the 13,605 symbols with
daily bars have no `universe` row at all** (delisted names, the 307-symbol wrapper backfill, ORB-broad symbols), so the
venue split is reliable only for the 54% that are current members. Practical read: an `imbalance`- or
`definition`-based study covers **about half** our tradeable names, and essentially none of the wrappers.

**(c) Our order stack is DAY-limit only, and we are blind to halts.**
`data_sources/alpaca_client.py` — `TimeInForce.DAY` is the **only** TIF anywhere in the repo (6 submit sites:
1380, 1855, 1970, 2038, 2094, 2152). **Zero occurrences of `opg`, `cls`, `ioc`, `fok`, `gtc`.** `MarketOrderRequest`
is imported (line 38) and never used. `extended_hours` is never set → RTH only. Order classes used: `BRACKET` and
`SIMPLE`; no OTO/OCO submitted. Alpaca's API *does* accept `opg` (MOO/LOO, before ~09:28 ET) and `cls` (MOC/LOC,
before ~15:50 ET) on **simple** market/limit orders — but they **cannot be a bracket/OTO/OCO leg**, so an auction exit
cannot reuse our bracket machinery; it needs a standalone order plus a prior cancel of the SL leg (the held-qty problem
already documented at `orb_engine.py:3840/3876`). This is a one-parameter change plus one new submit method — but it is
**unwritten and untested against Alpaca today**, which is why every auction candidate carries an executability gate.
**Latency is not the constraint:** HOD-break reacts **1–3 s after a bar close** (`hod_break_engine.py:535` drain thread,
`SETTLE_S=0.4`, pending polled every ~1 s) and StopMonitor's quote/trade handlers (`stop_monitor.py:3025/3084`) fire
**sub-second on every websocket message**. But repo-wide grep finds **no LULD band ingestion, no trading-status
subscription, and no halt/resume handler** — `halt` matches only `trading/system_state.py`, an account-level kill flag.
We would react to a reopening print as if it were any other tick.

**(d) The cost contract is the binding constraint, not the signal.** From `cost_by_outcome.md` / `cost_curve.md`:
blended expected cost **0.135 R/trade** (eod 0.083 / stop 0.194 / target 0.126); required `spread/R` for viability
**≈ 0.05**, versus 0.076 (>$50M/day names, midday) to 0.463 (<$5M/day, 09:35–10:00); the **first five minutes is the
worst window in every band**; a tight stop is penalised twice (1–2% stop → 0.190R cost and P(stop) 50.6%; 8%+ →
0.026R and 12.5%). **The best honest gross edge this program has ever measured is ≈0.1R.** Any candidate whose
published effect is a few bps on a liquid name is dead on arrival at our spreads; any candidate that forces a
first-five-minutes entry starts 0.19–0.36R in the hole.

**Power convention used below.** MDE (minimum detectable effect at gate G1, `t ≥ 2.0`) `= 2.0 × sd_R / sqrt(n)`, with
`sd_R = 1.2` — the working per-trade net-R dispersion of our stop-and-target books. This is an assumption from our own
books, not a fetched number; where a candidate is measured in bps/day rather than R, the MDE is stated in bps instead.

---

## 1. Ranked table

`Pub` = strength of published, out-of-sample-surviving, US-equity, post-2010 evidence (H/M/L).
`Exec` = can we place the order live on Alpaca today or after a small, testable change (H/M/L/None).
`Power` = MDE our data would buy at our book size. `$` = priced with `get_cost` (§A).

| # | Candidate | Pub | Exec | Power (MDE) | Data needed | $ | Prior near-neighbour (LOG.md verdict) |
|---|---|---|---|---|---|---|---|
| 1 | **LULD halt-resume, long and short** | **L** | **H** | **H** (≈0.03–0.07R) | XNAS.ITCH `status` ALL 2018-05→now; outcome bars from the SIP tape on disk / Alpaca | **$54.36** | **none — never tested anywhere in the tree** |
| 2 | **Closing-cross imbalance (NOII) → MOC/LOC** | M | **L** | H (≈0.11R booked; bps cross-section huge) | XNAS.ITCH `imbalance`, Nasdaq-listed subset, 2 yr | **$240** (pilot) | M20 EOD/closing-auction loser reversal — **dead, gross below cost** |
| 3 | **Point-in-time `definition` (listings, splits, test tickers)** | n/a (infrastructure) | n/a | n/a | XNAS.ITCH `definition` ALL 2024-07→now | **$35.20** | ZVZZT contaminated F6's TEST (+50.8R of +39.0R); ORB entered-only + ignition cohort look-aheads |
| 4 | **Opening-cross imbalance fade** | M | L | M (≈0.12R) | same file as #2 (both crosses are in it) | $0 marginal | M18 open fade — **CONFIRMED as a veto**, the only supportive prior in the tree |
| 5 | **MBO depth / hidden-fill / cancellation at the break level (veto)** | M | **L** (no live depth feed) | **H** (≈0.02R) | XNAS.ITCH `mbo`, 10-min windows around signals, 2 yr | **$30–60** | N1 (EQUS.MINI tbbo OFI at the ORB break): **+$300 from 12 cells, adjusted p≈0.5** |
| 6 | Level-1 OFI (CKS) redone on a real tape | M | **M** (computable from Alpaca quotes live) | H (≈0.02R) | XNAS.ITCH `mbp-1`/`tbbo` at signal windows | $20–40 | N1, same row — and N1's own stated fix was *"the same features on a consolidated tape (XNAS.ITCH for Nasdaq names)"* |
| 7 | Volume-profile / queue-depletion breakout confirmation | L | H | H | `ohlcv-1s` + `mbp-10` at signal windows | $30 | `rv_profile` is already inside HOD-break; M9 found **the paper's RV monotonicity absent** in our universe |
| 8 | `statistics` schema (official auction prices, halt reference data) | n/a | n/a | n/a | XNAS.ITCH `statistics` ALL 6.5 yr | $43.03 | none |
| 9 | Short-horizon reversal with microstructure conditioning | M | M | M | `mbp-1` + daily panel on disk | $40 | M36/M37 **−105…−174 bps**, M36 by vol-regime **no regime consistently positive** |
| 10 | Queue position / passive liquidity provision | M | **None** | H | `mbo` | — | B: resting fill only **56–88% queue-OK**; C: **H2 resting fill REJECTED, −0.026R _gross_ (t −61)** |
| 11 | ETF arbitrage (ETF vs underlying basket) | **H** | **None** | H | `mbp-1` on ETF + constituents | $100+ | lit-review rows 1/1b/4/13 (index-intraday, not arb) |
| 12 | Standalone iceberg / hidden-liquidity detection as alpha | L (futures only) | None | H | `mbo` | $30 | zero hits anywhere in the research tree |

**Bottom line of the table:** there is **no candidate that has both a strong published tradable effect and clean
executability on our stack**. The two channels XNAS.ITCH uniquely opens (`imbalance`, `status`) are the two with the
thinnest peer-reviewed *tradable* effect sizes; the channels with strong published effects (ETF arb, OFI price impact)
are either not executable by us or are contemporaneous-explanatory rather than predictive. That is the honest shape of
the opportunity set, and it is why the shortlist in §3 leads with the **cheapest untested channel in our own universe**
rather than with the best-cited paper.

---

## 2. Candidate by candidate

### 1. LULD halt-resume — **rank 1**

**Mechanism (one sentence).** A LULD halt freezes a stock after a 5-/10-/20-% band excursion in 5 minutes and reopens
it through a 5-minute reopening auction; the other side is (i) forced liquidation and margin-call flow that could not
exit during the halt, and (ii) the market makers who must quote the reopening cross with inventory they did not choose —
whoever supplies the reopening liquidity is paid for bearing that inventory, and whoever chases the reopening print is
paying for it.

**Published evidence — and this is the weak leg.** US single-stock LULD literature is about *market quality*, not about
a tradable post-resume return:
- Dalko, V. (2016), *"Limit Up–Limit Down: an effective response to the 'Flash Crash'?"*, **Journal of Financial
  Regulation and Compliance**, DOI `10.1108/jfrc-04-2016-0040`, 7 citations — policy assessment, no return effect size.
- McFarland, S., Jain, P. & McInish, T. (2022), *"The Effectiveness of Single Stock Circuit Breaker Designs: The
  Special Quote and Limit Up-Limit Down Rules"*, SSRN `10.2139/ssrn.4288811`, **0 citations** — unpublished, no
  replication.
- Lin, Y. (2017), *"Limit Up Limit Down, Exchange Access Fee and High Frequency Trading Around Price Limits"*,
  SSRN `10.2139/ssrn.3019986`, 1 citation.
- The large "magnet effect / post-limit reversal" body — Zeng, Wang & Tang (2024, *Finance Research Letters*,
  `10.1016/j.frl.2023.104803`); Curran & Mollica (2018, SSRN `3115844`); Dong et al. (2024, *EMFT*,
  `10.1080/1540496x.2024.2434042`) — is **China and Taiwan price-limit regimes, not US LULD**, and US LULD is a
  5-minute halt, not a daily limit. **Flagged: the published effect size does not transfer.**
- The closest US anchor already in our tree is Kavajecz & Odders-White (2004, *RFS*): support/resistance levels coincide
  with peaks in limit-order-book depth — the mechanism by which a reopening price sits where the book is thick.

So: **the sign of this effect is practitioner folklore, not published finance.** It ranks first anyway because the
other three factors dominate — and because an untested channel in exactly our universe is worth more than a
well-published effect we cannot trade.

**Data.** XNAS.ITCH `status`, `ALL_SYMBOLS`, 2018-05-01 → 2026-09-17 = **$54.36** (2025-01→now alone = $12.41). This is
the single cheapest deep-history buy available to us, and halts are rare enough that we need the years. Outcome bars:
2025-01→2026-09 is already on disk (`research/bf_zero/bars_sip.db`, consolidated SIP, 16 GB, free); pre-2025 halted
symbol-days can be fetched from **Alpaca** (consolidated, free, already wired) — do **not** use XNAS.ITCH `ohlcv-1m`
for outcomes (11% venue share; §0a), though it would cost only ~$11 (`$0.000365`/symbol-day).

**Executability — the best of any candidate.** Sub-second tick reaction already exists (`stop_monitor.py:3025/3084`);
DAY limit orders are all we need; no new live market-data subscription is required, because the *live* signal is "the
halt ended", which Alpaca's own SIP websocket exposes on its `statuses` channel (**needs a one-line verification
against our key** — we have never subscribed to it). The entry is a capped marketable limit on the first post-resume
bar, i.e. exactly the HOD-break engine's existing fill convention. **Gap:** we have zero halt code today, so a positive
result costs ~1 day of engine work, not a new data contract.

**Power.** Order-of-magnitude: US equities see roughly 10–20 LULD halts per session in names inside our band
(price ≥ $5, ADV20 ≥ 100K) — over 6.5 years that is ~15K–30K events, over 2025-01→2026-09 alone ~4K–8K. At n = 5,000
and a fat `sd_R = 2.5` (halt-resume returns are far more dispersed than a normal breakout), **MDE ≈ 0.071R**; at
n = 15,000, **MDE ≈ 0.041R**. This is the only candidate on the list whose MDE is comfortably *below* the 0.135R cost
line, i.e. the only one where a null would actually be informative.

**What would refute it.** Mean net R ≤ 0 on TRAIN with t < 2.0 on ≥ 5 trades/week; or a positive TRAIN that fails VAL's
55%-weeks-green; or — the specific failure mode to watch — an effect that lives entirely in the first post-resume
minute, where our `cost_curve.md` says spreads are 3–5× normal and `spread/R` will exceed 0.4.

**Prior.** **None.** `halt` / `LULD` appear in `lit_review_2026/HYPOTHESES.md`, `RUNBOOK.md` and `B_stocks_in_play.md`
as literature only, and in **no run anywhere in `research/`**. This is the largest genuinely untested surface we have.

---

### 2. Closing-cross imbalance (NOII) → MOC/LOC — **rank 2**

**Mechanism.** From 15:50 ET Nasdaq disseminates the closing-cross Net Order Imbalance Indicator every second; index
funds, rebalancers and MOC flow are price-insensitive and must print at the close, so a large one-sided imbalance is a
pre-announced liquidity demand — the money is in supplying the other side and unwinding the next morning.

**Published evidence.** The strongest citation is **Bogousslavsky, V. & Muravyev, D. (2023), *"Who trades at the close?
Implications for price discovery and liquidity"*, Journal of Financial Markets**, DOI `10.1016/j.finmar.2023.100852`,
29 citations — documents who supplies closing-auction liquidity and the price-discovery/liquidity trade-off; it is an
anatomy paper, **not a tradable-strategy paper with a reported Sharpe**. The mechanism-level evidence is
**Challet, D. & Gourianov, N. (2018), *"Dynamical regularities of US equities opening and closing auctions"*,
arXiv:1802.01921** — "the indicative match price is **strongly mean-reverting because the imbalance is**", and the
final auction price responds asymmetrically to order events at the open vs the close depending on imbalance direction.
Read carefully, Challet–Gourianov argues **against naively following the imbalance and for fading it**. Older
closing-auction work (Aitken, Comerton-Forde & Frino 2005, *Accounting & Finance*, `10.1111/j.1467-629x.2005.00155.x`,
43 cites) is **Australia, 1997 — flagged as neither US nor post-2010**. I could not locate a peer-reviewed US paper
reporting a tradable effect size for NOII specifically; **if one exists it is not indexed in Crossref/arXiv under the
obvious terms.** Treat the effect size as unquantified.

**Data.** XNAS.ITCH `imbalance` carries **both crosses** (opening 09:28–09:30, closing 15:50–16:00) in one file.
Measured rates: `ALL_SYMBOLS` 1 day = **$9.13** (so 2 years ≈ $4,565 — out of the question); **20 liquid symbols ×
2 years = $7.92** ⇒ ≈ **$0.40/symbol/2yr**, ≈ $1.45/symbol for the full 6.5 years. Thin names bill less.
**Pilot: 600 Nasdaq-listed names from our universe × 2 years ≈ $240.** Full active-Nasdaq coverage (2,908 names ×
2 yr) would be ≈ $1,160 and should only be bought after the pilot.

**Executability — the reason this is rank 2 and not rank 1.** Three separate obstacles, in order of severity:
1. **Live signal.** NOII is a real-time Nasdaq feed. To *act* on it at 15:55 we would need a **recurring Databento live
   subscription**, which is not in the historical price above. A historical study that we cannot feed live is a study
   we cannot ship.
2. **Order type.** MOC/LOC requires `tif='cls'` submitted before ~15:50, `simple` order class only. We have **zero
   `cls` code**; it cannot be a bracket leg, so it needs a new submit method plus a cancel-then-submit sequence on the
   shares the SL leg is holding.
3. **Routing.** Whether an Alpaca `cls` order actually participates in the **Nasdaq** closing cross for a Nasdaq-listed
   name (rather than being routed elsewhere) is unverified.
Each is settleable, but (1) is a recurring cost and (2)+(3) are a paper-trade experiment that must succeed *before* any
data is bought.

**Power.** Cross-sectionally enormous (600 names × 500 days = 300K symbol-day observations, MDE in single bps). But the
**book** is what matters: at 4 concurrent, one entry per day, 2 years ⇒ ~500 trades ⇒ **MDE ≈ 0.107R** — only just
under the 0.135R cost line. And the trade is measured in bps on a close-to-next-open horizon, where our
`cost_curve.md` gives the *cheapest* cell in the whole table (eod exit, 0.083R total cost) — this is the one candidate
whose exit is structurally cheap.

**What would refute it.** Imbalance-sorted quintiles that do not separate monotonically on TRAIN; or separation that is
entirely inside the top/bottom 1% of imbalance ratios (tail-dependence, which killed M41 and D0); or — most likely —
an effect smaller than the 4–7 bps the close→open leg costs once the MOC cannot be netted.

**Prior.** **Row 7 of `RESULTS.md` (M20)**: "EOD/closing-auction loser reversal (≤ −8% at 15:00 → enter 15:30, exit MOC)
— gross −0.7/+40.2/−3.2 bps, net −21/+20/−23 → **dead**: VAL's positive is period-wide (the winner control is +17 too),
and the deepest tail's +20 bps gross is **below cost**." That test used a **price-move trigger, not the imbalance
feed** — so the NOII-conditioned version is a genuinely new cell, but the prior on the horizon is poor and must be
disclosed in the pre-registration.

---

### 3. Point-in-time `definition` — **rank 3, and it is not a strategy**

**Mechanism.** None — this is infrastructure. It buys listing dates, delisting dates, security type, and split/
corporate-action factors at each point in time.

**Why it ranks this high.** Our program has been burned twice by exactly what this file fixes, both documented:
(i) **`ZVZZT`, a NASDAQ test ticker with 0 rows in `daily_bars`, contributed +50.80R of F6's +39.00R TEST result** —
ex-test-ticker TEST was −11.80R (`H/F6_reconcile`); (ii) survivorship and price-scale bugs are standing items in
CLAUDE.md's honesty checklist, and PLAN.md §1 now carries a hand-written exclusion rule (`^Z[A-Z]ZZT$`) that a
`definition` file would make unnecessary and complete. A $35 file that removes a whole *class* of false positive is
better value than most $240 alpha studies.

**Data and cost.** `definition`, `ALL_SYMBOLS`: **2024-07-01 → 2026-09-17 = $35.20** (matches our existing PIT daily
panel); full 2018-05-01 → now = **$120.94**. **Nasdaq-listed only** (§0b) — for NYSE/ARCA/BATS names we still rely on
`EQUS.SUMMARY` `ALL_SYMBOLS` daily (delisted included, already proven clean by N2: 99.0% of 200 keys within 0.01% of
Alpaca OHLC, median abs 0.0000%, raw/unadjusted).

**Recommendation:** buy the $35.20 slice regardless of which study runs. It is not on the shortlist because it
produces no P&L by itself.

---

### 4. Opening-cross imbalance fade — **rank 4**

**Mechanism.** Overnight news concentrates market orders into the 09:28–09:30 opening cross; the cross price is set by
whoever will absorb them, and Challet & Gourianov (arXiv:1802.01921) find the indicative match price **mean-reverts
because the imbalance does** — i.e. the opening print overshoots the imbalance and gives it back.

**Published evidence.** Same two citations as #2 (Challet & Gourianov 2018; Bogousslavsky & Muravyev 2023). Effect size
for a *US opening-cross fade* is unpublished so far as Crossref and arXiv index it.

**Why it is separated from #2.** It shares the data file (zero marginal cost) but it has **the only supportive prior in
our tree**: `RESULTS.md` row 10, **M18 open fade on prior-day attention names — "−60/−26/−20 bps open→10:30; minus
control −46 bps (t −4.9) TRAIN → CONFIRMED as a veto"**, and row 9, **M16: 0 of 27 gap × dollar-volume cells reach
+50 bps open→close; gap-ups > +10% run −60 to −190 bps — "the table says fade, not chase."** We already know our
universe fades at the open. The open question the imbalance file answers is whether the *size and side of the cross
imbalance* sharpens that veto into a tradable short, or merely re-describes it.

**Executability.** Worse than #2: an opening-cross trade needs `tif='opg'` **before 09:28**, and the fade leg is a
short in low-float small caps (borrow, locate, and our engines have no short path — LOG stage **G** built a mirrored
short study, not a short engine). More realistically this becomes a **veto/sizing input** to the existing books, which
is executable today and needs no new order type at all. **That is how it should be scoped.**

**Power / refutation.** As a veto: applied to the ORB and BF candidate streams (~2,460 distinct ORB symbols, thousands
of candidate symbol-days/yr), n is in the thousands, **MDE ≈ 0.04R**. Refuted if imbalance-conditioned vetoes do not
beat the M18 attention-veto baseline — and note the standing warning from LOG stage **L**: of 8 transfer filters across
6 books and 44 cells, **9 passed TRAIN+VAL, 4 improved TEST, 1 survived ex-top-5%, and the whole grid sat inside its
noise (p 0.794)**.

---

### 5. MBO depth / hidden-fill / cancellation at the break level — **rank 5**

**Mechanism.** Kavajecz & Odders-White (2004, *RFS*) — support/resistance levels coincide with **peaks in book depth**.
So a breakout level is where the resting sell book is thickest, and the question a breakout trader actually faces is
whether that book is being *consumed* (real demand) or *cancelled* (spoofed/withdrawn). Full MBO is the only schema
that distinguishes a fill from a cancel, and it is the only way to see hidden/iceberg replenishment behind the level.

**Published evidence.** Hidden liquidity: **Moinas, S. (2005), *"Hidden Limit Orders and Liquidity in Limit Order
Markets"*, SSRN `10.2139/ssrn.676564`** (20 cites) — concealment as informed-trader camouflage; **Frey, S. & Sandas, P.
(2009), *"The Impact of Hidden Liquidity in Limit Order Books"*, SSRN `10.2139/ssrn.1343538`** (7 cites) — *"when
concealed liquidity remains undetected, liquidity suppliers capture substantial surplus gains… once discovered,
competitive pressures erode these advantages"*, which is a direct statement that this edge decays on discovery.
Iceberg detection: **Zotikov, D. & Antonov, A. (2019), arXiv:1909.09495** — **CME futures, flagged**, and no effect size
in the abstract. Adverse selection on the passive side: **Albers, Cucuringu, Howison & Shestopaloff (2025),
*"The Market Maker's Dilemma"*, arXiv:2502.18625** — documents a **negative correlation between maker fill likelihood
and post-fill returns**, i.e. getting filled at the level is itself bad news.

**Data and cost — the cheapest thing on this list.** Windowed MBO on thin names is essentially free: measured
`mbo`, SNDL, a 10-minute window = **$0.0001**; AAPL 09:30–09:45 = **$0.040**; 20 liquid symbols for a full day = $3.27
(= $0.164/symbol-day). A study covering ~15,000 signal windows of 10 minutes each across 2 years on the Nasdaq-listed
half of our candidates prices at **$30–60**.

**Executability — the reason it is not rank 1 despite the price and the power.** Alpaca provides **no depth of book**.
Computing this signal live requires a **recurring Databento live XNAS.ITCH MBO subscription**, which is a new fixed
cost carried by a book that does not yet exist. And §0a applies in full: we would be measuring the depth and
cancellations on **~11% of the flow**, so an absence of signal is genuinely ambiguous between "no effect" and "wrong
venue".

**Power / refutation.** n ≈ 15,000 signal windows ⇒ **MDE ≈ 0.02R** — the best power on the list. Refuted if
depth-, hidden-fill- or cancellation-sorted buckets do not separate net R monotonically on TRAIN, or if the separation
dies ex-top-5%.

**Prior — and it is the right kind.** LOG stage **N1** tested CKS order-flow imbalance at the ORB breakout minute from
EQUS.MINI `tbbo`: *"No causal cell beats baseline by more than noise. Best = bottom-quintile `ofi_range` veto:
$14,729 vs $14,429 over 21 mo = **+$300 from removing 9 of 215 picks**; p 0.060 for one cell, ≈0.5 adjusted for 12."*
Crucially, **N1's own stated remedy is this study**: *"what would make it answerable: the same features on a
consolidated tape (XNAS.ITCH for Nasdaq names, or EQUS full), which would fix the 54% thin share and give a true
message-level OFI."* On the picks the ORB book actually takes, **54.4% had fewer than 20 prints in the entire 5-minute
opening range** on EQUS.MINI, and break-instant features were missing on 45–48% of them. So this is the
explicitly-recommended follow-up, not a re-litigation — **but** the honest counter is that it conditions books whose
*gross* edge is already near zero (`bf_zero2`: best families F8 +0.003R, F6 +0.007R/trade), and a 0.02R-resolution
measurement of a zero-gross book usually returns a precise zero.

---

### 6. Level-1 OFI (CKS) redone on a real tape — **rank 6**

**Mechanism.** Cont, Kukanov & Stoikov's order-flow imbalance aggregates best-bid/best-ask size changes; a one-sided
imbalance is net marketable demand that the quoted depth must absorb.

**Published evidence — with the critical caveat.** **Cont, R., Kukanov, A. & Stoikov, S. (2010/2014), *"The Price
Impact of Order Book Events"*, arXiv:1011.6402 (published in the *Journal of Financial Econometrics*)** — a linear
relation between OFI and price change with slope inversely proportional to depth. **Su, Sun, Li & Yuan (2021),
arXiv:2112.02947** report log-GOFI lifting R² from **32.89% to 83.57% at the 30-second scale**. **Cont, Cucuringu &
Zhang (2021), arXiv:2112.13213** extend to multi-level and cross-impact and report that *lagged* cross-impact
forecasts future returns. **The caveat that decides the rank: the headline R² figures are CONTEMPORANEOUS — OFI
explains the price change happening at the same instant, which is an identity of market clearing, not a forecast.**
Only the lagged cross-impact result is predictive, and Takahashi (2025, arXiv:2508.06788) reports that the one-second
shocks *"dissipate within seconds"* — far inside our 1–3 s bar-drain reaction and well inside our 0.135R cost.

**Executability — the best of the order-flow family.** Level-1 OFI needs only NBBO updates and trades, both of which
**Alpaca already streams to us** (`stop_monitor.py:3025/3084` subscribe quotes and trades). No new live subscription.
That is why it outranks full MBO on Exec even though MBO outranks it on information.

**Data.** `mbp-1` at signal windows (20 symbols × 1 day = $1.13 ⇒ $0.056/symbol-day) or `tbbo` ($0.024/symbol-day);
windowed to signals, **$20–40**. Same 11%-venue caveat.

**Prior.** N1, as above. Verdict was noise, and N1 named the tape as the reason.

---

### 7. Volume-profile / queue-depletion breakout confirmation — **rank 7**

**Mechanism.** A breakout that consumes the resting size at the level (queue depletion) differs from one that prints
through on thin size; `ohlcv-1s` plus `mbp-10` would let us measure how much of the level's displayed size was actually
traded in the breakout second.

**Published evidence.** Weak and mostly practitioner. The academic anchor is again Kavajecz & Odders-White (2004) for
depth-at-level; the deep-learning LOB line (**Ntakaris et al. 2019, arXiv:1904.05384**, on **TotalView-ITCH** US and
Nordic stocks; **Shabani et al. 2022, arXiv:2207.11577**) reports classification accuracy, not net-of-cost P&L, and
neither abstract discloses a Sharpe.

**Data/cost.** ~$30 windowed. **Executability** high as a *backtest* feature; live it needs depth (same subscription
problem as #5) unless restricted to `ohlcv-1s` volume, which Alpaca approximates already.

**Prior.** Relative volume is **already inside** the HOD-break spec as `rv_profile`, and `RESULTS.md` row 5 (M9)
records: *"RV monotonicity of ORB/HOD P&L on our universe — ≈0R in every RV bucket with the causal floor → the paper's
monotonicity is absent."* Four cells.

---

### 8. `statistics` schema — **rank 8, infrastructure**

Official opening/closing/auction prices, trading-reference data. **ALL_SYMBOLS, 6.5 years = $43.03.** Value is as a
cross-check on any auction study (#2, #4): it gives the *official* cross price, independent of our bar sources, which
is exactly the price-scale check PLAN.md §1 demands. No alpha by itself. Buy only alongside #2.

---

### 9. Short-horizon reversal with microstructure conditioning — **rank 9**

**Mechanism.** Nagel's liquidity-provision return: after a demand shock, expected returns to liquidity supply spike;
conditioning on microstructure (spread, depth, OFI) should identify *when* the reversal is compensated rather than
informed.

**Published evidence.** **Nagel, S. (2012), *"Evaporating Liquidity"*, Review of Financial Studies 25(7):2005–2039,
DOI `10.1093/rfs/hhs066`, 568 citations** — the strongest single citation in this whole review. But it is a
**liquidity-provision** return: it is earned by resting on the passive side across a reversal, which requires
maker-side execution we do not have, and Albers et al. (2025, arXiv:2502.18625) document that passive fills are
adversely selected. The overnight/intraday decomposition anchor — **Lou, D., Polk, C. & Skouras, S. (2019), *"A tug of
war: Overnight versus intraday expected returns"*, Journal of Financial Economics, DOI
`10.1016/j.jfineco.2019.03.011`, 313 citations** — is well-replicated but is a *monthly-horizon* cross-sectional
effect, not an intraday one.

**Prior — bad.** `RESULTS.md` row 6: M36/M37 large-loser and intraday-component reversal, **−105/−113/−9 and
−174/−93/−137 bps → dead**; row 6r, the same split by SPY vol tercile: **"no regime consistently positive"**; row 14,
M29 cross-sectional overnight continuation: **+7.4 (t 5.1) / +6.7 (t 3.5) / −13.4 (t −5.5) → fails; the classic
decayed-effect shape.**

---

### 10. Queue position / passive liquidity provision — **rank 10, not executable**

**Mechanism.** Huang, Pulido, Rosenbaum, Saliba & Sfendourakis (2019, arXiv:1902.10743) build a framework that values
the queue position of a limit order; the money is the queue-priority option.

**Why it is bottom-half despite good data.** We have **no control over queue position**: Alpaca routes for us, we set
no venue, we pay no rebates, and we cannot cancel-replace at microsecond cadence. Our own evidence says the same
thing twice: LOG stage **B** found the resting fill only **56–88% queue-OK**, and LOG stage **C** then **rejected H2
outright — the resting fill is −0.058R net (t −61) and −0.026R _gross_ (t −27)**. A queue-priority study would be
measuring an option we cannot buy.

---

### 11. ETF arbitrage — **rank 11, not executable**

**Mechanism / evidence.** **Box, T., Davis, R., Evans, R. & Lynch, A. (2021), *"Intraday arbitrage between ETFs and
their underlying portfolios"*, Journal of Financial Economics 141(3):1078–1095, DOI
`10.1016/j.jfineco.2021.04.023`, 76 citations** — the best-published, best-replicated candidate on this entire list,
and 27.4% of our universe (ARCA+BATS) is exactly this instrument class. **It is also the one we can least trade**:
capturing it requires simultaneous multi-leg execution across an ETF and its basket at sub-second latency, plus
authorised-participant creation/redemption for the residual. On a $50K account with DAY limit orders and a 1–3 s bar
loop, the realisable share of that effect is approximately zero. Ranked for completeness, not for action.

---

### 12. Standalone iceberg / hidden-liquidity detection as alpha — **rank 12**

Zero hits anywhere in our research tree, so it is untested — but the published evidence is CME futures
(Zotikov & Antonov 2019, arXiv:1909.09495, **flagged: futures-only**), Frey & Sandas (2009) explicitly say the surplus
**erodes once the hidden liquidity is discovered**, and we would be detecting icebergs on 11% of the flow with no live
feed to act on. Cheap ($30) but the lowest EV on the list.

---

## 3. Pre-registration-ready shortlist — at most 3, ordered by what I would run first

Splits are the fixed program splits (PLAN.md §1): **TRAIN 2025-01-02…2025-12-31 · VAL 2026-01-01…2026-05-31 ·
TEST 2026-06-01…2026-09-11**, early closes excluded, TEST read once after the stage's selection is frozen in writing.
Gates: **G1** TRAIN mean net R > 0, t ≥ 2.0, ≥ 5 trades/week · **G2** VAL mean net R > 0, t ≥ 1.0, ≥ 55% weeks green,
bar raised by 1 SE of weekly R per 10 cells that passed G1 · **G3** TEST read once and reported whatever it says.
Every G2 survivor additionally reports permutation search-adjusted p over all cells of the stage, tail removal
(top 1% / 5%), a winner cap at +3R, and a per-month table. Cost contract: `cost_by_outcome.md`
(entry half-spread always; target on a resting limit pays 0; stop 0.875×; close 0.412×).

### S1 — LULD halt-resume. **$54.36 + ~$11 = $66.** Run this first.

| | |
|---|---|
| **Step 0 (gate, $0.70)** | Pull `status` for `ALL_SYMBOLS` for **one month** and verify (a) LULD halt/resume actions appear, (b) whether NYSE-**listed** names carry actions on XNAS.ITCH. If (b) is false, the study is Nasdaq-listed-only ≈ 49.5% of the universe — proceed, with that stated in the report. Separately verify Alpaca's `statuses` websocket channel resolves on our key (this is the **live** feed; without it a positive result is unshippable). |
| **Data** | XNAS.ITCH `status` ALL_SYMBOLS 2018-05-01→2026-09-17 = **$54.36**. Outcome 1-min bars: SIP tape on disk for 2025-01→2026-09 (free); Alpaca for the 2018-2024 halted symbol-days (free) — **not** XNAS.ITCH `ohlcv-1m` (11% venue share). Budget **$11** if the Alpaca fetch proves impractical and we take the XNAS bars as a fallback, clearly labelled. |
| **Population** | Every LULD halt/resume pair on a symbol with prev-close ≥ $5 and ADV20 ≥ 100K at the halt date (both causal at the halt bar; `load_adv20_from_daily_bars`, the one shared definition). 2018–2024 is a **PRE split used for hypothesis generation only**, never for a gate. |
| **Cells — 12, pre-declared** | side {up-halt, down-halt} × rule {continuation: buy/short the first post-resume bar under a 0.6% cap; fade: take the opposite side} × horizon {+5 min, +30 min, 15:55 close} = 2 × 2 × 3 = **12**. No stop variant, no price-band variant, no sizing variant in this stage — those are Stage 2 if and only if a cell clears G1. |
| **Fill convention** | The engine's own: next bar's open under a cap; `low ≤ fill ≤ high` of the filling bar asserted on every trade; share of trades whose fill would differ from a touch reported. |
| **Power** | n ≈ 4K–8K events on 2025-01→2026-09 alone. At `sd_R = 2.5`, **MDE ≈ 0.056–0.079R** — below the 0.135R cost line, so a null here is informative. Reported alongside the result as PLAN.md §1 requires. |
| **Refuted if** | 0 of 12 cells clears G1; or the G1 survivor's edge sits entirely in the first post-resume minute where `spread/R` > 0.4; or it dies ex-top-5%. |
| **Why first** | Cheapest, highest executability (no new live data contract, sub-second reaction already built, DAY limit orders suffice), best power, **and the only candidate with no prior in `LOG.md` at all**. Its weakness — thin published evidence — is disclosed above and is the reason it is a 12-cell stage and not a program. |

### S2 — MBO / level-1 order flow at the break level, on the right tape. **$60.** Run second.

| | |
|---|---|
| **Step 0 (gate, $0)** | None on the data side. **But**: write down before running that a positive result requires a **recurring Databento live XNAS.ITCH subscription** to ship, and get the owner's price tolerance for that first. A study we cannot feed live is not worth $60. |
| **Data** | XNAS.ITCH `mbo`, 10-minute windows around each signal (level ± 5 min), Nasdaq-listed candidates only, 2025-01→2026-09. Measured rate: $0.0001/thin-name 10-min window, $0.040/liquid; ~15,000 windows ⇒ **$30–60**. |
| **Population** | Every F8 (opening-range break) and F6 (red-to-green) candidate in `candidates4` on a Nasdaq-listed symbol — the two base families `bf_zero2` left standing (+0.003R and +0.007R/trade gross). |
| **Cells — 6, pre-declared** | feature {displayed size at the level consumed vs cancelled; hidden-fill share in the breakout second; cancellation rate in the 60 s before the break} × use {bottom-quintile veto, composite score input} = 3 × 2 = **6**. |
| **Power** | n ≈ 15,000 ⇒ **MDE ≈ 0.02R**. The best-powered test available. |
| **Refuted if** | No feature separates net R monotonically across quintiles on TRAIN; or separation dies ex-top-5%; or the G1 survivor's lift is < 0.02R, i.e. inside the noise of a book whose gross edge is +0.003 to +0.007R. **Stated in advance: the most likely outcome is a precise zero**, because this measures a book with no gross edge — and that precise zero is itself worth $60, because it closes N1's explicitly-named open question. |
| **Prior disclosed** | N1: +$300 from 12 cells, adjusted p ≈ 0.5, on a tape where 54.4% of the picks had < 20 prints in the opening range. N1's own remedy is this study. |

### S3 — Closing-cross imbalance. **$240 (pilot), contingent.** Run third, and only if its $0 gate passes.

| | |
|---|---|
| **Step 0 (gate, $0 — do this before spending anything)** | Paper-trade a 1-share `tif='cls'` LOC order on a Nasdaq-listed name via our own key. If Alpaca rejects `cls`, or fills it away from the official Nasdaq closing print, **the study does not run and costs $0.** Second gate: price a Databento **live** `imbalance` subscription and confirm the owner will carry it. |
| **Data** | XNAS.ITCH `imbalance`, 600 Nasdaq-listed names from our universe, 2024-09-17→2026-09-17 ≈ **$240** (measured rate $0.40/symbol/2yr on liquid names; thinner names bill less). Add `statistics` ALL 6.5 yr = $43.03 only if the official cross price is needed as the price-scale check. Full 2,908-name coverage (≈ $1,160) only after the pilot separates. |
| **Cells — 8, pre-declared** | imbalance side {buy, sell} × |imbalance| quartile {Q1…Q4} = **8**, one rule (fade the imbalance at 15:55, exit next open), one horizon. Follow-the-imbalance is **not** a cell: Challet & Gourianov (arXiv:1802.01921) predicts fade, and M18/M16 say our universe fades — pre-committing to the published direction is the point. |
| **Power** | 300K symbol-day observations for the cross-sectional sort (MDE in single bps); ~500 booked trades at 4 concurrent ⇒ **MDE ≈ 0.107R**, only just under the 0.135R cost line. The exit is the cheapest in the cost table (eod, 0.083R). |
| **Refuted if** | Quartiles do not separate monotonically on TRAIN; or the separation is inside the top/bottom 1% of imbalance ratios; or the net effect is under the ~4–7 bps the close→open leg costs. |
| **Prior disclosed** | M20 (row 7 of `RESULTS.md`): the price-triggered version of this horizon is **dead, gross below cost**. This changes the trigger, not the horizon. Run it last for that reason. |

**Total if all three run: $366** ($66 + $60 + $240), plus **$35.20** for the `definition` file, which I would buy
regardless of any study because it retires a whole class of false positive (`ZVZZT` cost us a TEST result once).
**If only one thing is bought: S1, $54.36.**

---

## A. Every `get_cost` call behind the numbers above

XNAS.ITCH unless noted. Derived per-unit rates in **bold**.

| schema | scope | window | $ | derived |
|---|---|---|---|---|
| `status` | ALL_SYMBOLS | 2018-05-01→2026-09-17 | **54.36** | the cheapest deep-history buy we have |
| `status` | ALL_SYMBOLS | 2025-01-01→2026-09-17 | 12.41 | |
| `definition` | ALL_SYMBOLS | 2018-05-01→2026-09-17 | 120.94 | |
| `definition` | ALL_SYMBOLS | 2024-07-01→2026-09-17 | **35.20** | matches our PIT panel |
| `statistics` | ALL_SYMBOLS | 2018-05-01→2026-09-17 | 43.03 | |
| `imbalance` | ALL_SYMBOLS | 1 day | 9.13 | ⇒ 2 yr ALL ≈ **$4,565** — not viable |
| `imbalance` | ALL_SYMBOLS | 1 month | 200.86 | |
| `imbalance` | 20 liquid sym | 2 yr | 7.92 | **$0.40 / symbol / 2 yr** |
| `imbalance` | 20 liquid sym | 6.5 yr | 28.99 | **$1.45 / symbol / 6.5 yr** |
| `mbo` | 20 sym | 1 day | 3.27 | **$0.164 / symbol-day** |
| `mbo` | AAPL | 09:30–09:45, 1 day | 0.040 | |
| `mbo` | SNDL (thin) | 10 min | **0.0001** | windowed MBO on small caps is ~free |
| `mbp-10` | 20 sym | 1 day | 2.93 | $0.147 / symbol-day |
| `mbp-10` | SNDL | 10 min | 0.0002 | |
| `mbp-1` | 20 sym | 1 day | 1.13 | $0.056 / symbol-day |
| `tbbo` | 20 sym | 1 day | 0.47 | $0.024 / symbol-day |
| `bbo-1s` | 20 sym | 1 month | 2.35 | |
| `trades` | 20 sym | 1 month | 6.36 | |
| `ohlcv-1m` | 20 sym | 1 day | 0.0073 | **$0.000365 / symbol-day** |
| `ohlcv-1m` | ALL_SYMBOLS | 1 month | 15.96 | ⇒ 6.5 yr ≈ $1,245 |
| EQUS.SUMMARY `ohlcv-1d` | ALL_SYMBOLS | 1 yr | 4.53 | already on disk |

Symbol-resolution probe (cost only, no data): NYSE-listed `BAC`/`F`/`GE` and ARCA `SPY`/`XLF` **resolve** on
XNAS.ITCH for `trades`, `imbalance` and `status`; a bogus ticker returns `422 symbology_invalid_request`. **This proves
the symbols resolve, NOT that auction or trading-action records exist for them** — the `imbalance` and `status` quotes
for those names came back at what looks like a flat billing floor ($0.0172 for BAC/SPY/F vs $0.0194 for AAPL;
$0.000603 identically for all three on `status`), which is consistent with a minimum, not with a record count. Nasdaq
runs no opening or closing cross for a non-Nasdaq-listed security, so **assume no auction data for NYSE/ARCA/AMEX
listings until S3 step 0 says otherwise.**

## B. Cell count for this document

Candidates surveyed: **12**. Studies proposed: **3**, carrying **12 + 6 + 8 = 26 pre-declared cells** in total. No cell
was run, scored, or looked at; no split was read. The 26 cells are the multiplicity that the permutation-adjusted p of
each stage must be computed against.

## C. Sources

Verified this session via Crossref / arXiv APIs (Web search budget was exhausted; every citation below carries a DOI or
arXiv id that was returned by a live query, not recalled):

- Bogousslavsky, V. & Muravyev, D. (2023). "Who trades at the close? Implications for price discovery and liquidity." *Journal of Financial Markets*. [10.1016/j.finmar.2023.100852](https://doi.org/10.1016/j.finmar.2023.100852) — 29 cites.
- Challet, D. & Gourianov, N. (2018). "Dynamical regularities of US equities opening and closing auctions." [arXiv:1802.01921](https://arxiv.org/abs/1802.01921).
- Cont, R., Kukanov, A. & Stoikov, S. (2010). "The Price Impact of Order Book Events." [arXiv:1011.6402](https://arxiv.org/abs/1011.6402).
- Cont, R., Cucuringu, M. & Zhang, C. (2021). "Cross-Impact of Order Flow Imbalance in Equity Markets." [arXiv:2112.13213](https://arxiv.org/abs/2112.13213).
- Su, Y., Sun, Z., Li, J. & Yuan, X. (2021). "The Price Impact of Generalized Order Flow Imbalance." [arXiv:2112.02947](https://arxiv.org/abs/2112.02947).
- Takahashi, M. (2025). "Returns and Order Flow Imbalances: Intraday Dynamics and Macroeconomic News Effects." [arXiv:2508.06788](https://arxiv.org/abs/2508.06788).
- Nagel, S. (2012). "Evaporating Liquidity." *Review of Financial Studies* 25(7):2005–2039. [10.1093/rfs/hhs066](https://doi.org/10.1093/rfs/hhs066) — 568 cites.
- Lou, D., Polk, C. & Skouras, S. (2019). "A tug of war: Overnight versus intraday expected returns." *Journal of Financial Economics*. [10.1016/j.jfineco.2019.03.011](https://doi.org/10.1016/j.jfineco.2019.03.011) — 313 cites.
- Box, T., Davis, R., Evans, R. & Lynch, A. (2021). "Intraday arbitrage between ETFs and their underlying portfolios." *Journal of Financial Economics* 141(3):1078–1095. [10.1016/j.jfineco.2021.04.023](https://doi.org/10.1016/j.jfineco.2021.04.023) — 76 cites.
- Moinas, S. (2005). "Hidden Limit Orders and Liquidity in Limit Order Markets." SSRN. [10.2139/ssrn.676564](https://doi.org/10.2139/ssrn.676564) — 20 cites.
- Frey, S. & Sandas, P. (2009). "The Impact of Hidden Liquidity in Limit Order Books." SSRN. [10.2139/ssrn.1343538](https://doi.org/10.2139/ssrn.1343538) — 7 cites.
- Zotikov, D. & Antonov, A. (2019). "CME Iceberg Order Detection and Prediction." [arXiv:1909.09495](https://arxiv.org/abs/1909.09495) — **futures only**.
- Huang, W., Pulido, S., Rosenbaum, M., Saliba, P. & Sfendourakis, E. (2019). "From Glosten-Milgrom to the Whole Limit Order Book." [arXiv:1902.10743](https://arxiv.org/abs/1902.10743).
- Albers, J., Cucuringu, M., Howison, S. & Shestopaloff, A. (2025). "The Market Maker's Dilemma." [arXiv:2502.18625](https://arxiv.org/abs/2502.18625).
- Ntakaris, A., Mirone, G., Kanniainen, J., Gabbouj, M. & Iosifidis, A. (2019). "Feature Engineering for Mid-Price Prediction with Deep Learning." [arXiv:1904.05384](https://arxiv.org/abs/1904.05384).
- Dalko, V. (2016). "Limit Up–Limit Down: an effective response to the 'Flash Crash'?" *Journal of Financial Regulation and Compliance*. [10.1108/jfrc-04-2016-0040](https://doi.org/10.1108/jfrc-04-2016-0040) — 7 cites.
- McFarland, S., Jain, P. K. & McInish, T. (2022). "The Effectiveness of Single Stock Circuit Breaker Designs: The Special Quote and Limit Up-Limit Down Rules." SSRN. [10.2139/ssrn.4288811](https://doi.org/10.2139/ssrn.4288811) — **0 cites**.
- Lin, Y. (2017). "Limit Up Limit Down, Exchange Access Fee and High Frequency Trading Around Price Limits." SSRN. [10.2139/ssrn.3019986](https://doi.org/10.2139/ssrn.3019986) — 1 cite.
- Zeng, Z., Wang, G. & Tang, G. (2024). "Price Limits Hitting Effect and Cross-Sectional Stock Returns." *Finance Research Letters*. [10.1016/j.frl.2023.104803](https://doi.org/10.1016/j.frl.2023.104803) — **China**.
- Aitken, M., Comerton-Forde, C. & Frino, A. (2005). "Closing Call Auctions and Liquidity." *Accounting & Finance* 45(4):501–518. [10.1111/j.1467-629x.2005.00155.x](https://doi.org/10.1111/j.1467-629x.2005.00155.x) — **Australia, pre-2010**.

Internal (not re-verified here, cited as they stand in our tree): Kavajecz, K. & Odders-White, E. (2004), *RFS* —
support/resistance coincide with depth peaks; Osler, C. (2003) — stop clustering at round numbers and recent extremes.
Both are quoted from `research/fuckup_audit/probe_literature.md`.
