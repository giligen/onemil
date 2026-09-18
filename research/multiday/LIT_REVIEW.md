# Multi-day anomalies — adversarial literature review before the family runs

Written 2026-09-18, while `research/multiday/data/` is still building, so that anything it changes in
`PLAN.md` is still a **pre-registration and not a post-hoc edit**. No data was pulled, no code run, no
config touched. Every claim below carries a citation that was fetched from a bibliographic source
during this session (Crossref, OpenAlex, RePEc/IDEAS, NBER, or the publisher). Where a claim could
not be verified from a fetched page it is marked **NOT VERIFIED** and is not used to decide anything.

Phrasing rule (PLAN §1): nothing here says "no edge exists". It says what a named study detected, in
a named universe, over a named window, at a named cost — and where the study is silent, it says so.

---

## 0. The question, and why the literature mostly cannot answer it directly

`PLAN.md` asks two different questions and rightly separates them. This review adds a third that the
plan currently leaves implicit and that decides most of the verdicts below.

1. **Does the published effect still exist in 2016–2026 US equities?** The meta-literature says
   assume decay by default. McLean & Pontiff (2016, *Journal of Finance* 71(1) 5–32, DOI
   10.1111/jofi.12365) studied 97 published cross-sectional predictors: *"Portfolio returns are 26%
   lower out-of-sample and 58% lower post-publication."* They also report that *"returns are higher
   for portfolios concentrated in stocks with high idiosyncratic risk and low liquidity"* — i.e. the
   surviving return is where our $5 / $10M-ADV floors forbid us to go. Chordia, Subrahmanyam & Tong
   (2014, *Journal of Accounting and Economics* 58(1) 41–58, DOI 10.1016/j.jacceco.2014.06.001) is
   more specific about the era: *"the majority of the anomalies have attenuated and the average
   returns from a portfolio strategy based on prominent anomalies have approximately halved after
   decimalization"*, attributed to hedge-fund AUM, short interest and turnover. So the honest prior
   for **every** family below, before a single number is computed, is *half the published effect or
   less*, and all six of our families were published between 1989 and 2019.

2. **Can this account harvest it?** Our constraints are: a $50–66K book; long-only unless the name is
   `easy_to_borrow`; no leverage beyond Reg-T; entries/exits in the opening/closing auction via
   `opg`/`cls` (which exist in the Alpaca SDK and are **unused in this repo** — a build, not a knob);
   the declared cost model (no quoted spread on auction fills, SEC+TAF ≈0.4 bps on sells, impact
   ≈10 bps per 1% of ADV, GC borrow 0.3%/yr).

3. **The third question, which is the crux: how much of the published effect lives in the LONG leg?**
   Every headline number in this literature is a long–short decile spread over 500–1,000 names. We can
   hold ~20. So the relevant quantity is not the spread, it is the top-decile long leg's return in
   excess of the market, inside a universe that already excludes microcaps. §1 collects what the
   literature actually reports on this, and it is much less than one would like.

### 0b. What our own program has already settled, and must not be re-run as if new

- `research/fuckup_audit/R_daily/REPORT_v2.md`: five home-made long-only multi-day rules (gap
  continuation, 52-week-high-on-volume, short-term reversal, overnight continuation, uptrend
  pullback), 1–10 session holds, split-adjusted point-in-time Nasdaq panel, 2019–2025. **0 of 20
  cells positive on TRAIN, 0 on VAL, permutation p = 1.00, MDE 6.3–30 bps in 10 of 20 cells.** In the
  powered cells this is a negative result, not a low-power null. Two of those five overlap our six:
  K2 is a crude F5 (52-week high) and K4 is a crude F6 (overnight). They were home-made rules, not the
  published constructions — which is exactly why `PLAN.md` exists — but a family that was *also*
  strongly negative in its crude form starts below zero prior, not at zero.
- `research/lit_review_2026/RESULTS.md` (the July/September queue) already killed, at a 4–50 name book
  and 2–10 bps auction costs: **M22** volume-shock overnight; **M29** cross-sectional overnight
  continuation (TRAIN +7.4 bps t 5.1, VAL +6.7 t 3.5, **TEST −13.4 t −5.5** — the decay shape);
  **M36/M37** large-loser and intraday-component reversal; **M41** new-252-day-high momentum (TRAIN
  +104 bps t 2.0, VAL +389 t 3.4, **TEST −183 t −2.3**); **M20** end-of-day loser reversal.
  `overnight_auction.md` ran the overnight family *at auction costs already* — top-25 and top-50 books
  are +25 bps TRAIN, +15 VAL, **−12 TEST**. None of these is re-proposed below without saying why.
- `research/lit_review_2026/cost_curve.md` / `cost_by_outcome.md`: the reason the program moved to
  multi-day holds at all — spread/R ≈ 0.02 on a multi-day hold vs 0.14 intraday. That arithmetic is
  correct and is the one structural advantage the multi-day line has.
- Stage O's standing rule, which applies to every cell here: **a band-table cost constant is a
  hypothesis, not a cost.** Any cohort outside the population the cost table was measured on must have
  its own spread/impact measured before a cell is reported.

---

## 1. The crux: how much of each effect is in the long leg?

This is the single most decision-relevant number and the literature reports it for very few families.
What is verified:

- **Israel & Moskowitz (2013, *Journal of Financial Economics* 108(2) 275–301, DOI
  10.1016/j.jfineco.2012.11.005)** is the one paper that decomposes directly. Abstract, verbatim:
  *"We find that long positions make up almost all of size, 60% of value, and half of momentum
  profits. Shorting becomes less important for momentum and more important for value as firm size
  decreases. The value premium decreases with firm size and is weak among the largest stocks.
  Momentum profits, however, exhibit no reliable relation with size."* → For **F3 (12-1 momentum)**
  the long leg is worth about **half** the L-S spread, and — importantly for us — momentum's size
  relation is not reliable, so the effect is not a microcap artefact. This is the best news in this
  review and it is family-specific: it does **not** transfer to PEAD, reversal, 52-week-high or
  overnight.
- **Stambaugh, Yu & Yuan (2012, *JFE* 104(2) 288–302)** points the other way for a broad set of
  eleven anomalies: anomalies strengthen after high sentiment, *"short positions in strategies become
  more profitable after sentiment spikes, and long positions show no sentiment correlation"* — the
  time-varying, exploitable component sits in the leg we cannot trade. Their eleven anomalies do not
  include PEAD or short-term reversal, so this is a prior about the class, not a measurement of our
  families.
- **Fama & French (2008, *Journal of Finance* 63, 1653–1678, DOI
  10.1111/j.1540-6261.2008.01371.x)**: *"The anomalous returns associated with net stock issues,
  accruals, and momentum are pervasive; they show up in all size groups (micro, small, and big)."*
  This is the cite that keeps **momentum** and **net share issuance** alive inside our $5/$10M floors,
  and it is why net issuance appears in §3 as an addition.
- **Hou, Xue & Zhang (2020, *Review of Financial Studies*; NBER w23394)** is the general warning:
  *"With microcaps alleviated via New York Stock Exchange breakpoints and value-weighted returns, 286
  anomalies (64%) … are insignificant at the conventional 5% level."* Any family whose published
  magnitude comes from equal-weighted portfolios that include microcaps should be assumed to shrink by
  most of its size inside our floors.
- **Novy-Marx & Velikov (2016, *RFS*; NBER w20721)** is the cost gate: *"Most of the anomalies that we
  consider with one-sided monthly turnover lower than 50% continue to generate statistically
  significant net spreads… Few of the strategies with higher turnover do."* This single sentence
  decides the verdict for the weekly-rebalanced family (F4) and for any daily round-trip (F6's
  executable form).

**Standing requirement this review adds to `PLAN.md`:** because the literature so rarely gives the
long-leg share, *we must measure it ourselves* and report it as a first-class column — top-decile
long-leg excess return as a fraction of the L-S spread, per family per split. It is the number that
decides whether a published effect is a book for this account or a fact about a portfolio we cannot
hold.

---

## 2. Verdict on the six pre-registered families

| # | family | seminal cite (verified) | most recent replication / decay cite (verified) | long-leg share | microcap risk | **verdict** |
|---|---|---|---|---|---|---|
| F1 | PEAD / SUE | Bernard & Thomas 1989, *JAR* 27, p.1, DOI 10.2307/2491062 | **Martineau 2022, *Critical Finance Review* 11(3-4) 613–646, DOI 10.1561/104.00000122** — sample 1984–2019 (analyst) / 1973–2019 (random walk): *"no statistically significant PEAD for all-but-microcap stocks after 2006"*, microcaps after 2016; 2016–19 coefficients −0.002 / +0.002, both insignificant | not reported in any source I could verify | fatal — the residual drift Martineau finds lived in microcaps until 2016 | **DROP as pre-registered** (see below) |
| F2 | earnings-announcement-return drift | Chan, Jegadeesh & Lakonishok 1996, *JF* 51(5) 1681–1713, DOI 10.1111/j.1540-6261.1996.tb05222.x | none verified post-2015. **Martineau 2022 does NOT test this measure** (confirmed against the paper text) — it tests analyst and random-walk surprise only | not reported | unmeasured | **KEEP, promote to first priority** |
| F3 | 12-1 cross-sectional momentum | Jegadeesh & Titman 1993, *JF* 48(1) 65–91, DOI 10.1111/j.1540-6261.1993.tb04702.x — *"yields 1.31% per month"* (1.49% with a one-week lag) | Asness, Moskowitz & Pedersen 2013, *JF* 68(3) 929–985; Daniel & Moskowitz 2016, *JFE* 122(2) 221–247 (crashes in "panic states"); Fama & French 2008 — momentum *"pervasive… in all size groups (micro, small, and big)"* | **~50%** (Israel & Moskowitz 2013) — the only family with a published number | low — no reliable size relation (Israel & Moskowitz 2013) | **KEEP AS IS**, add one variant (see §5) |
| F4 | short-term reversal, industry-adjusted | **Da, Liu & Schaumburg 2014, *Management Science* 60(3) 658–674, DOI 10.1287/mnsc.2013.1766** (PLAN gives no venue; it is *Management Science*, not RFS) | Cheng, Hameed, Subrahmanyam & Titman 2017, *JFQA* 52(1) 143–173 — *"price declines over the previous quarter produce stronger reversals"*, tied to institutional exits | DLS split the mechanism: **liquidity shocks dominate the LONG leg** (forced selling), sentiment the short leg — the long leg is the liquidity-provision side, which is the good news | moderate | **MODIFY — cut to one long-only cell** |
| F5 | 52-week high | George & Hwang 2004, *JF* 59(5) 2145–2176 — (6,6) **0.45%/month raw, 1.23%/month ex-January**, 0.86% risk-adjusted | **George, Hwang & Li 2018, *JFE* 128(1) 148–163, DOI 10.1016/j.jfineco.2018.01.005** — the **q-factor model explains the price-to-high anomaly**; price-to-high predicts future profitability and investment growth. The original authors reclassify their own anomaly as risk-based | not reported | moderate | **MODIFY — re-label as a factor tilt, not alpha** |
| F6 | overnight vs intraday | Lou, Polk & Skouras 2019, *JFE* 134(1) 192–213 — profits earned *"either entirely overnight… or entirely intraday, typically with profits of opposite signs"* | Akbas, Boehmer, Jiang & Koch 2022, *JFE* 145(3) 850–875 — positive overnight returns are *routinely followed by daytime reversals*; Bogousslavsky 2021, *JFE* 141(1) 172–194; Hendershott, Livdan & Rösch 2020, *JFE* 138(3) 635–662 (beta priced **positively overnight, negatively intraday**); Lachance 2023, *Rev. Fin. Econ.* 41(4) 347–363 | the executable form is a **daily round trip**, so the leg question is moot — cost is the binding constraint | high | **MODIFY — measurement cell only, no tradable cell** |

### F1 — why DROP is the honest verdict, and what replaces it
Martineau (2022) is the decisive cite and it is unusually clean: the announcement-date response *grew*
from **20 bps (1984–1990) to 120 bps (2016–2019)** for all-but-microcap stocks, with announcement-day
R² rising **1.7% → 11.7%**, and the 60-day post-announcement drift went to zero for exactly the
non-microcap universe our $5 / $10M floors force us into. In plain terms: the market now prices the
surprise on the day, so the thing our F1 proposes to hold for 20–60 sessions has already happened
before our `opg` order is accepted the next morning. Our whole surprise-construction apparatus (XBRL
`EarningsPerShareDiluted` seasonal random walk, per Foster–Olsen–Shevlin / Bernard–Thomas) is a
*weaker* surprise measure than the analyst-based one Martineau uses — Livnat & Mendenhall (2006, *JAR*
44(1) 177–205) report drift is **significantly larger** with analyst forecasts than with time-series
models — so our version is measuring the deader of the two variants with the worse instrument.
This is not "PEAD never existed"; it is "no SUE-based drift was detectable for non-microcap US stocks
after 2006 in Martineau's 1984–2019 sample, and we would be testing a weaker surprise measure on a
later window than his."
**Replacement, not deletion:** the 8-K event table the data stage is building right now is exactly what
F2 and the two event-anchored additions in §3 need. F1's cells move there.

### F2 — the strongest survivor, and the reason to re-order the run
F2 is the only one of the six whose most-cited modern critic explicitly does **not** cover it. CJL
(1996) found *"past return and past earnings surprise each predict large drifts in future returns
after controlling for the other"* with little subsequent reversal. Brandt, Kishore, Santa-Clara &
Venkatachalam, "Earnings Announcements are Full of Surprises" (SSRN 2006, DOI 10.2139/ssrn.909563) is
the standard cite for the announcement-return version — **its abstract could not be retrieved from any
open source** (SSRN returns 403), so it is listed as a pointer, not as evidence. The honest status of
F2 is therefore: *published, not verifiably replicated since, and not covered by the study that killed
its sibling.* That is the profile of the highest-information cell in the program, and it needs no
accounting data at all — only the 8-K timestamp and prices, both of which the data stage already has.

### F4 — the mechanism is better than expected, the turnover is the problem
DLS's decomposition is genuinely favourable to us: the **long** leg of short-term reversal is the
liquidity-provision side (buying forced sellers), not the sentiment side. But the rebalance is weekly,
which puts one-sided monthly turnover near 100%, and Novy-Marx & Velikov (2016) is explicit that
*"few"* strategies above 50% one-sided monthly turnover generate significant net spreads. Against that,
our auction cost model is much cheaper than theirs — which is precisely the kind of claim Stage O's
standing rule says must be *measured*, not assumed. Our own three attempts at buying recent losers
(M20 end-of-day reversal, M36/M37 large-loser reversal, M18 open fade) were all negative, but all three
were intraday books on movers at retail spreads, not a weekly industry-residual book on ≥$10M/day names
at auction prices — a different population, so they lower the prior without settling it.
**Keep one cell: long-only, weekly, industry-residual, auction fills, with the break-even cost
reported.** Drop the published L-S cell, which we could not trade even if it worked.

### F5 — still there, but it is a factor tilt with a January problem
Two facts change what F5 means. First, George & Hwang's own magnitude is **0.45%/month raw but
1.23%/month excluding January** — most of the raw effect is a seasonal we cannot count on and that
overlaps the tax-loss literature. Second, George, Hwang & Li (2018) show the q-factor model *explains*
price-to-high, and that the ratio predicts future profitability and investment growth. For a
long-only book the distinction between risk premium and mispricing does not change the P&L, but it
changes the honest expectation: we would be buying a profitability/investment tilt and inheriting its
drawdowns, not harvesting an inefficiency that a small account is uniquely placed to collect.
**Keep the cells, re-label the hypothesis, and report the ex-January number separately** — otherwise a
January-driven result will be read as a year-round edge.

### F6 — the decomposition is worth measuring; the trade is not
Four independent verified findings say the *pattern* is real: LPS 2019 (profits are entirely overnight
or entirely intraday, with opposite signs), Hendershott–Livdan–Rösch 2020 (beta is priced positively
overnight and negatively intraday), Bogousslavsky 2021, Akbas et al. 2022 (overnight gains are
*routinely followed by daytime reversals*). What none of them establishes is that a retail account can
collect it, and the one source that addresses cost directly says the opposite: Haghani, Ragulin &
Dewey, "Night Moves" (Elm Partners, 20 June 2022) simulate the long-short overnight portfolio at 38%
p.a. gross and report that **1 bp of round-trip cost removes about 5 percentage points a year**, that
an investor paying 1 bp *"would not have made any money in the last 8 years"*, and that market impact
at meaningful size can reach *"200% per annum"*. The executable form of F6 is a **daily** MOC-buy /
MOO-sell round trip — roughly 250 round trips a year — which is the single most cost-exposed shape in
this entire review, and it sits above Novy-Marx & Velikov's turnover cliff by an order of magnitude.
Our own `overnight_auction.md` already ran this at 2–10 bps auction costs: top-25/top-50 books at
+25 bps TRAIN, +15 VAL, **−12 TEST**; and `RESULTS.md` M29 was +7.4 (t 5.1) / +6.7 (t 3.5) /
**−13.4 (t −5.5)**. One more note the literature adds and our plan should absorb: Boyarchenko, Larsen
& Whelan (2023, *RFS* 36(9) 3502–3547, DOI 10.1093/rfs/hhad020) find the index-level overnight drift is
concentrated *"during the opening hours of European markets"* — a window an `opg`/`cls` account cannot
reach at all, so even the market-level version of this effect is largely not addressable by us.
**Keep F6 as a measurement/decomposition cell** (it is free, it is diagnostic, and the overnight/intraday
split is a useful covariate for every other family). **Do not declare a tradable F6 cell.**

### Citation corrections `PLAN.md` must absorb
- **F4**: Da, Liu & Schaumburg is ***Management Science* 60(3) 658–674 (2014)**, DOI 10.1287/mnsc.2013.1766.
  More important than the venue: **the paper's headline construction purges cash-flow news using analyst
  revisions**, which we do not have. Our 2-digit-SIC industry residual is a *proxy for* their
  construction, not their construction, and the report must say so.
- **F1**: Livnat & Mendenhall is *JAR* **44(1) 177–205 (2006)**, and its finding cuts against us — drift
  is *significantly larger* with analyst forecasts than with time-series models, and time-series is all
  we have.
- **F3**: Daniel & Moskowitz is *JFE* **122(2) 221–247 (2016)**.

---

## 3. What is missing from the six — four proposed additions

Ranked by expected harvestable edge **for this account** (long-only, ~20 names, auction fills, $5 /
$10M floors), not by published effect size. Each has: mechanism, seminal cite, a verified recent
re-test, why the long leg is the tradable leg, and the data position.

### A1 (rank 1) — Earnings-announcement **premium** (hold *into* the announcement, not after it)
- **Mechanism.** Liquidity providers and liquidity-motivated sellers both step away ahead of a scheduled
  announcement, so the pre-announcement price is bid up by a temporary, predictable liquidity premium;
  the buyer of that liquidity is paid for bearing announcement risk over a date known months in advance.
- **Cites.** Frazzini & Lamont, "The Earnings Announcement Premium and Trading Volume", **NBER WP 13090
  (2007)** — *"On average, stock prices rise around scheduled earnings announcement dates… stocks with
  high past announcement period volume earn the highest announcement premium"*. ⚠ **This paper was never
  published in a journal** (no Crossref/OpenAlex journal record); cite it as a working paper.
  Published support: Barber, De George, Lehavy & Trueman, *JFE* **108(1) 118–138 (2013)** — *"U.S. stocks
  have been shown to earn higher returns during earnings announcement months than during non-announcement
  months"* (9 of 20 countries significantly positive; the US magnitude is **NOT VERIFIED** from the
  abstract). Recent re-tests: Levi & Zhang, *JFE* **118(2) 383–398 (2015)** — pre-announcement liquidity
  sales *"are a significant driver"* of the premium; Chapman, *JAE* **66(1) 222–243 (2018)**.
- **The adversarial cite, which changes the design.** Johnson & So, *JAR* **56(1) 217–263 (2018)**, DOI
  10.1111/1475-679x.12189: *"The cost of trading on negative news, relative to positive news, increases
  before earnings announcements… This asymmetry generates a predictable upward price bias that increases
  preannouncement and subsequently reverses."* So the premium is at least partly a liquidity artefact
  **that reverses after the event**. The cell must therefore be *buy N days before, sell at or
  immediately after the announcement* — a post-announcement hold would give the premium straight back.
  This is the opposite trade to F1, on the same event table.
- **Why long-only works.** The premium is a *level* effect on the announcing stock, not a cross-sectional
  spread — there is no short leg to miss. This is the only family in this review for which the long-only
  form is the published form.
- **Data.** Zero marginal cost: the 8-K Item 2.02 `acceptanceDateTime` table now being written to
  `research/multiday/data/edgar/` **is** the event calendar, and Alpaca daily opens/closes are already
  pulled. One caveat that must be handled: the 8-K is filed *at* the announcement, so a
  buy-before-the-announcement rule needs the *expected* date (the same fiscal quarter's filing lag from
  prior years), which is computable from the same table and must pass the availability audit.

### A2 (rank 2) — **Low short interest** in actively traded names (the long side, explicitly)
- **Mechanism.** Short interest is a costly, informed signal. Its *absence* in a heavily traded name says
  the informed community has looked and declined to bet against it; the literature finds that the
  positive abnormal return to that group is at least as large as the negative return to heavily shorted
  names — and it is on the side we can trade.
- **Cite.** Boehmer, Huszár & **Jordan**, "The good news in short interest", ***Journal of Financial
  Economics* 96(1) 80–97 (2010)**, DOI 10.1016/j.jfineco.2009.12.002 (⚠ the third author is Jordan, not
  Jones, and the venue is JFE, not RFS). Verbatim: *"relatively heavily traded stocks with low short
  interest experience both statistically and economically significant positive abnormal returns"*, while
  the negative returns to high short interest *"can be transient and of debatable economic
  significance"*, with the low-short-interest positive returns *"often larger (in absolute value)"*.
  Aggregate-level corroboration that the signal carries information: Rapach, Ringgenberg & Zhou, *JFE*
  **121(1) 46–65 (2016)** — short interest *"outperforms a host of popular return predictors both in and
  out of sample, with annual R² statistics of 12.89% and 13.24%"*.
- **Fit to our constraints.** The published sort is *conditional on heavy trading* — i.e. it lives in
  exactly the liquid band our $10M-ADV floor forces us into, the opposite of the usual microcap problem.
  Rebalance is semi-monthly, so turnover sits under Novy-Marx & Velikov's 50% cliff.
- **Data.** Short interest is **not** in Alpaca. FINRA and the exchanges publish semi-monthly equity
  short-interest files free of charge; this is a new but small pipeline (a bulk download plus a symbol
  join), and it has a hard availability trap: the *settlement* date and the *dissemination* date differ
  by roughly eight business days, so every position must be keyed to the dissemination date or the cell
  is a look-ahead. Days-to-cover needs ADV, which we already have.
- **Decay status: NOT VERIFIED.** No post-2015 re-test of the low-short-interest long leg was found in
  these indexes. That is a gap, not a clearance, and it should be stated in the report.

### A3 (rank 3) — **Net share issuance** (the long leg = firms shrinking their share count)
- **Mechanism.** Managers issue stock when it is expensive and retire it when it is cheap; the market
  under-reacts to the share-count change itself, which is a cleaner and slower signal than any
  announcement.
- **Cites.** Pontiff & Woodgate, *Journal of Finance* **63(2) 921–945 (2008)** — *"Post-1970, share
  issuance exhibits a strong cross-sectional ability to predict stock returns. This predictive ability is
  more statistically significant than the individual predictive ability of size, book-to-market, or
  momentum."* Fama & French, *JF* 63 (2008) — net stock issues are *"pervasive; they show up in all size
  groups (micro, small, and big)"*, which is the specific reason this survives our floors. Recent:
  Goto, Wang & Yan, *Financial Analysts Journal* **76(1) 63–81**, DOI 10.1080/0015198X.2019.1682427 — the
  effect intensifies with managerial incentives and a hybrid strategy *"generated significant returns
  even after accounting for transaction costs"*.
- **Reject the buyback-announcement form of this idea.** Ikenberry, Lakonishok & Vermaelen, *JFE*
  **39(2–3) 181–208 (1995)** report *"the average abnormal four-year buy-and-hold return… is 12.1
  percent"* — a **four-year** horizon, outside this program's ≤12-month scope and ≈3%/yr. And Bargeron,
  Bonaimé & Thomas, *JFQA* **52(2) 491–517 (2017)** find the long-run repurchase return is *"not
  persistent drifts but rather step functions"* driven by later authorizations and takeovers. So trade
  the share count, not the announcement.
- **Why the long leg.** The *long* leg is low/negative issuance. Fama–French's "pervasive in big" is the
  cite that makes a large-cap-only long leg plausible; a published long-leg *share* is **NOT VERIFIED**.
- **Data.** Near-zero marginal cost: the EDGAR submissions/XBRL machinery is already wired in
  `research/multiday/data/edgar_*.py`; the addition is one more `companyconcept` — the cover-page
  `dei:EntityCommonStockSharesOutstanding` (or `us-gaap:CommonStockSharesOutstanding`) with its `filed`
  dates, which the availability audit already covers for EPS.

### A4 (rank 4) — **Dividend-month premium**
- **Mechanism.** Dividend-seeking investors buy predictably in the month a firm is expected to pay, and
  the resulting price pressure is a return to whoever supplies that liquidity in advance.
- **Cite.** Hartzmark & Solomon, "The dividend month premium", *JFE* **109(3) 640–660 (2013)**, DOI
  10.1016/j.jfineco.2013.02.015: *"companies have positive abnormal returns in months when they are
  predicted to issue a dividend"*; the magnitude *"rivals the value premium while showing lower
  volatility"*, attributed to *"price pressure from dividend-seeking investors"*.
- **Fit.** Long-only by construction (the anomaly is defined on the predicted-dividend month, not a
  spread), predictable dates, monthly turnover, and it lives in dividend payers, which are large and
  liquid — the same direction of size bias that helps us for once.
- **Data.** Free and almost already there: `research/fuckup_audit/R_daily/fetch_splits.py` already calls
  Alpaca's `CorporateActionsClient` with `TYPES = ['forward_split','reverse_split','unit_split',
  'stock_dividend']`; adding `cash_dividend` is a one-line change and gives the declared/ex-date history
  the Hartzmark–Solomon prediction rule needs.
- **Why it ranks last, honestly.** **No post-2016 replication or decay test was found** in these indexes.
  Under McLean & Pontiff's 58% post-publication haircut, an unreplicated 2013 effect is the weakest
  evidence base of the four, and it is ranked accordingly rather than dropped.

### Also proposed, but as a MODIFICATION to F3 rather than a new family
**Residual / idiosyncratic momentum.** Blitz, Huij & Martens, *Journal of Empirical Finance* **18(3)
506–521 (2011)**: *"residual momentum earns risk-adjusted profits that are about twice as large as those
associated with total return momentum"*; Blitz, Hanauer & Vidojevic, ***International Review of
Economics & Finance* 69, 932–957 (2020)** — *"idiosyncratic momentum generates robust returns across a
range of developed and emerging markets"* (an out-of-sample re-test by the original author). It ranks
high on evidence but needs **no new data** (daily returns plus Ken French's free factor files) and is the
same signal family as F3, so it belongs as two extra F3 cells, not as a fifth family. Its specific appeal
here is Daniel–Moskowitz's crash problem: residualising removes the dynamic beta that produces the
crashes, which matters far more to a compounding $50K account than to a paper portfolio.

---

## 4. What the literature says we should NOT expect

**(a) Effects that are gone or halved — do not spend cells on them.**
- The general rate: McLean & Pontiff (2016) — **26% lower out-of-sample, 58% lower post-publication**;
  Chordia, Subrahmanyam & Tong (2014) — prominent anomalies *"approximately halved after
  decimalization"*. All six of our families predate that.
- **SUE-based PEAD**: Martineau (2022) — no significant drift for all-but-microcap stocks **after 2006**.
- **Index addition / deletion / rebalance flow**: Greenwood & Sammon, "The Disappearing Index Effect"
  (NBER w30748; *Journal of Finance* 2025) — the S&P 500 addition abnormal return fell from *"3.4% in the
  1980s and 7.6% in the 1990s to 0.8% over the past decade"*, deletions to −0.6%. **Rejected** as a
  family: the effect is ~0 and the auction it trades in is the most crowded print of the year.
- **Pre-FOMC drift**: Lucca & Moench, *JF* **70(1) 329–371 (2015)** (the NY Fed staff-report version puts
  the pre-announcement window at *"more than 80 percent of the equity premium"*), but Kurov, Halova Wolfe
  & Gilbert, *Finance Research Letters* **40, 101781 (2021)**: *"the pre-FOMC drift essentially
  disappeared after 2015"*. **Rejected.** The FOMC-cycle even-week pattern (Cieslak, Morse &
  Vissing-Jorgensen, *JF* **74(5) 2201–2248, 2019** — *"the equity premium is earned entirely in weeks 0,
  2, 4, and 6 in FOMC cycle time"*) has **no verified post-2019 replication**, which at ~8 events a year
  is too thin an evidence base to declare a cell on.
- **Time-series momentum on ETFs** (the one explicitly-requested ETF book): Moskowitz, Ooi & Pedersen,
  *JFE* **104(2) 228–250 (2012)** is contradicted by Huang, Li, Wang & Zhou, "Time series momentum: Is it
  there?", *JFE* **135(3) 774–794 (2020)**: *"Asset-by-asset time series regressions reveal little
  evidence of TSM, both in- and out-of-sample. While the t-statistic in a pooled regression appears
  large, it is not statistically reliable."* **Rejected** — an ETF book is attractive for execution
  reasons, but this particular ETF book has a top-journal replication failure against it.
- **Accruals**: Green, Hand & Soliman, *Management Science* **57(5) 797–816 (2011)**, titled *"Going,
  Going, Gone? The Apparent Demise of the Accruals Anomaly"* — listed only as a reminder of how fast this
  class of accounting anomaly died; it is not one of ours.

**(b) Effects whose published alpha is in stocks we cannot buy.**
- The class-wide warning: Hou, Xue & Zhang — *"With microcaps alleviated via NYSE breakpoints and
  value-weighted returns, 286 anomalies (64%) … are insignificant"*.
- **Betting-against-beta / low-volatility long leg** (explicitly asked about): Novy-Marx & Velikov,
  "Betting against betting against beta", *JFE* **143(1) 80–106 (2022)**: *"For each dollar invested in
  BAB, the strategy commits on average $1.05 to stocks in the bottom 1% of total market
  capitalization."* Our $5 / $10M-ADV floors remove exactly that dollar. **Rejected.**
- Martineau's residual PEAD survived *only* in microcaps, and only until 2016.
- McLean & Pontiff's own finding that surviving predictability concentrates in *"high idiosyncratic risk
  and low liquidity"* stocks is the general form of this problem, and it is the single strongest reason
  to expect every number in this program to come in below its published counterpart.

**(c) Effects that need a long-short book we cannot run.**
- Stambaugh, Yu & Yuan (2012): across eleven anomalies, *"short positions… become more profitable after
  sentiment spikes, and long positions show no sentiment correlation"* — the state-dependent, exploitable
  half is the half we cannot hold.
- The **overnight** long-short is a ~250-round-trip-a-year book; Haghani, Ragulin & Dewey ("Night Moves",
  Elm Partners, 20 Jun 2022) simulate it at 38% p.a. gross and report that **1 bp of round-trip cost
  removes ~5 points a year** and that an investor paying 1 bp *"would not have made any money in the last
  8 years"*.
- **Failures-to-deliver** (explicitly asked about): the one study with a tradable implication, Autore,
  Boulton & Braga-Alves, *Financial Review* **50(2) 143–172 (2015)**, finds *"stocks reaching threshold
  levels of failures become significantly overvalued"* with *"extreme overpricing and subsequent
  reversals"* — a **short**-side signal. Fotak, Raman & Yadav, *JFE* **114(3) 493–516 (2014)**, find
  *"greater FTDs lead to higher liquidity and pricing efficiency"* and no price distortion. **Rejected**
  for a long-only account.
- **Analyst-revision / guidance-change drift** (explicitly asked about): the effect is real in the
  seminal work — Chan, Jegadeesh & Lakonishok (1996) find *"past return and past earnings surprise each
  predict large drifts in future returns after controlling for the other"* — but every published
  construction needs a consensus-estimate vendor (I/B/E/S or equivalent). We have none, and the free
  EDGAR route gives reported EPS, not expectations. **Rejected on data, not on evidence.** The same
  constraint degrades F4, whose published form purges cash-flow news with analyst revisions.

**(d) Seasonality — real enough to report, too thin to be a family.**
- Turn-of-month: Etula, Rinne, Suominen & Vaittinen, **"Dash for Cash: Monthly Market Impact of
  Institutional Liquidity Needs", *Review of Financial Studies* 33(1) 75–111 (2020)** (⚠ RFS, not JFQA,
  and the title differs from the one in circulation) — they *"document temporary increases in the costs
  of debt and equity capital that coincide with key dates associated with month-end cash needs"*.
  Magnitude **NOT VERIFIED**.
- Sell-in-May: Zhang & Jacobsen, *Journal of International Money and Finance* **110, 102268 (2021)** — on
  62,962 observations of all world indices, returns are *"on average 4% higher during November–April"*
  with summer excess returns *"around −1%"*. It survives out of sample, but 4%/yr at index level on a
  $50–66K book is ~$2–2.6K/yr with an enormous variance, and our own M40 turn-of-month test had **n = 4–12
  events** — meaningless, as `RESULTS.md` recorded.
- January: Cheema, Ding & Wang, *Journal of Asset Management* **24(6) 513–530 (2023)** — long-short
  portfolios earn *"over 20 times higher returns in January"* and, notably for us, *"85% of the
  cross-sectional January effect comes from its long legs"*. That is the most long-leg-favourable finding
  in this review, and it is a *seasonal amplifier of other signals*, not a standalone book.
- **Therefore**: no seasonality family. Instead, a reporting rule — **every family's result must be
  reported with January shown separately and with a month-end dummy**, because George–Hwang's own F5
  number is 0.45%/month raw versus 1.23%/month ex-January, and a January-loaded result read as a
  year-round edge is exactly the kind of mistake this program exists to stop.

**(e) One requested candidate that is neither accepted nor rejected: 52-week-high × earnings.**
No study testing that specific interaction was verified. The two nearest verified works are Chen, Stivers
& Sun, *Journal of Empirical Finance* **79, 101556 (2024)** (reversals weaken and shift toward momentum as
turnover and the price-to-52-week-high ratio rise) and Byun & Jeon, *Financial Analysts Journal* **79(2)
120–139 (2023)** (52-week-high-*neutral* momentum *"substantially attenuates crashes"*). Both are
covariate findings. **Recommendation: carry price-to-52-week-high as a reported covariate on F2 and F3,
not as a family.** Declaring an untested interaction as a cell is how a program buys multiplicity for
nothing.

---

## 5. Recommendation on `PLAN.md`'s 26 cells

**Yes, `PLAN.md` should be edited, and it must be edited *now* — before the first family run — so the
change is a pre-registration and not a post-hoc selection.** Two things are wrong with the grid as
written and one thing is missing. First, **the arithmetic: the declared grid enumerates 22 cells, not
26** (F1 2×3 = 6, F2 2×3 = 6, F3 2×2 = 4, F4 2, F5 2, F6 2). A mis-stated cell count is not cosmetic —
it is the denominator of the permutation adjustment and of the "70 cumulative" figure, so the plan
currently over-states its own multiplicity correction budget while under-stating its enumeration.
Second, **two families are spending six cells on questions the literature has already answered**: F1's
six cells test a SUE drift that Martineau (2022) reports as absent for non-microcap US stocks after
2006, using a weaker surprise measure than his (Livnat & Mendenhall 2006), and F6's tradable cell is a
~250-round-trip-a-year book that Haghani et al. show dies at 1 bp of cost and that our own
`overnight_auction.md` and M29 already ran to a negative read-once split. Third, **the crux column is
missing**: the long-leg share of the L-S spread is the number that decides everything for this account
and the literature supplies it for exactly one of the six (≈50% for momentum, Israel & Moskowitz 2013),
so we must measure it ourselves.

**The exact edit (keeps the declared budget at 26, so the permutation denominator and the "70
cumulative" line both become true rather than aspirational):** F1 **6 → 2** (keep only the 60-session L-S
and long-only cells, and re-label them in the plan as a *declared replication of a published null*, not
an edge hypothesis); F2 **6 → 6**, and it moves to the **front** of the run order as the highest-information
family; F3 **4 → 6** (add the residual-momentum L-S and long-only 12-1 variants, no new data); F4 **2 → 1**
(long-only weekly industry-residual only, with the break-even cost reported per Stage O's rule, and a
written note that our SIC residual is a proxy for DLS's analyst-revision construction); F5 **2 → 2** with
the ex-January number reported separately and the hypothesis re-labelled as a profitability/investment
tilt per George–Hwang–Li (2018); F6 **2 → 1**, measurement/decomposition only, no tradable cell; then the
four additions at 2 cells each — **A1 earnings-announcement premium 2, A2 low-short-interest long leg 2,
A3 net share issuance 2, A4 dividend-month premium 2**. Total **2+6+6+1+2+1+2+2+2+2 = 26**. Add to the
Deliverables section three mandatory columns for every cell — **long-leg share of the L-S spread, the
break-even cost, and the ex-January result** — and one new data line: A2 needs a free FINRA semi-monthly
short-interest download keyed to the **dissemination** date (not the settlement date) or it is a
look-ahead; A3 and A4 need only one extra EDGAR `companyconcept` and one extra Alpaca corporate-action
type respectively, both free and both on pipelines the data stage has already built.

---

## 6. Sources

Verified this session via Crossref, OpenAlex, RePEc/IDEAS, NBER, OSF and the publisher pages named
inline. Every DOI above was returned by one of those services; the items marked **NOT VERIFIED** were
reachable only as metadata and no quantitative claim rests on them.

- [McLean & Pontiff 2016, JF 71(1) 5–32](https://doi.org/10.1111/jofi.12365) ·
  [Chordia, Subrahmanyam & Tong 2014, JAE 58(1) 41–58](https://doi.org/10.1016/j.jacceco.2014.06.001) ·
  [Hou, Xue & Zhang, NBER w23394](https://www.nber.org/papers/w23394) ·
  [Novy-Marx & Velikov, NBER w20721](https://www.nber.org/papers/w20721)
- [Israel & Moskowitz 2013, JFE 108(2) 275–301](https://doi.org/10.1016/j.jfineco.2012.11.005) ·
  [Stambaugh, Yu & Yuan 2012, JFE 104(2) 288–302](https://ideas.repec.org/a/eee/jfinec/v104y2012i2p288-302.html) ·
  [Fama & French 2008, JF 63 1653–1678](https://doi.org/10.1111/j.1540-6261.2008.01371.x)
- [Bernard & Thomas 1989, JAR 27](https://doi.org/10.2307/2491062) ·
  [Livnat & Mendenhall 2006, JAR 44(1) 177–205](https://doi.org/10.1111/j.1475-679x.2006.00196.x) ·
  [Martineau 2022, CFR 11(3-4) 613–646](https://doi.org/10.1561/104.00000122) ·
  [preprint](https://api.osf.io/v2/preprints/z7k3p/)
- [Chan, Jegadeesh & Lakonishok 1996, JF 51(5)](https://www.nber.org/papers/w5375) ·
  [Jegadeesh & Titman 1993, JF 48(1) 65–91](https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html) ·
  [Asness, Moskowitz & Pedersen 2013, JF 68(3)](https://doi.org/10.1111/jofi.12021) ·
  [Daniel & Moskowitz 2016, JFE 122(2) 221–247](https://www.nber.org/papers/w20439)
- [Da, Liu & Schaumburg 2014, Mgmt Sci 60(3) 658–674](https://ideas.repec.org/a/inm/ormnsc/v60y2014i3p658-674.html) ·
  [Cheng, Hameed, Subrahmanyam & Titman 2017, JFQA 52(1)](https://doi.org/10.1017/s0022109016000958)
- [George & Hwang 2004, JF 59(5)](https://doi.org/10.1111/j.1540-6261.2004.00695.x) ·
  [George, Hwang & Li 2018, JFE 128(1) 148–163](https://ideas.repec.org/a/eee/jfinec/v128y2018i1p148-163.html) ·
  [Chen, Stivers & Sun 2024, JEF 79](https://doi.org/10.1016/j.jempfin.2024.101556) ·
  [Byun & Jeon 2023, FAJ 79(2)](https://doi.org/10.1080/0015198x.2023.2183706)
- [Lou, Polk & Skouras 2019, JFE 134(1) 192–213](https://ideas.repec.org/a/eee/jfinec/v134y2019i1p192-213.html) ·
  [Bogousslavsky 2021, JFE 141(1) 172–194](https://ideas.repec.org/a/eee/jfinec/v141y2021i1p172-194.html) ·
  [Hendershott, Livdan & Rösch 2020, JFE 138(3) 635–662](https://ideas.repec.org/a/eee/jfinec/v138y2020i3p635-662.html) ·
  [Akbas, Boehmer, Jiang & Koch 2022, JFE 145(3) 850–875](https://ideas.repec.org/a/eee/jfinec/v145y2022i3p850-875.html) ·
  [Lachance 2023, RFE 41(4)](https://doi.org/10.1002/rfe.1180) ·
  [Boyarchenko, Larsen & Whelan 2023, RFS 36(9) 3502–3547](https://ideas.repec.org/a/oup/rfinst/v36y2023i9p3502-3547..html) ·
  [Haghani, Ragulin & Dewey 2022, "Night Moves"](https://elmwealth.com/night-moves/)
- [Frazzini & Lamont, NBER w13090](https://www.nber.org/papers/w13090) ·
  [Barber, De George, Lehavy & Trueman 2013, JFE 108(1) 118–138](https://ideas.repec.org/a/eee/jfinec/v108y2013i1p118-138.html) ·
  [Levi & Zhang 2015, JFE 118(2)](https://ideas.repec.org/a/eee/jfinec/v118y2015i2p383-398.html) ·
  [Chapman 2018, JAE 66(1)](https://ideas.repec.org/a/eee/jaecon/v66y2018i1p222-243.html) ·
  [Johnson & So 2018, JAR 56(1) 217–263](https://doi.org/10.1111/1475-679x.12189)
- [Boehmer, Huszár & Jordan 2010, JFE 96(1) 80–97](https://ideas.repec.org/a/eee/jfinec/v96y2010i1p80-97.html) ·
  [Rapach, Ringgenberg & Zhou 2016, JFE 121(1)](https://ideas.repec.org/a/eee/jfinec/v121y2016i1p46-65.html) ·
  [Autore, Boulton & Braga-Alves 2015, Financial Review 50(2)](https://ideas.repec.org/a/bla/finrev/v50y2015i2p143-172.html) ·
  [Fotak, Raman & Yadav 2014, JFE 114(3)](https://ideas.repec.org/a/eee/jfinec/v114y2014i3p493-516.html)
- [Pontiff & Woodgate 2008, JF 63(2) 921–945](https://doi.org/10.1111/j.1540-6261.2008.01335.x) ·
  [Ikenberry, Lakonishok & Vermaelen 1995, JFE 39(2-3)](https://ideas.repec.org/a/eee/jfinec/v39y1995i2-3p181-208.html) ·
  [Bargeron, Bonaimé & Thomas 2017, JFQA 52(2)](https://ideas.repec.org/a/cup/jfinqa/v52y2017i02p491-517_00.html) ·
  [Goto, Wang & Yan, FAJ 76(1)](https://doi.org/10.1080/0015198x.2019.1682427)
- [Hartzmark & Solomon 2013, JFE 109(3) 640–660](https://ideas.repec.org/a/eee/jfinec/v109y2013i3p640-660.html) ·
  [Blitz, Huij & Martens 2011, JEF 18(3)](https://ideas.repec.org/a/eee/empfin/v18y2011i3p506-521.html) ·
  [Blitz, Hanauer & Vidojevic 2020, IREF 69](https://ideas.repec.org/a/eee/reveco/v69y2020icp932-957.html)
- [Greenwood & Sammon, NBER w30748](https://www.nber.org/papers/w30748) ·
  [Lucca & Moench 2015, JF 70(1)](https://ideas.repec.org/a/bla/jfinan/v70y2015i1p329-371.html) ·
  [Kurov, Halova Wolfe & Gilbert 2021, FRL 40](https://ideas.repec.org/a/eee/finlet/v40y2021ics1544612320315956.html) ·
  [Cieslak, Morse & Vissing-Jorgensen 2019, JF 74(5)](https://ideas.repec.org/a/bla/jfinan/v74y2019i5p2201-2248.html)
- [Moskowitz, Ooi & Pedersen 2012, JFE 104(2)](https://ideas.repec.org/a/eee/jfinec/v104y2012i2p228-250.html) ·
  [Huang, Li, Wang & Zhou 2020, JFE 135(3) 774–794](https://ideas.repec.org/a/eee/jfinec/v135y2020i3p774-794.html) ·
  [Frazzini & Pedersen 2014, JFE 111(1)](https://ideas.repec.org/a/eee/jfinec/v111y2014i1p1-25.html) ·
  [Novy-Marx & Velikov 2022, JFE 143(1) 80–106](https://ideas.repec.org/a/eee/jfinec/v143y2022i1p80-106.html)
- [Etula, Rinne, Suominen & Vaittinen 2020, RFS 33(1) 75–111](https://ideas.repec.org/a/oup/rfinst/v33y2020i1p75-111..html) ·
  [Zhang & Jacobsen 2021, JIMF 110](https://ideas.repec.org/a/eee/jimfin/v110y2021ics0261560620302242.html) ·
  [Bouman & Jacobsen 2002, AER 92(5)](https://www.aeaweb.org/articles?id=10.1257/000282802762024683) ·
  [Cheema, Ding & Wang 2023, JAM 24(6)](https://ideas.repec.org/a/pal/assmgt/v24y2023i6d10.1057_s41260-023-00324-1.html) ·
  [Green, Hand & Soliman 2011, Mgmt Sci 57(5)](https://doi.org/10.1287/mnsc.1110.1320)
