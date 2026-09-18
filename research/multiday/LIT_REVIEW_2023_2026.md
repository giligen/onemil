# Multi-day literature — closing the 2023–2026 gap

Written 2026-09-18, while `research/multiday/data/` is still building, so anything this changes in
`PLAN.md` is still a **pre-registration and not a post-hoc edit**. Companion to
`research/multiday/LIT_REVIEW.md`, whose newest citation was 2022 — the gap the owner caught.
No data was pulled, no code run, no config touched.

> **SECOND PASS, 2026-09-18 (same day, still before any family was scored).** The six items §7 listed as
> unclosed were worked; **five closed**, including all three previously unreadable abstracts and both
> regulatory items. The results are in **§7 (rewritten)** and **AMENDMENT 3 (§6b)**. Headline: the cell
> count stays at **26** and the run order is unchanged, but F4's prior drops, A1's age disclosure is
> withdrawn, A4 gains a **verified two-regime ex-date rule** with a discontinuity **inside the TEST
> split**, and the Reg NMS tick/fee reform is confirmed **NOT in force until November 2027** — so §2.2's
> hypothesised spread-compression tailwind is **withdrawn**, not banked. Sections 1, 2.1, 2.2, 6(c),
> 6(e), 7 and 8 were amended; §§3, 4, 5 are unchanged.

Evidence standard is `research/fuckup_audit/PLAN.md` §1: every claim carries author, year, venue and a
DOI/URL that was **fetched during this session**. Anything not fetched is marked **NOT VERIFIED** and
decides nothing. The assistant's own recollection of 2023–2026 papers is not a citation and is not used.

---

## 0. Method, and what this review could NOT check (read this before trusting any row)

**The session's WebSearch budget was already exhausted (200/200) before this task began.** Zero general
web searches were available. All discovery was done by querying bibliographic APIs through WebFetch:
OpenAlex (`api.openalex.org`), Crossref (`api.crossref.org`), the arXiv API (`export.arxiv.org`), and
`sec.gov` directly. ~40 such fetches were made.

Consequences, stated plainly because they bias every "no post-2022 evidence found" verdict below:

1. **Coverage is biased toward indexed journal articles with retrievable metadata.** SSRN returns 403
   to this tool, and SSRN is where a large share of 2025–2026 finance research still sits. Several
   directly on-topic working papers were visible only as titles (0DTE market-quality papers, the
   earnings-announcement-premium re-tests, several PEAD items). They are listed as NOT VERIFIED.
2. **Elsevier/ScienceDirect abstracts are frequently absent from both Crossref and OpenAlex.** Three
   papers that are squarely on our families could be confirmed to exist (title, authors, journal, DOI)
   but their content could not be read: *Maxing out short-term reversals in weekly stock returns* (JEF
   2025), *Intraday and overnight return anomalies* (FRL 2025), and the JFE version of Lopez-Lira &
   Tang (the arXiv version was used instead, and that substitution is flagged where it matters).
3. **OpenAlex serves abstracts as an inverted index.** Reconstructing prose from it is lossy — the
   reconstructions came back with occasional dropped words. So: **quotes sourced from OpenAlex
   reconstructions are marked "(reconstructed)" and are paraphrase-risk, not verbatim.** Only
   Crossref-deposited abstracts, arXiv abstracts and sec.gov pages are quoted as exact.
4. **Not checked at all, for want of a general web search**: practitioner and industry sources (Elm,
   Robeco, AQR, exchange research), exchange-published closing-auction volume statistics, 0DTE market
   share data, and the **current status of the September-2024 Reg NMS compliance dates** (both dates
   are now in the past — see §2.2, this is the one gap I would close first).
   **UPDATE, second pass 2026-09-18:** items 2 and 4's Reg NMS gap are **closed** — the three Elsevier
   abstracts via the `colab.ws` Crossref mirror (secondary source, caveat in §7.1) and the Reg NMS status
   from five Federal Register primary documents. The closing-auction volume share and the 0DTE and
   practitioner gaps **remain open** and still need a general web search (§7.3).

**The single most important number this review adds, and it is a discount, not an opportunity:**
Chen & Velikov, *Zeroing In on the Expected Returns of Anomalies*, **JFQA** (published online 2022,
DOI [10.1017/S0022109022000874](https://doi.org/10.1017/s0022109022000874)) — Crossref-deposited
abstract, exact: across **204** anomalies, after effective spreads, post-publication effects and
modern trading technology, *"the average anomaly's expected return is a measly 4 bps per month. The
strongest anomalies net, at best, 10 bps after controlling for data mining. Several methods combining
anomalies net around 20 bps."* And they note this **omits price impact**.

4–10 bps/month is 0.5–1.2%/yr. On a $50–66K book that is **$250–800/yr**. Every cell in `PLAN.md`
should be read against that number before it is read against its published magnitude. This does not
say the program is pointless — it says the honest prior for a *found* survivor is a few hundred dollars
a year, and a cell that reports much more is more likely mis-specified than exceptional.

The counterweight, also verified: Jensen, Kelly & Pedersen, *Is There a Replication Crisis in Finance?*,
**Journal of Finance 2023**, DOI [10.1111/jofi.13249](https://doi.org/10.1111/jofi.13249)
(reconstructed): the majority of asset-pricing factors **do** replicate, cluster into 13 themes, and
work out-of-sample in a new 93-country data set. So the factors are real; Chen–Velikov says what is
left after you pay to trade them. Both are true at once, and the gap between them is exactly the
long-leg/cost question `LIT_REVIEW.md` §1 identified.

---

## 1. Per-family table — newest VERIFIED evidence, 2023–2026

"Changes cells?" is answered against the AMENDED grid in `PLAN.md` (F1 2, F2 6, F3 6, F4 1, F5 2,
F6 1, A1 2, A2 2, A3 2, A4 2 = 26).

| # | family | newest verified evidence | yr | what it says | changes cells / rank? |
|---|---|---|---|---|---|
| F1 | PEAD / SUE | *Earnings Autocorrelation and the PEAD: Experimental Evidence*, **JFQA**, DOI [10.1017/S0022109023000881](https://doi.org/10.1017/s0022109023000881) | 2023 | A **laboratory** study of the mechanism. Says nothing about whether the field effect is tradable in 2016–2026 US equities. | **No.** Martineau 2022 remains the decisive field cite. F1 stays 2 cells as a declared null-replication. |
| F2 | earnings-**announcement-return** drift (CJL 1996) | Jegadeesh, Luo, Subrahmanyam & Titman, *Short-Term Reversals and Longer-Term Momentum around the World*, **RFS**, DOI [10.1093/rfs/hhaf057](https://doi.org/10.1093/rfs/hhaf057) — Crossref/OpenAlex agree, abstract exact | 2025 | Model + US and international evidence; novel prediction (a), supported: **reversals are attenuated after earnings announcements**. I.e. post-announcement returns behave less mean-reverting than the unconditional short-horizon pattern. | **No cell change — but it is the first post-2022 evidence pointing F2's way**, and it is indirect (it does not sort on the announcement return). F2 keeps 6 cells and the front of the run order. |
| F3 | 12-1 momentum + residual momentum | Jensen, Kelly & Pedersen, **JF 2023**, DOI [10.1111/jofi.13249](https://doi.org/10.1111/jofi.13249); Chen & Velikov, **JFQA**, DOI [10.1017/S0022109022000874](https://doi.org/10.1017/s0022109022000874) | 2023 / 2022 | Factors incl. momentum replicate out-of-sample across 93 countries; and the average anomaly nets 4 bps/mo after costs. **No post-2022 study specific to US 12-1 decay was found. No post-2022 re-test of residual momentum was found** (Blitz, Hanauer & Vidojevic 2020 remains newest). | **No.** F3 stays at 6. The residual-momentum cells rest on 2020 evidence — say so in the report. |
| F4 | short-term reversal, industry-adj. | **Chen, Cohen, Liang & Sun, *Maxing out short-term reversals in weekly stock returns*, Journal of Empirical Finance 82, 101608 (June 2025)**, DOI [10.1016/j.jempfin.2025.101608](https://doi.org/10.1016/j.jempfin.2025.101608) — abstract **CLOSED 2026-09-18 via the colab.ws Crossref mirror**, see §7.1 for the provenance caveat. Second verified item: *Evolution of short-term contrarian profits*, **Studies in Economics and Finance 2023**, DOI [10.1108/sef-12-2022-0599](https://doi.org/10.1108/sef-12-2022-0599). Third: Jegadeesh, Luo, Subrahmanyam & Titman, **RFS 2025**, DOI [10.1093/rfs/hhaf057](https://doi.org/10.1093/rfs/hhaf057) | **2025** / 2023 | JEF 2025 is **exactly our F4 horizon** (weekly, US). The reversal it finds is **conditional on MAX**: among high-MAX ("lottery-like", elevated recent maximum daily return) stocks, past 1-week losers/winners earn positive/negative next-week returns worth **≈1.66%/week**, vs **0.65%/week** applying the same strategy to low-MAX stocks; and the profit **"emerges exclusively during periods of highest retail order imbalance."** SEF 2023 says the *unconditional* US effect is gone: short-term contrarian profits **"have largely disappeared in past two decades"**, vanished since 2000, and the cause is a fall in the level of overreaction, not the cross-sectional or lead-lag components. RFS 2025 supplies the mechanism (reversals larger with more noise trading). | **No cell change — and the prior moves DOWN, not up.** The one live 2023–2026 form of weekly reversal is (a) **conditional on high MAX** — low-priced, high-volatility lottery names, which our raw-close ≥ $5 and $10M-ADV floors delete — and (b) **conditional on extreme retail order imbalance**. **Neither paper reports transaction costs; neither splits the long leg from the short leg** — and a 100%-per-week turnover strategy is the textbook case Detzel, Novy-Marx & Velikov (**JF 78(3) 1743–1775, 2023**, DOI [10.1111/jofi.13225](https://doi.org/10.1111/jofi.13225)) say inflates expensive factors when costs are neglected. F4 keeps **1** long-only cell and its second-to-last slot in the run order; see AMENDMENT 3(a) for the reporting requirement it now carries. |
| F5 | 52-week high | **None.** A full 2022→2026 sweep returned only covariate work already in `LIT_REVIEW.md` (Byun & Jeon FAJ 2023; Chen, Stivers & Sun JEF 2024) plus anchoring/insider-behaviour studies. | — | **No post-2022 re-test of the George–Hwang return premium was found.** The newest direct evidence remains George, Hwang & Li 2018 (q-factors explain it). | **No.** F5 stays at 2 with the ex-January split. Note in the report that its evidence base is now 8 years old. |
| F6 | overnight vs intraday | **Still none on-topic.** The flagged candidate — Zirk-Sadowski & Hryckiewicz, *Intraday and overnight return anomalies: Evidence from 11.6 million price observations*, **FRL 86, 108638 (Dec 2025)**, DOI [10.1016/j.frl.2025.108638](https://doi.org/10.1016/j.frl.2025.108638) — was **READ on 2026-09-18** and is **NOT about our question**. | 2025 | The FRL paper is a **time-of-day / day-of-week seasonality** study on **small-capitalisation NYSE stocks**, ~12M observations at 30-second-to-60-minute intervals, bootstrapped ANOVA: an 11 AM hour effect, anomalies persisting Tue–Thu, a reversed morning effect at 10 AM on Mondays. It contains **no cross-sectional overnight-vs-intraday return decomposition**, no Lou–Polk–Skouras replication, and no cost analysis. | **No — and one false lead is now retired.** F6 stays at 1 measurement cell. A **second, independent** 2023+ sweep for an LPS-style US cross-sectional re-test (overnight/intraday, tug-of-war, institutional-vs-retail) returned only China, Korea, sector-ETF and European-sentiment papers. **The post-2022 silence on F6 is now a searched-for absence across two sweeps, not an indexing gap.** |
| A1 | earnings-announcement **premium** | **CLOSED 2026-09-18.** Tsafack, Becker & Han, *Earnings announcement premium and return volatility: Is it consistent with risk-return trade-off?*, **Pacific-Basin Finance Journal 79, 102029 (June 2023)**, DOI [10.1016/j.pacfin.2023.102029](https://doi.org/10.1016/j.pacfin.2023.102029) — abstract read via the colab.ws mirror (§7.1 caveat). Secondary: *Lottery Demand and Stock Returns Preceding Earnings Announcements*, **Journal of Business Finance & Accounting 2025**, DOI [10.1111/jbfa.70008](https://doi.org/10.1111/jbfa.70008) (OpenAlex, reconstructed). | **2023** / 2025 | PBFJ 2023: the earnings-announcement premium — positive average returns during announcements — **persists in post-financial-crisis data**; it is **positively** related to *expected* volatility (**"a risk-return tradeoff"**) and **inversely** related to *realized* volatility, on both positive and negative surprises. Despite the journal, the study is on the US sample and offers Asia-Pacific markets as **future** work. JBFA 2025: stocks with the most extreme returns around *past* announcements beat the least-extreme by **≈85 bps over the 10 days preceding** the current announcement — a lottery-demand channel in the pre-announcement window. | **YES — one disclosure is withdrawn; cells and rank unchanged.** A1 is no longer an "evidence is 8 years old" cell: it has 2023 evidence that the premium **survives post-crisis**, which supports its rank-2 placement. Two honest riders replace the age disclosure: (i) PBFJ frames the premium as **compensation for expected volatility risk**, not mispricing — a long-only harvester earns it *and bears the risk*, so it is not free alpha and must be reported next to its realized volatility; (ii) **neither paper reports transaction costs or a long-only implementation**, and JBFA's pre-announcement excess again lives in the **lottery/high-MAX** corner our floors delete. A1 keeps **2** cells. |
| A2 | low short interest, liquid names (long side) | *Short Interest and Aggregate Stock Returns: International Evidence*, **Review of Asset Pricing Studies 2023**, DOI [10.1093/rapstu/raad007](https://doi.org/10.1093/rapstu/raad007) (reconstructed) | 2023 | Short interest **negatively predicts aggregate** returns in 24 of 32 countries, survives out-of-sample. This is the **time-series/market-level** signal (the Rapach–Ringgenberg–Zhou result, extended internationally). | **No.** It corroborates that short interest carries information; it does **not** re-test Boehmer–Huszár–Jordan's cross-sectional **long** leg, which remains 2010 evidence with **no post-2022 replication found**. A2 stays at 2 cells; the "decay status NOT VERIFIED" flag in `LIT_REVIEW.md` §3 stands and is now a deliberate, searched-for null. |
| A3 | net share issuance | **None US.** Nearest: a **pre-registered** buyback-anomaly replication in **Japan**, Pacific-Basin Finance Journal 2025, DOI [10.1016/j.pacfin.2025.102666](https://doi.org/10.1016/j.pacfin.2025.102666) (reconstructed); and a risk-based explanation of composite equity issuance, IRFA 2024, DOI [10.1016/j.irfa.2024.103288](https://doi.org/10.1016/j.irfa.2024.103288) (**abstract NOT VERIFIED**). | 2024–25 | **No post-2022 US replication of the share-issuance premium was found.** | **No.** A3 stays at 2. |
| A4 | dividend-month premium | **Hartzmark & Solomon, *Market-Wide Predictable Price Pressure*, American Economic Review 115(9) 3171–3213**, DOI [10.1257/aer.20231725](https://doi.org/10.1257/aer.20231725) (reconstructed; authors/venue/pages exact from Crossref) | **2025** | The **same authors**, same mechanism, in a top-5 journal: predictable uninformed cash flows (dividend payments announced weeks ahead) forecast returns; top-quintile payment days earn ~4× the bottom quintile; holds internationally; estimated market-level price multiplier 1.9. | **YES — rank change.** `PLAN.md` ranks A4 last with "run only if A1–A3 leave budget", justified by "no post-2016 replication found". That justification is now **false**: the mechanism has a 2025 AER publication. The *cross-sectional* dividend-month premium still has no post-2016 replication, so the cells stay at 2 — but the "run only if budget" rider should be removed and A4 moved up the run order. |

**Two honest summary statements about Priority 1 — REVISED 2026-09-18 after the second pass.**
(i) The evidence-age finding shrinks from five families to **four**: for **F1 (field), F5, F6 and A3 no
post-2022 evidence of any kind could be verified**, now across *two independent sweeps each* for F5, F6
and A3 — a searched-for absence, not an indexing gap. **A1 leaves that list** (PBFJ 2023 verifies the
premium persists post-crisis). Those four cells are still being run on evidence 6–20 years old, in a
literature whose own meta-studies say to expect 26%/58% haircuts, and the report must say so next to each
number. (ii) The genuine 2023–2026 upgrades are now **two**: A4's mechanism (AER 2025) and A1's
persistence (PBFJ 2023). The one 2023–2026 *supportive* signal for the highest-priority family remains
indirect (RFS 2025's "attenuated reversals after earnings announcements").

**A third statement, new, and it is the most useful thing this second pass produced.**
**Every live 2023–2026 form of the two effects we could finally read sits in the lottery / high-MAX /
extreme-retail-order-imbalance corner of the cross-section** — JEF 2025 gets weekly reversal only in
high-MAX names and only in the top retail-order-imbalance periods; JBFA 2025 gets the pre-announcement
excess from extreme past-announcement returns. That corner is **low-priced, high-volatility, and exactly
what `PLAN.md`'s raw-close ≥ $5 gate and the $10M-ADV liquidity floor delete.** It is the same shape as
Avramov, Cheng & Metzker's finding that excluding microcaps and high-volatility names "considerably
attenuates" ML profitability, and the same shape as Lopez-Lira & Tang's drift living in small stocks.
**Pre-committed expectation, recorded before any cell was scored: our long-only, ≥ $5, liquid-name
variants of F4 and A1 should measure materially LESS than the published magnitudes. A cell that measures
close to the published number is evidence of a floor leak or a construction error first, and an
exceptional result second.**

---

## 2. US market structure 2023–2026, and what it does to our cost model

### 2.1 T+1 settlement — in force since 2024-05-28
SEC press release [2023-29](https://www.sec.gov/newsroom/press-releases/2023-29), fetched: the
standard settlement cycle was shortened from T+2 to T+1, and *"the compliance date for the final rules
is May 28, 2024."* (exact).

**Implication for us: essentially none for cost, one real trap for data.** Auction fill prices are not
a function of the settlement cycle. But our sample spans the change, and two things shift at that
boundary:
- Cash-account buying power recycles one day sooner from 2024-05-28 — relevant to the *engine build*
  (a long-only $50–66K book turning over monthly), not to the backtest's returns.
- **The ex-dividend date / record date relationship changed mechanically when the cycle shortened.**
  **CLOSED 2026-09-18 from a primary source.** **FINRA Rule 11140(b)(1)**
  ([finra.org rulebook](https://www.finra.org/rules-guidance/rulebooks/finra-rules/11140), fetched):
  *"the date designated as the 'ex-dividend date' shall be the record date if the record date falls on a
  business day"* (and the first business day preceding it if the record date is not a delivery day). The
  rulebook page records the amendment as **SR-FINRA-2023-017, effective May 28, 2024** — the T+1
  compliance date. So:
  - **On and after 2024-05-28: ex-date = record date** (record date a business day).
  - **Before 2024-05-28 (T+2): ex-date = one business day BEFORE the record date.**

  This is no longer a hypothesis. It is a **date-construction rule with a discontinuity inside our TEST
  split** (TEST is 2024-01 → 2026-09, so the break falls four months into it), and **A4's whole signal is
  a predicted payment month built from prior-year dividend dates that straddle it.** A single-formula
  ex-date construction applied across 2024-05-28 mis-dates every pre-break event by one business day —
  which, for a signal whose resolution *is* the ex-date, is a first-order error, not rounding. See
  AMENDMENT 3(c) for the exact pre-registered rule.

### 2.2 Reg NMS amendments — adopted 2024-09-18, **status CLOSED 2026-09-18: NOT IN FORCE, twice postponed**
SEC press release [2024-137](https://www.sec.gov/newsroom/press-releases/2024-137) and rulemaking
record **Release 34-101070, File S7-30-22**, both fetched. Adopted (quotes exact):
- A new **$0.005 minimum pricing increment** for NMS stocks priced ≥ $1.00 that are tick-constrained,
  assigned every six months from Time Weighted Average Quoted Spread over a three-month evaluation.
- **Access fee cap cut to $0.001/share** for stocks ≥ $1.00 (from the long-standing $0.003), and fees
  must be *"determinable at the time of execution."*
- Identification of the **best-priced odd-lot orders**.
- **Compliance: "the first business day of November 2025"** for Rule 612, Rule 610 and the round-lot
  definition; **"the first business day of May 2026"** for odd-lot information.

**CLOSED 2026-09-18 from the Federal Register API and the SEC's own exemptive orders, read in full text.
Neither compliance date took effect. Both were postponed by SEC temporary exemptive relief, and the tick
/ access-fee package has now been postponed twice.** Status table, every row from a fetched primary
document:

| amended rule | what it does | original compliance date | current status |
|---|---|---|---|
| **Rule 612** ($0.005 min. pricing increment) | half-penny quoting for tick-constrained stocks ≥ $1 | first business day of **Nov 2025** | exempted to Nov 2026 (FR **2025-19926**, pub. 2025-11-17), then exempted again to the **first business day of November 2027** (FR **2026-11997**, pub. 2026-06-15). **NOT in force.** |
| **Rule 610(c)** ($0.001 access fee cap) | cuts the cap from $0.003 | first business day of **Nov 2025** | same two orders → **first business day of November 2027**. **NOT in force.** |
| **Rule 600(b)(89)(i)(F)** (round-lot tier) | new round-lot definition | first business day of **Nov 2025** | same two orders → **first business day of November 2027**. **NOT in force.** |
| **Rule 610(d)** (fees determinable at execution) | fees must be knowable at execution | first business day of **Nov 2025** | extended only to the **first business day of February 2026** (FR 2025-19926) and **not** renewed in the June-2026 order, which covers only 610(c), 612 and 600(b)(89)(i)(F). **Inference: in force since ~2026-02-02.** Marked as inference — no document was fetched that affirmatively states it took effect. |
| **Rule 600(b)(69)(ii)** (odd-lot information dissemination) | best-priced odd-lot data on the SIP | first business day of **May 2026** | exempted for the Plans' participants (19 exchanges + FINRA) to the **first business day of May 2028** (FR **2026-01013**, pub. 2026-01-21). **NOT in force.** |

**Implication — this REVERSES the tentative "good news" reading above, and the correct direction is that
a hypothesised tailwind is withdrawn, not that a cost is added.** The half-penny tick and the $0.001
access-fee cap are the two changes that would have compressed quoted spreads in exactly the liquid,
higher-priced names our $5 / $10M-ADV floors push us into. They are not in force, and on the current
schedule they will not be before **November 2027** — which is **after** the last date in our TEST split
(2026-09) and after any plausible live-trading window this program produces. Therefore:

1. **The entire sample, 2016-01 → 2026-09, sits under the OLD regime**: $0.01 minimum increment,
   $0.003 access-fee cap, 100-share round lots. **No regime break needs to be modelled inside the
   backtest, and none may be assumed as an out-of-sample improvement.** The declared cost model is
   unchanged by this section.
2. The secondary "5 bps/side flat" continuous-market arm gets **no** discount for tick reform. A spread
   estimate calibrated on the sample is valid for the sample *and* for live trading until Nov 2027.
3. Unchanged either way: **an auction fill pays no quoted spread**, so the primary `cls` arm was never
   exposed to this rule. §2.3 (Goyal–Jegadeesh–Wu) remains the binding cost evidence.
4. Forward-looking, noted and **not** acted on: the SEC published a **proposed** rule on 2026-06-17
   (FR **2026-12163**) to **rescind the trade-through rule (Rule 611) and the locked-and-crossed
   provisions** of Reg NMS. Proposed only; it changes nothing in-sample and nothing in the cost model.
   Flagged here so it is on the record before the runs, not discovered after them.

### 2.3 The finding that actually changes our plan: opening auctions are illiquid
**Goyal, Jegadeesh & Wu, *Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets:
A Benchmark for Cost of Trading on Anomalies*, Journal of Financial and Quantitative Analysis, online
2026-03-05, pp. 1–36, DOI [10.1017/S0022109026102592](https://doi.org/10.1017/s0022109026102592).**
Abstract retrieved from Crossref (exact) and independently from OpenAlex (agrees). Verbatim, the four
sentences that matter:

> "Closing auctions account for about 10% of daily trading volume… the price impact is lower in closing
> auctions than in the continuous market for all stocks except Nasdaq microcaps. Opening auctions are
> illiquid… The annualized trading costs for long/short portfolios based on financial ratios such as
> profitability and investment range from 17 to 41 basis points (bps)… Excluding microcaps, these costs
> fall to 9–21 bps in closing auctions."

This is a top-tier, 2026, directly-on-point measurement of the exact question our cost model guesses
at, and it splits two ways:

- **It confirms the closing-auction half of our model.** `cls` exits paying no quoted spread and low
  impact, in non-microcap names, is supported — and our $5/$10M floors put us outside the one exception
  (Nasdaq microcaps).
- **It refutes the opening-auction half.** `PLAN.md` currently specifies **MOO entry / MOC exit** for
  F1's executable variant, and **MOC buy / MOO sell** for F6. "Opening auctions are illiquid" says the
  `opg` leg is the *expensive* leg, not a free one. Our declared cost model charges the same
  impact allowance at both auctions. **That is now a known error, with a citation, before a single
  cell has been scored.**
- **It gives us a number to calibrate against.** 9–21 bps *annualized* for a monthly-ish rebalanced
  non-microcap long/short book executed in closing auctions. Our own arithmetic, flagged as ours and
  not theirs: a long-only book does roughly half the round trips of an equivalent long/short, so the
  same construction should land on the order of **5–12 bps/yr** — against a Chen–Velikov expected gross
  of 4–10 bps/*month*. That is the first time in this program that cost and expected return have been
  bracketed by two independent published measurements, and it says a **low-turnover, close-executed,
  long-only book is viable on cost** — and that F4 (weekly) and any daily round trip are not.

### 2.4 Retail flow / PFOF / fragmentation
*The Demise of the NYSE and Nasdaq: Market Quality in the Age of Market Fragmentation*, **JFQA 2023**,
DOI [10.1017/S0022109022001545](https://doi.org/10.1017/s0022109022001545) (reconstructed): market
quality has generally improved, but *most of the improvement accrued to the largest stocks*, producing
a bifurcation by size. **Implication: our liquidity floors sit in the corner where execution quality
improved**, which is the one structural tailwind a small long-only account has. No 2023–2026 study of
PFOF-specific retail execution quality in US equities could be verified; a directly relevant paper
exists for Xetra (SSRN 2026) and was unreadable. **NOT VERIFIED.**

### 2.5 0DTE and its spillover to single stocks
**Nothing verifiable.** A full sweep returned SSRN-only titles (*Does 0DTE Options Trading Increase
Volatility?*, *0DTE Option Pricing*, *The Factor Structure of 0DTE Option Returns*, *Retail Traders Love
0DTE Options… But Should They?*) with no retrievable abstracts, plus two 2026 items on hedging
(*Deep hedging 0DTE options*, J. Financial Stability, DOI
[10.1016/j.jfs.2026.101535](https://doi.org/10.1016/j.jfs.2026.101535)) that say nothing about equity
costs. **No verified evidence that 0DTE growth has changed single-stock equity liquidity or auction
costs.** Given our book is long-only multi-day equity executed at the close, this is also low priority —
but it is unchecked, not cleared.

### 2.6 Closing-auction volume share
The only verified figure in this review is Goyal–Jegadeesh–Wu's **~10% of daily volume**. Exchange-published
2024–2026 statistics could not be fetched (no general web search). **Any claim that the close is now
13%/15%/20% of volume is NOT VERIFIED and must not be used.**

---

## 3. New candidate families, 2023–2026, same evidence standard

**Only one clears the bar, and it is not a stock-selection family.** I searched for 2023–2026
anomalies executable long-only at auction prices by a $50–66K account, outside the grid and outside the
rejected list. The honest result: the 2023–2026 journal literature I could reach is overwhelmingly
mechanism papers, international samples, ML methodology, and text/LLM signals — not new,
implementable, long-leg-separable US cross-sectional anomalies.

### N1 (rank 1, and the only one) — Predictable uninformed cash-flow price pressure (market-level)
- **Cite.** Hartzmark & Solomon, *Market-Wide Predictable Price Pressure*, **American Economic Review
  115(9) 3171–3213 (2025)**, DOI [10.1257/aer.20231725](https://doi.org/10.1257/aer.20231725).
  Authors, venue, volume, issue, pages verified from Crossref; abstract reconstructed from OpenAlex.
- **What it says (reconstructed).** Buying pressure from dividend payments — announced weeks in
  advance — predicts higher value-weighted **market** returns; the top quintile of payment days earns
  about four times the bottom quintile; it holds internationally, is stronger when reinvestment is high
  and liquidity is low; estimated market-level price multiplier 1.9.
- **The long-leg answer.** Not applicable in the usual sense, and that is the point: this is a
  **level/timing** effect on the whole market, so the published form **is** the long-only form. There
  is no short leg to miss — the same structural property that made A1 attractive.
- **Data we would need.** Effectively already budgeted: the same one-line change to
  `fetch_splits.py` that A4 needs (`cash_dividend` on Alpaca's `CorporateActionsClient`) to get
  declared/ex/pay dates, plus SPY daily opens/closes we already pull. **Zero new pipelines.**
- **Why I am NOT recommending it as a cell now.** It is a market-timing overlay, not a
  cross-sectional selection rule; putting it inside the 26 would mix two different books and corrupt the
  permutation denominator that `LIT_REVIEW.md` just fixed. And the honest magnitude for us is small:
  an index-level daily effect harvested by a $50–66K account is a few basis points on a few dozen days.
  **Recommendation: log it as a separate one-cell pre-registration with its own gate, to be run after
  the 26, not inside them.** It is recorded here, dated, before the data stage finished — so running it
  later is still pre-registered.

### What was looked for and NOT found (so the absence is on the record)
No verifiable 2023–2026 addition in: accruals-family successors, profitability/investment executable
variants, institutional-flow or ETF-rebalance signals, or any new event-anchored long-only US anomaly.
The one long-only-implementation cost study found was non-US and weak (*Polish equity risk factors and
their implementation costs*, Bank i Kredyt 2025, DOI
[10.5604/01.3001.0055.3041](https://doi.org/10.5604/01.3001.0055.3041), reconstructed) — it reports that
L-S factor returns "deteriorate substantially" after costs while long-only implementations with
turnover constraints and liquidity filters "can still add value". Directionally consistent with our
plan; too weak and too far from our market to carry weight.

**The crux column remains unanswered by the literature.** I specifically searched 2023–2026 for the
long-leg share of published L-S spreads. Nothing was found. `LIT_REVIEW.md` §1's standing requirement —
*we must measure it ourselves* — is therefore not just still valid, it is now a searched-for gap rather
than an assumed one.

---

## 4. SOTA algorithms 2023–2026 — does a modern method beat a simple sort for a small long-only account?

Constraints being scored against: **2 CPUs, no GPU, ~13–15K training rows per book, long-only, auction
execution, $50–66K.**

**Headline answer: no. Not one of the seven lines below has verified 2023–2026 evidence of beating a
simple sort, long-only, net of costs, at anything like our scale.** Two lines have verified evidence of
the *opposite*. One line has a single verified sentence that is mildly encouraging about long legs.

### 4.1 Time-series foundation models (TimesFM, Chronos, Moirai, TimeGPT, Lag-Llama, TTM)
The owner's precise question — are they ever evaluated on return **direction / trading metrics**, or
only on level MAE where a random walk already wins? — now has a 2026 answer, and it is damning.

- **FinVerse: Financial Time-Series Benchmark**, arXiv [2608.03259](https://arxiv.org/abs/2608.03259)
  (2026-08-04), abstract fetched exact. 116,897 financial series, 171.1M observations, **43 public
  time-series foundation models**, 11 metric families / 78 metrics chosen per series by economic
  meaning. Their finding, verbatim: *"strong performance under generic forecasting criteria does not
  necessarily translate into useful financial forecasts."* They state the reason explicitly: in stock
  forecasting, predicting **whether a price will rise or fall** is more relevant to realized returns
  than minimizing point-wise error.
- **Forecast Collapse in Time-Series Foundation Models**, arXiv
  [2608.14106](https://arxiv.org/abs/2608.14106) (2026-08-14), abstract fetched exact. Forecasting
  hourly returns for **1,000 US equities**, TSFM predictions *"become nearly flat and show poor stock
  ranking, as measured by cross-sectional correlation"* — they name this **forecast collapse**. It
  *"largely disappears when forecasting trading volume under the same setting"*, i.e. it is tied to the
  low predictability of the target, not to a bug. They identify a **calibration–ranking tradeoff**:
  optimizing squared error produces flat predictions; optimizing cross-sectional correlation improves
  ranking but can inflate forecast amplitude by more than an order of magnitude.
- A third item (*Hybrid Neural-Classical Correction for Frozen TSFMs*, arXiv 2608.08825) claims
  correlation improvements over frozen TimesFM on technology stocks — its abstract was **not fetched**;
  **NOT VERIFIED**.

**Verdict for this account: no cell, and a rule.** The literature's own 2026 benchmark says a paper
reporting only MAE/MAPE/RMSE on prices is evidence of nothing for us — exactly as the brief suspected.
And the one paper that *does* evaluate on cross-sectional ranking, on US equities, finds collapse. These
models also need a GPU we do not have. **Data needed: n/a. Worth a pre-registered cell: no.**

### 4.2 Transformers / deep nets for cross-sectional returns, vs gradient boosting, net of turnover
- **Avramov, Cheng & Metzker, *Machine Learning vs. Economic Restrictions: Evidence from Stock Return
  Predictability*, Management Science 69(5) 2587–2619 (2023)**, DOI
  [10.1287/mnsc.2022.4449](https://doi.org/10.1287/mnsc.2022.4449). Abstract reconstructed (OpenAlex),
  authors/volume/issue/pages exact (Crossref). This is the decisive cite and it cuts both ways.
  Against: deep-learning signals *extract profitability from difficult-to-arbitrage stocks and during
  high limits-to-arbitrage states*; **excluding microcaps, distressed stocks, or high-volatility
  episodes "considerably attenuates profitability"**; performance *"further deteriorates in the presence
  of reasonable trading costs because of turnover and extreme positions"*. Our $5/$10M floors **are**
  those exclusions. For: *"deep learning signals are profitable in long positions during recent years
  and command low downside risk."* That single clause is the only verified 2023–2026 sentence in this
  entire review that is positive about an ML **long leg**, and it deserves to be quoted honestly rather
  than suppressed — it says the long leg is not obviously worse, **not** that it beats a sort.
- **Blitz, Hanauer, Hoogteijling & Howard, *The Term Structure of Machine Learning Alpha*, Journal of
  Financial Data Science 5(4) 40–65 (2023)**, DOI
  [10.3905/jfds.2023.1.135](https://doi.org/10.3905/jfds.2023.1.135), authors/volume/pages exact
  (Crossref), abstract reconstructed: ML models trained on **one-month** forward returns show impressive
  gross alphas but **net-of-cost performance post-2004 is "close to zero"**. Training on **longer**
  horizons with efficient portfolio construction recovers significant net returns — and those
  longer-horizon strategies *"select slower signals and load more on traditional asset pricing
  factors."*
- **No verified head-to-head transformer-vs-gradient-boosting study on the US cross-section was
  found.** An arXiv sweep returned loss-function and diffusion-model papers, none comparing the two.
  **NOT VERIFIED.**

**Verdict: no cell.** The only version of ML with verified net-of-cost survival is the low-turnover,
long-horizon version — which by the authors' own description converges toward traditional factor
exposure, i.e. toward a simple sort. With ~13–15K rows and 2 CPUs we cannot even reach the regime where
the published gains were measured. **Cost-aware / turnover-aware training is the right idea and is
exactly what the JFDS paper implements; it is not a reason to add a cell to a 26-cell plan.**

### 4.3 LLMs in asset pricing (bears directly on F2/A1 and our 8-K corpus)
- **Lopez-Lira & Tang, *Can ChatGPT forecast stock price movements? Return predictability and large
  language models*, Journal of Financial Economics 184, 104335 (2026)**, DOI
  [10.1016/j.jfineco.2026.104335](https://doi.org/10.1016/j.jfineco.2026.104335). Title/authors/volume/
  pages exact from Crossref; the JFE abstract was not deposited, so the content is quoted from the
  **arXiv version, [2304.07619](https://arxiv.org/abs/2304.07619), v. dated 2025-10-28** — that
  substitution is a caveat, stated. Verbatim: GPT-4 scores *"significantly predict the subsequent
  drift, especially for small stocks and negative news"*, and *"Strategy returns decline as LLM
  adoption rises, consistent with improved price efficiency."*
  **Long-leg answer, bluntly: the reported strength is in NEGATIVE news and SMALL stocks — the short
  leg, below our floors — and the paper documents its own decay as adoption rises.** For a long-only
  book with a $5/$10M floor, that is close to the worst possible profile.
- **Look-ahead is a documented, named failure mode for this exact method.** *Assessing Look-Ahead Bias
  in Stock Return Predictions Generated by GPT Sentiment Analysis*, **Journal of Financial Data Science
  2023**, DOI [10.3905/jfds.2023.1.143](https://doi.org/10.3905/jfds.2023.1.143) (abstract
  reconstructed; **authors NOT VERIFIED**): backtesting LLM sentiment is biased when training periods
  overlap the sample, in two ways — **look-ahead bias** (the model may know the returns that followed
  an article) and a **distraction effect** (general knowledge of the named company interferes with
  sentiment measurement). They find anonymized headlines *outperform* originals, i.e. the distraction
  effect dominates, and the effect is strongest for larger companies.
- *Sentiment trading with large language models*, **Finance Research Letters 2024**, DOI
  [10.1016/j.frl.2024.105227](https://doi.org/10.1016/j.frl.2024.105227) (reconstructed): 965,375 US
  news articles 2010–2023; a long-short strategy with 10 bps costs reports Sharpe 3.05. **No long-leg
  split, no size/liquidity split reported** — treat as weak.

**Verdict: no cell, and one standing rule worth more than the cell would have been.**
**An LLM must never be used to score a document dated before that model's training cutoff.** Our
`research/multiday/data/edgar/` 8-K corpus runs 2016→2026 and every frontier model's cutoff falls
inside it. Any LLM scoring of those filings is contaminated by construction, and the JFDS 2023 paper
shows the contamination is not only the obvious kind. This rule belongs in `PLAN.md`.

### 4.4 Meta-labelling / triple-barrier / purged CV — bears on `research/meta_label/PREREG.md`
I searched 2022–2026 for "meta-labeling", "meta labeling", "triple barrier", "purged cross-validation".
**Result: there is essentially no independent peer-reviewed out-of-sample evidence in equities.** What
exists:
- *Ensemble Meta-Labeling*, **Journal of Financial Data Science 2022**, DOI
  [10.3905/jfds.2022.1.114](https://doi.org/10.3905/jfds.2022.1.114) (reconstructed) — a framework
  paper for model selection; describes when ensembles help (multiple nonlinear regimes) and explicitly
  positions itself as *"a starting point for further research"*. It is advocacy plus experiments, from
  the practitioner line that originated the method.
- *Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling … Pair Trading in Cryptocurrency Markets*,
  **Mathematics (MDPI) 2024**, DOI [10.3390/math12050780](https://doi.org/10.3390/math12050780)
  (reconstructed) — the only item found reporting out-of-sample trading metrics (Sep 2022–Dec 2023,
  crypto pair trading; +51.42% profitability on one signal class, −73.24% MDD on another). **Crypto,
  pairs, MDPI, single study.**

**Verdict, blunt, because it changes how a currently-running job must be read:**
`research/meta_label/PREREG.md` is running xgboost on ~13K ORB candidates right now. **There is no
external evidence base entitling it to a favourable prior.** Meta-labelling is not a validated edge; it
is an unvalidated labelling convention with a practitioner literature and one crypto paper. That means
**our run is the evidence, not a confirmation of someone else's**, and it must therefore carry the full
apparatus on its own: G1/G2, TEST read once, permutation across every cell looked at, tail removal, and
the availability audit. Separately: **purged/embargoed CV is a correctness requirement, not an edge** —
it can only ever lower a measured result, and a book that needs un-purged CV to look good is already
dead. **Worth a pre-registered cell: it already is one; the correction is to the prior, not the plan.**

### 4.5 Reinforcement learning — execution vs sizing (separated, as asked)
- The only peer-reviewed anchor found is a **survey**: *Recent Advances in Reinforcement Learning in
  Finance*, **Mathematical Finance 2023**, DOI [10.1111/mafi.12382](https://doi.org/10.1111/mafi.12382)
  (reconstructed; **authors NOT VERIFIED**), which covers optimal execution and order routing among
  applications but is not itself evidence of out-of-sample performance.
- **No verified out-of-sample, retail-scale, US-equity evidence was found for either application.**
  Searches returned overwhelmingly non-finance RL (robotics, LLM reasoning) or simulation-only work.

**Verdict for this account: no cell for either, for different reasons.**
**Execution RL is the more credible of the two in the literature and the more useless to us**: its whole
value is optimizing a *child-order schedule* across time, and our execution is a single order resting in
a single auction — there is no schedule to learn. **Sizing RL has neither evidence nor sample**: with
~13–15K rows, 4 concurrent slots and one decision per position, the effective sample for a sizing policy
is in the hundreds, which cannot identify a policy that a fixed fractional rule would not.

### 4.6 Conformal prediction / uncertainty quantification for position sizing
One directly on-point, **pre-registered**, and **negative** study:
**Conformal Kelly: Conformal Prediction Intervals as the Scale in Fractional Kelly Position Sizing**,
arXiv [2608.01494](https://arxiv.org/abs/2608.01494) (2026-08-02), abstract fetched exact. A 75%
conformal interval scales a fractional-Kelly position (wider interval → smaller position). Development
window 2016–2021, with trading costs and leverage caps: 28.5% annualized net log growth, Sharpe 1.34,
27.7% max drawdown, vs 15.9% for the S&P 500. The configurations were then **sealed from 2022 onward and
pre-registered** before evaluation. Verbatim result: *"Calibration held (0.745 coverage against 0.750…);
growth did not: the two configurations earned 8.5% and 7.0% per year, below the passive benchmarks."*

**Verdict: no cell.** This is the most methodologically honest paper in the whole sweep — the search was
run by an LLM agent over 200 configurations, the data were sealed, the outcome was reported as
pre-registered — and the honest outcome was that the interval-scaling idea did not survive. Caveat:
single-author arXiv preprint, not peer reviewed. Its **method** is worth copying (seal, pre-register,
report whatever comes out); its **signal** is not worth a cell.

### 4.7 Ensembles and factor-zoo shrinkage (Kozak–Nagel–Santosh line and successors)
**No verified 2023–2026 evidence that shrinkage-based portfolio construction beats a simple sort
long-only net of costs.** What the sweep actually returned was a different question — shrinkage of the
**mean vector / covariance** in mean-variance optimization vs the 1/N rule (*Optimal Shrinkage of Means
in the Markowitz Model*, IRFA 2025, DOI
[10.1016/j.irfa.2025.104136](https://doi.org/10.1016/j.irfa.2025.104136); *Distribution-Free Shrinkage
of High-Dimensional Mean Vector*, JBES 2026, DOI
[10.1080/07350015.2026.2638490](https://doi.org/10.1080/07350015.2026.2638490); both reconstructed).
Both report better out-of-sample Sharpe than benchmarks; **neither treats trading costs seriously and
neither is a long-only-vs-sort comparison.** The nearest theory item, *When Do Cross-Sectional Asset
Pricing Factors Span the Stochastic Discount Factor?*, **NBER w31275 (2023)**, DOI
[10.3386/w31275](https://doi.org/10.3386/w31275) (reconstructed), argues that heuristically sorted
characteristic portfolios do **not** span the SDF unless the return covariance has a special structure,
and that the conditions are *more* likely satisfied when researchers use **large numbers of
characteristics simultaneously** — i.e. the fix requires more data and more names than we have.

**Verdict: no cell.** A ~20-name long-only book has no meaningful covariance-estimation problem to
shrink, and the published gains are portfolio-optimization gains at hundreds-to-thousands of names.

### 4.8 SOTA section, one paragraph
The blunt answer the brief asked for: **the published gains in this literature are long-short portfolios
of hundreds-to-thousands of names, rebalanced monthly or faster, and they do not survive our
constraints.** Where the papers do impose our constraints, they say so themselves: Avramov et al. lose
most of the profitability when microcaps, distressed and high-vol names are excluded; Blitz et al. put
one-month-horizon ML net alpha at "close to zero" post-2004; Lopez-Lira & Tang's LLM drift lives in
small stocks and negative news; the 2026 TSFM work finds outright forecast collapse on US equity returns;
the one pre-registered conformal-sizing test failed out of sample. **The correct SOTA move for this
account is not a model — it is the low-turnover, close-executed, long-only construction that §2.3 just
gave us a cost number for.**

---

## 5. Rejected-list reversals

**None. Zero of the seven rejected items reversed.** Four were actively re-checked and either stand or
were reinforced; three returned nothing.

| rejected item | 2023–2026 check | outcome |
|---|---|---|
| **Index addition** | Greenwood & Sammon, *The Disappearing Index Effect*, **Journal of Finance 2024**, DOI [10.1111/jofi.13410](https://doi.org/10.1111/jofi.13410) (reconstructed) — the S&P 500 addition abnormal return fell from ~7.4% in the 1990s to **less than 1%** in the past decade; deletions to ~0.1% over 2010–2020. | **STANDS**, now with a **published** cite. **Correction to `LIT_REVIEW.md`: it is JF 2024, not JF 2025**, and the published magnitudes supersede the NBER working-paper figures quoted there. |
| **Pre-FOMC drift** | Only a 2024 University of Toronto thesis (drift realized on a minority of FOMC days preceded by macro releases; neither drift nor premium on the other ~two-thirds) and an SSRN item on long-term Treasuries. Neither is a citable peer-reviewed reversal. | **STANDS. NOT VERIFIED** either way; Kurov et al. 2021 remains the newest peer-reviewed word. |
| **TSMOM / ETF trend** | Searched 2021→2026 for any rebuttal of Huang, Li, Wang & Zhou (2020 JFE). **None found.** | **STANDS.** |
| **Betting-against-beta long leg** | *Betting against beta with intraday and overnight signals*, **International Review of Financial Analysis 2023**, DOI [10.1016/j.irfa.2023.102542](https://doi.org/10.1016/j.irfa.2023.102542) (reconstructed): all variants show positive cumulative returns, but *abnormal returns are mainly due to nano- and micro-cap stocks, which tend to outperform large-caps*. | **STANDS — and is now reinforced by post-2022 evidence**, independently of Novy-Marx & Velikov 2022. |
| **Failures-to-deliver** | Searched 2022→2026. No study found with a **long-side** tradable FTD signal. | **STANDS.** |
| **Analyst-revision drift** | Searched 2022→2026. No post-2022 US evidence found either way. The rejection was on **data** (no consensus vendor), and that is unchanged. | **STANDS.** |
| **Seasonality as a standalone family** | Searched 2023→2026. Nothing that upgrades Cheema, Ding & Wang (2023) beyond its existing role as a *reporting rule* (January shown separately). | **STANDS.** |

A reversal with a citation would have been worth more than a new family, as the brief says. There is
not one. That is a clean, searched-for result and it should raise, not lower, confidence in the
rejected list.

---

## 6. The single recommended `PLAN.md` amendment

**Recommendation: amend — one amendment, no change to the 26-cell budget, no new families.**

The cell count, the families and the gates all survive this review. What does **not** survive is the
**execution specification**, and it is wrong in the same direction for every executable cell, which is
exactly the kind of error that is cheap to fix now and expensive to fix after the run.

> ### AMENDMENT 2 — proposed 2026-09-18 by `research/multiday/LIT_REVIEW_2023_2026.md`, BEFORE any family was scored
>
> **(a) Execution — the substantive change. `opg` is no longer assumed cheap.**
> Goyal, Jegadeesh & Wu (JFQA 2026, DOI 10.1017/S0022109026102592) measure it directly: price impact is
> **lower in closing auctions than in the continuous market for all but Nasdaq microcaps**, and
> **"opening auctions are illiquid."** Therefore:
> - Every **executable** cell's default execution becomes **close-to-close** (`cls` both legs), not
>   MOO-entry/MOC-exit. Where the published construction requires an open (F1's next-open entry after the
>   8-K, F6's MOO sell), the cell is run **both ways** and reports both.
> - The declared cost model gains an explicit **opening-auction impact premium** as a separate, named
>   parameter — not the same allowance as the close. Its value is a **hypothesis until measured on our
>   own population** (Stage O's standing rule applies in full).
> - Each executable cell reports its **break-even cost** (already mandatory) **against the published
>   benchmark**: 9–21 bps/yr annualized for a non-microcap L/S book executed in closing auctions, and
>   our own halved estimate of ~5–12 bps/yr for the long-only analogue — flagged in the report as *our*
>   arithmetic, not theirs.
>
> **(b) Expected-magnitude framing, pre-committed now.** Every cell's result is reported next to
> Chen & Velikov's net benchmark (JFQA, DOI 10.1017/S0022109022000874): the average of 204 anomalies nets
> **4 bps/month**, the strongest **~10 bps**, combinations **~20 bps** — before price impact. A cell
> reporting materially more than that is to be treated as **suspect first and exceptional second**, and
> the report must say which checks were run before it was believed.
>
> **(c) A4 is un-deferred; run order changes, cells do not.** The "no post-2016 replication found" rider
> that put A4 last is superseded: Hartzmark & Solomon, **AER 115(9) 3171–3213 (2025)**, publish the same
> mechanism in a top-5 journal. A4 keeps **2 cells** and moves up the run order to follow A2. New order:
> **F2 → A1 → F3 → A2 → A4 → A3 → F1 → F5 → F4 → F6.**
> **Data check A4 must pass first:** the ex-dividend/record-date relationship changed with T+1 on
> **2024-05-28** (SEC press release 2023-29). A4's predicted-payment-month rule is built from prior-year
> dividend dates that straddle that boundary. ~~This is flagged as NOT VERIFIED~~ — **SUPERSEDED
> 2026-09-18: now VERIFIED from FINRA Rule 11140(b)(1) (amended SR-FINRA-2023-017, effective
> 2024-05-28). The rule and the pre-registered date construction are in AMENDMENT 3(c) below.**
>
> **(d) Standing rule — no LLM may score a document dated before that model's training cutoff.**
> Our 8-K corpus spans 2016→2026 and every frontier model's cutoff falls inside it. Look-ahead and the
> documented "distraction effect" (JFDS 2023, DOI 10.3905/jfds.2023.1.143) make any such score
> contaminated by construction. This applies to F2, A1 and anything else built on the EDGAR corpus.
>
> **(e) Evidence-age disclosure.** For **F1 (field), F5, F6, ~~A1~~ and A3 no post-2022 evidence of any
> kind could be verified.** Each of those cells must carry, in `REPORT.md`, the year of its newest
> verified evidence next to its number. A 2018-vintage prior in a literature that documents 26%/58%
> haircuts is a material fact about the cell. **AMENDED 2026-09-18: A1 is REMOVED from this list — it now
> has 2023 verified evidence (PBFJ 79, 102029) that the premium persists post-crisis. A1 carries the two
> riders in AMENDMENT 3(b) instead. F5, F6 and A3 stay, and their silence is now confirmed by a second
> independent sweep each.**
>
> **(f) Cell count unchanged at 26.** N1 (market-wide predictable price pressure, AER 2025) is
> **logged as a separate one-cell pre-registration**, dated today, to be run **after** the 26 with its own
> gate — deliberately outside the permutation denominator this program just corrected.
>
> **(g) Citation correction.** Greenwood & Sammon, *The Disappearing Index Effect*, is **Journal of
> Finance 2024**, DOI 10.1111/jofi.13410 — not "JF 2025" as `LIT_REVIEW.md` §4 has it — and the published
> magnitudes (~7.4% in the 1990s → <1% in the past decade; deletions ~0.1%) supersede the working-paper
> figures quoted there.

**What is deliberately NOT changed:** no family is added or dropped; no cell count moves; the three
mandatory columns stand; the rejected list stands in full (§5 found zero reversals); the gates,
splits and TEST-once discipline are untouched. "No change" was a legitimate outcome and it is what
Priorities 1, 3 and 4 delivered — the amendment exists almost entirely because of Priority 2.

### 6b. AMENDMENT 3 — added 2026-09-18 after the gap-closing pass, still BEFORE any family was scored

The six items §7 listed as unclosed were worked. **Five closed.** Nothing found adds or removes a family,
and **the cell count stays at 26**. Three things change, and one hypothesised tailwind is withdrawn.

> **(a) F4 — one mandatory reporting split, because the only live published form is below our floors.**
> The JEF 2025 weekly-reversal result (our exact horizon) is **conditional on high MAX and on extreme
> retail order imbalance**; the unconditional US effect is reported dead since 2000 (SEF 2023). Neither
> reports costs; neither splits the long leg. Therefore F4's single cell must report, alongside its
> number: (i) the result **split by MAX quintile** of the entry universe, (ii) the **share of its P&L
> from names that would fail a $10 price floor**, and (iii) its **break-even cost against the ~100%/week
> turnover it requires**. F4 keeps **1** cell and its second-to-last run slot. If the cell's profit is
> concentrated in the top MAX quintile, that is a **negative** result for this account, not a positive
> one, and must be reported as such.
>
> **(b) A1 — age disclosure withdrawn, two riders added.** A1 no longer reports "newest evidence 2018".
> It reports **PBFJ 79, 102029 (2023): the premium persists in post-crisis data.** The two riders, both
> from that paper and from JBFA 2025, are mandatory in `REPORT.md`: (i) the premium is framed there as
> **compensation for expected volatility**, so A1's number must be reported **next to the realized
> volatility of the position**, not as free alpha; (ii) the 2025 pre-announcement excess is a
> **lottery-demand** effect, so A1 carries the same **MAX-quintile split** as F4(a). Cells: **2**,
> unchanged. Run order: **unchanged at rank 2** — this evidence supports the existing placement.
>
> **(c) A4 — the ex-date rule is now VERIFIED and is hereby pre-registered as a two-regime construction.**
> Per FINRA Rule 11140(b)(1) as amended by SR-FINRA-2023-017 effective **2024-05-28**:
> - for a record date **on or after 2024-05-28**: **ex-date = record date** (first preceding business day
>   if the record date is not a business day);
> - for a record date **before 2024-05-28**: **ex-date = record date − 1 business day.**
>
> A4's predicted-payment-month construction must implement **both branches**, and `DATA.md` must report a
> **reconciliation on both sides of 2024-05-28** — the count of events whose ex-date shifts by one
> business day under the wrong single-formula rule — before any A4 return is looked at. The break falls
> **four months inside the TEST split**, so a single-formula implementation silently mis-dates the
> majority of TEST events. This replaces the NOT VERIFIED flag in AMENDMENT 2(c).
>
> **(d) Cost model — the Reg NMS tick/fee reform is NOT in force and may not be assumed.** Rule 612's
> $0.005 increment, Rule 610(c)'s $0.001 access-fee cap and the new round-lot tier were postponed twice
> and now carry a **November 2027** compliance date; odd-lot dissemination is postponed to **May 2028**
> (§2.2, five primary documents). **Consequence, pre-committed: the whole sample 2016-01 → 2026-09 is one
> regime — $0.01 tick, $0.003 cap, 100-share round lots. No regime break is modelled, and no
> out-of-sample spread compression may be credited to tick reform in any break-even-cost statement.**
> The §2.2 sentence "good news for the secondary cost arm" is **withdrawn**. AMENDMENT 2(a) and (b) are
> otherwise untouched: the auction arm was never exposed to this rule and §2.3 remains binding.
>
> **(e) Pre-committed magnitude expectation for F4 and A1 (§1, third summary statement).** Both families'
> live 2023–2026 evidence sits in the lottery / high-MAX / high-retail-order-imbalance corner that our
> ≥ $5 raw-close gate and $10M-ADV floor delete. **Our long-only, liquid-name variants are therefore
> expected to measure materially LESS than the published magnitudes. A cell landing near the published
> number is to be treated as a floor leak or construction error first and an exceptional result second**,
> and the report must say which checks were run before it was believed. This is the same discipline as
> AMENDMENT 2(b), applied to two specific families with a specific reason.
>
> **(f) Nothing else moves.** No family added or dropped. **26 cells.** Run order unchanged from
> AMENDMENT 2(c): **F2 → A1 → F3 → A2 → A4 → A3 → F1 → F5 → F4 → F6.** The rejected list still stands —
> the 2023–2026 re-sweep found **zero** reversals (§5, and §7.2 below for the two items re-checked here).
> The three mandatory columns, the frequency/additivity column, the gates, the splits and TEST-once are
> untouched.

---

## 7. Gap-closing pass, 2026-09-18 — what closed, and what genuinely remains

Same constraint as the first pass: **zero WebSearch budget.** Everything below was fetched through
bibliographic and government APIs, plus two Crossref-mirror sites. ~25 fetches.

### 7.1 CLOSED — five of the six items, with the provenance of each

| # | item | status | what it turned out to be | what it changes |
|---|---|---|---|---|
| 1 | **JEF 2025 weekly reversal** (our exact F4 horizon) | **CLOSED** | Chen, Cohen, Liang & Sun, JEF 82, 101608. Weekly US reversal exists **only conditional on high MAX** (≈1.66%/wk vs 0.65%/wk low-MAX) and **only in the top retail-order-imbalance periods**. Corroborated by SEF 2023 (10.1108/sef-12-2022-0599): unconditional US contrarian profits **vanished since 2000**. | **F4's prior goes DOWN.** AMENDMENT 3(a): mandatory MAX-quintile split, price-floor P&L share, and break-even cost at ~100%/wk turnover. |
| 2 | **FRL 2025 overnight** (F6) | **CLOSED — and it was a false lead** | Zirk-Sadowski & Hryckiewicz, FRL 86, 108638: a **time-of-day/day-of-week seasonality** study on small-cap NYSE stocks at 30 s–60 min intervals. **No cross-sectional overnight-vs-intraday decomposition at all.** | Nothing for F6 except the retirement of a "highest-value fetch". A **second** independent sweep for an LPS-style US re-test returned only China/Korea/ETF/Europe papers: **the F6 silence is real.** |
| 3 | **PBFJ 2023 earnings-announcement premium** (A1) | **CLOSED** | Tsafack, Becker & Han, PBFJ 79, 102029: the premium **persists in post-crisis data**; positively related to *expected* volatility (**a risk-return trade-off**), inversely to *realized* volatility. US sample; Asia-Pacific offered as future work. Plus JBFA 2025 (10.1111/jbfa.70008): ≈85 bps in the 10 days **before** announcements, from lottery demand. | **A1 leaves the "evidence is 8 years old" list.** AMENDMENT 3(b): report next to realized volatility; MAX-quintile split. Cells and rank 2 unchanged. |
| 4 | **Reg NMS 2024 compliance dates** | **CLOSED, five primary documents** | **Neither date took effect.** Rule 612 / 610(c) / round lot: Nov 2025 → Nov 2026 (FR 2025-19926) → **Nov 2027** (FR 2026-11997). Odd-lot info: May 2026 → **May 2028** (FR 2026-01013). Rule 610(d) alone ran to Feb 2026 and was not renewed (**inference** that it is in force). Also found: a **proposed** June-2026 rescission of the trade-through rule (FR 2026-12163). | **AMENDMENT 3(d): a tailwind is withdrawn.** The sample is ONE regime ($0.01 tick, $0.003 cap); no spread compression may be credited. §2.2's "good news" sentence is withdrawn. Auction arm untouched. |
| 5 | **T+1 ex-date / record-date mechanics** (A4) | **CLOSED, primary source** | **FINRA Rule 11140(b)(1)**, amended **SR-FINRA-2023-017, effective 2024-05-28**: *"the 'ex-dividend date' shall be the record date if the record date falls on a business day."* Under T+2 it was one business day earlier. | **AMENDMENT 3(c): a pre-registered two-regime ex-date construction**, plus a mandatory `DATA.md` reconciliation across 2024-05-28. The break falls **four months inside TEST**. |

**Provenance caveat, stated because it is the weak link in items 1–3.** Elsevier deposits no abstracts to
Crossref, OpenAlex, OpenAIRE or Semantic Scholar for these three DOIs (all four checked, all four
returned null), ScienceDirect and `ouci.dntb.gov.ua` both return **HTTP 403** to this tool, and SSRN is
still 403. The three abstracts were finally read from **`colab.ws`**, a Crossref-metadata mirror that
renders a summary of the publisher abstract. **That is a secondary source and its rendering is
paraphrase-risk.** Phrases inside quotation marks in §1 and §7 are as the mirror presented them in
quotation marks; everything else is that mirror's summary, not the authors' words. **Treated as
VERIFIED-VIA-MIRROR, not as a publisher-verbatim abstract.** Nothing in AMENDMENT 3 depends on a single
word of these three: 3(a) is a *reporting split*, 3(b) is a *disclosure change*, and both are
conservative — they make the cells harder to pass, not easier. **If a later session gets publisher access,
re-read all three; the direction of 3(a)/3(b) should not flip, and if it does, this note is where to
start.**

### 7.2 Item 6 — rejected-family reversals and the remaining silences

**Reversals: still zero.** The two items §5 could not check either way were re-swept.
- **Pre-FOMC drift**: one peer-reviewed 2024 item found — *The pre-FOMC announcement drift: short-lived
  or long-lasting?*, **Applied Economics 2024**, DOI
  [10.1080/00036846.2024.2322573](https://doi.org/10.1080/00036846.2024.2322573) (OpenAlex,
  reconstructed): a pre-announcement positive excess return **survives before press-conference
  announcements but is "short-lived, becoming insignificant shortly after the disclosure,"** explicitly
  contrasting with Lucca–Moench 2015. **Weaker and shorter, not reversed. The rejection STANDS**, now
  with a peer-reviewed post-2022 cite rather than a thesis.
- **Analyst-revision drift / failures-to-deliver / index addition / TSMOM / BAB / seasonality**: no new
  evidence; §5 stands unchanged.

**Silences, per family, and whether each is a real absence or an indexing gap:**

| family | second sweep run 2026-09-18 | verdict on the silence |
|---|---|---|
| **F5** 52-week high | "52-week high momentum anomaly stock returns", 2023+ — 629 hits, **none** a US return-premium re-test. What exists is anchoring psychology (JBEF 2024), insider exploitation of the anchor (Financial Review 2023, 10.1111/fire.12371), fundamental-strength moderators (RQFA 2023), and a **corporate-bond** 52-week-high paper (SSRN 2024). | **REAL ABSENCE.** The topic is actively published; nobody is re-testing the George–Hwang *return premium* in US equities. Newest direct evidence stays George–Hwang–Li 2018. |
| **F6** overnight vs intraday | "overnight intraday cross-section tug of war institutional retail", 2023+ — results are China (JEDC 2024), Korea (AFR 2025), sector ETFs (Risks 2026), EJF/JIFMIM sentiment "night and day" papers. **No US LPS re-test.** | **REAL ABSENCE** for the US cross-section; the international work is an indexing *presence*, which makes the US gap harder to explain away. |
| **A3** net share issuance | "net share issuance anomaly … United States", 2023+ — returns anomaly-*aggregate* papers (Review of Finance 2023, 10.1093/rof/rfad025), global ML anomaly work, Forest-through-the-Trees (JF 2025). **No US issuance re-test.** | **REAL ABSENCE**, consistent with the first sweep's Japan-only and risk-explanation hits. A3 stays at 2 cells on 2008-vintage evidence and must say so. |
| **A1** | — | **NO LONGER SILENT** (§7.1 item 3). |
| **the crux column** (long-leg share of an L-S spread) | "long leg short leg decomposition … long-only implementable", 2023+ — the only long-only-with-costs hit is a **mutual-fund-selection** paper (JFE 2023, 10.1016/j.jfineco.2023.103737, ≈2.4%/yr net, long-only, but fund picking, not the stock cross-section). | **REAL ABSENCE, now across two sweeps.** `LIT_REVIEW.md` §1's requirement — **we must measure the long-leg share ourselves** — is confirmed as a searched-for gap. This is the single most load-bearing unknown in the program. |

### 7.3 Genuinely still open, and why each could not be closed

1. **The publisher-verbatim text of the three Elsevier abstracts** (JEF 2025, FRL 2025, PBFJ 2023).
   Content is closed via a mirror; **wording is not**. Blocked by: no Elsevier deposit to any of the four
   open aggregators, HTTP 403 from ScienceDirect, OUCI and SSRN. **Needs institutional access or a
   general web search — not closable by any API route tried.** Risk is bounded (§7.1 caveat).
2. **Rule 610(d)'s in-force status is an inference**, not a fetched statement. It was extended only to
   Feb 2026 and omitted from the June-2026 order. Low stakes: 610(d) is a fee-disclosure rule and does
   not enter our cost model.
3. **0DTE spillover to single-stock equity liquidity** — still no verified evidence either way. The
   literature is SSRN-only and SSRN is 403. Low priority for a long-only close-executed book, but
   unchecked, not cleared.
4. **Closing-auction volume share 2024–2026** — only Goyal–Jegadeesh–Wu's academic **~10%** is verified.
   Exchange statistics need a general web search. Any 13%/15%/20% figure remains unusable.
5. **Order Competition Rule and Reg Best Execution status** — not re-attempted this pass; the Federal
   Register sweep was scoped to the S7-30-22 package. **NOT VERIFIED.** Neither bears on an auction fill.
6. **PFOF-era retail execution quality, US equities 2023–2026** — nothing verifiable; SSRN-bound.
7. **Transformer vs gradient boosting, head to head, US cross-section** — no such study found in either
   sweep. **NOT VERIFIED**, and §4.2's "no cell" verdict does not depend on it.

**Phrasing discipline, applied to this section.** None of the rows above supports "no such effect
exists". Each supports: *no post-2022 evidence for this family was detectable in OpenAlex, Crossref,
arXiv, OpenAIRE, Semantic Scholar or the Federal Register, searched on these terms, through an indexing
layer that excludes SSRN entirely and carries no Elsevier abstracts.* For F5, F6, A3 and the long-leg
column that null now rests on **two independent sweeps each**, which raises confidence in the absence
without converting it into proof.

---

## 8. Sources fetched this session

SEC (exact quotes): [press release 2023-29, T+1](https://www.sec.gov/newsroom/press-releases/2023-29) ·
[press release 2024-137, Reg NMS amendments](https://www.sec.gov/newsroom/press-releases/2024-137) ·
[rulemaking record, Release 34-101070 / File S7-30-22](https://www.sec.gov/rules-regulations/rulemaking-activity?year=2024&month=9)

Crossref-deposited or arXiv abstracts (exact):
[Goyal, Jegadeesh & Wu, JFQA 2026](https://doi.org/10.1017/s0022109026102592) ·
[Chen & Velikov, JFQA](https://doi.org/10.1017/s0022109022000874) ·
[Jegadeesh, Luo, Subrahmanyam & Titman, RFS 2025](https://doi.org/10.1093/rfs/hhaf057) ·
[Lopez-Lira & Tang, arXiv 2304.07619 (JFE 184, 104335, 2026)](https://arxiv.org/abs/2304.07619) ·
[FinVerse, arXiv 2608.03259](https://arxiv.org/abs/2608.03259) ·
[Forecast Collapse, arXiv 2608.14106](https://arxiv.org/abs/2608.14106) ·
[Conformal Kelly, arXiv 2608.01494](https://arxiv.org/abs/2608.01494)

Metadata exact, abstract reconstructed from OpenAlex (paraphrase-risk, flagged in text):
[Jensen, Kelly & Pedersen, JF 2023](https://doi.org/10.1111/jofi.13249) ·
[Hartzmark & Solomon, AER 115(9) 2025](https://doi.org/10.1257/aer.20231725) ·
[Avramov, Cheng & Metzker, Mgmt Sci 69(5) 2023](https://doi.org/10.1287/mnsc.2022.4449) ·
[Blitz, Hanauer, Hoogteijling & Howard, JFDS 5(4) 2023](https://doi.org/10.3905/jfds.2023.1.135) ·
[Greenwood & Sammon, JF 2024](https://doi.org/10.1111/jofi.13410) ·
[short interest, RAPS 2023](https://doi.org/10.1093/rapstu/raad007) ·
[BAB intraday/overnight, IRFA 2023](https://doi.org/10.1016/j.irfa.2023.102542) ·
[fragmentation, JFQA 2023](https://doi.org/10.1017/s0022109022001545) ·
[PEAD experimental, JFQA 2023](https://doi.org/10.1017/s0022109023000881) ·
[Ensemble Meta-Labeling, JFDS 2022](https://doi.org/10.3905/jfds.2022.1.114) ·
[triple-barrier crypto, Mathematics 2024](https://doi.org/10.3390/math12050780) ·
[LLM look-ahead, JFDS 2023](https://doi.org/10.3905/jfds.2023.1.143) ·
[sentiment trading with LLMs, FRL 2024](https://doi.org/10.1016/j.frl.2024.105227) ·
[RL in finance survey, Math. Finance 2023](https://doi.org/10.1111/mafi.12382) ·
[Chen & Zimmermann, Open Source Cross-Sectional Asset Pricing, CFR 2022](https://doi.org/10.1561/104.00000112) ·
[Publication Bias in Asset Pricing Research, Oxford Res. Encyc. 2023](https://doi.org/10.1093/acrefore/9780190625979.013.888) ·
[NBER w31275, SDF spanning](https://doi.org/10.3386/w31275) ·
[Polish factor implementation costs, Bank i Kredyt 2025](https://doi.org/10.5604/01.3001.0055.3041)

Existence confirmed, content NOT VERIFIED:
[Japan buyback pre-registered replication, PBFJ 2025](https://doi.org/10.1016/j.pacfin.2025.102666) ·
[composite equity issuance, IRFA 2024](https://doi.org/10.1016/j.irfa.2024.103288)

### 8b. Added by the gap-closing pass, 2026-09-18

**Primary regulatory documents, full text fetched (exact quotes):**
[FINRA Rule 11140 — ex-dividend date, amended SR-FINRA-2023-017 eff. 2024-05-28](https://www.finra.org/rules-guidance/rulebooks/finra-rules/11140) ·
[FR 2025-19926 — SEC temporary exemptive relief, Rules 600(b)(89)(i)(F), 610(c), 610(d), 612](https://www.federalregister.gov/documents/full_text/text/2025/11/17/2025-19926.txt) ·
[FR 2026-11997 — SEC temporary exemptive relief to Nov 2027, Rules 600(b)(89)(i)(F), 610(c), 612](https://www.federalregister.gov/documents/full_text/text/2026/06/15/2026-11997.txt) ·
[FR 2026-01013 — SEC temporary exemptive relief to May 2028, Rule 600(b)(69)(ii) odd-lot information](https://www.federalregister.gov/documents/full_text/text/2026/01/21/2026-01013.txt) ·
[FR 2024-21867 — the adopting release as published](https://www.federalregister.gov/documents/2024/10/08/2024-21867/regulation-nms-minimum-pricing-increments-access-fees-and-transparency-of-better-priced-orders) ·
FR 2026-12163 — **proposed** rescission of the trade-through rule, pub. 2026-06-17 (metadata only)

**Metadata exact (Crossref/OpenAlex), abstract read via the `colab.ws` Crossref mirror — secondary
source, paraphrase-risk, see §7.1:**
[Chen, Cohen, Liang & Sun, JEF 82, 101608 (2025)](https://doi.org/10.1016/j.jempfin.2025.101608) ·
[Zirk-Sadowski & Hryckiewicz, FRL 86, 108638 (2025)](https://doi.org/10.1016/j.frl.2025.108638) ·
[Tsafack, Becker & Han, PBFJ 79, 102029 (2023)](https://doi.org/10.1016/j.pacfin.2023.102029)

**Metadata exact, abstract reconstructed from OpenAlex (paraphrase-risk):**
[Detzel, Novy-Marx & Velikov, *Model Comparison with Transaction Costs*, JF 78(3) 1743–1775 (2023)](https://doi.org/10.1111/jofi.13225) ·
[*Evolution of short-term contrarian profits*, Studies in Economics and Finance 2023](https://doi.org/10.1108/sef-12-2022-0599) ·
[*Lottery Demand and Stock Returns Preceding Earnings Announcements*, JBFA 2025](https://doi.org/10.1111/jbfa.70008) ·
[*The pre-FOMC announcement drift: short-lived or long-lasting?*, Applied Economics 2024](https://doi.org/10.1080/00036846.2024.2322573) ·
[*Corporate insiders' exploitation of investors' anchoring bias at the 52-week high and low*, Financial Review 2023](https://doi.org/10.1111/fire.12371) ·
[*Machine learning and fund characteristics help to select mutual funds with positive alpha*, JFE 2023](https://doi.org/10.1016/j.jfineco.2023.103737)

**Confirmed as returning no abstract for the three Elsevier DOIs** (so the mirror route was necessary,
and so the next session does not repeat the attempt): Crossref `api.crossref.org/works/{doi}`,
OpenAlex `api.openalex.org/works/doi:{doi}`, OpenAIRE `api.openaire.eu/search/publications?doi=`,
Semantic Scholar `api.semanticscholar.org/graph/v1/paper/DOI:`. **HTTP 403 to this tool:**
`sciencedirect.com`, `ouci.dntb.gov.ua`, SSRN. **Returned only the search form, no results:**
`econpapers.repec.org/scripts/search.pf`.
