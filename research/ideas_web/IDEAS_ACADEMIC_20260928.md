# Academic return-predictability scan — 2026-09-28

Web research scan only (no code, no backtest, no independent rebuild yet — every row below still needs the
5-point check in CLAUDE.md §"No research claim ships without an independent check" before any capital or even a
PREREG). Budget: 40 WebSearch/WebFetch calls, all SSRN abstract pages (`papers.ssrn.com`) returned HTTP 403 to
automated fetch — effect sizes for SSRN-only papers are taken from search-snippet text, not the primary PDF, unless
noted "PDF verified." Two PDFs downloaded directly from author sites and text-extracted locally (`pypdf`): the
End-of-Day Reversal paper (verified) and the Imperial College closing-auction thesis (extraction failed, dropped).

**Stack assumed:** Alpaca (US equities incl. shorts, options L3, crypto, 24/5 overnight), Databento (OPRA 2013→,
XNAS/XNYS ticks incl. auction imbalances + halt status, daily EQUS), EDGAR filing cache, ~$65K capital, no futures.

**Excluded (already closed on this codebase, not re-proposed):** intraday breakout entries on small caps in any
form, opening-range breakouts, overnight holds of breakouts/52-week highs, index put-write/put-spreads, EDGAR
item-code event studies, post-earnings drift, leveraged-ETF decay pairs, news sentiment, insider-filing signals,
crypto trend, index intraday ORB. Where a lead below borders one of these (overnight premium, leveraged-ETF flow)
the paragraph states explicitly why the mechanism differs from the closed version.

---

## Ranked table (top 15, by effect-size-after-costs × robustness × feasibility)

| # | Idea | Paper / link | Effect size & horizon | Sample / years | OOS or post-pub status | Cost treatment | Data needed (have it?) | Retail feasibility |
|---|---|---|---|---|---|---|---|---|
| 1 | End-of-day cross-sectional reversal (last 30 min) | Baltussen, Da & Soebhag, "End-of-Day Reversal," SSRN 5039009 / [PDF](https://academicweb.nd.edu/~zda/EOD.pdf), Apr 2025 | Low-minus-high ROD3-sorted quintile spread on the last-30-min return = **3.78 bps/day raw**; a timing strategy nets ≈2.79 bps/day (per SSRN abstract) | NYSE/NASDAQ/AMEX common stocks, TAQ+CRSP, **Jan 1993–Dec 2019** (27 yr), PDF-verified | Robust across Fama–MacBeth, subsamples, winsorization choices (PDF-verified); **no post-2019 extension found** in the sections read (55 pp, read pp.1–12 of 55); 2nd place Quantpedia Awards 2025 is practitioner recognition, not an OOS test | **Not found** in the sections reviewed — flag before sizing; 3–4 bps/day is inside round-trip cost for many names, echoes this book's own "cost model is the null" lesson | XNAS/XNYS intraday trade ticks only (Databento) — no options, no EDGAR | **High** — pure tick signal, executable at the close on Alpaca equities, directly matches "last 30 minutes" theme |
| 2 | Closing-auction (MOC) imbalance reversal | Wu, "Closing Auction, Passive Investing, and Stock Prices," SSRN 3440239 (older working paper, **posting date NOT independently confirmed — SSRN ID numbering suggests ≈2019–2020, may predate the 2024–2026 window**); cost-side confirmed current by Goyal, Jegadeesh & Wu, "Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets," **JFQA, forthcoming 2026** | Long/short MOC-imbalance-reversal strategy: **13.2 bps/day risk-adjusted** (Wu, unconfirmed vintage); JFQA 2026 separately confirms price impact is **smallest in the closing auction** of the three venues (continuous, open, close) | Wu paper: sample unconfirmed. JFQA 2026: benchmarks execution cost for anomaly strategies across venues, current | Wu number needs reconfirmation against the primary PDF (blocked by SSRN 403) before use; JFQA 2026 is the up-to-date, peer-reviewed cost benchmark | JFQA 2026 IS the cost-treatment paper — direct evidence closing-auction execution is the cheapest of the three venues | Databento closing-auction-imbalance messages (XNAS/XNYS have these) | **High** for the cost/venue-choice finding (JFQA 2026); **medium** for the 13.2 bps reversal number until the primary source is re-read |
| 3 | Half-day option return momentum + reversal | Bali, Goyal, Moerke & Weigert, "In Search of Seasonality in Intraday and Overnight Option Returns," SSRN 5386128 (2025) | **0.22%–0.45% per half-day**, momentum within the same half-day period across days, reversal across opposite (intraday vs. overnight) periods, persists **≥20 business days** | Not confirmed beyond abstract (SSRN 403) | Not confirmed | Not confirmed | Databento OPRA options quotes at intraday/half-day resolution | **Medium-high** — needs an OPRA pull at finer granularity than daily, but the data exists in-stack |
| 4 | VIX term-structure carry via ETPs (no futures) | Wang et al., "VIX constant maturity futures trading strategy: a walk-forward machine learning study," *PLoS One* 19(4):e0302289 (2024), PDF-verified via PMC | Prediction IC mean **0.037** (4/7 models >0.02); Information Ratio **0.40 (long-short)–0.62 (C-MVO)**, linear-regression C-MVO IR **2.29** / 15.0% annualized | Train Dec2005–Jun2010, val Jul–Dec2010, **walk-forward OOS Jan2011–Aug2022** (11 yr), PDF-verified | Genuine walk-forward expanding-window OOS, PDF-verified | **Not addressed** in the backtest methodology — explicit gap, PDF-verified | VIX futures term structure (CBOE) — **we have no futures access**; must substitute VIX ETPs (VXX/SVXY/UVXY) as Alpaca equities, which introduces tracking-error/decay vs. the paper's pure futures-roll construction | **Medium** — implementable only via the ETP proxy, not the instrument tested |
| 5 | Opening-auction retail-flow reversal | Brown, "The Quote Not Taken: Inefficient Price Discovery in Opening Auctions," SSRN 5498938 (2025) | Abstract claims a long/short strategy on public retail order-flow data generates "significant abnormal returns" at the open — **exact bps not confirmed** (SSRN 403 blocked the PDF) | Not confirmed | Not confirmed | Not confirmed | Databento XNAS/XNYS ticks (sub-penny/odd-lot classification for retail flow) | **Medium** — mechanism and data fit are good, but this row needs a primary read before it earns a PREREG |
| 6 | Retail order-flow predictability — CAUTION, partially closed in the literature itself | (a) Sun & Moneta, "Retail Trading, Liquidity, and the Decline in Stock Return Predictability," SSRN 7396498 (2026); (b) Barber, Lin & Odean, "Resolving a Paradox: Retail Trades Positively Predict Returns but Are Not Profitable," *JFQA* 59(6) 2547–2581 (2024), PDF-partially-verified via Cambridge Core | (a) Retail order-flow predictability was real 2014–2016, **decayed substantially 2017–2022** as liquidity improved; (b) long-short on retail order imbalance: **−14.8% annualized** among heavily-retail-traded names vs. **+6.6%** among others — sign flips by stock population (composition bias: retail buys attention-grabbing names that then underperform) | (a) 2014–2022; (b) Robinhood-era sample | Both are themselves the "OOS / decay" check on the broader retail-order-flow theme | (b): failure is NOT purely transaction costs — it's a stock-selection/composition effect, per PDF-verified abstract | XNAS/XNYS ticks + retail classification (Boehmer-Jones-Zhang-Zhang / sub-penny method, per Barber, Huang, Jorion, Odean & Schwarz, "A (Sub)penny for Your Thoughts," *J. Finance* 2024 — the data-construction reference underlying rows 1, 5, 6) | **Do not build a naive long-retail-imbalance strategy** — the literature's own newest results argue against it. The actionable residual is the SHORT side of high-attention/heavy-retail names, which overlaps row 1's mechanism |
| 7 | Cross-asset order-flow imbalance (OFI) microstructure alpha | Kethan S E, "Predictive Order Flow Imbalance: Cross-Asset Microstructure Alpha," SSRN 7053198 (2026) | Mean Information Coefficient **+0.0044** at short horizons, statistically significant | 12 US equity/ETF/futures instruments, **Apr 2024–Apr 2025** | Not confirmed beyond abstract | Not confirmed | Requires L2/L3 order-book data (Databento MBO/MBP) | **Medium** — small effect size in IC terms (not yet translated to bps), some of the 12 instruments are futures we can't trade; equity/ETF subset usable |
| 8 | Short-squeeze probability from short-interest × attention | Allen, Haas, Pirovano & Tengulov, "How Prevalent Are Short Squeezes? Evidence from the US and Europe," SSRN 4526147 / *J. Banking & Finance* (2025) | Quarterly incidence of squeeze events: **9.9% of US stocks, 12.3% of EU stocks**; a **+1pp** increase in short interest during periods of heightened attention → **≈+5pp** increase in squeeze probability | US and Europe, dates not confirmed from abstract | Not confirmed | Not confirmed | **Return magnitude of the squeeze itself is not given in the abstract — only probability lift.** Do not size a trade off this row without the primary tables | FINRA short-interest files + an attention proxy (search/social) — EDGAR doesn't carry this; would need a separate attention data source | **Medium** — signal construction is feasible, but the paper as scanned gives a probability, not a return |
| 9 | Overnight (Blue Ocean/24-hour) session cost structure — infrastructure, not an edge | Lim, "Overnight Adverse Selection: Evidence from Blue Ocean ATS and NASDAQ Regular Trading Hours," SSRN 6610883 (2026) | Effective spreads overnight average **7¢/share higher** than RTH; 10-sec price-impact component **2.4¢/share higher**; only ~1/3 of the premium is adverse selection, the rest is thin single-venue liquidity; retail-popular names have **60% smaller** overnight premia; ETFs have **larger** price-impact premia | 30 actively-traded symbols, 24 matched sessions, **Sep 2025–Mar 2026** | Current, single study | This IS a cost paper — no separate cost layer needed | Blue Ocean ATS overnight ticks (does Databento carry BOATS? confirm before relying on this) | **This is not a trading signal** — it's a warning that overnight execution on Alpaca's 24/5 session is expensive outside retail-popular, liquid names. Relevant to sizing/venue choice for any other overnight idea, not investable on its own |
| 10 | Market-level (not breakout-conditioned) overnight risk premium | "Day and night expected returns under overnight information shocks," *ScienceDirect* (2025); "Intraday and overnight return anomalies: evidence from 11.6M price observations," *ScienceDirect* (2025) | Not confirmed — paywalled, abstracts only | 2025 journal-dated | Not confirmed | Not confirmed | Daily/intraday bars (have) | **Distinct from the closed "overnight holds of breakouts" topic**: this is an unconditional close-to-open premium (e.g., always-long SPY/QQQ overnight), not a signal conditioned on an intraday breakout. Flagged low-confidence pending primary read — do not conflate with the closed line of research when re-reading this row later |
| 11 | Crypto–equity lead-lag (BTC → MSTR and other BTC-treasury equities) | Aufiero, Briola, Salarin, Bartolucci, Caccioli & Aste, "Cryptocurrencies in the Balance Sheet: Insights from (Micro)Strategy," arXiv:2505.14655 (2025), PDF-verified | Transfer entropy BTC→MSTR **0.0241 bits** (significant in 31.6% of rolling windows) vs. MSTR→BTC **0.0191 bits** (13.8%); same-day correlation 0.660; MSTR beta to BTC 1.37; across 39 BTC-treasury firms, avg beta 0.62 | Apr 2023–Apr 2025 primary window, extended history from Aug 2020, PDF-verified | Observational only | **No trading strategy is backtested and no costs are discussed** — PDF-verified. This is a mechanism paper, not a strategy paper | Alpaca crypto (BTC) + equities (MSTR etc.) | **Low today, medium with work** — the mechanism (BTC leads, MSTR sometimes leads on firm-specific events) is real and the instruments are tradeable, but translating transfer-entropy bits into an actual entry/exit rule and measuring costs is undone work |
| 12 | Off-exchange/dark-pool short-volume signal | Practitioner-grade but **not peer-reviewed**: Equibles short-volume predictability study (FINRA short-sale files, 2020–Jul 2026, 660,246 windows / 6,959 stocks / 156 settlement dates); academic base is dated — Boulton & Braga-Alves, "Short Selling and Dark Pool Volume," SSRN 3882880 (2021, **pre-window**) | Not confirmed to academic standard | Equibles: 2020–2026, out-of-window academic base is 2021 | No 2024–2026 peer-reviewed paper found despite a dedicated search | Not confirmed | FINRA short-sale volume files + Databento | **High data feasibility, low evidentiary confidence** — rank low until a real academic 2024–2026 paper is found or the practitioner methodology is independently rebuilt |
| 13 | Prediction-market-implied probabilities vs. LLM forecast calibration | "Do Large Language Models Know What They Don't Know? Evaluating Epistemic Calibration via Prediction Markets," arXiv:2512.16030 (2025); KalshiBench; HINDCAST, arXiv:2607.14051 | Claude Opus 4.5 Brier score **0.227** vs. superforecaster ECE **0.03–0.05** — LLMs are measurably worse-calibrated than top humans, especially in mid-confidence regions | Live/contemporary prediction-market questions through late 2025 | These ARE the OOS/calibration tests (the whole point of the benchmarks) | N/A (forecasting accuracy, not a cost-bearing trade) | **Not in our stack** — Polymarket/Kalshi are not Alpaca products | **Low feasibility** — would require opening a separate prediction-market account outside the stated stack; included because explicitly requested, ranked last of the "real" findings |
| 14 | VIX 9-day vs. spot/3-month term-structure measure | Lim, "The Front End of the VIX Term Structure and Forward Realised Volatility," SSRN 6752518 (2026) | 9-day-inclusive term-structure measures "dominate" the conventional spot-vs-3-month measure across horizons (forecasting forward realized vol, not returns) — no bps/Sharpe given in the snippet | Not confirmed | Not confirmed | Not confirmed | CBOE VIX9D + VIX + VIX3M | **Low** — this is a vol-forecasting paper, not a P&L strategy; would need to be paired with an options overlay to monetize |
| 15 | Trading-halt / LULD reopening-auction drift | — no qualifying paper found — | — | — | — | — | Databento halt-status + XNAS/XNYS ticks would support this if a paper existed | **Gap, not a finding.** Two targeted searches ("trading halt LULD reopening auction price reaction predictability 2024 2025", "trading halt reopening auction return drift academic study LULD 2024 2025") returned only practitioner blogs and SEC rule-change filings, no 2024–2026 academic study. Absence of hits is a claim about this search, not about the phenomenon — a deeper pass (SSRN direct search UI, Google Scholar, market-microstructure journals specifically) is needed before concluding there's nothing here |

---

## Paragraphs

**1. End-of-day reversal.** Baltussen, Da & Soebhag show that a stock's own early-day return (their "ROD3" predictor:
overnight + first half-hour + midday) cross-sectionally *negatively* predicts its return in the last 30 minutes —
opposite in sign and mechanism from the well-known market-level intraday-momentum effect (Gao-Han-Li-Zhou 2018,
Baltussen-Da-Lammers-Martens 2021). PDF-verified: sample is NYSE/NASDAQ/AMEX 1993–2019 via TAQ+CRSP, robust to
Fama-MacBeth, winsorization, and subsample choices. Two proposed mechanisms — attention-induced retail buying
pushing up intraday losers into the close, and short-sellers de-risking before the close — both fit data we can
observe directly (retail sub-penny prints, short-volume flags). The open question is net-of-cost viability: 3–4
bps/day on a 30-minute holding period is thin, and the paper's own text (in the pages read) does not net out
transaction costs, so this needs our own measured-NBBO-cost test before it's a candidate for anything beyond a
PREREG.

**2. Closing-auction imbalance reversal + the JFQA 2026 cost benchmark.** Two related but distinct claims: an
older Yanbin Wu working paper reports a large (13.2 bps/day) reversal strategy off market-on-close imbalances,
but its vintage could not be confirmed (SSRN blocked the fetch, and the ID numbering suggests it may predate 2024).
Separately, a **confirmed 2026 JFQA** paper by Goyal, Jegadeesh & Wu benchmarks price impact across continuous
trading, opening auctions, and closing auctions for anomaly-strategy execution, and finds the closing auction is
the *cheapest* of the three. That second result is current, peer-reviewed, and directly useful regardless of the
first number's age: it says that whatever mean-reversion signal we trade at the close, executing it through the
closing auction (which Databento's XNAS/XNYS feed carries as imbalance messages) is the least expensive way to
do it.

**3. Half-day option return seasonality.** Bali, Goyal, Moerke & Weigert find that option returns over a given
half-day interval (e.g., "first half hour," "last half hour") momentum with the same interval on subsequent days
but reverse against the opposite session (intraday vs. overnight), with magnitudes of 0.22%–0.45% per half-day
persisting up to 20 sessions. This is a genuinely options-native signal — a good fit for Databento's OPRA feed —
but the exact sample, statistical significance, and cost treatment could not be confirmed beyond the abstract in
this pass (SSRN 403).

**4. VIX term-structure carry, ETP-only.** The clearest quantitatively-backed carry result found (PLoS One 2024,
genuine 11-year walk-forward OOS, PDF-verified) is built on VIX constant-maturity *futures*, which are outside our
stack. The substitute — VXX/SVXY/UVXY as Alpaca equities — introduces daily-rebalancing tracking error that is a
different (and separately studied) phenomenon; this is not the same mechanism as the leveraged-single-stock-ETF
decay pairs this book already closed (cd5b779), because the tradeable instrument here is a volatility-index ETP,
not a 2x/3x single-name wrapper, and the "decay" being harvested is roll yield on the VIX curve, not daily
compounding drag. Still, this needs its own cost model before it's more than a research idea — the PLoS One paper
explicitly does not address transaction costs.

**5–8. Order-flow and attention cluster.** These four rows (opening-auction retail reversal, the retail
predictability decay/composition-bias pair, cross-asset OFI, and short-squeeze probability) all sit on the same
underlying data (TAQ-derived retail/odd-lot classification, order-book imbalance, short interest) and the same
underlying caution: the *literature itself* now argues that naive retail-order-flow-following strategies do not
survive contact with 2017–2026 market structure (Sun & Moneta) or with realistic position sizing (Barber-Lin-Odean's
composition-bias result, PDF-verified via Cambridge Core: the retail-imbalance long-short spread is +6.6%
annualized on non-heavily-retail names but **−14.8%** on the heavily-retail-traded names where the signal is
strongest — i.e., the easiest-to-observe version of this trade loses money). Any work in this cluster should start
from the short side of high-attention/heavy-retail names (which also mechanically overlaps Row 1's short-sellers'
end-of-day channel) rather than a long-only retail-imbalance-follow strategy.

**9–10. Overnight session.** The Blue Ocean ATS paper is a cost/microstructure study, not an alpha signal — it
quantifies that overnight execution outside liquid, retail-popular names is expensive (7¢/share wider effective
spread than RTH), which matters for sizing any other overnight idea on Alpaca's 24/5 session but is not investable
by itself. Two 2025 ScienceDirect papers on the general (unconditional) overnight risk premium look promising by
title but could not be read past the abstract in this pass; they are flagged as **distinct from the closed
"overnight holds of breakouts" research line** — this would be an always-on close-to-open position, not a
breakout-conditioned one — precisely so a future reader doesn't mistake it for previously-closed work.

**11. Crypto–equity lead-lag.** The transfer-entropy evidence that BTC leads MSTR (and 39 other BTC-treasury
equities) more than the reverse is real and PDF-verified, but the paper is explicitly observational — no backtest,
no entry/exit rule, no costs. This is the least "shovel-ready" idea on the list: the mechanism is credible and the
instruments (Alpaca crypto + equities) are directly tradeable, but someone has to do the work of turning
"0.0241 bits of transfer entropy" into an actual signal.

**12–15. Weak or gap rows.** Dark-pool/TRF short-volume has a strong practitioner writeup but no peer-reviewed
2024–2026 paper found; LLM/prediction-market calibration is well-studied but Kalshi/Polymarket are outside our
broker stack; the VIX9D term-structure paper forecasts volatility, not returns; and trading-halt/LULD reopening
drift returned no qualifying academic paper in two searches — logged as a gap for a deeper follow-up pass, not as
a closed question.

---

## Search log (themes covered, for audit)
Closing/opening auction imbalances; retail order flow (sub-penny/odd-lot); 0DTE options and intraday vol premium;
scheduled-macro-event vol crush (FOMC/CPI — found only practitioner content, no qualifying academic paper);
VIX term-structure carry; leveraged/index ETF flow and rebalancing pressure; overnight/Blue Ocean 24-hour session;
halt and LULD auctions (no qualifying paper); short-squeeze/days-to-cover; crypto–equity lead-lag; intraday
seasonality (lunch, last 30 min); dark-pool/TRF share (no fresh qualifying paper); prediction markets + LLM
calibration; attention-induced reversal. 22 WebSearch + 15 WebFetch calls used; 6 SSRN WebFetches were blocked by
HTTP 403 (search-snippet text used instead, flagged per-row above); 2 PDFs pulled directly from author/publisher
sites and text-extracted locally, 1 succeeded (End-of-Day Reversal), 1 failed to extract cleanly (Imperial closing-
auction thesis, dropped from the table rather than reported with a guessed number).
