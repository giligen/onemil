# Topic E — Intraday return predictability from ML and microstructure signals (2022–2026 evidence)

Scope: what a small shop with **1-minute SIP bars, NBBO quotes (fetchable historically, not stored in bulk), Databento MBP-1 per window at cost, Alpaca SIP websocket, no colocation, ~1–3 s decision latency, one position at a time** could actually compute and capture. Every reported number below is tagged **[verified]** (read from the paper text or its abstract during this review) or **[unverified/from memory]**. Where a paper reports only gross numbers I say so.

Date of review: 2026-09-16. Author: research agent (no code run, no repo changes beyond this file).

---

## 0. Executive read-out (what the literature actually says)

1. **Short-horizon predictability is real, ubiquitous, and dies inside the spread.** With only TAQ level-1 quotes and trades, Aït-Sahalia–Fan–Xue–Zhou get a median **10.5% out-of-sample R² for 5-second returns** and 64% direction accuracy on S&P 100 stocks, but predictability "vanishes to 0 (a binary directional prediction becomes no better than a coin toss) in just five minutes, and approximately 2,000 transactions", and **~80% of it needs the most recent 10 ms** [verified]. Cont–Cucuringu–Zhang's 1-minute-ahead OFI forecasts on the 100 largest S&P 500 stocks have out-of-sample R² ≈ **−0.10%** (cross-impact) and −0.37% (own OFI) — i.e., zero — and the forecast-implied P&L is **0.2–0.4% per year before costs** [verified]. Nothing in this literature supports a 1-minute-holding directional strategy at 1–3 s latency paying a retail spread.
2. **Deep-LOB models forecast the next few book events, not tradable returns.** Lucchese et al.: predictability exists up to ~50 order-book events ahead, gone by 500–1000 events, costs not modeled [verified]. Prata et al. (LOBCAST): F1 drops from 82.6 (FI-2010) to 59–61 on 2021–22 LOBSTER data, "profitability is far from guaranteed" [verified]. Briola et al.: MCC 0.29 for large-tick vs 0.11 for small-tick stocks at 10 events; small-tick at 100 events = random; "high forecasting power does not necessarily correspond to actionable trading signals" [verified]. TLOB: F1 collapses when the trend threshold is set to the average spread, and Intel predictability fell 6.7 F1 points between 2012 and 2015 [verified]. None of this needs to be replicated by us; it is the negative result.
3. **The only 2022–2026 results with net-of-cost numbers on an instrument we can trade are at the 15–30 minute horizon on index ETFs/futures:** (a) Aleti–Bollerslev–Siggaard (Management Science 2025): 15-minute SPY timing from the lagged factor zoo, OOS R² 0.21%, **net-of-half-spread annualized intraday Sharpe 1.37, FF6 alpha 20.8%/yr, 2004–2020** [verified]; (b) Baltussen–Da–Lammers–Martens (JFE 2021) last-30-minute market intraday momentum: pooled equity-futures β = 0.0418, OOS R² 2.88%, and on negative-net-gamma days for the S&P 500 β = 0.0663 (t = 4.78, R² 3.58%) vs insignificant on positive-gamma days; costs not modeled but a positive net Sharpe survives a 1-tick cost in ES [verified]. Both are **market-level, once-or-a-few-times-a-day, liquid-ETF trades** — the opposite of our 1-minute small-cap microstructure.
4. **Gamma / 0DTE effects on SPY are second-order and mostly about volatility, not direction.** Amaya–Garcia-Ares–Pearson–Vasquez (Cboe proprietary OMM positions, 1-minute gamma, Jul 2020–Jun 2023): maximum gamma-induced increase in annualized daily vol **3.3 pp**, in 30-minute vol **6.4 pp** — "not large" relative to normal daily changes (s.d. 4.5 pp) [verified]. Dim–Eraker–Vilkov: 0DTE open-interest gamma does **not** propagate past volatility; 0DTE VRP is ~0.01%/day, "hardly profitable" after hedging costs [verified]. The one directional hook is Baltussen's NGE × rest-of-day-return interaction (and Park–Zhao's break-even-range version for single stocks).
5. **Retail-flow signals do not give an intraday rule.** BJZZ's 10 bps/week retail-imbalance effect [verified] is weekly, was measured 2010–2015, mis-signs 28% of trades (Barber et al. 2024) [verified], **no longer predicts or profits in 2016–2021** (Ardia–Aymard–Cenesizoglu) [verified], and in attention/retail-heavy stocks (our universe) the retail-imbalance long-short returns **−14.8%/yr** (Barber–Odean et al., JFQA) [verified]. The only usable direction is as a *veto* on names retail is piling into.
6. **VPIN is dead as a signal; bulk-volume classification is worse than a tick rule.** Andersen–Bondarenko (RoF 2015): VPIN has no incremental power for future volatility once trading intensity is controlled, and its apparent power is a BVC classification artefact [verified]; Jurkatis (JFM 2022): BVC "does not offer improvements", a quote-based algorithm halves Lee–Ready misclassification at second timestamps [verified]. Trade-sign imbalance from 1-minute bars (close-vs-VWAP, or up-vol/down-vol) is a BVC-class proxy and inherits the problem.

**Bottom line for OneMil:** the capturable content of this literature is (i) a last-30-minute SPY/ES momentum overlay gated on dealer gamma, (ii) L1 quote-imbalance as an *entry-timing filter* (not a signal) for the HOD-break — with a strong prior that at 1–3 s latency in $5–50 names it adds ≤ a few percentage points of win rate, and (iii) a retail-flow *veto*. Everything sub-5-minute is spread-bounded for us by construction.

---

## 1. Paper-by-paper review

Format: citation · instrument/universe · data · span · exact signal & horizon · model · costs · result (net vs gross) · OOS evidence · critiques.

### A. Order-flow imbalance from level-1 quotes

**E1. Cont, Kukanov, Stoikov (2014), "The price impact of order book events", J. Financial Econometrics 12(1). arXiv:1011.6402.**
- Universe/data: 50 US stocks, NYSE TAQ (best bid/ask + sizes) [verified]; April 2010, 10-second buckets [unverified/from memory].
- Signal: OFI = Σ over quote updates of (Δ in bid-side demand − Δ in ask-side supply) at the best level, i.e. `e_n = 1{P_b ≥ P_b,prev}·q_b − 1{P_b ≤ P_b,prev}·q_b,prev − 1{P_a ≤ P_a,prev}·q_a + 1{P_a ≥ P_a,prev}·q_a,prev`. Horizon: **contemporaneous** price change in the same bucket.
- Model: linear regression, slope inversely proportional to market depth.
- Result: linear relation, R² of contemporaneous price change on OFI on the order of 65% on average [unverified/from memory]. **No forward-looking return prediction is claimed** [verified from abstract]. Costs: n/a.
- Critique: this is the impact model everything else builds on; it says nothing about predicting the next minute. It is computable from NBBO quote updates only — which we can stream but do not store in bulk.

**E2. Cont, Cucuringu, Zhang (2023), "Cross-impact of order flow imbalance in equity markets", Quantitative Finance 23(10):1373–1393. arXiv:2112.13213; SSRN 3993561.**
- Universe/data: 100 largest S&P 500 stocks by market cap, Nasdaq ITCH via LOBSTER (10 levels), **2017-01-01 → 2019-12-31**; 1-minute buckets for forecasting, 30-minute windows for contemporaneous fits [verified].
- Signal: best-level OFI and an "integrated OFI" = first principal component of 10-level OFIs (explains >89% of variance). Horizon: contemporaneous, and **1-minute-ahead** (plus multi-minute) returns from lagged own- and cross-asset OFI [verified].
- Model: LASSO with/without cross-asset terms.
- Result: contemporaneous in-sample R² 71% (best-level) → 87% (integrated); OOS 83.8%. **Forward 1-minute OOS R² ≈ −0.10% (cross-impact) vs −0.37% (own OFI only)** — cross-asset terms help only "at short horizons and decay rapidly"; "performance deteriorates substantially beyond 5–10 minutes". Forecast-implied annualized P&L **0.39–0.43% (integrated, cross) vs 0.21–0.23% (own), ignoring transaction costs** — **gross only**, and economically nil [verified].
- OOS: rolling; yes.
- Critique: the definitive "OFI explains contemporaneous moves, forecasts nothing tradable at 1 minute" result on large caps. Stronger cross-impact for high tick-to-price stocks (large tick) — our $5–50 names are small-tick, i.e., the worse case.

**E3. Kolm, Turiel, Westray (2023), "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book", Mathematical Finance 33(4). doi:10.1111/mafi.12413; SSRN 3900141.**
- Universe/data: 115 Nasdaq stocks, LOBSTER full-depth book (2019) [verified: 115 stocks, granular order book; year from memory].
- Signal: multi-level OFI vectors as stationary inputs; targets = mid-price returns at multiple horizons measured in **price-change events**, not clock time.
- Model: off-the-shelf ANNs (MLP/LSTM/CNN); OF inputs beat raw-book inputs.
- Result: state-of-the-art OOS R² at the shortest horizons, decaying so that "the effective horizon of stock-specific forecasts is approximately **two average price changes**"; "information-rich" stocks are more predictable (cross-sectional regression on microstructure characteristics) [verified from abstract]. Per-horizon R² values: [unverified — not read].
- Costs: **none** (gross, mid-price).
- Critique: horizon of two mid-price changes is seconds for liquid names; needs L2 data and sub-second execution. Not our regime.

**E4. Aït-Sahalia, Fan, Xue, Zhou (2022 NBER WP 30366; Management Science 2025 as Aït-Sahalia, Fan, Xue, Zhu), "How and when are high-frequency stock returns predictable?"** https://www.nber.org/papers/w30366
- Universe/data: **101 S&P 100 constituents, NYSE TAQ trades + level-1 NBBO quotes only (no depth), Jan 2019–Dec 2020** [verified].
- Signal: 13 predictors × 9 look-back spans × 3 clocks (calendar 0.1 s–25.6 s, trade, volume): breadth, immediacy, volume stats, Lambda (Kyle-type), **LobImbalance (L1 size imbalance), TxnImbalance (Lee–Ready signed volume), PastReturn**, turnover, autocov, quoted & effective spread [verified]. Targets: average transaction return over the next Δ (5 s baseline), direction, durations.
- Model: LASSO and random forests (horse race incl. ridge, GBT — "the specific method makes little difference").
- Result [verified]: median stock **OOS R² 10.5% for 5-second returns, 64% direction accuracy**; important predictors = L1 imbalance, transaction imbalance, past returns. **Predictability → 0 in ~5 minutes / ~2,000 trades.** ~80% of it comes from the most recent 10 ms / 10 transactions; an artificial processing lag "decreases sharply the accuracy". More predictable: low nominal price, less liquid, less volatile, low market correlation. Simulated look-ahead at next-trade sign lifts 5-s R² 14.0% → 27.1%.
- Costs: **none — gross statistical predictability**; the paper explicitly is "not about" profitability.
- Critique: the cleanest quantification of the latency tax. For a 1–3 s decision loop, the paper implies we forfeit most of the 5-second R² and all of it by 5 minutes. Its "less liquid, lower-priced stocks are more predictable" finding is the one crumb for our universe — but those are exactly the names whose spread exceeds the predicted move.

**E5. Takahashi (2025), "Returns and order flow imbalances: intraday dynamics and macroeconomic news effects", arXiv:2508.06788.**
- Instrument/data: S&P 500 E-mini futures, 1-second data, structural VAR identified via heteroskedasticity, estimated per 15-minute interval [verified].
- Result: price impact and flow impact both significant at the 1-second horizon; **"shocks dissipate almost entirely within a second"**; price impact rises and flow impact falls around macro news [verified]. No costs, no trading.
- Critique: reinforces E4 — the OFI→return channel on ES is a sub-second phenomenon.

**E6. Stoikov (2018), "The micro-price: a high-frequency estimator of future prices", Quantitative Finance 18(12). SSRN 2970694.** [background; pre-2022]
- Signal: L1 imbalance I = q_b/(q_b+q_a) and spread → Markov-chain adjustment of the mid; the micro-price is a better predictor of the next mid move than the mid or the size-weighted mid [verified from abstract summary]. Follow-ups (Sfendourakis & Muni Toke 2023; Blakely 2024 arXiv:2411.13594) confirm the monotone imbalance→next-move relation [verified from search summaries].
- Horizon: next mid-price change (ticks to seconds). Costs: n/a.
- Critique: computable from a single NBBO snapshot, which we *can* fetch at decision time. Most useful for large-tick names (spread = 1 tick); for our small-tick, wide-spread names the imbalance is noisy (Briola, E10).

**E7. Rahman & Upadhye (2024), "Hybrid VAR + NN for OFI prediction in HFT", arXiv:2411.08382** — Binance data, forecasts OFI itself (not returns); no return/P&L evidence [verified]. Listed for completeness; not relevant.

**E8. Cucuringu et al. / "ClusterLOB" (2025), arXiv:2504.20349** — one year of Nasdaq **market-by-order** data; K-means on order features → directional / opportunistic / market-making clusters; per-cluster OFI in 30-minute buckets improves Sharpe vs un-clustered OFI; transaction costs not addressed [verified]. Needs MBO — not computable from L1.

### B. Deep-learning LOB forecasting with honest cost accounting

**E9. Lucchese, Pakkanen, Veraart (2024), "The short-term predictability of returns in order book markets: a deep learning perspective", Int. J. Forecasting 40(4):1587–1621. arXiv:2211.13777.**
- Universe/data: 10 Nasdaq stocks chosen for liquidity spread (LILAK, QRTEA, XRAY, CHTR, PCAR, EXC, AAL, WBA, ATVI, AAPL), LOBSTER 10-level, **2019-01-02 → 2020-01-31** [verified].
- Signal/horizon: 3-class mid-price move at h ∈ {10,…,1000} **order-book events**; DeepLOB / deepOF / new "volume representation" (deepVOL), Siamese + multi-head attention.
- Result: predictability "systematically present" up to ~50 events at 99% confidence for most stocks; AAPL at 10–20 events, LILAK fades by 30; gone by 500–1000 events. Quoted spreads 0.99 bp (AAPL) to 15.9 bp (LILAK) [verified].
- Costs: **not discussed** — statistical only.
- Critique: 50 events ≈ seconds for liquid names; a 1–3 s decision loop is already at the edge of the predictable window.

**E10. Briola, Bartolucci, Aste (2025), "Deep limit order book forecasting: a microstructural guide", Quantitative Finance 25(7):1101–1131. arXiv:2403.09267; LOBFrame code.**
- Universe/data: 15 Nasdaq stocks 2017–2019, grouped by spread/tick: small-tick (spread ≥ 3 ticks: CHTR, GOOG, GS, IBM, MCD, NVDA), medium (AAPL, ABBV, PM), large-tick (≤1.5 ticks: BAC, CSCO, KO, ORCL, PFE, VZ) [verified].
- Model: DeepLOB; horizons H10/H50/H100 events.
- Result [verified]: MCC at H10 — small-tick 0.11, medium 0.13, **large-tick 0.29**; at H100 small-tick 0.01 ("random"), large-tick 0.26. New metric pT (probability of a correct *transaction*) shows "high forecasting power does not necessarily correspond to actionable trading signals"; small-tick failure attributed to noise, 39% up/down confusion at H50, wider spreads eating thin margins, and 100 events ≥ 10 s.
- Costs: modeled through the spread-aware pT construction, no P&L table.
- Critique: the "spread-bounded" evidence in one figure — predictability lives where the spread is one tick, which is where there is nothing to earn. Our universe ($5–50, spreads often 20–100 bps) is the small-tick corner.

**E11. Prata et al. (2024), "LOB-based deep learning models for stock price trend prediction: a benchmark study" (LOBCAST), Artificial Intelligence Review. arXiv:2308.01915.**
- 15 SOTA models; FI-2010 (5 Finnish stocks, June 2010) vs LOBSTER LOB-2021 (Jul 1–15 2021) and LOB-2022 (Feb 1–15 2022) on SOFI, NFLX, CSCO, WING, SHLS, LSTR; k ∈ {1,2,3,5,10} [verified].
- Result: best model BINCTABL F1 82.6% (FI-2010) → 61.2% / 59.2% (LOBSTER); average drop ≈19.6 pts; backtests via Backtesting.py; "the models' profitability is far from guaranteed" [verified]. Cost assumptions: [unverified — truncated].
- Critique: generalization failure across time and venue is the headline; gross-only.

**E12. Berti & Kasneci (2025), "TLOB: a novel transformer model with dual attention for price trend prediction with LOB data", arXiv:2502.15757.**
- Data: FI-2010; TSLA/INTC LOBSTER Jan 2015; BTC perpetuals Jan 2023. F1 on TSLA h=10 60.5% → h=100 39.8%; INTC 80.2% → 50.1% [verified].
- Cost experiment: redefining the "trend" threshold as the **average spread** (the real transaction cost) collapses TSLA F1 to 41.4% (h=50) / 36.5% (h=100) / 30.8% (h=200); INTC F1 66.9% (2012) → 60.2% (2015), "predictability has declined over time"; authors: models "not sufficiently mature for practical deployment" [verified].
- Critique: the most explicit "predictability but no profit after the spread" statement in the 2025 crop.

**E13. Zhang, Zohren, Roberts (2019) DeepLOB, IEEE TSP 67(11); Zhang & Zohren (2021) multi-horizon; Zhang, Lim, Zohren (2021) MBO models arXiv:2102.08811.** [background]
- The simulated trading in DeepLOB executes at mid and omits costs — "only gross predictive value" [verified from search summary]. Successors (E9–E12) are the honest re-tests. Do not build.

### C. Cross-sectional / time-series intraday predictability at 1–60 minutes

**E14. Chinco, Clark-Joseph, Ye (2019), "Sparse signals in the cross-section of returns", J. Finance 74(1):449–492. https://www.alexchinco.com/sparse-signals-in-cross-section.pdf** [pre-2022, but the reference point]
- Universe/data: 250 randomly chosen NYSE stocks per day, 1-minute returns, **Jan 2005–Dec 2012**; predictors = lagged 1–3 minute returns of all ~2,000 NYSE stocks; rolling 30-minute estimation windows; 1-minute-ahead forecasts [verified].
- Model: LASSO (glmnet) vs OLS benchmark with own/market lags.
- Result: LASSO adds **≥ 1.2 pp OOS R²** over the benchmark; forecast-implied strategy **annualized Sharpe 1.8 "net of trading costs"** (their cost model, details not read here); predictors are sparse (≈12.7 per stock), short-lived (<5% used > 15 minutes), and tied to fundamental news in *other* firms [verified].
- Critique: 2005–2012 sample; the cost model is not the retail half-spread; requires the full live cross-section every minute. Aleti et al. note Aït-Sahalia's finding that this style of predictability "essentially vanished over longer five-minute return intervals" [verified].

**E15. Aleti, Bollerslev, Siggaard (2025), "Intraday market return predictability culled from the factor zoo", Management Science 71(9):7731–7751. SSRN 4388560; https://public.econ.duke.edu/~boller/Papers/MS_2025.pdf**
- Universe/data: 272 high-frequency portfolios (218 characteristic-sorted factors, 48 industries, FF6) at **15-minute** frequency, 1996–2020; target = next 15-minute Fama–French market return; trades **SPY** (also QQQ, IVV, IWM in appendix) [verified].
- Signal: HAR-style lags (15-min, hourly, daily) of all factor returns, jump/continuous split (truncation); 816–1,632 regressors.
- Model: ridge/LASSO/ENet/PCR/PLS/FNN/RF/GBRT + ensemble; expanding train/valid, annual refit, **OOS 2004–2020 (111,280 obs)**.
- Result [verified]: OOS R² best **0.212%** (Ensemble, 15-minute) — vs a theoretical risk-premium bound of 0.0078% — so the source is frictions/slow information, not risk premia. Trading: S-Sign strategy (rebalance only when |forecast| > half-spread, else hold) **net of half the TAQ time-weighted quoted spread and 2%/yr borrow: intraday annualized return 19.9%, Sharpe 1.37, FF6 alpha 20.8% (t = 5.7)**; long-only S-Positive Sharpe 0.99; SPY intraday Sharpe over the same period 0.09. Gains concentrate in recessions/high uncertainty, around 10:45 and 14:15–14:45, and load on turnover/liquidity factors.
- Critique: net numbers are real but assume half-spread fills on SPY every 15 minutes and the full factor-zoo intraday panel (CRSP-level whole-market 15-minute returns). Infrastructure, not latency, is our barrier.

**E16. Bogousslavsky (2021), "The cross-section of intraday and overnight returns", JFE 141(1):172–194. SSRN 2869624.**
- 30 years of US intraday returns; anomalies accrue at different times of day: **size and illiquidity premia are earned in the last 30 minutes**; profitability/idiosyncratic-vol accrue during the day and reverse overnight; a mispricing factor is positive intraday and **performs poorly at the end of the day** (arbitrageurs cut positions before the close; overnight margin/lending costs) [verified from abstract].
- Costs: long-short factor returns, gross. Horizon: half-hour buckets.
- Critique: relevant to *when* to be flat: anomaly-longs are sold into the close; small/illiquid names get a last-30-minute bid.

**E17. Heston, Korajczyk, Sadka (2010), "Intraday patterns in the cross-section of stock returns", J. Finance 65(4).** [background] Returns continue in the same half-hour interval on subsequent days, lasting up to 40 days [verified via Baltussen's citation]. Gross; a periodicity effect of a few bps per half-hour that is spread-bounded for single names.

### D. Intraday time-series momentum, dealer gamma, 0DTE

**E18. Gao, Han, Li, Zhou (2018), "Market intraday momentum", JFE 129(2).** [background]
- SPY 1993–2013: first-half-hour return (and second-to-last half-hour) predicts the last-half-hour return; "persists after accounting for reasonable transaction costs"; stronger on volatile, high-volume, recession and macro-news days; holds on 10 other liquid ETFs with often larger OOS R² [verified from search summary].

**E19. Baltussen, Da, Lammers, Martens (2021), "Hedging demand and market intraday momentum", JFE 142(1):377–403. SSRN 3760365.**
- Universe/data: 17 equity-index, 16 bond, 21 commodity, 8 currency futures; Tick Data 1-minute bars, **Dec 1974–May 2020**; S&P 500 NGE from OptionMetrics 1996–2017 + SqueezeMetrics to May 2020 (assumes MMs short all puts, long all calls) [verified].
- Signal: `r_LH = α + β·r_ROD` where r_ROD = previous close → 30 min before close; r_LH = last 30 minutes.
- Result [verified]: pooled equity futures **β_ROD = 4.18 (×100), t = 7.29, OOS R² 2.88%** (expanding window, Clark–West); positive & significant OOS R² for 14/17 equity contracts. **S&P 500 futures: NGE_{t−1} < 0 → β = 6.63 (t = 4.78), R² 3.58%; NGE ≥ 0 → β = 0.82 (t = 1.03), R² 0.05%.** Interaction NGE × r_ROD = −123 (t = −3.42). r_LH reverts over the next 3 days (transitory price pressure). Sharpe of naive timing strategies 0.87–1.73 by asset class, **gross**; "we do not consider transaction costs … exploiting the effect in the S&P 500 futures yields a positive net Sharpe ratio when we assume transaction cost equal to a tick."
- OOS: yes (expanding-window OOS R²). Robust to controlling for lagged r_LH (i.e., distinct from Heston et al. seasonality).
- Critique: effect size per day is small (β ≈ 0.04–0.07 × a ~0.5–1% rest-of-day move ≈ 2–7 bps), so viability rests on very low costs and size. Dim–Eraker–Vilkov re-examine and find that for SPX "the latter is not significantly related to net gamma but is negatively and significantly related to the interaction of net gamma and lagged return" [verified] — i.e., the gate must be NGE × r_ROD, not NGE alone.

**E20. Park & Zhao (2025), "Inelastic hedging demand and intraday momentum" (working paper, Sept 6 2025), NFA portal.**
- Universe/data: US single stocks; TAQ midquotes (MTAQ, Holden–Jacobsen), ISE open/close volume by account type to estimate MM dollar gamma (MM OI = −(Firm + Customer)), OptionMetrics 30-day ATM Greeks [verified].
- Signal: Gao-style `r_{390,360} on r_{30,0}` with interactions D_ShortGamma and D_GTBR_Hit, where the gamma–theta break-even range **GTBR = ±√(−θ/(365·50Γ)) ≈ ±σ_imp/√365** (daily move at which short-gamma hedgers' gamma loss exceeds theta) [verified].
- Result [verified]: base intraday-momentum coefficient 1.05 (×100); **MM short gamma adds +68%** (0.71), **breaching the GTBR adds +91%** (0.96), both beyond each other; momentum absent inside the range even when MMs are short gamma; MMs keep delta-hedging rather than unwind at the range. Stronger on down moves and when active option traders are also short gamma. Out-of-sample R² section exists [not read].
- Costs: none (gross, midquote returns).
- Critique: for our small caps with daily moves of 10–50%, the GTBR (≈ 3–6% for 60–100% IV) is breached by construction on every candidate day; the operative question is whether MMs are short gamma in that name, which needs ISE/OPRA-type positioning data we do not have — Alpaca's options chain gives OI and IV only.

**E21. Li, Sakkas, Urquhart (2021/2022), "Intraday time series momentum: global evidence and links to market characteristics", J. Financial Markets 57.** [background] Global ETF evidence for first-half-hour → last-half-hour, linked to volatility/volume; details not fetched (repository blocked) [unverified].

**E22. Amaya, Garcia-Ares, Pearson, Vasquez (2025), "0DTE index options and market volatility: how large is their impact?" (Jan 25 2025; Cboe research publication).** https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf
- Data: **all Cboe SPX/SPXW trades with trading capacity (442.6M records) Jan 2020–Jun 2023**, cumulated to the aggregate OMM net position per series at 1-minute frequency; Black–Scholes gammas from Algoseek 1-minute BBO IVs; ES 1-minute returns; analysis Jul 2020–Jun 2023 [verified].
- Model: Engle–Sokalska/MIDAS GARCH on 1-minute returns with 5 lags of OMM gamma; linear squared-return models as robustness; counterfactual simulation with γ = 0.
- Result [verified]: OMM gamma typically positive (mean of daily means 341×10⁹; after Tue/Thu expirations were added in Apr/May 2022 the median daily minimum is negative, i.e. gamma is negative at some point on ≥ half of days); conditional variance negatively related to lagged gamma; **maximum gamma-induced increase: +3.3 pp annualized daily vol, +6.4 pp annualized 30-minute vol**, vs. s.d. of daily changes in realized vol of 4.5 pp and >5% of 30-minute changes exceeding 6.4 pp → "not large". 0DTE = 34.8% of SPX/SPXW volume; 68.7% of volume is customer-vs-market-maker.
- Costs/trading: none — a volatility-attribution study.
- Critique: the best-identified gamma dataset says the *volatility* effect is modest. Direction is not tested.

**E23. Dim, Eraker, Vilkov (2024), "0DTEs: trading, gamma risk and volatility propagation", SSRN 4692190 (May 14 2024 draft).**
- Data: SPXW and SPY options ≤ 30 DTE, Cboe DataShop 30-minute NBBO bars + OPRA transactions (2012–Jun 2023), SPX/SPY/ES 1-minute bars [verified].
- Result [verified]: 0DTE open-interest dollar-gamma **does not propagate past volatility**; for > 1 DTE, OI gamma is associated with *lower* intraday RV; intraday 0DTE volume shocks do not amplify past index returns (difference across eras = 0.15 s.d. of vol, "economically negligible"); 0DTE–underlying volume correlation rose from 0.25–0.30 (pre-2021) to 0.59 (2023); 0DTE VRP ≈ **0.01%/day**, "hardly profitable" after hedging and costs; 0/1-DTE volume drops before and rebounds after FOMC.
- Critique: directly contradicts the "0DTE gamma squeezes drive SPY" narrative for the aggregate; the residual effect is the interaction term in E19.

**E24. Brogaard, Han, Won (2023/24), "Does 0DTE options trading increase volatility?", SSRN 4426358.** 1 s.d. increase in 0DTE option trading → **+9.1% volatility relative to its mean** (S&P 500 ETF return volatility vs 0DTE share of SPX option volume), driven by speculative retail even after controlling for OMM gamma hedging [verified from search summary]. Gross/no trading. Critique: volume-share, not gamma; Amaya et al. and Dim et al. find the gamma channel small.

**E25. Adams, Fontaine, Ornthanalai (2024), "The market for 0DTE: the role of liquidity providers in volatility attenuation", SSRN 4881008.** Identification from the Apr/May 2022 introduction of Tue/Thu SPXW expiries: **volatility is lower on days when 0DTE options are available**; volatility related to sign/size of OMM gamma [verified from Amaya's summary]. No trading.

**E26. Beckmeyer, Branger, Gayda (2023), "Retail traders love 0DTE options… but should they?", SSRN 4404704.** >75% of retail SPX option trades are 0DTE; retail lost **$241K per average day** Feb 2021–Sep 2023 [verified from search summary]. Implication: the retail side of 0DTE is the liquidity-providing edge's counterparty, not a signal for us.

**E27. Vilkov (2026), "0DTE trading rules", SSRN 4641356 (repo vilkovgr/0dte-strategies).**
- SPXW 0DTE structures (calls/puts/straddles/strangles/iron flies/condors/ratio spreads/risk reversals), Cboe 30-minute NBBO bars + ThetaData 1-minute SPX/VIX, **Sep 2016–Jan 2026**; entries 10:00 ET held to 16:00 [verified].
- Result [verified]: unconditional VRP 10:00→expiry ≈ 0.0011% of spot (tiny); risk reversals the only structure with consistently positive mean/median (~0.01% of spot). **After half-spread + 0.5 bp costs** most strategies deteriorate. Conditional logistic direction classifier, OOS Apr 2019–Feb 2026: put ratio spreads gross SR 1.18 / **net 0.93**; iron fly/condor net **−0.20**; top-3 basket net SR 0.82; equal-weight all net 0.25. ES₁% 0.58–1.58% of underlying — tail capital dwarfs mean P&L.
- Critique: an options-selling sleeve, not an underlying-direction signal; retail option spreads are far wider than the half-spread assumed.

### E. Retail order flow

**E28. Boehmer, Jones, Zhang, Zhang (2021), "Tracking retail investor activity", J. Finance 76(5). SSRN 2822105.** Sub-penny TAQ prints (price ends in .x1–.x40 → retail sell, .x60–.x99 → retail buy); **stocks with net retail buying outperform by ~10 bps over the following week**, 2010–2015; less than half explained by flow persistence [verified from search summary]. Weekly horizon; gross long-short.

**E29. Barber, Huang, Jorion, Odean, Schwarz (2024), "A (sub)penny for your thoughts: tracking retail investor activity in TAQ", J. Finance 79(4):2403–2427.** 85,000 real retail trades Dec 2021–Jun 2022: BJZZ identifies 35% of them, **mis-signs 28%**, gives uninformative imbalance for 30% of stocks; the **quote-midpoint sign rule** cuts sign errors to 5% [verified].

**E30. Ardia, Aymard, Cenesizoglu (2024), "Revisiting Boehmer et al. (2021): recent period, alternative method, different conclusions", arXiv:2403.17095.** With QMP and BJZZ on **2016–2021**: past retail order imbalance **no longer predicts weekly returns on large caps; the long-short strategy is no longer profitable** [verified].

**E31. Barber, Odean et al. (2024), "Resolving a paradox: retail trades positively predict returns but are not profitable", JFQA.** Extreme-quintile retail-imbalance long-short returns **−14.8%/yr in stocks with heavy retail concentration vs +6.6% elsewhere**; attention-driven buying of visible stocks that subsequently underperform [verified].

**E32. "The information content of retail order flow: evidence from fragmented markets", J. Banking & Finance (2024), doi S0378426624001894.** Retail flow is more informed than off-exchange institutional flow, but **on high-Robinhood-activity days its information content drops** [verified from search summary]. Horizon: daily/weekly; no intraday rule.

### F. VPIN and trade-sign inference from bars

**E33. Easley, López de Prado, O'Hara (2012), "Flow toxicity and liquidity in a high-frequency world", RFS 25(5).** VPIN from volume buckets with bulk-volume classification; claimed to lead the Flash Crash [background; unverified numbers].

**E34. Andersen & Bondarenko (2015), "Assessing measures of order flow toxicity and early warning signals for market turbulence", Review of Finance 19(1):1–54.** BVC "is inferior to a standard tick rule"; **VPIN predicts volatility solely because rising volatility induces systematic BVC classification errors**; controlling for trading intensity and volatility, no incremental power; "unsuitable for capturing order flow toxicity" [verified].

**E35. Jurkatis (2022), "Inferring trade directions in fast markets", J. Financial Markets 58 (BoE WP 896).** New quote-timing-based algorithm **halves misclassification vs Lee–Ready at second-stamped data; BVC offers no improvement**; a risk-averse investor would pay 33 bps/yr for the better cost estimates [verified].

### G. Realized-volatility signatures (for sizing, not direction)

**E36. Zhang, Zhang, Cucuringu, Qian (2024), "Volatility forecasting with machine learning and intraday commonality", J. Financial Econometrics. arXiv:2202.08962.** Pooled cross-stock NN/tree/linear models for intraday RV with market-vol proxies beat HAR-type baselines OOS; a "universal volatility mechanism" transfers to stocks excluded from training; time-of-day effects when using intraday RV for 1-day-ahead forecasts [verified from abstract]. No return prediction; usable for stop width / risk sizing.

**E37. "Time series momentum and reversal: intraday information from realized semivariance", J. Empirical Finance (2023), doi S0927539823000245.** Intraday up/down semivariance improves daily TSMOM/reversal signals [not read; unverified].

---

## 2. Hypothesis table

Inputs legend: **B** = 1-min SIP bars (OHLCV, trade_count, vwap); **Q** = NBBO quote snapshot at decision time (streamable live, fetchable historically per window); **MBP** = Databento MBP-1 per window (cost); **OPT** = Alpaca options chain (OI, IV) if enabled; **X** = whole-market intraday panel (we do not have it).

| H-id | Rule (one sentence) | Prior effect (net, per trade) & mapping | Inputs | Falsification criterion (pre-committed) | Cost sensitivity | Capturable by a 1-position retail account at 1–3 s? |
|---|---|---|---|---|---|---|
| **H-E1 Gamma-gated last-30-min SPY momentum** | At 15:30 ET, if sign(r_ROD) matches and the dealer-gamma proxy is negative (NGE×r_ROD < 0), go long/short SPY (or ES) to 15:58. | E19: β≈0.066 on neg-NGE days → E[r_LH] ≈ 0.066 × |r_ROD|; for |r_ROD| = 0.7% ≈ **4.6 bps gross**, SPY round-trip cost ≈ 1–2 bps → **~3 bps net**, R² 3.6%; 0–1 trades/day. Dim et al. imply the interaction, not NGE level, is the gate. | B (SPY 1-min), OPT (SPX/SPY chain OI for a crude NGE = Σ OI×γ×S², puts short / calls long convention) | 250 trading days of shadow: net mean r_LH on gated days ≤ 0 at t < 1.5, or gated-day β not > un-gated β. Also require the effect to reverse over next 3 days (E19 mechanism). | Moderate: 3 bps net needs ≤ 1.5 bps all-in; marketable limit at 15:30 on SPY is fine; slippage at 15:58 MOC is the risk. | **Yes, mechanically** (SPY is deep, latency irrelevant). Biggest reason it may not pay: the effect is ~3 bps/day — at our $150 risk sizing it is dollars; only worth it as a full-notional sleeve, and the post-2022 0DTE regime (E22/E23) may have shrunk the negative-gamma channel. |
| **H-E2 L1 imbalance entry filter for HOD-break** | Take the HOD-break entry only if NBBO size imbalance I = q_b/(q_b+q_a) ≥ 0.6 at the decision snapshot; skip if ≤ 0.4. | E6/E4: imbalance is the strongest L1 predictor of the *next mid move* (seconds). Prior for a 1–3 s-stale snapshot in $5–50 small-tick names: +2–4 pp win rate on the first 1–2 minutes, ≈ 0 by 5 min; maps to ≈ +5–15 bps on a trade whose spread is 20–100 bps — a fill-quality effect, not alpha. | Q at decision time (already fetchable), B | 200 dry-run/live entries with the snapshot logged: conditional WR(I≥0.6) − WR(I≤0.4) < 3 pp or first-2-minute MFE difference < half the spread → drop. Also test on Databento MBP-1 windows for 300 historical candidates. | Low as a filter (no extra trades); high if used as a standalone signal (spread-bounded). | **Partially.** Biggest reason it fails: at 1–3 s latency in small-tick names the imbalance has already been traded (E4: 80% of predictability in the first 10 ms; E10: small-tick MCC 0.11 → 0.01). Expect a fill-quality edge at best. |
| **H-E3 Bar trade-sign imbalance (BVC-class) → next 1–5 min return** | Sign 1-min volume by close-vs-VWAP (or up/down volume) and buy when the last-3-bar imbalance z > 2. | E2: 1-min forward OOS R² ≈ 0 on large caps; E4: gone by 5 min; E34/E35: BVC signing is worse than a tick rule. Prior net effect **≤ 0 bps** after a 20–100 bps spread. | B | OOS R² of next-1/3/5-min return on the imbalance ≤ 0.2% on 2025–26 cache, or expected move < quoted half-spread at entry → dead. | Fatal: any 1–5 min hold pays the full spread. | **No.** Spread-bounded by construction; do not build as a signal. Reuse only as a diagnostic feature. |
| **H-E4 Factor-zoo 15-min SPY timing (Aleti)** | Every 15 min forecast the next market return from lagged 15-min returns of 200+ characteristic portfolios; trade SPY only when |forecast| > half-spread. | E15: OOS R² 0.21%, net Sharpe 1.37, ≈ 19.9%/yr intraday; ≈ 1–3 bps per rebalanced interval. | **X** (whole-market 15-min returns for ~5,000 stocks + characteristics) — not available; Databento daily parquet is daily only. | Cannot be run until a whole-market 15-min panel exists; if built: OOS R² < 0.1% or net S-Sign Sharpe < 0.5 on 2021–26 → drop. | Moderate (half-spread on SPY assumed). | **Not now** — infrastructure-gated, not latency-gated. Biggest reason: we would need to construct 218 daily-rebalanced characteristic portfolios and their 15-min returns; the paper's edge is also concentrated in recessions. |
| **H-E5 Cross-sectional 1-min LASSO (Chinco)** | Each minute, LASSO the next-minute return of a target on the last 3 minutes of returns of the whole cross-section; trade when |forecast| > half-spread. | E14: +1.2 pp OOS R², net Sharpe 1.8 in 2005–2012 with the authors' cost model; predictability at 1-min "essentially vanished" at 5-min per E4/E15 [verified]. Prior for 2025–26 on our small caps: **negative net**. | B live for ~2,000 symbols (SIP websocket can carry it; compute is feasible) | 3-month shadow on 250 random symbols: OOS ΔR² < 0.5 pp or net-of-quoted-half-spread Sharpe < 0.5 → dead. | Fatal at retail spreads; the paper's names are NYSE large/mid caps. | **No.** 1-minute holds at 1–3 s latency in wide-spread names. Biggest reason: cost model — half the quoted spread on our names is 10–50 bps vs a 1-minute expected move of a few bps. |
| **H-E6 GTBR gamma-squeeze continuation (single stocks)** | For optionable HOD-break names, if |day return| > σ_imp/√365 and dealers are short gamma (proxy: net customer long calls/short puts via OI skew), expect stronger last-hour continuation; hold to 15:55 instead of TP. | E20: momentum coefficient 1.05 (×100) base, ≈ 2.7 (×100) with short gamma + GTBR breach; on a +10% day → ≈ **+27 bps gross** last-30-min continuation, gated. | OPT (OI, IV), B | 60 gated trades: last-30-min return on gated vs un-gated names not > 0 at t < 1.5 → drop. | Low (we are already in the position; only the exit timing changes). | **Partially.** Biggest reason: we cannot observe MM positioning (ISE account-type data); OI alone does not give the sign of dealer gamma; many of our names have no listed options. |
| **H-E7 End-of-day anomaly reversal / size premium (Bogousslavsky)** | Do not hold "mispricing-long" (high-attention, high-profitability) names into the last 30 min; do hold small/illiquid names, which earn their size premium 15:30–16:00. | E16: factor-level effects of a few bps per day; for a single small cap at 10–50% daily range this is noise. Prior **≈ 0 bps** at our sizing. | B, DB trades | 6 months of HOD-break exits: 15:30–15:55 P&L contribution by name-size tercile; if no tercile differs at t > 2, ignore. | Low. | **Yes but immaterial.** Biggest reason: effect size. |
| **H-E8 Retail-flow veto (sub-penny prints)** | Veto a long entry if the intraday retail order imbalance (quote-midpoint-signed sub-penny prints, Barber et al.) is in the top decile — attention crowding predicts underperformance. | E31: −14.8%/yr for the extreme quintile in retail-heavy names (weekly horizon, gross); E30: predictability gone in large caps 2016–21. Prior for intraday: **unknown sign, weekly-horizon evidence only**; treat as a veto study, not alpha. | Alpaca historical trades (per candidate symbol-day, sub-penny prices) + Q for midpoint signing | Compute on 300 HOD-break/BF candidates: top-decile retail-imbalance names' next-2-hour return not < others at t < 2 → drop. | None (a filter). | **Yes** (offline feature). Biggest reason it fails: the documented effect is weekly and attention-driven; our names are *all* attention names, so the cross-sectional contrast may be absent. |
| **H-E9 VPIN / flow toxicity** | Size down or stand aside when VPIN is in its top 5%. | E34: no incremental power; artefact of BVC. Prior **0**. | B | Do not build. Pre-committed: dead on prior evidence. | — | **No.** |
| **H-E10 Intraday RV forecast for stop width / sizing** | Replace fixed % stops with a HAR/ML forecast of the next-hour RV from 1-min bars (pooled across names, E36). | Not a return signal. Expect stop-out rate to fall without P&L loss; HAR-class 1-step RV OOS R² is typically 0.3–0.6 [unverified/from memory]. | B | On the HOD-break cache: RV-scaled stops must not reduce mean R at a 2σ level while reducing stop-outs ≥ 15%; else keep fixed stops. | None. | **Yes.** Cheap, and the one ML-microstructure result that transfers to us unchanged. |
| **H-E11 Negative-gamma-day vol sizing for SPY overlays** | On days when the NGE proxy is negative, halve size on any SPY overlay (H-E1) and widen stops. | E22: max gamma-induced 30-min vol +6.4 pp; typical effect much smaller. Sizing effect, ≈ 0 bps alpha. | OPT, B | Realized 30-min RV on neg-NGE days not > pos-NGE days at t > 2 → drop the sizing rule. | None. | **Yes**, trivially. |
| **H-E12 Deep-LOB direction model** | Train DeepLOB/TLOB-class model on Databento MBP-1 for our names to time entries. | E9–E12: predictability only at ≤ 50 events, small-tick MCC ≈ 0.1, collapses at the spread threshold. Prior **≤ 0 net**. | MBP (cost), sub-second execution (not available) | Do not build. | Fatal. | **No.** Biggest reason: needs L2/L3 and sub-100 ms execution; the papers themselves say not deployable. |

**Ranking for OneMil (best first):** H-E10 (free, robust), H-E1 (small but real, needs a gamma proxy and full-notional sizing), H-E2 (filter, likely a fill-quality effect), H-E8 (veto study), H-E6 (only if a dealer-gamma sign proxy can be built). Everything else is either infrastructure-gated (H-E4, H-E5) or falsified on prior evidence (H-E3, H-E9, H-E12).

---

## 3. What we would have to build to test the top three

- **H-E1**: nightly pull of SPX/SPY option OI per strike (Alpaca options chain or Cboe delayed OI), Black–Scholes γ from IV, NGE = Σ(OI_calls·γ − OI_puts·γ)·S²·100 with the Baltussen convention (dealers long calls, short puts); a 15:30 cron that reads r_ROD from SPY 1-min bars and logs gated/un-gated; 250 days of paper before a $ sleeve. Pre-registered stat: gated-day mean r_LH and its next-3-day reversal.
- **H-E2**: log the NBBO snapshot (bid, ask, sizes, timestamp lag) at every `[HOD DRY] WOULD BUY`; the engine already fetches quotes for the cap/skip rule, so this is a logging change; evaluate after 200 events, and in parallel run the same test on 300 historical candidates using Databento MBP-1 windows around the breakout minute (this is the OFI-confirmation plan already in memory, `project_hod_break_ofi_filter_plan.md`; the literature prior for it is: fill-quality edge, ≤ 3–4 pp WR, nothing at 5 min).
- **H-E10**: pooled HAR (or LightGBM) on 1-min RV from the HOD-break cache, next-60-min RV target; compare stop-out rate and mean R against the consolidation-low stop.

---

## 4. Sources (URLs used)

- Cont–Kukanov–Stoikov: https://arxiv.org/abs/1011.6402
- Cont–Cucuringu–Zhang: https://arxiv.org/abs/2112.13213 · https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159
- Kolm–Turiel–Westray: https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141
- Aït-Sahalia–Fan–Xue–Zhou: https://www.nber.org/papers/w30366 · https://pubsonline.informs.org/doi/10.1287/mnsc.2022.02435
- Takahashi: https://arxiv.org/abs/2508.06788
- Stoikov: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
- Lucchese–Pakkanen–Veraart: https://arxiv.org/abs/2211.13777 · https://www.sciencedirect.com/science/article/pii/S0169207024000062
- Briola–Bartolucci–Aste: https://arxiv.org/abs/2403.09267 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/
- Prata et al. (LOBCAST): https://arxiv.org/abs/2308.01915
- Berti–Kasneci (TLOB): https://arxiv.org/abs/2502.15757
- ClusterLOB: https://arxiv.org/abs/2504.20349
- Chinco–Clark-Joseph–Ye: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12733 · https://www.alexchinco.com/sparse-signals-in-cross-section.pdf
- Aleti–Bollerslev–Siggaard: https://pubsonline.informs.org/doi/10.1287/mnsc.2023.01657 · https://public.econ.duke.edu/~boller/Papers/MS_2025.pdf
- Bogousslavsky: https://www.sciencedirect.com/science/article/abs/pii/S0304405X21000854
- Gao–Han–Li–Zhou: https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351
- Baltussen–Da–Lammers–Martens: https://www.sciencedirect.com/science/article/abs/pii/S0304405X21001598 · https://academicweb.nd.edu/~zda/intramom.pdf
- Park–Zhao: https://portal.northernfinanceassociation.org/viewp.php?n=2240183764
- Li–Sakkas–Urquhart: https://www.sciencedirect.com/science/article/abs/pii/S138641812100001X
- Amaya–Garcia-Ares–Pearson–Vasquez: https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf
- Dim–Eraker–Vilkov: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4692190 · https://westernfinance-portal.org/viewpaper?n=950096
- Brogaard–Han–Won: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4426358
- Adams–Fontaine–Ornthanalai: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4881008
- Beckmeyer–Branger–Gayda: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704
- Vilkov 0DTE rules: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4641356 · https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md
- Boehmer–Jones–Zhang–Zhang: https://onlinelibrary.wiley.com/doi/10.1111/jofi.13033
- Barber–Huang–Jorion–Odean–Schwarz: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13334
- Ardia–Aymard–Cenesizoglu: https://arxiv.org/abs/2403.17095
- Barber–Odean et al. (paradox): https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/resolving-a-paradox-retail-trades-positively-predict-returns-but-are-not-profitable/6AAA9078F50C2597F44D73FA6A8E3F0D
- Retail flow / fragmented markets (JBF 2024): https://www.sciencedirect.com/science/article/abs/pii/S0378426624001894
- Andersen–Bondarenko: https://academic.oup.com/rof/article-abstract/19/1/1/2886427
- Jurkatis: https://www.sciencedirect.com/science/article/abs/pii/S1386418121000173
- Zhang–Zhang–Cucuringu–Qian: https://arxiv.org/abs/2202.08962
- Realized semivariance & TSMOM: https://www.sciencedirect.com/science/article/abs/pii/S0927539823000245
