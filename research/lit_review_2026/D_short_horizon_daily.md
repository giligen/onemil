# Topic D — Short-horizon (1–10 day) equity anomalies tradable on daily bars

Literature review, 2026-09-16. Scope: effects with a 1–10 trading-day holding period that a
small, long-mostly account could trade from daily bars (plus our SPY 1-min and the
≥5%-range symbol-day 1-min subset). Emphasis on 2022–2026 out-of-sample evidence and
post-publication decay.

Our constraints, restated so the hypothesis table is honest about them:
- Tradable band: price ≥ $5, ADV ≥ 100K shares. Effects that live in microcaps / sub-$5 /
  illiquid names are FLAGGED and usually not usable.
- Long-only in practice (shorting is expensive/unavailable for us). Every long-short paper is
  re-read for what the LONG leg alone delivers.
- Data on hand: point-in-time daily OHLCV for all US symbols 2025–2026 incl. delisted
  (~10K symbols × 420 days); 1-min for SPY and for symbol-days with ≥5% range; Alpaca news
  API. NO fundamentals, NO earnings calendar (hypotheses that need one are marked).
- 4-position book. Costs: our own live telemetry shows entry slippage on thin names runs
  multiples of any academic model (README), so cost sensitivity is scored per hypothesis.

Sourcing note: SSRN/ScienceDirect/OUP full texts were blocked (HTTP 403) and no PDF-text
tool exists on this node, so several entries rely on abstracts, publisher summaries and
secondary sources. Numbers I could not confirm from a fetched page are marked
`[recall — verify]`. Do not quote those into a ship decision without opening the paper.

---

## 1. Meta-evidence: how much of a published effect survives

### M1. McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?", JF 71(1). SSRN 2156623.
- 97 published cross-sectional predictors, US CRSP. Compares in-sample, post-sample-pre-publication, and post-publication long-short returns.
- Result: **−26% out-of-sample (upper bound on data-mining), −58% post-publication**; the 32-pt gap is attributed to publication-informed trading. Decay is larger for predictors with higher in-sample returns and in more liquid/lower-idiosyncratic-risk stocks (cheaper to arbitrage).
- Replicated many times (Jacobs & Müller 2020; Chen & Zimmermann 2020; JKP 2023). Jacobs & Müller (JFE 2020, 241 anomalies × 39 markets): **the US is the only country with a reliable post-publication decline** — arbitrage capital is a US phenomenon.
- Critique: sample ends 2013; monthly-rebalanced, mostly fundamental anomalies; says little directly about daily-horizon effects.

### M2. Chen & Velikov (2023), "Zeroing In on the Expected Returns of Anomalies", JFQA 58(1). SSRN 3073681; FEDS 2020-039.
- 204 anomalies, effective-spread costs (Hasbrouck/Corwin-Schultz), post-publication, post-2005 modern-market era.
- Result: **average anomaly nets 4–8 bps/month** (FEDS draft 4 bps, JFQA 8 bps); the strongest net 10–20 bps/month; combinations ~20 bps. Gross post-publication decay ≈50%, **≈72% if stale pre-2005 post-publication data is excluded, ≈93% net of costs**.
- Read for us: monthly-rebalanced anomalies are dead net of costs for a retail account. Only effects whose gross premium is compensation for liquidity/risk taken (reversal, overnight, closing-auction flows) have a structural reason to persist — and those are exactly the ones whose costs are highest.

### M3. Chen & Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing", Critical Finance Review 11(2), 207–264. SSRN 3604626; openassetpricing.com; github OpenSourceAP/CrossSection.
- 319 characteristics reproduced. For the 161 clearly-significant originals, **98% reproduce with t > 1.96**; regression of reproduced on original t-stat slope 0.88, R² 0.82.
- Companion: Chen & Zimmermann (2020, RAPS) "Publication Bias and the Cross-Section of Stock Returns": bias-corrected in-sample returns are only ~12% below reported `[recall — verify]`; Chen (2022, arXiv 2206.15365) "Most claimed statistical findings in cross-sectional return predictability are likely true".
- Read: the *replication* crisis is not the problem; *decay and costs* are.

### M4. Jensen, Kelly & Pedersen (2023), "Is There a Replication Crisis in Finance?", JF 78(5), 2465–2518. NBER w28432; github bkelly-lab/ReplicationCrisis.
- 153 factors, 93 countries, Bayesian hierarchical model with theme-level shrinkage.
- Result: majority replicate in-sample and OOS internationally; 13 themes, most in the tangency portfolio; evidence *strengthened* by the multiplicity once shrinkage is applied. Short-term reversal is one of the 13 themes and replicates internationally; the "seasonality" theme is among the weakest `[recall — verify]`.
- Post-publication US decline is present and of the McLean-Pontiff magnitude `[recall — verify]`; their point is that decay ≠ false discovery.

### M5. Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns", RFS 29(1). SSRN 2249314.
- Multiple-testing framework over 300+ published factors; recommends **t > 3.0** for a new factor; "most claimed research findings are likely false" in the naive sense.
- Counter: Chen (2022/2024, arXiv 2204.10275) "Do t-Statistic Hurdles Need to be Raised?" — the hurdle question is weakly identified because rejected results are unobserved; empirical-Bayes shrinkage / FDR on the published set is better identified than a raised hurdle.

### Calibration I will use in the hypothesis table
- **Gross survival fraction ≈ 0.40–0.50** of the published gross long-short premium for a US
  anomaly in its post-publication window (MP 42%; Chen-Velikov 50%, 28% excluding stale
  years). Use 0.4 as the prior on gross.
- **Net-of-cost survival for monthly-rebalanced fundamental anomalies ≈ 0.05–0.10** (Chen-Velikov).
  Not applicable to us — we do not have fundamentals anyway.
- For **daily-horizon liquidity-provision effects** (reversal, end-of-day, volume-shock overnight)
  the *gross* premium persists because it is payment for immediacy (Dai-Medhat-Novy-Marx;
  Nagel 2012), but it has trended down with spreads and is **regime-dependent (VIX)**. Prior:
  gross ≈ 0.5× the pre-2010 published level in the $5+/ADV≥100K band; net ≈ gross − 2×
  half-spread − our measured slippage.
- Decision rule: only test a hypothesis whose *haircut* gross (0.4×published) is ≥ 3× the
  round-trip cost in our band; otherwise it cannot be distinguished from zero on 420 days.

---

## 2. Short-term reversal and refinements

### R1. Da, Liu & Schaumburg (2014), "A Closer Look at the Short-Term Return Reversal", Management Science 60(3). NY Fed SR 513; SSRN 1926851. nd.edu/~zda/Reversal.pdf
- Universe: CRSP common stocks (NYSE/AMEX/NASDAQ), 1982–2009, monthly; returns decomposed with analyst-forecast cash-flow news (needs I/B/E/S — we don't have it).
- Rule: monthly reversal on the RESIDUAL of last-month return after removing (i) across-industry momentum, (ii) within-industry expected-return variation, (iii) within-industry cash-flow news. Long bottom / short top decile of residual, hold 1 month.
- Result: residual reversal ≈ **3× the standard reversal** risk-adjusted; standard reversal's decomposed components other than the residual are insignificant. Long side driven by liquidity shocks, short side by sentiment.
- OOS: sample ends 2009; the industry-adjusted idea is confirmed by Hameed & Mian (2015, JFQA "Industries and Stock Return Reversals": within-industry reversal roughly doubles the profit; profits concentrate in illiquid stocks and high-VIX months `[recall — verify]`) and by Blitz et al. (R3) through 2022.
- Critique: monthly formation; cash-flow-news step needs analyst data. What survives for us: **industry/market-adjusted 1–5 day residual**, which needs only prices.

### R2. Dai, Medhat & Novy-Marx (2023), "Reversals and the Returns to Liquidity Provision", NBER w30917. rochester.edu/novy-marx RRLP.pdf; Dimensional Q&A 2023.
- US CRSP 1970s–2020s, monthly reversal sorted by volatility, turnover, size.
- Findings (from the authors' Q&A): more volatile stocks → faster, initially stronger but short-lived reversals; **lower-turnover stocks → more persistent, ultimately stronger reversals** (inventory sits longer); smaller stocks → stronger. Reversal returns "have come down over the last 20 years, consistent with improved market liquidity" but "remain reliable". They purge PEAD and short-run industry momentum from the signal.
- Read: this is the mechanism paper. For us: condition on turnover (low-turnover names revert; high-turnover names don't — see R4) and on volatility (fast reversal = 1–3 day horizon in high-vol names, which is our band).

### R3. Blitz, van der Grient & Honarvar (2023), "Reversing the Trend of Short-Term Reversal", SSRN 4575689; Robeco Insights Oct 2023.
- Global developed markets, multi-decade to ~2022, monthly.
- Finding: **classic 1-month reversal has steadily weakened and "vanished entirely in most regions"**; it is revived by removing its implicit bet against short-term industry and factor momentum (i.e., residualise the past-month return on industry/factor returns). Enhanced version: >2× risk-adjusted performance, lower risk, "remains effective over time". Interpreted as liquidity provision.
- Critique: institutional universe (large/mid caps); monthly. Same message as R1: only the residual reverses.

### R4. Medhat & Schmeling (2022), "Short-term Momentum", RFS 35(3), 1480–1526.
- CRSP common non-financial 1963–2018 + 22 developed markets. Double sort last-month return × share turnover.
- Result: **reversal exists only among LOW-turnover stocks; HIGH-turnover stocks show short-term MOMENTUM** (last month's winners continue), as profitable and persistent as 12-month momentum, survives costs, **strongest in the largest, most liquid, most covered stocks**. Reversal indistinguishable from zero after 2 months.
- Read for us: our band (ADV≥100K, high-turnover movers) is on the *momentum* side of this split. Naively buying last-month losers among high-turnover names is the wrong sign. A 1–5 day version of this split is testable on daily bars (turnover = volume/shares outstanding — we lack shares outstanding; use volume/ADV or $-volume rank as proxy).

### R5. Avramov, Chordia & Goyal (2006), "Liquidity and Autocorrelations in Individual Stock Returns", JF 61(5).
- Weekly/monthly; negative autocorrelation (reversal) is driven by illiquid stocks and is strongest when the initial move came with high turnover `[recall — verify]`. Consistent with reversal = liquidity provision.
- Flag: small/illiquid concentration.

### R6. Nagel (2012), "Evaporating Liquidity", RFS 25(7).
- Reversal-strategy returns are the returns to liquidity provision; **they load strongly on VIX** (high-VIX periods → reversal profits 3–5× normal `[recall — verify]`), and shrank in calm regimes.
- Read: a 2025–2026 test must be split by VIX regime; a null in calm months is not a refutation.

### R7. Cheng, Hameed, Subrahmanyam & Titman (2017), "Short-Term Reversals: The Effects of Past Returns and Institutional Exits", JFQA 52(1).
- Reversal following large one-month declines is concentrated in stocks with institutional selling (exits) and is weaker after declines in strong-return stocks `[recall — verify]`.
- Relevance: "reversal after large moves" is conditional on WHO sold — a non-news, high-volume decline reverts; a news-driven one does not (see N1, N2).

### R8. Chen, Cohen, Liang & Sun (2025), "Maxing Out Short-Term Reversals in Weekly Stock Returns", J. Empirical Finance (S0927539825000301); SSRN 4622831 (Nov 2023).
- CRSP NYSE/AMEX/NASDAQ common stocks, July 1963–Dec 2022, **weekly**.
- Rule: within top-MAX (max daily return last month) stocks, long last week's losers / short winners; hold 1 week.
- Result: **1.66%/week among high-MAX vs 0.65%/week among low-MAX** long-short. Mechanism: lottery demand amplifies overreaction to news.
- Critique: high-MAX stocks are disproportionately small, low-priced, high-spread; the long leg (buying last week's losers among lottery stocks) is exactly the inventory nobody wants. Costs will eat most of it below $5; above $5 unknown — that is the test.

### R9. Baltussen, Da & Soebhag (2024), "End-of-Day Reversal", SSRN 5039009; EFMA 2024.
- US individual stocks, intraday; the return in the **last 30 minutes reverses the intraday return up to 15:30**; "economically and statistically highly significant"; comes mainly from positive price pressure on intraday LOSERS (i.e. the LONG leg).
- Related: Brogaard, Han & Kim (2024), "Intraday Residual Reversal in the U.S. Stock Market", SSRN 4731947 — residual intraday reversal earns "162.3% annualized" gross, i.e. a number that only exists before costs.
- For us: our 1-min data for ≥5%-range symbol-days is *exactly* the intraday-loser sample. Long-only: buy at 15:30 the day's big losers, sell MOC/at close. Cost sensitivity extreme (two trades in 30 min).

### R10. Barardehi, Bogousslavsky & Muravyev (2026), "What Drives Momentum and Reversal? Evidence from Day and Night Signals", RFS forthcoming; SSRN 4069509.
- CRSP 1926–2019 + international. Decompose past returns into intraday (open→close) and overnight (close→open).
- Result: portfolios on **past INTRADAY returns show short-term reversal and momentum without long-term reversal; past OVERNIGHT returns show no momentum, only long-term reversal**. Interpretation: under-reaction to private information revealed through trading (Hong-Stein).
- For us: daily OHLC gives both components. The 1–5 day reversal signal should be built on the open→close component, not close→close.

---

## 3. Volume shocks / high-volume return premium

### V1. Gervais, Kaniel & Mingelgrin (2001), "The High-Volume Return Premium", JF 56(3), 877–919. SSRN 146468.
- NYSE 1963–1996, daily and weekly formation. Rule: a stock whose volume on formation day is in the top (bottom) 10% of its prior 49 days is "high (low) volume"; hold 1–20 days (also 50-day reference-return version).
- Result: high-volume stocks outperform low-volume over the next 20 days by roughly 0.5–1% (daily formation), stronger in small stocks but present in large `[recall — verify]`. Visibility/attention mechanism.
- OOS: Kaniel, Ozoguz & Starks (2012 JFE) — 41 countries, persistent. Wang, Wang & Xue (2020 JFE, "The high volume return premium and economic fundamentals") — the premium is linked to fundamentals/analyst revisions, not just visibility.
- Critique: post-2000 US evidence thin at the daily horizon; small-stock heavy. Long-only feasible (long high-volume).

### V2. Cartea, Cucuringu, Jin & Wilson (2025), "Volume Shocks and Overnight Returns", SSRN 5156605 (Oxford, Feb 2025).
- US stocks, long sample; intraday volume shock (volume in the session relative to its own history) → **positive OVERNIGHT return, no effect in the next intraday session**; holds across size buckets and sub-periods; positively skewed, fat right tail.
- Practitioner replication (Quantitativo, 2025): rank by volume through 15:45, buy top decile MOC, sell next open; skip orders >10% of daily volume; IB tiered commissions ×2. **Russell 3000: ~14%/yr, Sharpe 0.89 net; Nasdaq Biotech: 36.1%/yr, Sharpe 1.52, MDD −36.6%, 68% positive months, ~0 correlation to SPX.**
- Related: Perreten & Wallmeier (2024, SSRN 5004991) — S&P 500 2008–2023, intraday volume timing predicts overnight returns; "Persistence or reversal? The effects of abnormal trading volume on stock returns" (Eur. J. Finance 2024, doi 10.1080/1351847X.2024.2303092); Lee, Kim & Kim (SSRN 2812010) — ATV's short-run return predictability comes only from stocks whose ATV *persists*.
- Critique: the overnight leg needs the CLOSE fill and the OPEN fill — both auctions, both fine for size ≤ 1% ADV. Requires intraday volume by ~15:45 live (our scanner has it) but our BT has only daily volume for most names (≤ 15 min of look-ahead volume; small but must be stated).
- **This is the strongest daily-bar candidate in the review.**

---

## 4. Earnings: PEAD and the announcement premium (need an earnings calendar)

### E1. Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift", Critical Finance Review 11(3–4), 613–646. SSRN 3111607.
- US, 1990s–2010s; prices now fully reflect the surprise on the announcement day. **PEAD non-existent for large stocks since 2006, disappeared later for microcaps.** Mechanism: decimalisation, HFT, Reg NMS.
- Contested by Meursault, Liang, Routledge & Scanlon (2023, text-based surprise, 2008–2019), Dickerson-Julliard-Mueller (2025) and Hirshleifer-Peng-Wang (2025, t ≈ 14). **Subrahmanyam (UCLA Anderson Review, 2025) reconciles: replicating Dickerson with all stocks t = 2.18; excluding microcaps t = 1.43.** The "revival" is microcaps (3% of market value).
- For our band ($5+/ADV≥100K): treat daily PEAD as dead. Needs earnings calendar anyway.

### E2. Heitz, Narayanamoorthy & Zekhnini (2020/2025), "The Disappearing Earnings Announcement Premium", SSRN 3296537.
- US; the Frazzini-Lamont (2007) premium (long expected announcers, ~1.5%/mo, Sharpe ~0.9 on 1973–2004) **disappeared after the 2004 Form 8-K disclosure regulation**; the premium migrated to 8-K filing days; robust internationally (Barber, De George, Lehavy & Trueman 2013 JFE, 46 countries, >11%/yr).
- Daily-horizon variants: Linnainmaa & Zhang "Earnings Announcement Return Cycle"; Ertan et al. "Earnings Announcement Return Extrapolation" (long high-recent-EA-return firms into the next EA earns 16–18 bps/day post-EA in the five-factor model `[from abstract]`); Gamm (SSRN 3293638) — positive OVERNIGHT returns for weeks after both large positive and negative surprises.
- Needs: earnings calendar + 8-K feed. Prior for a US daily long-only EA-premium trade: ~0 since 2004.

### E3. Brandt, Kishore, Santa-Clara & Venkatachalam (2008), "Earnings Announcements are Full of Surprises" (Quantpedia PEAD page).
- 1987–2004, price ≥ $5, ~1,000 stocks, long top-SUE∩top-EAR quintile, hold 60 days: ~12.5–15%/yr long-short, MDD −11%. Long-side driven; small-cap heavy. Superseded by E1.

---

## 5. Lottery / MAX at short horizons

### L1. Bali, Cakici & Whitelaw (2011), "Maxing Out", JFE 99(2).
- CRSP 1962–2005, monthly: low-minus-high MAX decile >1%/month. Gorman, Akhtar, Durand & Gould (2022, CFR 11(3–4)) — it is overreaction embedded in the high-MAX month, not lottery preference per se; event-study evidence. Journal of Behavioral Finance (2022) — MAX effect concentrated in stocks with the WORST recent returns and absent in recent winners.
- Long-only at 1–10 days: the profitable side is *avoiding/shorting* high-MAX names; the tradable long variant is R8 (weekly loser reversal inside high-MAX). Post-2011 decay: not quantified in what I could fetch; JKP class it inside the "low-risk"/"skewness" theme which replicates but with shrunken alpha `[recall — verify]`.

---

## 6. 52-week high / breakouts

### B1. George & Hwang (2004), "The 52-Week High and Momentum Investing", JF 59(5).
- Monthly, 6–12 month holding; nearness to 52-week high beats price momentum. Hong, Jordan & Liu (industry version, Quantpedia): 1963–2009, 0.93%/mo long-short, Sharpe 0.7, MDD −54%; **Quantpedia OOS tracking: "slightly negative performance in recent periods", confidence "Moderate"**.
- Daily-horizon breakout evidence: none in the peer-reviewed literature I could find for 2022+ (the "JFM 2023: 72% continuation, +11.4% over 31 days on >150% volume breakouts" claim that surfaced in search is from a retail blog and could not be traced to a paper — treat as unverified). Our own clean-sheet study (`research/bf_zero/`) found the intraday HOD-break survives at +0.27–0.33R while the daily 52-week-high proximity is a 6–12-month signal, not a 1–10-day one.
- Prior for a daily 52-week-high breakout hold 1–10 days in our band: weak (≤ 5 bps/day gross), high cost (buying highs = paying the spread into momentum flow).

---

## 7. Calendar seasonalities

### S1. McConnell & Xu (2008), "Equity Returns at the Turn of the Month", FAJ 64(2); SSRN 917884; Quantpedia "Turn of the Month in Equity Indexes".
- 1926–2005 US; buy close of last trading day (or T−4), sell close of T+3: ~7.2%/yr on the window, Sharpe 1.04, MDD −21%; 31 of 35 countries.
- **2025 update (QuantSeeker, US ETFs through early 2025): classical [0,+3] window is "a few bps higher than other days", NOT significant for SPY/QQQ/IWM; the broader [−3,+3] window is +5–12 bps/day and significant but with lower Sharpe than buy-and-hold; rolling averages show the classical effect "gradually diminished to zero over the last decade".** Vidal & Vidal-García (2022, SSRN 4106003) similar; a 2023 study of 2018–2023 finds no TOM effect.
- Long-only trivially feasible (SPY). Prior: ~5 bps/day in the wide window, non-robust.

### S2. Birru (2018), "Day of the Week and the Cross-Section of Returns", JFE 130(1), 182–214.
- CRSP; speculative stocks (high vol, low price, high MAX, distressed) earn LOW returns on Monday and HIGH on Friday relative to non-speculative; Monday/Friday alone account for >100% of anomaly returns whose speculative leg is short/long; robust to sub-samples and news; mood explanation. International confirmation (Economics Letters 2019).
- Post-2018 OOS: not found. "Reversal of Monday returns: it is the afternoon that matters" (Finance Research Letters 2024) reports the Monday effect is now an intraday phenomenon `[from title/abstract only]`.
- Long-only variant: hold speculative names Thu close → Fri close; stay out Monday. Prior 10–20 bps/day differential gross `[recall — verify]`, unshrunk.

---

## 8. Large price change + volume: continuation vs reversal

### P1. Pritamani & Singal (2001), "Return predictability following large price changes and information releases", JBF 25(4).
- NYSE/AMEX 1990–1992, |daily return| ≥ 10% (or 3σ); condition on volume and on news (WSJ/Dow Jones) `[recall — verify]`. **Large move + high volume + news → continuation** over 1–20 days (~1–2% cumulative); large move without volume/news → reversal.
- Recent: "Price reversals and price continuations following large price movements", JBF 2019 (sciencedirect S014829631830420X; authors not fetched — see link) confirms firm-specific information drives continuation, non-information moves revert. Chiang & Kirby, "Short-Term Reversals, Short-Term Momentum, and News-Driven Trading Activity" — news-driven volume separates continuation from reversal.
- OUR OWN OOS EVIDENCE (2025–2026, point-in-time, 647,796 symbol-days, `research/bf_zero/REPORT.md`; ORB veto study): the raw large-move detectors (Cameron flag, raw ORB breakout) have NO edge at any exit in the whole market (ORB raw −0.18R 2025 / −0.04R 2026; BF raw −0.01R / −0.07R). That is a direct, current refutation of unconditional "big move + volume → continuation" in our band; the selection stack (news catalyst, PM$ volume, prev-day range) is where the edge was.

### N1. Jiang, Li & Wang (2021), "Pervasive underreaction: Evidence from high-frequency data", JFE 141(2), 573–599. SSRN 2679614.
- US, high-frequency decomposition of daily returns into news- and non-news components (RavenPack timestamps). **Prices drift in the direction of the initial news reaction for several days without reversal**; a strategy on the news-driven component is profitable after costs `[abstract]`; drift stronger when investors are distracted; analysts adjust slowly.
- Long-only: buy positive-news-driven up-moves at next open, hold 1–5 days. Needs timestamped news (Alpaca/Benzinga suffices for a crude version).

### N2. Lopez-Lira & Tang (2023, rev. 2024), "Can ChatGPT Forecast Stock Price Movements?", SSRN 4412788; arXiv 2304.07619.
- >50K headlines Oct 2021–May 2024 (post-training-cutoff), NYSE/Nasdaq/small-cap exchange; GPT-4 scores headline good/bad; long positives / short negatives at next open. **~700% cumulative gross L/S Oct 2021–May 2024; ~90% portfolio-day hit rate on the (non-tradable) initial reaction; the tradable drift is "especially for small stocks and negative news"; returns decline with LLM adoption.**
- Critique: gross; small-cap; negative-news short side is where most of the drift is (not feasible for us); the effect is being arbitraged in real time.
- Ryan & Taffler (2004, JBFA): ≥65% of large FTSE-350 price/volume moves are explained by public news — the anchor for "condition on news".

---

## 9. Close-of-day flows: leveraged ETFs, market intraday momentum, index events

### F1. Baltussen, Da, Lammers & Martens (2021), "Hedging demand and market intraday momentum", JFE 142(1). SSRN 3760365; nd.edu/~zda/intramom.pdf.
- 60+ futures 1974–2020. **The last-30-minute return is positively predicted by the return from prior close to 15:30; reverts over the next days;** linked to gamma-hedging of option market makers and leveraged-ETF rebalancing. Complements Gao, Han, Li & Zhou (2018 JFE, "Market intraday momentum": SPY 1993–2013, first-half-hour return predicts last-half-hour; timing strategy Sharpe ≈ 1 gross `[recall — verify]`). MDPI Risks 12(11):180 (2024) "Market Predictability Before the Closing Bell Rings" — recent-sample confirmation on the index `[title only; fetch blocked]`.
- We hold SPY 1-min. Long-only feasible (long SPY 15:30→close on up-days). Prior: 2–4 bps/day gross, Sharpe ~1 gross, SPY spread ~0.5 bp → net positive but tiny in $; it is a *daily sizing* overlay, not a book.

### F2. Barbon, Beckmeyer, Buraschi & Moerke (2022/2024), "Liquidity Provision to Leveraged ETFs and Equity Options Rebalancing Flows: Evidence from End-of-Day Stock Prices", SSRN 3925725 (earlier: Beckmeyer & Moerke, "The Role of Leveraged ETFs and Option Market Imbalances on End-of-Day Price Dynamics", 2012–2019).
- Single-stock level. A 1-σ increase in LETF rebalancing flow raises the last-30-min return by ~430% of its (tiny) average; gamma-hedging pressure by −113%. **LETF effects have DECREASED over time** as liquidity providers front-run them; gamma effects persist. Ivanov & Lenkey (2018, JFM "Do leveraged ETFs really amplify late-day returns and volatility?") — after correcting the flow measure, LETF rebalancing explains little.
- Lenkey (2024, Quantitative Finance and Economics 8(4), 815–840, survey): associations are statistically significant but **economically insignificant, and most papers have "potentially serious methodological errors"**.
- For us: single-stock 2× wrappers now cover TSLA/NVDA/MSTR/COIN etc.; we have the wrapper map (`data/research/orb_asset_class_map_20260711.csv`) and 1-min bars on the ≥5%-range days, which are the days rebalancing demand is largest. Prior: low and shrinking; front-run by pros.

### F3. Greenwood & Sammon (2022/2025), "The Disappearing Index Effect", NBER w30748 / HBS 23-025 / JF forthcoming.
- S&P 500 additions/deletions 1980s–2020: the announcement-to-effective abnormal return fell from ~7–8% (1990s) to ~0 in the 2010s `[recall — verify]`, while mechanical index demand *rose*; explanations: pre-positioning, more liquid closing auctions, offsetting mid-cap fund selling. Bennett, Stulz & Wang (2022) and S&P DJI (2021, 1995–2021) concur. Russell reconstitution effects also shrank after the Nasdaq closing cross was adopted; from 2026 recon is semiannual.
- Needs event lists we don't have. Prior at daily horizon: ~0 net for additions; deletions/pressure-day reversals are short-side or need borrow. Deprioritised.

---

## 10. Hypothesis table

Conventions: "prior bps/day" = expected GROSS return per day held on the long leg alone in
our band after the 0.4× survival haircut; "Sharpe" = prior net Sharpe of a stand-alone book if
capacity were unlimited; cost sensitivity: L (auction fills, ≥ $10 names), M, H (thin names,
two trades inside 30 min). Trades/week assume a 4-slot book, 1 entry per slot per hold.

| H-id | Rule (one sentence) | Prior effect (bps/day held; Sharpe) | Long-only feasible? | Data needed | Falsification on 2025–2026 daily bars | Cost sens. | Trades/wk (4 slots) |
|---|---|---|---|---|---|---|---|
| **D1 Volume-shock overnight** (V2, V1) | At 15:45 rank all $5+/ADV≥100K stocks by today's volume ÷ ADV20; buy top decile (cap 4 names by highest ratio, skip if order >1% ADV) MOC; sell at next open. | 8–15 bps per overnight gross (Quantitativo R3000 net ≈ 5.5 bps/day incl. IB costs; biotech subset 14 bps); Sharpe 0.6–0.9 net | YES — the whole effect is on the long side | Daily OHLCV (BT uses full-day volume = ≤15 min look-ahead; state it); live scanner volume at 15:45 | Mean close→open return of top-decile minus same-day decile-5 control ≤ 0 or t < 2 on ≥ 400 event-days; or sign not positive in BOTH 2025 and 2026 halves; or gain < 2× (half-spread at close + at open) | L–M (two auctions) | ~15–20 (daily turnover of all 4 slots) |
| **D2 Residual reversal after large non-news down day** (R1, R2, R6, R7, R10, P1) | Buy at next open a stock whose OPEN→CLOSE return today is ≤ −8% (or bottom 1% of industry/market-residual) on volume ≥ 2× ADV20 with NO Alpaca news in [prev close, now]; hold to close of day +2 (or +1 in high-vol names); skip if 1-day return ≤ −25% (halt/news risk). | 15–30 bps/day held gross in $5+ names (published residual reversal ≈ 7 bps/day L/S monthly → event-conditional 1–3 day long leg larger); Sharpe 0.5–0.8 net; **VIX-dependent** | YES — long leg = liquidity provision to forced sellers | Daily OHLCV; Alpaca news flag (crude); VIX for regime split; no fundamentals needed | Event mean day+1..+2 return ≤ 0 or t < 2 on ≥ 300 events; effect must be ≥ 0 within BOTH VIX halves (a positive-only-in-high-VIX result is a *conditional* pass, not a fail); must not be driven by <$10 names | M–H (buying a falling knife at the open; use limit at open) | ~4–8 |
| **D3 High-MAX weekly-loser reversal** (R8, L1) | Each Friday close, among the top-decile MAX (max daily return last 20 d) stocks with price ≥ $5/ADV ≥ 100K, buy the bottom-decile past-week losers; hold 5 days. | Published 1.66%/wk L/S → long leg ≈ 0.8%/wk gross → 0.4× haircut ≈ 6–8 bps/day; Sharpe 0.3–0.5 net | YES (long leg) but the names are the illiquid tail of our band | Daily OHLCV only | 5-day return of the long leg minus equal-weighted high-MAX universe ≤ 0 or t < 2 on ≥ 60 weekly formations; or the effect vanishes when price ≥ $10 | H | ~4 (weekly rebalance) |
| **D4 SPY market intraday momentum** (F1) | If SPY return prev-close→15:30 > +0.3%, buy SPY at 15:30, sell MOC; (mirror: stay flat/hedge when < −0.3%). | 2–4 bps/day gross, Sharpe ~0.8–1.0 gross, ~0.6 net; tiny $ | YES | SPY 1-min (have) | Mean last-30-min return conditional on sign of the day-so-far return ≤ 0 or t < 2 over 420 days; or Sharpe < 0.4 net | L | ~2–3 (signal fires ~40–60% of days) — an overlay, not a book |
| **D5 Single-stock end-of-day reversal** (R9) | At 15:30 buy the day's ≥ −5% intraday losers (price ≥ $5, ADV ≥ 100K, no halt, no news), sell MOC. | Published "highly significant" gross; positive pressure on losers; prior 10–20 bps per 30-min hold gross; net ≈ 0–10 bps after two spread crossings; Sharpe ≤ 0.5 net | YES | Our ≥5%-range symbol-day 1-min bars (exactly the loser sample); live quote for spread | Mean 15:30→close return of qualifying losers ≤ 2× half-spread or t < 2 on ≥ 500 events; must survive excluding <$10 names | H | ~10–15 |
| **D6 News-driven continuation** (N1, N2, P1) | At open, buy stocks with ≥ +5% overnight gap AND an Alpaca/Benzinga headline in [prev 16:00, 09:25] scored positive (LLM or keyword), hold to close of day +1 to +3. | 10–20 bps/day gross for small names, ≤ 5 bps in mid/large; drift is concentrated in NEGATIVE news (not ours); Sharpe 0.3–0.6 net; decaying with LLM adoption | YES for the positive-news leg only (the weaker leg) | Daily OHLCV + Alpaca news + an LLM scorer (Claude API) — parity with our ORB catalyst flag | Day+1..+3 return of positive-news gappers minus no-news gappers ≤ 0 or t < 2 on ≥ 300 events (this is our ORB catalyst-veto finding re-tested at the daily horizon) | M | ~4–8 |
| **D7 Day-of-week speculative long** (S2) | Buy Thursday close a basket of "speculative" names (top-quintile 20-d vol, price $5–20, high MAX), sell Friday close; never hold speculative names over Monday. | 10–20 bps/day differential gross `[recall]`, unshrunk; Sharpe 0.3–0.5 net | YES | Daily OHLCV | Fri-minus-other-days return of the speculative basket ≤ 0 or t < 2 over ~88 Fridays; Monday-minus-other ≥ 0 (i.e. the pattern absent) | M | 4 (one weekly cycle) |
| **D8 Turn-of-month SPY** (S1) | Buy SPY at close T−4 (or T−1), sell at close T+3. | Classical window ≈ 0–3 bps/day, insignificant 2015–2025; wide window +5–12 bps/day; Sharpe < buy-and-hold | YES | SPY daily | Window return minus non-window ≤ 0 or t < 1.5 over 20 months — likely FAILS; keep as a sizing calendar only | L | ~1 |
| **D9 Large up-move + volume continuation (Pritamani-Singal, daily)** (P1) | Buy at close a stock up ≥ +10% close-to-close on volume ≥ 3× ADV20 with a news catalyst; hold 1–5 days with −1R stop. | Published ~1–2% over 20 days in 1990s; **our own 2025–26 evidence: raw ≈ 0 or negative; catalyst-conditioned ≈ small +**; prior ≤ 5 bps/day; Sharpe ≤ 0.3 | YES | Daily OHLCV + news flag | Already largely falsified by `research/bf_zero` and the ORB veto study (raw ORB −0.18R/−0.04R). Only re-test the catalyst+PM$-conditioned version at the daily horizon | M | ~4–8 |
| **D10 Daily 52-week-high breakout** (B1) | Buy at close a stock closing at a new 252-d high on volume ≥ 1.5× ADV20, hold 5–10 days with −1R stop. | ≤ 5 bps/day gross; Quantpedia OOS slightly negative; Sharpe ≤ 0.3 | YES | Daily OHLCV | 5- and 10-day post-breakout return minus universe ≤ 0 or t < 2 on ≥ 300 breakouts — expected to fail | M | ~4 |
| **D11 Single-stock LETF close rebalancing** (F2) | On days when an underlying with a 2× wrapper is up/down ≥ 5% at 15:30, buy (if up) the underlying at 15:30, sell MOC. | Published effect economically small and decreasing; prior 0–5 bps per 30 min; Sharpe ≤ 0.3 | YES for the up-day side | Our wrapper map + ≥5%-range 1-min bars | Mean 15:30→close return on wrapper-underlying big-up days minus non-wrapper big-up days ≤ 0 or t < 2 — expected to fail | H | ~2–5 |
| **D12 Intraday-component reversal, 1 day** (R10) | Buy at close the bottom-decile OPEN→CLOSE residual return (market- and 20-d-vol-adjusted), sell next close; exclude names whose CLOSE→OPEN was also large (news). | Reversal on the intraday component only; prior 5–10 bps/day gross, Sharpe 0.4–0.6 net | YES | Daily OHLCV | Next-day return of bottom-decile intraday-residual minus middle decile ≤ 0 or t < 2 over 420 days; must hold in ≥ $10 names | M | ~15–20 |
| **D13 PEAD, daily (day+1..+5)** (E1) | Buy at open day+1 stocks with announcement-day return ≥ +8% on ≥ 3× volume, hold 5 days. | Dead in non-microcaps since 2006 (t = 1.43 ex-microcaps); prior ≈ 0 | YES | **Earnings calendar (not available)** | Skip until calendar exists; if tested: day+1..+5 drift ≤ 0 or t < 2 | M | ~4 in season, 0 otherwise |
| **D14 Earnings-announcement premium, daily** (E2) | Buy at close T−1 stocks with a scheduled announcement at T (AMC) or T+1 (BMO), sell close T+1. | Disappeared in the US post-2004; prior ≈ 0 | YES | **Earnings calendar (not available)** | Skip; if tested: announcement-window return minus non-window ≤ 0 | M | ~4 in season |
| **D15 Index-event pressure/reversal** (F3) | Buy at close of effective day a deletion (pressure reversal) / fade additions. | ≈ 0 for additions since 2010s; deletion reversal is a short/borrow story | Partly | **Event lists (not available)** | Skip | M | <1 |

### Priority ranking for the 2025–2026 falsification runs
1. **D1 Volume-shock overnight** — strongest 2025 evidence, long-side effect, auction fills, all-daily-bar data, highest trade count (statistical power on 420 days).
2. **D2 Residual reversal after large non-news down day** — mechanism paper (R2) + our band is high-vol (fast, 1–3 day reversal); split by VIX. Highest per-event bps; execution risk is the falling-knife open.
3. **D12 Intraday-component reversal** — the cleanest OHLC-only test of R10; if D12 is flat, D2 is probably noise too.
4. D4 (SPY overlay), D6 (needs LLM scorer — we already run a catalyst flag), D5 (uses the 1-min subset we have).
5. D3, D7, D8, D9, D10, D11 — cheap to test, low prior; D13–D15 blocked on data.

### Cross-cutting test protocol (pre-committed, same as `research/bf_zero/DESIGN.md`)
- Point-in-time universe incl. delisted; price ≥ $5 and ADV ≥ 100K at signal time.
- Report event returns vs. a same-day matched control (same price/ADV bucket), not vs. zero.
- Two eras (2025 / 2026 YTD) must both be ≥ 0; t ≥ 2 on the pooled sample; effect must survive
  dropping the < $10 bucket (else it is the illiquid tail).
- Net = gross − 2 × half-spread (from the 1-min subset or live quotes) − our measured slippage
  (`trades` DB), never an academic 10 bps.
- Expect ~0.4× of published gross; a result *above* published is a bug (look-ahead) until proven otherwise.
