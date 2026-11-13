# Wide signal search: candidates (2026-11-13)

Brief: `research/signal_search/BRIEF_20261113.md` (owner: "go wide and find it"). 59 candidates in 10 families (A–J),
none from the CLOSED list. Where a candidate is near a closed line, the entry says what is different.

**Evidence convention.** A number without a tag was seen in a source during this session: a search-result summary of
the paper, or the paper's own text (PDF extracted for Baltussen–Da–Soebhag and OFR WP 23-08). Search summaries are
second-hand, so every headline number gets re-read in the paper before a PREREG. **[m]** = from memory and NOT
verified this session; treat it as a hypothesis about the paper, not a fact. "Gross" means before costs; most academic
numbers are long-short decile spreads, factor-adjusted, on CRSP-era samples. Every one of them is a claim about a
PAST sample, and the literature on post-publication decay (McLean–Pontiff 2016 [m]: roughly −58 % after publication)
applies to all of them.

Cost reference (brief): round trip 15–40 bps on small caps, 5–10 bps on liquid names. MOC/MOO auction fills pay no
quoted spread. L/S = long-short. R% = R as a share of price.

---

## A. Overnight vs intraday clienteles (the "tug of war")

**A1. Overnight momentum (buy the close, sell the open, on overnight-return winners)**
Source: Lou, Polk, Skouras, "A tug of war: overnight versus intraday expected returns", JFE 134 (2019) 192–213, http://personal.lse.ac.uk/polk/research/TugOfWar.pdf.
Mechanism: overnight clientele (retail, open-auction demand) and intraday clientele (institutions) push in opposite directions persistently. Momentum profit accrues entirely overnight; the intraday component is negative.
Gross: of 14 strategies, momentum and reversal earn everything overnight and the others earn intraday, usually with opposite signs (seen). Magnitudes: momentum overnight leg ≈ +3 %/month, intraday leg negative [m].
Frequency: daily close→open on ~100 names, monthly formation. Data: Alpaca daily open/close, free. Cost: two auction fills per night (MOC + MOO) means no quoted spread, but auction impact applies. Edge per night ≈ 10+ bps [m], so it survives only on auction fills.
Out of regime: sample 1993–2013 [m]; 2022 unknown. Not the closed "overnight hold of closers-at-high": that was a day-of selection on a gapper universe, while this sorts on the 12-month overnight-return history.
Prior: STRONG (JFE, replicated internationally [m]).

**A2. Frequency of overnight-up / daytime-down reversals predicts next month**
Source: Akbas, Boehmer, Jiang, Koch, "Overnight returns, daytime reversals, and future stock returns", JFE 145(3) (2022) 850–875, https://ink.library.smu.edu.sg/lkcsb_research/7712/.
Mechanism: arbitrageurs over-correct persistent overnight noise-trader pressure, so stocks with frequent "+overnight / −day" days are underpriced.
Gross: a higher count of such reversals in month t predicts higher returns in t+1, for stocks and for the market (seen). L/S ≈ 0.5–1 %/month [m].
Frequency: monthly cross-section, holdings of weeks. Data: Alpaca daily bars, free. Cost: fine at monthly turnover. R% not applicable (portfolio sleeve).
Out of regime: holds in the US; does not hold uniformly in other G7 markets (seen).
Prior: MEDIUM-STRONG.

**A3. Retail-attention open inflation: short at the open, cover at the close**
Source: Berkman, Koch, Tuttle, Zhang, "Paying attention: overnight returns and the hidden cost of buying at the open", JFQA 47(4) (2012), https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1625495.
Mechanism: retail net buying at the open inflates the opening price of high-attention stocks (large absolute return the day before). Liquidity providers sell it, and the price reverts during the day. Retail pays.
Gross: a strong overnight-positive, intraday-reversal pattern concentrated in high-attention, hard-to-value stocks. The implicit cost frequently exceeds the effective half-spread (seen). Decile spreads ≈ 30–60 bps intraday [m].
Frequency: 5–20 names/day, open→close. Data: Alpaca daily and 1-min, free. Cost: short at the open auction (no quoted spread), cover with MOC. R% ≥ 1 % on attention names.
Out of regime: sample 1996–2008 [m]; stronger when retail sentiment is high (seen), so it should be larger in 2020–21 and smaller in 2022.
Note: this sits next to the closed gapper lines. It is a cross-sectional SHORT at the open sorted on attention, not a breakout. Kill it at once if it reproduces the gapper population.
Prior: MEDIUM.

**A4. Arbitrageur unwinding at the close (mispricing factor loses in the last 30 min)**
Source: Bogousslavsky, "The cross-section of intraday and overnight returns", JFE 141(1) (2021) 172–194, https://econpapers.repec.org/article/eeejfinec/v_3a141_3ay_3a2021_3ai_3a1_3ap_3a172-194.htm.
Mechanism: arbitrageurs cut positions before the close to avoid overnight risk, so mispricing widens at the end of the day. Size and illiquidity premia are realised in the last 30 minutes, when small-cap turnover spikes (seen).
Gross: the mispricing factor is positive through the day and negative at the end of the day (seen). Magnitude in bps [m].
Frequency: daily, 30-minute hold. Data: free 1-min bars plus a composite of daily anomaly scores. Cost: one leg can exit through the auction. Gross per day is probably a few bps, which is a cost risk.
Prior: MEDIUM (one JFE paper; consistent with A5).

**A5. Single-stock end-of-day reversal: buy intraday losers at 15:30, sell MOC**
Source: Baltussen, Da, Soebhag, "End-of-Day Reversal" (EFMA 2024 WP), https://academicweb.nd.edu/~zda/EOD.pdf (text extracted).
Mechanism: retail "buy-the-dip" attention purchases near the close, plus short sellers' end-of-day risk management, put positive price pressure on intraday LOSERS. The effect is asymmetric and comes entirely from negative ROD3 returns (seen).
Gross: ROD3 (close t−1 → 15:00) negatively predicts the last-half-hour return. Portfolio-sort t > 10. L/S earns **3.55–6.15 bps/day (8.9–15.4 %/yr) since 1993**. Present in almost every 5-year rolling window and in the largest and most liquid stocks. Not bid-ask bounce: the paper skips 15:00–15:30. The LH pressure reverts the next day (seen).
Frequency: daily, hundreds of names, 30-minute hold. Data: free 1-min bars. Cost: **fails at the average**. 4–6 bps/day L/S against 5–10 bps per liquid round trip. It survives only if the extreme-loser tail carries ≥ 15 bps and the exit is MOC (no spread) while the entry rests at the bid.
Out of regime: rolling windows are stable (seen).
Prior: MEDIUM-STRONG on existence, WEAK on net tradability.

**A6. Fade the closing pressure overnight (the reverse leg of A5 and B1)**
Source: same paper (the LH return "contains a transitory price pressure that reverts the next day", seen), and B1.
Mechanism: the same flow, collected on the next morning's open.
Gross: per-name magnitude [m]. Frequency: daily. Data: free. Cost: MOC + MOO, so auction fills only.
Prior: MEDIUM.

**A7. Same-half-hour periodicity (Heston–Korajczyk–Sadka)**
Source: Heston, Korajczyk, Sadka, "Intraday patterns in the cross-section of stock returns", JF 65 (2010) 1369–1407, https://ar5iv.arxiv.org/html/1005.3535.
Mechanism: recurring institutional order splitting at the same time of day.
Gross: continuation at lags of 13, 26, 39 half-hours, lasting ≥ 40 days (seen). L/S ≈ 1–3 bps per half-hour [m].
Frequency: very high. Data: free 1-min bars. Cost: **fails**. A few bps per half-hour does not cover even liquid spreads.
Prior: STRONG on existence, REJECT on cost (kept as a timing overlay for other entries).

**A8. Same-weekday / daily return seasonalities**
Source: Keloharju, Linnainmaa, Nyberg, "Return seasonalities", JF (2016) https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12398; Da & Zhang, "Same-Weekday Momentum" https://afajof.org/management/viewp.php?n=181972.
Mechanism: recurring flows and mood. A stock's relative performance on a weekday recurs on the same weekday (seen).
Gross: weekday spread magnitude [m]. Frequency: daily rebalance. Data: free. Cost: likely < 10 bps/day, so at risk.
Prior: MEDIUM.

**A9. Overnight drift at the European open (SPY, 02:00–03:00 ET)**
Source: Boyarchenko, Larsen, Whelan, "The overnight drift", RFS (2023), https://research.cbs.dk/en/publications/the-overnight-drift-in-us-equity-returns/; NY Fed Liberty Street, "The disappearing overnight drift", https://ideas.repec.org/p/fip/fednls/103479.html.
Gross: 3.6–3.7 %/yr annualised 1998–2019, **≈ 0 since 2021** (seen).
Data: needs the futures or the Alpaca overnight session [m].
Prior: REJECT (decayed; listed so it is not re-proposed).

---

## B. Closing auction, calendar and dealer-flow effects

**B1. Closing-auction (MOC imbalance) price-pressure reversal**
Source: "Who trades at the close? Implications for price discovery and liquidity" (AEA 2021), https://www.aeaweb.org/conference/2021/preliminary/paper/H9T4hef7; EFMA 2023 closing-auction paper, https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2023-UK/papers/EFMA%202023_stage-4455_question-Full%20Paper_id-228.pdf.
Mechanism: index and benchmark funds trading AT the close move the auction price, and the next open is "sticky" toward the pre-auction value.
Gross: **21–43 % of the closing-auction return reverses the next day (NYSE vs Nasdaq)**. Price impact of last-15-min imbalances ≈ 83 % reversed over 3–5 days (seen in a search summary; exact attribution to each paper still to be checked).
Frequency: daily, many names (filter on |auction return| or |imbalance|). Data: the Nasdaq/NYSE imbalance feeds are public in real time after 15:50 but historical files are NOT free [m]. The auction return itself is free (15:59 bar vs official close) from Alpaca bars. Cost: MOC entry, MOO exit, so no quoted spread. R% is set by the auction move.
Out of regime: [m].
Prior: MEDIUM.

**B2. Quarter-end mark-up ("leaning for the tape"), small caps**
Source: Carhart, Kaniel, Musto, Reed, JF 57 (2002) 661–693, https://www.ssrn.com/abstract=309177; coded replication: https://paperswithbacktest.com/strategies/window-dressing.
Mechanism: fund managers buy holdings in the last minutes of the quarter to inflate NAV, and the price reverts the next morning.
Gross: **Russell 2000 +0.28 % on the quarter's last day and −0.10 % the next day over 152 quarter-ends; 1988–99 +0.67 % / −0.22 %**. 80 % of funds beat the S&P on the year's last day, 37 % the next day (seen).
Frequency: 4 events/yr, so it **fails 3 fills/week alone** (a stackable sleeve). Data: free. Cost: fine for ETF or basket auction fills.
Out of regime: decayed after the 2000s SEC scrutiny [m]. Duong–Meschke document the "rise and fall of portfolio pumping" (https://www.cfr-cologne.de/download/researchseminar/WS0708/DuongMeschkePumping.pdf).
Prior: STRONG on history, WEAK on today's size.

**B3. Month-end pension and balanced-fund rebalancing (front-run the predictable side)**
Source: Harvey, Mazzoleni, Melone, "The unintended consequences of rebalancing", NBER WP 33554 (2025), https://ideas.repec.org/p/nbr/nberwo/33554.html.
Mechanism: calendar- and threshold-based rebalancers sell the month's equity winner in the last days of the month, and front-runners profit.
Gross: when stocks are overweight, equity returns are lower around rebalancing dates. Aggregate cost to funds ≈ 8 bps/yr, about $16 B (seen). Per-event SPY return differential ≈ 15–20 bps [m].
Frequency: monthly, index-level (SPY/IWM/TLT), so it fails cadence alone. Data: free. Cost: liquid ETFs at 1–2 bps.
Prior: MEDIUM (one NBER WP, 2025, practitioner-acknowledged).

**B4. Turn-of-month liquidity demand ("dash for cash")**
Source: Etula, Rinne, Suominen, Vaittinen, RFS 33(1) (2020) 75–111, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2528692.
Mechanism: month-end cash needs force selling before the month turns, and prices rebound after.
Gross: equity returns are temporarily depressed before month-end payment dates and high around the turn (seen qualitatively). Magnitude [m].
Frequency: monthly. Prior: STRONG on existence, low frequency.

**B5. Option-expiration-week effect in option-active large caps**
Source: Stivers & Sun, JBF 37 (2013) 4226–4240, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1571786; Quantpedia https://quantpedia.com/strategies/option-expiration-week-effect.
Mechanism: option market makers unwind hedges into the third Friday.
Gross: **28 option-active S&P 100 names averaged +0.45 %/week in OPEX weeks vs +0.12 % in other weeks, 1996–2008** (seen).
Frequency: monthly, 1 week. Data: free. Cost: liquid, fine.
Out of regime: weekly options (2010+) and 0DTE dilute the monthly concentration, which predicts decay [m].
Prior: MEDIUM.

**B6. Expiration-day pinning toward high-OI strikes**
Source: Ni, Pearson, Poteshman, "Stock price clustering on option expiration dates", JFE 78 (2005) 49–87, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=519044.
Mechanism: delta-hedge rebalancing by long-gamma market makers, plus some manipulation.
Gross: **returns of optionable stocks are altered by ≥ 16.5 bps on average on expiration dates** (seen); sample 1996–2002.
Frequency: weekly (weekly expiries now), dozens of names. Data: strikes and OI. Current OI comes from Alpaca contracts; historical OI is not free [m]. Cost: a 16.5 bp average vs 5–10 bps is thin.
Prior: STRONG on existence, MEDIUM on tradability.

**B7. Market-level expiry reversal**
Source: Baltussen, Terstegge, Whelan (2023), cited in A5 [m: title, numbers].
Mechanism: overnight and intraday market reversals around index option expiries. Index-level and monthly.
Prior: WEAK until read.

**B8. Pre-FOMC drift**
Source: Lucca & Moench, JF 70(1) (2015), https://www.emanuelmoench.com/documents/Lucca_Moench(JF2015).pdf.
Gross: **+49 bps in the 24 h before scheduled FOMC, 1994–2011; disappeared after 2015** (seen).
Prior: REJECT (decayed; listed so it is not re-proposed).

**B9. Pre-holiday and turn-of-month index seasonals**
Source: Ariel (1990) [m]; Lakonishok–Smidt (1988) [m]; McConnell–Xu (2008) [m].
Gross: [m]. Frequency: ~20 days/yr, index-level. Prior: WEAK-MEDIUM. Stackable only.

---

## C. Index reconstitution and fund-flow mechanics

**C1. S&P 500 additions, post-announcement fade (short)**
Source: Greenwood & Sammon, "The disappearing index effect", https://www.hbs.edu/ris/download.aspx?name=ssrn-4294297.pdf; event study https://blog.tradingstudio.finance/sp500-additions-event-study/.
Mechanism: arbitrageurs front-run and index demand is absorbed. Recent additions overshoot and revert.
Gross: abnormal return 7.39 % (1990s) → 5.16 % (2000s) → **0.83 % (2010–20)**. Median 2011–21 −0.04 %. 2016–20 "upward additions" −2.48 % over 3 days. **T+5 mean −0.80 % (t −3.38)** (seen).
Frequency: ~20–30 adds/yr, so it fails cadence alone. Data: S&P press releases, free. Cost: liquid.
Prior: MEDIUM (a decayed effect turned reversal; small n).

**C2. Russell reconstitution (rank-day predicted adds and deletes)**
Source: Madhavan, FAJ 59(4) (2003), https://www.hillsdaleinv.com/uploads/The_Russell_Reconstitution_Effect,_Ananth_Madhaven,_Financial_Analysts_Journal,_JulyAugust_2003,_Pages_51-64.pdf; Onayev (EFMA 2008) on predatory trading.
Gross: **long adds / short deletes earned +14.94 % in June, 1996–2002**. R1000 adds +10.9 %, R2000-Growth deletes −6.6 % (seen). The trade has compressed since; deletions earn more than additions (seen).
Frequency: once a year (semi-annual from 2026 [m]), hundreds of names. Data: free bands from market caps [m]. Prior: STRONG history, WEAK today. Stackable.

**C3. Leveraged single-stock ETF close rebalancing (TSLL, NVDL, MSTU, …)**
Source: Zhao, "Preying on leveraged ETFs", arXiv 2608.03703 (2026), https://arxiv.org/abs/2608.03703. Context: Baltussen et al. JFE 2021 (index level, closed for us).
Mechanism: an L× daily ETF must buy L(L−1)·AUM·r of the underlying at the close (mechanical, size known in advance), and the price impact reverts.
Gross: in Korea, LETF-tracked stocks reversed **≈ 75 % of the first-day response by the next close** (seen). There is no US single-name number; that is the open gap.
Frequency: daily on ~20–40 underlyings with large LETF AUM. Data: AUM and leverage from issuer sites (free), bars free. Cost: liquid underlyings at 5–10 bps. MOC/MOO legs.
Out of regime: none. Not the closed "leveraged single-stock ETF decay" (that held the ETF short). This trades the UNDERLYING against the predictable flow.
Prior: WEAK-MEDIUM (mechanical flow; evidence foreign).

**C4. ETF premium/discount mean reversion (illiquid ETFs)**
Source: Petajisto, "Inefficiencies in the pricing of exchange-traded funds", FAJ 73 (2017), http://www.petajisto.net/papers/etf28.pdf.
Mechanism: authorised-participant arbitrage is slow in illiquid or international ETFs.
Gross: prices fluctuate in a band of ~200 bps around NAV (~100 bps on average). Active strategies earn "substantial abnormal returns before transaction costs" (seen; the number is [m]).
Data: intraday iNAV/NAV is not in Alpaca; daily NAV is on issuer sites (free scrape). Cost: illiquid-ETF spreads are often 20–100 bps. Prior: MEDIUM, implementability WEAK.

**C5. Closed-end fund discount mean reversion**
Source: Kohl, SSRN 2294410, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294410; Pontiff (1995) [m].
Gross: strategies earn **> 10 %/yr abnormal; one mean-reversion spec 18.2 %/yr, Sharpe 1.92; net-of-cost 7.4–14.6 %/yr, of which discount reversion is 2.2–9.3 %** (seen in a search summary).
Frequency: ~400 US CEFs, holding weeks. Data: daily NAV from CEFConnect (free scrape), bars free. Cost: CEF spreads 10–50 bps.
Out of regime: discounts blow out in crises (2008, 2020, 2022), the classic liquidity-provision tail.
Prior: MEDIUM-STRONG (long literature). R% fits a swing book, not the intraday engine.

**C6. ETF → constituent intraday lead-lag**
Source: Cartea, Cucuringu, Shi, "Intraday lead-lag relationships in idiosyncratic stock returns", SSRN 6811380, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6811380; Ernst, "Stock-specific price discovery from ETFs", https://www.mit.edu/~ternst/docs/jmp.pdf.
Gross: lead-lag is strong and persistent intraday and vanishes at daily frequency (seen). A Granger claim of sector-ETF minute t → thin-stock minute t+3 is in a practitioner memo (seen; WEAK). Bps [m].
Cost: a minute-scale edge is usually below the spread on thin names. Prior: MEDIUM existence, WEAK net.

---

## D. Retail flow and attention

**D1. Retail marketable order imbalance from sub-penny SIP prints (BJZZ)**
Source: Boehmer, Jones, Zhang, Zhang, "Tracking retail investor activity", JF 76(5) (2021) 2249–2305, https://ideas.repec.org/a/bla/jfinan/v76y2021i5p2249-2305.html. Measurement critique: Barber et al., arXiv 2403.17095, https://arxiv.org/pdf/2403.17095.
Mechanism: wholesaler price improvement prints retail at sub-penny prices. Retail flow is persistent and (in this sample) informed.
Gross: **stocks with net retail buying beat those with net selling by ≈ 10 bps over the next week**; less than half comes from flow persistence (seen). Sample 2010–2015 [m].
Frequency: weekly cross-section, thousands of names. Data: SIP trades with sub-penny prices from Alpaca's trades endpoint, free but heavy. Cost: **10 bps/week fails** on its own; a candidate filter for D2/E.
Prior: STRONG on the measure, WEAK as a standalone money book.

**D2. Fade retail herding (short the top retail-bought names, 1–5 days)**
Source: Barber, Huang, Odean, Schwarz, "Attention-induced trading and returns: evidence from Robinhood users", JF 77(6) (2022) 3141–3190, https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13183.
Mechanism: attention-driven herding on app "top movers" lists inflates prices, and later buyers pay.
Gross: **top stocks bought by RH users each day: −4.7 % abnormal over 20 days; −3 % over 5 days (−6 % for the most extreme herding)** (seen). Sample 2018–2020 (Robintrack) [m].
Frequency: daily events, 1–5/day. Data: Robintrack is dead. Proxy = D1 sub-penny buy-imbalance spike on top-gainer names (free, heavy). Cost: small caps at 15–40 bps vs a 300 bp 5-day gross survives if the proxy holds. Borrow: 5,209 shortable names, but herded names are often HTB.
Out of regime: 2018–20 only [m].
Prior: MEDIUM-STRONG (JF), proxy risk HIGH.

**D3. Search-volume attention (Google SVI)**
Source: Da, Engelberg, Gao, "In search of attention", JF (2011) [m].
Gross: abnormal SVI predicts higher prices over 2 weeks and reversal within a year [m].
Data: Google Trends, free but rate-limited, sample-normalised. Prior: MEDIUM, implementability WEAK.

**D4. No-news extreme moves reverse, news moves drift (small caps)**
Source: Chan, JFE 70(2) (2003) 223–260, https://ideas.repec.org/a/eee/jfinec/v70y2003i2p223-260.html; Tetlock, "All the news that's fit to reprint", RFS (2011), https://business.columbia.edu/sites/default/files-efs/pubfiles/3099/Tetlock%20Fit%20to%20Reprint%2010%2010.pdf.
Mechanism: an uninformed liquidity shock reverts and an informed one does not. Stale-news reversal is more than twice as large in small firms and larger in names with many small trades (seen).
Gross: drift after bad news; reversal after extreme no-news moves, mainly in small illiquid stocks (seen). Monthly magnitudes ~1–2 % [m].
Frequency: daily, tens of names. Data: Alpaca News API (Benzinga, from ~2015 [m]) free; bars free. Cost: small caps at 15–40 bps vs a %-scale gross.
Out of regime: Nagel predicts larger in stress (2008, 2020, 2022). Not the closed reversal: that was UNCONDITIONAL on the liquid top-1000, while this is CONDITIONED on no news, below the top-1000.
Prior: STRONG (two top-journal papers plus D5).

**D5. Nagel liquidity-provision reversal in SMALL caps (VIX-scaled)**
Source: Nagel, "Evaporating liquidity", RFS 25(7) (2012) 2005–2039, https://www.nber.org/system/files/working_papers/w17653/w17653.pdf.
Mechanism: the reversal return is the price of liquidity, highest for small, illiquid, high-volatility names and when VIX spikes (seen).
Gross: a daily reversal at tens of bps/day in high-VIX periods [m]. Khandani–Lo document the decline over time, with the effect concentrated in small illiquid stocks (seen).
Frequency: daily, holding 1–5 days. Data: free. Cost: the crux; small-cap round trips of 15–40 bps.
Note: the brief explicitly opens this (we only tested the liquid top-1000).
Prior: STRONG on existence.

**D6. Odd-lot / small-trade share as a retail-intensity conditioner**
Source: the Tetlock (2011) small-trade result (seen); odd-lot literature [m].
Use: a conditioner on D4/D5, not a standalone. Prior: WEAK standalone.

---

## E. Short-selling flows and constraints

**E1. FINRA daily short-volume ratio (off-exchange), liquid names, 1–5 days**
Source: Vortex Capital, "The 50 % short volume myth" (Oct 2026; 7,505 stocks, 1,200 sessions 2022-01 → 2026-10, 5.7 M stock-days), https://www.vortexcapitalgroup.com/insights/the-50-percent-short-volume-myth-what-the-daily-short-sale-file-actually-predicts; academic precedent Diether, Lee, Werner, "Short-sale strategies and return predictability", RFS (2009), https://diether.org/papers/short_strategies.pdf.
Mechanism: informed and liquidity-providing short sellers sell into temporary buying pressure, and high relative shorting precedes low returns.
Gross (> $100 M/day tier): **top-decile minus bottom-decile SVR −26.7 bp over 5 sessions (t −4.1); next session −12.4 bp (t −6.4)**. By year: **2022 −40, 2023 −9, 2024 −18, 2025 −40, 2026 YTD −26, "never positive"**. The $10–100 M tier gives −13.5 bp (t −2.4). The $1–10 M tier is nothing. Below $1 M it flips sign (+36 bp, mostly next-day bounce). DLW: 1 SD of relative short sales → −2.9 bps next day (seen).
Frequency: daily decile on the liquid tier, holding 1–5 days. Data: the FINRA Reg SHO daily files, free and T+1. Cost: a 27 bp weekly spread "is inside the cost of running it" (the source's own caveat). Liquid legs at 5–10 bps each leave a thin net.
Out of regime: **2022 was the strongest year (−40)**. Caveats: the file covers ~32 % of volume (off-exchange); results are gross and exclude bounce.
Prior: MEDIUM-STRONG (RFS precedent plus a 2022–26 out-of-sample replication by a third party).

**E2. Rule 201 (SSR) trigger discontinuity: buy just-triggered −10 % names**
Source: OFR Working Paper 23-08, "Are short-selling restrictions effective?", https://www.financialresearch.gov/working-papers/files/OFRwp-23-08-are-short-selling-restrictions-effective.pdf (text extracted). Counter-evidence: Cato Regulation (2011), https://www.cato.org/sites/cato.org/files/serials/files/regulation/2011/7/regv34n2-7.pdf.
Mechanism: the alternative uptick rule throttles aggressive shorting for the rest of the day and the next day.
Gross: **restricted stocks (intraday low just below −10 %) have subsequent daily returns ≈ 35 bps higher than stocks just above −10 % in the same hour**. Short volume is 8 % lower, seller-initiated volume 5 % lower. Placebo cutoffs at −5 % and −15 % show nothing (seen). The early (2011) Cato number is next-day close return after −10 % days, +0.46 % before vs −0.46 % after the rule (seen), so the unconditional rebound is NOT positive.
Frequency: dozens of triggers/day. Data: free 1-min bars (a −10 % cross is observable live). Cost: 35 bps relative vs 15–40 bps; R% large. The RD is a DIFFERENCE, so the tradable leg must be the triggered name hedged with a near-miss name, or the level is unknown.
Prior: MEDIUM (clean RD design, one paper).

**E3. Days-to-cover**
Source: Hong, Li, Ni, Scheinkman, Yan, "Days to cover and stock returns", NBER w21166, https://www.nber.org/system/files/working_papers/w21166/w21166.pdf.
Mechanism: DTC = short interest / turnover, the cost of exiting a crowded short. High-DTC names are the most overpriced.
Gross: **EW L/S 1.19 %/month (t 6.67), Sharpe 1.3** (seen); sample 1988–2012 [m].
Frequency: semi-monthly FINRA short interest (free), monthly rebalance. Cost: low turnover, fine. The short leg is HTB.
Prior: STRONG (published, replicated by Alpha Architect, https://alphaarchitect.com/2015/07/30/betting-against-days-to-cover).

**E4. Short-selling risk / borrow-fee spikes**
Source: Engelberg, Reed, Ringgenberg, "Short-selling risk", JF (2018) [m].
Gross: high short-fee-variance names underperform [m]. Data: borrow fees are forward-only (iBorrowDesk scrape); Alpaca gives an ETB/HTB flag only [m]. Prior: MEDIUM, implementability WEAK (no history).

**E5. Squeeze mechanics (high SI + rising borrow + positive catalyst, long)**
Source: practitioner anecdote; the academic evidence (E3, E4) says high-SI names underperform on average.
Prior: WEAK. Listed only to record that the documented prior points the other way.

---

## F. Option-market signals and dealer positioning

**F1. Option-to-stock volume ratio (O/S), weekly**
Source: Johnson & So, "The option to stock volume ratio and future returns", JFE 106 (2012), https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1624062; follow-up: "Why does the option to stock volume ratio predict stock returns?", JFE (2016), https://www.sciencedirect.com/science/article/abs/pii/S0304405X16000167.
Mechanism: short-sale costs push informed negative views into options, so high O/S = bad news coming.
Gross: **lowest-O/S decile beats highest by 0.34 %/week (19.3 %/yr); 1.47 %/month risk-adjusted**. Short-lived (weekly) (seen); sample 1996–2010 [m].
Frequency: weekly sort over ~2,000+ optionable names. Data: option volume from Alpaca 1-min option trade bars (free; history only from ~2024-02 [m]) plus stock volume (free). Cost: 34 bps/week L/S vs 2 × 5–10 bps survives at the published size.
Out of regime: Alpaca history covers no 2022. Retail call buying since 2020 changes what O/S means; it may weaken or flip.
Prior: STRONG (JFE, two papers).

**F2. Pre-earnings long straddle (3 days before → announcement)**
Source: Gao, Xing, Zhang, "Anticipating uncertainty: straddles around earnings announcements", JFQA 53(6) (2018) 2587–2617, https://ideas.repec.org/a/cup/jfinqa/v53y2018i06p2587-2617_00.html.
Mechanism: investors underestimate the uncertainty build-up, and implied vol runs up into the event.
Gross: **ATM straddle from t−3 to the announcement earns +3.34 %**, larger for small, volatile, high-kurtosis, high-cost names (seen).
Frequency: earnings season, tens per week. Data: option trades free (Alpaca); historical option QUOTES are not free, so entry prices are trade prints only. Cost: option spreads of 2–10 % of premium on small names can exceed 3.34 %. The edge is where costs are highest, a red flag.
Not the closed "pre-earnings run-up" (that was a stock drift); this is a volatility trade.
Prior: MEDIUM.

**F3. Single-stock end-of-day momentum under negative dealer gamma**
Source: "End-of-day momentum in the cross-section and option hedging" (EFMA 2021), https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2021-Leeds/papers/EFMA%202021_stage-2049_question-Full%20Paper_id-237.pdf; Ni, Pearson, Poteshman, White, RFS 34(4) (2021) 1952–1986.
Mechanism: negative aggregate dealer gamma in a stock means hedging chases the day's move into the close, and the effect reverts at the next open (seen). Positive gamma means end-of-day reversal (seen).
Gross: bps [m]. A5 finds the cross-sectional effect overall is REVERSAL, not explained by gamma, so this holds only in the negative-gamma subset.
Data: needs per-stock OI and a dealer-side assumption. Historical OI is not free; current OI comes from the Alpaca contracts endpoint, so collect forward.
Not the closed index intraday momentum: this is single-name and gamma-conditioned.
Prior: MEDIUM-WEAK.

**F4. Volatility spread / IV smirk (put-call parity deviations)**
Source: Cremers & Weinbaum, JFQA (2010) [m: ≈ 50 bps/week L/S]; Xing, Zhang, Zhao, JFQA (2010) [m: smirk ≈ 10.9 %/yr].
Data: needs historical option QUOTES (OPRA). That is Databento, so the owner's GO is needed. Forward collection from Alpaca snapshots is possible.
Prior: STRONG on paper, BLOCKED on data.

**F5. 0DTE flow effects on single names**
Source: Adams et al., "Do S&P 500 options increase market volatility? Evidence from 0DTEs", https://www.jean-sebastienfontaine.com/papers/0dte-options-volatility.pdf (index level; no single-name evidence found).
Prior: WEAK (no documented single-name gross edge).

---

## G. Corporate events and filings (EDGAR, free)

**G1. Opportunistic insider purchases (Form 4), entered on filing**
Source: Cohen, Malloy, Pomorski, "Decoding inside information", JF (2012), https://static1.squarespace.com/static/5f6349cae367a31c2bc8d676/t/5f74a5522faa5a39493dd96d/1601480022374/pomalco.pdf; microcap update arXiv 2602.06198, https://arxiv.org/html/2602.06198v1.
Mechanism: insiders who break their routine calendar trade on information. Routine trades carry zero information.
Gross: **opportunistic trades: 82 bps/month value-weighted abnormal; routine ≈ 0** (seen). Filing-day and 5-day reaction for open-market cluster buys ≈ 1–2 % in small caps [m].
Frequency: open-market purchases ~20–60/week market-wide [m]; opportunistic clusters fewer. Data: EDGAR Form 4 XML, free, filed within 2 business days. Cost: small caps at 15–40 bps vs a %-scale 1-month gross survives. R% fine.
Out of regime: [m]. Not PEAD.
Prior: STRONG (JF plus a long literature).

**G2. IPO lock-up expiry (short)**
Source: Field & Hanka, JF (2001) [m]; 2004–2023 sample of 1,914 lockups (seen in a search summary; source to be pinned); IHS Markit note https://cdn.ihs.com/www/blog/commentary/pdf/Shorting-IPO-lockup-expiration.pdf.
Mechanism: a known supply shock with insufficient anticipation; VC-backed names show it most.
Gross: **three-day abnormal −1.5 % with a permanent 40 % volume increase (classic); −0.7 % over 3 days in 2004–2023 (1,914 lockups)** (seen).
Frequency: 1–5/week in active IPO years, 0–1 in dead years. Data: EDGAR S-1/424B4 (lockup term), free. Cost: borrow on recent IPOs can be HTB. 70 bps gross over 3 days vs 15–40 bps is thin.
Prior: MEDIUM (decayed).

**G3. Dividend month premium / ex-day effect**
Source: Hartzmark & Solomon, "The dividend month premium", JFE (2013), https://www.sciencedirect.com/science/article/abs/pii/S0304405X13000585; "Predictable price pressure", https://www.nber.org/system/files/working_papers/w30688/w30688.pdf.
Mechanism: dividend-seeking investors buy before the dividend and reinvestment flows land on payment.
Gross: **long predicted-dividend-month names vs the same names in other months +37 bps/month abnormal (up to 53–115 bps in some specs); actual announcement day +12 bps; ex-day +26 bps; announcement → ex +17 bps; −72 bps over the 40 days after the ex-day; +15 bps more in recessions** (seen).
Frequency: thousands of payer-events per quarter, i.e. tens of names a day. Data: Alpaca corporate-actions API (free [m]) and EDGAR. Cost: liquid payers at 5–10 bps. A **two-sided** structure: long into ex-day, short or underweight for 40 days after.
Out of regime: higher in recessions and high volatility (seen).
Prior: STRONG (JFE, replicated in Germany and the Nordics).

**G4. Activist 13D filing reaction**
Source: Brav, Jiang, Partnoy, Thomas, JF (2008) [m: ~7 % abnormal −20..+20 days; ~2 % in the filing window]; Bebchuk et al. on pre-filing accumulation [m].
Mechanism: the market under-reacts to an activist stake. Real-time EDGAR SC 13D is free.
Gross after the public filing: [m], likely small. Frequency: ~5–10/week [m]. Prior: MEDIUM-WEAK post-filing.

**G5. Buyback announcement drift**
Source: Ikenberry, Lakonishok, Vermaelen (1995) [m]; post-2003 decline [m].
Horizon: years, not days. Prior: WEAK for this engine.

**G6. Customer–supplier momentum (economic links)**
Source: Cohen & Frazzini, "Economic links and predictable returns", JF (2008), https://pages.stern.nyu.edu/~afrazzin/pdf/Economic%20Links%20and%20Predictable%20Returns%20-%20Cohen%20and%20Frazzini.pdf; update "Customer momentum", arXiv 2301.11394, https://arxiv.org/pdf/2301.11394.
Mechanism: limited attention to linked firms. A customer's news moves the supplier with a lag.
Gross: **customer-momentum L/S > 150 bps/month alpha**; customers average 20 % of supplier sales (seen).
Frequency: monthly, or event-driven (big customer move → supplier next day). Data: 10-K major-customer segment disclosure (free, hard text parse). Cost: fine at monthly turnover.
Prior: STRONG (JF), decay [m].

**G7. FDA PDUFA calendar run-up and fade**
Source: practitioner anecdote only [m]. Prior: WEAK.

**G8. Stock-split announcement drift**
Source: Ikenberry & Ramnath (2002) [m]. Horizon: months. Prior: WEAK for this engine.

**G9. After-hours earnings reaction → next-day open-to-close**
Source: Jiang, Likitapiwat, McInish (2012) [m].
Mechanism: thin after-hours price discovery is incomplete at the open.
Overlap with the closed PEAD/reaction work: high. Run it only if the PEAD autopsy did not already look at open-to-close on day 1.
Prior: WEAK-MEDIUM.

---

## H. Volume, attention and seasonality in the cross-section

**H1. High-volume return premium (volume shock without an extreme return)**
Source: Gervais, Kaniel, Mingelgrin, JF 56(3) (2001) 877–919, https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00349; cross-country evidence (seen in results).
Mechanism: visibility. A volume shock raises the investor base and demand.
Gross: **weekly abnormal high-volume premiums up to 0.50 %/week, 1962–2005**. Stronger when the volume shock is NOT accompanied by an extreme return; not explained by risk or announcements (seen).
Frequency: weekly sort, thousands of names. Data: free. Cost: 50 bps/week gross covers 5–10 bps liquid, marginal on small caps.
Note: not the closed "ignition/momentum bursts" (those required a price burst); this explicitly selects volume WITHOUT a price move.
Prior: STRONG on history; post-2005 [m].

**H2. Calendar-month return seasonality**
Source: Heston & Sadka, JFE (2008), https://www.sciencedirect.com/science/article/abs/pii/S0304405X0700195X; Keloharju, Linnainmaa, Nyberg (2016).
Gross: same-calendar-month winners beat losers for up to 20 years; **≈ 13 %/yr** (seen).
Frequency: monthly. Data: free. Prior: STRONG. A slow sleeve, stackable with MOM, but check overlap with MOM first.

**H3. MAX / lottery-stock underperformance**
Source: Bali, Cakici, Whitelaw, JFE (2011) [m: ≈ 1 %/month L/S].
Mechanism: the retail lottery preference overprices names that just had a max-return day. Short-heavy. Prior: STRONG [m], a short-side sleeve.

**H4. 52-week LOW capitulation (long)**
Source: no peer-reviewed gross edge found this session; the George–Hwang literature is about highs (closed) [m].
Prior: WEAK.

---

## I. Cross-asset lead-lag and relative value

**I1. Distance-method pairs (daily) in liquid names**
Source: Gatev, Goetzmann, Rouwenhorst, RFS (2006) [1.4 %/month seen in summary]; Do & Faff (2010, declining); Zhu (Yale 2024) replication 6.2 %/yr, https://economics.yale.edu/sites/default/files/2024-05/Zhu_Pairs_Trading.pdf; **US large caps 2014-06 → 2024-05: no portfolio significant, −0.79 % to +0.44 %/yr**, https://github.com/Softpandaa/distance-pairs-modern (seen).
Prior: MEDIUM-NEGATIVE (the classic spec is dead in large caps).

**I2. Sector ETF / large-cap → laggard small-cap (industry lead-lag)**
Source: Hou, RFS (2007) [m: big firms lead small within the industry]; C6 for the intraday form.
Prior: MEDIUM existence, WEAK net.

**I3. Commodity → producer equities (GLD→GDX/juniors, USO→E&P, TLT→KRE)**
Source: no peer-reviewed gross edge found this session.
Prior: WEAK. Listed for completeness.

**I4. BTC → crypto equities (COIN, MSTR, miners)**
Source: arXiv 2505.14655, https://arxiv.org/html/2505.14655v1: "no consistent linear lead-lag pattern" between BTC and treasury-strategy equities (seen).
Prior: WEAK (refuted at the linear level).

**I5. ADR / foreign-close lead-lag**
Source: older ADR-arbitrage literature [m]; the open prices in foreign information.
Prior: WEAK.

---

## J. ETF short-term mean reversion (practitioner)

**J1. RSI(2) / IBS pullback-buy in index and sector ETFs (SPY, QQQ, IWM, XL*)**
Source: QuantifiedStrategies RSI-2 guide, https://www.quantifiedstrategies.com/rsi-2-strategy/; coded backtests: https://backtestify.com/strategies/connors-rsi2, https://www.loomiai.io/learn/rsi-2-mean-reversion-strategy/.
Mechanism: index-level liquidity provision. Index ETFs mean-revert at 1–5 days, while single stocks (closed) did not show it in the liquid top-1000.
Gross: **SPY / S&P 500 stocks 75–79 % win rate, 3–5 day hold; "broad indices" PF 2.45 over 406 trades; "slight decay 2015–2025 out of sample"** (seen; practitioner, unaudited).
Frequency: ~1–3 signals/week across ~15 ETFs. Data: free. Cost: 1–2 bps per ETF, fine. R% ≥ 1 %.
Out of regime: worked in 2008 [m]; 2022 bear rallies are the test.
Prior: WEAK-MEDIUM (practitioner, survivor-picked; no peer-reviewed replication found).

---

## Count

| Family | n |
|---|---|
| A Overnight/intraday clientele | 9 |
| B Auction / calendar / dealer flow | 9 |
| C Index and fund-flow mechanics | 6 |
| D Retail flow and attention | 6 |
| E Short-selling | 5 |
| F Options / dealer positioning | 5 |
| G Corporate events (EDGAR) | 9 |
| H Volume / seasonality | 4 |
| I Cross-asset lead-lag | 5 |
| J ETF mean reversion | 1 |
| **Total** | **59** |

Rejected on arrival (listed only to block re-proposal): A7 on cost, A9 and B8 decayed, I1 dead in large caps, I4 refuted, F5 with no evidence.

---

## (1) Top 10, ranked by gross evidence × frequency × cost survival × free-data implementability

1. **E1 FINRA short-volume ratio, liquid tier**: −26.7 bp / 5 d (t −4.1), never positive 2022–26 including 2022, free T+1 data, daily, liquid names. Weakness: gross ≈ cost.
2. **D4+D5 no-news extreme-move reversal in small caps (Chan/Tetlock/Nagel)**: %-scale gross in the literature, daily frequency, free news plus bars, opened by the brief. Weakness: small-cap cost; no-news classification quality.
3. **F1 O/S ratio weekly (Johnson–So)**: 34 bps/week L/S, JFE, two papers, free option volume. Weakness: Alpaca option history only from ~2024 [m], so no 2022.
4. **G3 dividend month / ex-day premium**: +37–53 bps/month, ex-day +26 bps, −72 bps post-ex, liquid payers, tens of events/day, free corporate actions.
5. **G1 opportunistic insider buys (Form 4)**: 82 bps/month VW, free EDGAR, real-time, small-cap gross ≫ cost.
6. **H1 volume shock without a price move**: up to 50 bps/week, free, thousands of names. Weakness: post-2005 decay unknown.
7. **E2 Rule 201 trigger RD**: +35 bps relative, daily dozens, free 1-min bars; needs a hedged (triggered vs near-miss) construction.
8. **D2 fade retail herding**: −3 % / 5 d (JF). Large gross, but the Robinhood data is dead and the sub-penny proxy is unproven and heavy to build.
9. **E3 days-to-cover**: 1.19 %/month (t 6.67), free semi-monthly SI; slow, and the short leg is HTB.
10. **A5 end-of-day reversal (losers at 15:30 → MOC)**: t > 10, 3.6–6.2 bps/day, robust since 1993 incl. large caps; survives only in the extreme tail with an auction exit.

## (2) Stage-1 rejection-only designs for the top 3 (free data; PREREG + FREEZE before scoring)

### #1 E1 FINRA short-volume ratio (test first)
* **Universe**: common stocks with 20-day median dollar volume > $100 M (point-in-time from Alpaca daily bars), plus the $10–100 M tier reported separately. Exclude ETFs, `^Z[A-Z]ZZT$`, and anything absent from `daily_bars`.
* **Dates**: TRAIN 2016-01 → 2020-12; VAL 2021-01 → 2023-12 (includes 2022); TEST 2024-01 → 2026-10 sealed. FINRA daily files are on the free CDN [m: back to 2009].
* **Signal**: SVR_t = ShortVolume / TotalVolume from the consolidated NMS file for day t, published after 18:00 ET [m]. It is known only after day t, so entry is at day t+1 OPEN (auction) and the exit at the t+5 close (MOC).
* **The number**: mean equal-weight bottom-decile-minus-top-decile return, open(t+1) → close(t+5), day-clustered / Newey-West SE, GROSS. Report it next to (a) the next-day open-to-close piece and (b) the per-fill cost from measured NBBO at the 09:30–09:35 minute.
* **MDE**: weekly-equivalent L/S SD ≈ 1.2 % [m]. On ~750 VAL days of overlapping 5-day portfolios (NW lag 5), SE ≈ 9 bps, so MDE (80 %, α 5 %) ≈ 25 bps / 5 d. That is at the published effect size, so VAL is adequately powered only if the effect is real at full size. State that up front.
* **Kill**: VAL gross L/S ≥ 0 or |t| < 2; TRAIN halves of opposite sign; ex-top-5 % of name-days ≤ 0; net of measured cost (two legs, auction fills) ≤ 0; or < 3 fills/week on the long-only bottom-decile version vs a sector-matched hedge.

### #2 D4/D5 no-news extreme-loser reversal, below the top-1000
* **Universe**: rank 1,001–3,000 by 20-day dollar volume (point-in-time), price ≥ $3, plus ADV ≥ $2 M.
* **Event**: close-to-close return ≤ −8 % (and a sector-ETF-residual version) on day t, with NO Alpaca News item for the symbol from 16:00 t−1 to 16:00 t. A news item = any headline tagged to the symbol, timestamps checked as causal.
* **Dates**: 2016–2020 TRAIN, 2021–2023 VAL (2022 inside), 2024–2026 TEST sealed.
* **Entry**: t+1 open (auction). Exit: t+1 close and t+3 close (both pre-registered; the primary is t+1 close).
* **The number**: mean GROSS return of no-news events minus news events (placebo = the same name on a random no-event day, chosen AFTER the event window), day-clustered SE. Also report the level of the no-news leg alone, VIX-tercile split (the Nagel prediction: larger in the high-VIX tercile), and the R% of a stop at 1 ATR.
* **MDE**: expect ~5–15 events/day, ~1,500/yr. With per-event SD ≈ 6 % and day-clustering (effective n ~ 600 days in VAL), MDE ≈ 2.8 × 6 %/√(600 × ~3 effective) ≈ 40 bps.
* **Kill**: no-news leg gross ≤ +40 bps on VAL, or not above the news leg; high-VIX tercile not the largest (mechanism fails); ex-top-5 % ≤ 0; measured-NBBO cost at the 09:30 auction ≥ gross.

### #3 F1 O/S ratio, weekly
* **Universe**: optionable common stocks with option volume on ≥ 15 of the last 20 days and stock ADV ≥ $5 M.
* **Data**: Alpaca option 1-min trade bars summed to daily contract volume × 100 / stock share volume. History from ~2024-02 [m]; check coverage ≥ 80 % before scoring (availability rail).
* **Dates**: Johnson–So is the TRAIN prior. Here VAL = 2024-02 → 2025-06 and TEST = 2025-07 → 2026-10 sealed. No 2022 is possible on free data; record that as a known gap. The out-of-regime check needs OPRA volume (Databento) and therefore the owner's GO.
* **Signal**: O/S over week w (Mon–Fri), formed Friday after the close; enter at Monday's open, exit at Friday's close.
* **The number**: GROSS weekly return, lowest-O/S decile minus highest, equal-weight, with a week-clustered t. Also report the long-only low-O/S leg vs a size/sector-matched benchmark.
* **MDE**: ~70 VAL weeks with a weekly L/S SD ≈ 1.5 % gives SE ≈ 18 bps and MDE ≈ 50 bps/week, above the published 34 bps. **Stage 1 can therefore only reject a large effect**. Say so; a pass needs the TEST weeks pooled after FREEZE.
* **Kill**: VAL L/S ≤ 0; ex-top-5 % ≤ 0; sign flips between halves; or the effect lives entirely in names with spreads > 30 bps.

## Recommendation: test E1 first
It is the only candidate with an independent third-party replication on 2022–2026 data. That window includes the 2022 regime that killed our other books, and the sign held every year. The data is free, T+1 and already daily (no heavy fetch). It lives in liquid names where cost is measured at 5–10 bps. Stage 1 can be built and scored in one session.

Its weakness, gross ≈ cost, is exactly what a rejection-only Stage 1 resolves fastest: if our own rebuild cannot clear the measured auction cost on VAL, it dies in a day. If it survives, it is a ≥ 3-fills/week market-neutral sleeve and stacks with MOM.

D4/D5 is the higher-gross second test, with more classification risk.
