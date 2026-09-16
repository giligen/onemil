# Topic B — Single-stock intraday strategies driven by stock SELECTION ("stocks in play")

Literature review, 2026-09-16. Scope: relative volume at the open, gap size, pre-market
volume, news/catalyst, short interest, float, lottery features; single-stock ORB;
gap-and-go / gap-fill; first-30-minutes -> rest-of-day continuation and reversal;
VWAP-anchored rules; "day trading for a living" profitability studies. 2022–2026 focus
plus the foundational papers each of them leans on.

Reading posture: skeptical. Our own 2026-09-15 clean-sheet study found NO net edge for
the 5-min ORB, the HOD-break, or the flag on >= 5%-range days after a 40-bps spread.
Section 3 explains why the strongest paper in this literature does not contradict that
result — it is measured in a different universe, with a stop the size of our spread,
and with a cost model 30–60x cheaper than ours.

Conventions. "Causal at entry" = every input to the selection and the entry is known at
the moment the order is placed. "R" = the stop distance of the paper in question (they
differ by an order of magnitude across papers — never compare R across papers without
converting to bps of price). Our cost anchor: median effective spread ~40 bps round
trip on $5+ US stocks whose day range is >= 5% (measured on our own fills).

Search budget note: 200 web searches were spent; every number below is either quoted
from the paper text (Zarattini-Barbon-Aziz, Baltussen-Da-Soebhag, Lou-Polk-Skouras,
the SEC DERA LULD paper and the Shanghai halts paper were read in full text) or from
the abstract / a named replication. Where I could only get an abstract I say so.

---

## 1. Paper-by-paper

### 1.1 Zarattini, Barbon, Aziz (2024) — "A Profitable Day Trading Strategy For The U.S. Equity Market"
SSRN 4729284 (Feb 16 2024), also SFI WP 24-98. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284

**The exact spec (from the paper text, Sections 2.1 and 4):**
- Universe: all NYSE + Nasdaq listings 2016-01-01 → 2023-12-31 (~7,000 names, CRSP
  membership incl. delisted, so survivorship-free; IQFeed 1-min bars, unadjusted).
- Daily eligibility filters (all known before the open):
  1. opening price > $5;
  2. 14-day average daily volume >= 1,000,000 shares;
  3. 14-day ATR > $0.50.
- Stocks-in-play filter (computed at 09:35 ET, causal):
  `RelativeVolume(t,j) = ORVolume(t,j) / mean_{i=1..14} ORVolume(t-i,j)` where
  ORVolume is the 09:30–09:35 volume. Require RelativeVolume >= 100% (>= 1.0x) and
  trade only the **top 20 by RelativeVolume** that day.
- Direction: the first 5-min candle's colour. Bullish (close > open) -> long only;
  bearish -> short only; doji -> no order.
- Entry: a **stop order at the 5-min high (long) / 5-min low (short)**, placed at
  09:35. Fill assumed AT the level (no slippage modelled).
- Stop-loss: **10% of the 14-day ATR from the executed entry**. That is the whole
  risk unit R (their BLDR example: ATR $5 -> stop $0.50).
- Target: none. Exit at the stop or at **16:00 ET** (hold to close).
- Sizing: shares such that a stop-out loses 1% of capital, **capped at 4x leverage**
  (FINRA day-trading margin). With 20 names and a 0.1-ATR stop the 4x cap binds
  and the 1% never does — realised risk per position is a few bps of equity.
- Costs: **commission only, $0.0035/share** (IBKR Pro tiered). No spread, no
  slippage, no borrow fee on the short half, no market impact. Start capital $25K.

**Results (Table 2, net of the $0.0035 commission):**

| Strategy | Total return | IRR | Vol | Sharpe | daily hit ratio | MDD | worst day | alpha | beta |
|---|---|---|---|---|---|---|---|---|---|
| ORB Base (all eligible stocks, no RV filter) | 29% | 3.2% | 6.6% | 0.48 | 41.4% | 13% | −0.8% | 3.3% | 0.01 |
| ORB + RelVol top-20 | **1,637%** | **41.6%** | 14.8% | **2.81** | 48.4% | 12% | −1.61% | **35.8%** | 0.00 |
| S&P 500 | 198% | 14.2% | 18.3% | 0.78 | 54.9% | 34% | −10.9% | 0 | 1 |

Per-trade PnL by relative-volume bucket (Figure 4, base filters, net of commission):
RV < 1.0x -> **−0.02R**; RV > 1.0x -> **+0.08R**; RV > 30x -> **+0.38R**.
Time frames (Table 3, all with RV >= 1 measured over the matching window): 5-min
1,637% / Sharpe 2.81; 15-min 272% / 1.43; 30-min 21% / 0.21; 60-min 39% / 0.40;
equal-weight COMBO 234% / 1.99. Best cumulative-R names: DDD, FSLR, NVDA, SWBI, RCL,
W, VIR, EXAS, ALK, TSLA, AMD, ADBE (per-name win ratio 17–24%). Worst: CMC, TRGP,
CSX, CNP, BJ.

**Causal-information check:** clean. ADV, ATR, price are T-1; RV and candle colour
are known at 09:35; the entry is a resting stop order. Nothing in the selection uses
end-of-day information. (This is better than our own >= 5%-range universe, which is
selected on the day's realised range — see 3.3.)

**Out-of-sample:** none in the paper. 2016–2023 is one in-sample block; the authors
argue "parameters were minimal and based on economic rationale". The genuine OOS
window is **2024-01 → today**, which is exactly our data window.

**Critiques:**
- Cost model. With ATR/price of 2–4% for these liquid names, R = 0.1 ATR is
  **20–40 bps of price**. The bucket edge of +0.08R is therefore ~2–3 bps per trade
  gross-of-spread; +0.38R at RV > 30x is ~8–15 bps. The effective spread on NVDA/AMD/
  TSLA/ADBE in 2016–2023 was 1–3 bps and the commission is ~1.4 bps round trip on a
  $50 stock, so the edge survives THERE. Any universe with a 10+ bps spread eats the
  whole thing; at our 40 bps the paper predicts a **loss of ~1R per trade**.
- Stop-order fills at the level with no slippage on a breakout of a stock that is
  top-20 in the whole market by relative volume — the exact moment queue-jumping and
  latency arbitrage are worst. The Brusco replication of the sister QQQ paper (1.2)
  found break-even at 2.2 cents/share on a ~$300 ETF (< 1 bp) — the same fragility
  applies here, only more so.
- Half the book is short (bearish first candle). Borrow availability and fees for
  the day's top-20-RV names are not modelled; hard-to-borrow names are exactly the
  ones in play.
- Reported hit ratio is DAILY portfolio hit ratio (48.4% of days positive), not the
  per-trade win rate, which the per-name tables put at ~17–24%. The book is a
  right-tail machine: hold-to-close on 0.1-ATR stops means most trades lose ~1R and a
  few make 5–15R.
- Leverage arithmetic: 20 names x 4x cap x (0.08–0.4R x 30 bps) ≈ 10–40 bps/day is
  the whole IRR. The 41.6% is a leverage number, not a per-trade-edge number.
- Concentration in a few high-beta retail names (Table 4) and one regime (2020–2021)
  cannot be excluded — no yearly table is given.
- Replications: QuantConnect (1,000 most liquid names, same rules) reports Sharpe
  2.40 vs 0.84 SPY, beta −0.04, and "68% of the 5–25-min / 500–1,500-universe grid
  beat the benchmark" — but with realistic costs the community threads report a
  large haircut, and the QC page quotes a ~17% win rate. No published independent
  replication with a spread model exists as of this review.

### 1.2 Zarattini & Aziz (2023) — "Can Day Trading Really Be Profitable?" (QQQ/TQQQ 5-min ORB)
SSRN 4416622. Independent replication: Brusco (2025), github.com/giovannibrusco/zarattini-2023-orb-qqq.
- Rule: at 09:35 enter in the direction of the first 5-min QQQ candle (no breakout
  wait); stop = opposite extreme of the candle (1R); target 10R (hit 2–3% of the time);
  else close at 16:00. 1% risk, 4x cap, $0.0005/share commission, **no slippage**.
- Paper: 1,795 trades, Sharpe 1.12, QQQ 676% / TQQQ 1,484% 2016–2023, win rate ~24%,
  ~+0.13R per trade. Replication: 1,775 trades, Sharpe 1.06, net PnL $138,639 at zero
  slippage → **$4,860 at 2 cents/share** (97% erosion); **break-even ≈ 2.2 c/share**.
  Bootstrap 95% CI on the filtered Sharpe [0.05, 1.41] overlaps buy-and-hold
  [−0.03, 1.47]; 76% of the NQ-filtered PnL came from 2022; negative in 2017, 2020,
  early 2023.
- Lesson for us: the per-trade edge of a first-candle ORB on the most liquid
  instrument in the world is < 1 bp. It is a leverage-and-frequency product with a
  single-regime dependence.

### 1.3 Zarattini & Aziz (2023) — "VWAP: The Holy Grail for Day Trading Systems"
SSRN 4631351. QQQ/TQQQ, 2018-01-02 → 2023-09-28. Long above VWAP, short below (trend
following on VWAP); QQQ 671% / MDD 9.4% / Sharpe 2.1 vs buy-and-hold 126% / MDD 37%.
Commissions "deducted" but not quantified; no slippage; no stop rule; no trade count.
Index-ETF only — no single-stock VWAP evidence in the academic literature that I could
find for 2022–2026. The retail "VWAP bounce / fade" statistics (e.g. a 2022
QuantConnect note: short at +2σ VWAP band 63% WR, long at −2σ 61% WR on 100 liquid
Nasdaq names) are not peer reviewed and carry no cost model.

### 1.4 Chague, De-Losso, Giovannetti (2020) — "Day Trading for a Living?"
SSRN 3423101. CVM records for **19,646 individuals** who began day-trading mini-Ibovespa
futures 2013–2015, followed to 2017. Of the 1,551 who persisted >= 300 days, **97% lost
money**; 0.4% earned more than a bank teller (US$54/day); the best earned ~$310/day with
a daily s.d. of $2,560. Fraction profitable by tenure: 1 day 29.8% → 2–50 days 15.5%
→ 51–100 8.9% → 101–200 6.8% → 201–300 5.4% → 300+ 3.0%. Persistence itself is
adverse selection: the longer you stay, the worse you are on average. Not a
strategy paper; a base rate for any discretionary intraday book.

### 1.5 Barber, Lee, Liu, Odean — Taiwan day traders
(a) "The Cross-Section of Speculator Skill: Evidence from Day Trading", J. Fin. Markets
2014 (Taiwan 1992–2006, ~450K day traders); (b) "Do Day Traders Rationally Learn About
Their Ability?" (2017/2020 WP). Findings: of ~277K day traders with > NT$600K activity,
~20% profit gross, **< 1% earn reliable positive abnormal returns net of fees**. The top
500 by prior-year rank go on to earn **49.5 bps/day gross, 28.1 bps/day net** on the
day-trading portfolio (Sharpe of the top group only 0.21); traders with a history of
losses earn −17.5 gross / −34.2 net bps/day. Over half of day-trading volume comes from
experienced traders with a history of losses. Costs assumed: 10 bps round-trip
commission + 30 bps sales tax. The persistence result is the only academic evidence
that a skilled minority exists; it is a population result, not a rule.

### 1.6 Jordan & Diltz (2003, FAJ) and Garvey & Murphy (2005)
US retail day traders at a direct-access broker (J&D: 324 accounts, 1998–1999; G&M:
1,386 traders, 3 months). J&D: ~twice as many lose as win; ~20% "more than
marginally" profitable; profits track the Nasdaq. G&M: about half profitable after
commissions; winners made $9.5M, losers lost $4.6M in the sample — a heavier right tail
than Taiwan/Brazil, on a 1999–2000 sample. Cheng et al. (Heliyon 2023) on Taiwan's
2016 relaxation: day trading is more profitable when VIX is high; institutions avoid
high-day-trade-ratio names, retail prefers them.

### 1.7 Barber, Huang, Odean, Schwarz (2022, JF) — "Attention-Induced Trading and Returns: Evidence from Robinhood Users"
J. Finance 77(6) 3141–3190; SSRN 3715077. Robintrack holdings 2018–2020. Robinhood
"herding episodes" (top stocks by daily user-count increase) are followed by
**−4.7% average 20-day abnormal returns**; outages cut trading in high-attention names.
Selection (user-count growth) is a T-1 signal, causal. Not intraday, but it is the
single best-identified estimate that retail-attention buying in exactly our universe
(cheap, volatile, in-the-news) is followed by negative returns. Implication: on the
day AFTER a retail-attention spike, the long side is negative-expectancy; catalyst
presence without institutional volume is a sell signal, not a buy signal.

### 1.8 Berkman, Koch, Tuttle, Zhang (2012, JFQA) — "Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open"
JFQA 47(4) 715–741; SSRN 1625495. US stocks 1996–2008. High-attention stocks (recent
extreme returns / volume / news) have **positive overnight returns followed by intraday
reversals** — the opening price is high relative to the rest of the day. The pattern is
strongest in hard-to-value, hard-to-arbitrage names and in high-sentiment periods; the
implicit cost of buying such a name at the open "frequently exceeds the effective half
spread". Baltussen-Da-Soebhag (1.10, Table 11) re-confirm it on 1993–2019: overnight
return loads +1.43 on the prior day's rest-of-day return for winners and, for losers,
(1.43 − 2.29) < 0, i.e. attention pushes the open up for BOTH extreme winners and
extreme losers. This is the academic form of "gap-and-fade".

### 1.9 Lou, Polk, Skouras (2019, JFE) — "A Tug of War: Overnight Versus Intraday Expected Returns"
JFE 134(1) 192–213. CRSP 1993–2013, > $5, ex bottom NYSE size quintile. Firm-level
**overnight returns continue overnight** (past-month overnight winner-minus-loser
decile: +3.47%/month overnight, t = 16.8) and **intraday returns continue intraday**
(+2.41%/month 3-factor intraday alpha, t = 7.7), with an offsetting **cross-period
reversal** (overnight winners lose −3.02%/month intraday; intraday winners lose
−1.77%/month overnight), persisting for years. Every one of 14 anomaly strategies earns
entirely overnight or entirely intraday. Interpretation: two clienteles (retail at the
open, institutions intraday). For us: a stock's history of *intraday* strength is the
right conditioning variable for an intraday-hold rule; overnight strength (gaps) is the
wrong one.

### 1.10 Baltussen, Da, Soebhag (2024) — "End-of-Day Reversal"
SSRN 5039009; Quantpedia Awards 2025 runner-up. US stocks 1993–2019, TAQ, ex bottom
10% NYSE size, cap-weighted regressions. Signal: ROD3 = return from prior close to
15:00. **Last-half-hour (15:30–16:00) return reverses ROD3**, t-stats > 10 in sorts.
Long-short quintile: **3.78 bps/day VW (9.5%/yr), 6.86 bps/day EW (17.3%/yr)**,
6-factor alpha 3.71 bps/day; **smallest size quintile 14.71 bps/day (t = 27.2)**,
largest quintile 3.41 bps/day. "> 20% annualised alpha gross". Driven by the LONG side
(intraday losers rally into the close); winners contribute ~0. Mechanism: attention-
induced retail buy-the-dip in the last half hour (three retail proxies agree) plus
short sellers covering losers into the close; NOT gamma hedging, NOT liquidity. Costs:
gross; the authors do not net a spread. At 40 bps our round trip is 6–10x the average
daily spread — only the extreme tail of ROD3 losers can clear it, and that tail is
exactly what our >= 5%-range cache contains (see H-B5).

### 1.11 Bogousslavsky (2021, JFE) — "The Cross-Section of Intraday and Overnight Returns"
JFE 141(1) 172–194. 30 years of US intraday returns. Anomalies accrue over the day in
radically different ways: several earn through the day and lose in the last 30 min
(mispricing correcting, then worsening at the close); size and illiquidity premia are
realised in the last 30 minutes; profitability and IVOL accrue gradually. Reinforces
1.9/1.10: the last half hour is a different market.

### 1.12 Barardehi, Bogousslavsky, Muravyev (RFS 2026) — "What Drives Momentum and Reversal? Evidence from Day and Night Signals"
SSRN 4069509. US 1926–2019. Portfolios formed on past **intraday** returns show
short-term reversal AND momentum without long-run reversal; portfolios formed on past
**overnight** returns show no momentum, only long-run reversal. Overnight returns are
news; intraday returns are trading; investors underreact to other investors' trades.
For us: continuation signals should be built from intraday (open-to-close) history,
never from gap history.

### 1.13 Heston, Korajczyk, Sadka (2010, JF) — "Intraday Patterns in the Cross-Section of Stock Returns"
JF 65(4); arXiv 1005.3535. US 2001–2005, 13 half-hours/day. Return continuation at
lags that are exact multiples of a day, out to 40 days; strongest in the first and last
half hours; ~1–3 bps per half hour for mid/large caps, > 5 bps for small caps; volume,
imbalance, spreads do not explain it. Gross of costs — a few bps per half hour is far
below any retail-accessible spread.

### 1.14 Gao, Han, Li, Zhou (2018, JFE) — "Market Intraday Momentum"; Chen, Haboub, Khan, Mahmud (2024, RQFA) — cross-section version
Gao et al.: SPY 1993–2013, first half-hour return (from prior close) predicts the last
half-hour return; stronger on volatile / high-volume / macro-news days. Index-level;
Brusco-style replications find it regime-dependent. Chen et al. (Rev. Quant. Fin. Acc.
2024): the cross-sectional analogue holds in UK and Brazil, is weak in China A-shares
(retail-dominated) and strengthens after MSCI inclusion — **institutional** flow, not
retail, drives intraday momentum in single stocks. Implication: in a retail-dominated
small-cap universe the "first-30-min predicts rest of day" prior is weak-to-absent;
Baltussen et al. (1.10) actually find the opposite sign at the stock level in the last
half hour.

### 1.15 Brogaard, Han, Kim (2024) — "Intraday Residual Reversal in the U.S. Stock Market"
SSRN 4731947. Factor-model residuals over intraday intervals; buy negative-residual /
sell positive-residual stocks next interval: **162.3% annualised gross** — explicitly
"the returns to liquidity provision to the transitory component". This is the market-
maker's edge measured as a strategy; it is not accessible at 40 bps and it is the
mirror image of any breakout rule (breakouts are the positive residuals it shorts).

### 1.16 Bahcivan, Dam, Gonenc (2023/2025, J. Behav. Exp. Finance) — "Dark Side of the Day: Overnight Price Jumps and Short-Term Return Predictability"
SSRN 4335622 / 5648748. **9,283** NYSE/AMEX/Nasdaq stocks. Overnight jump returns
(positive AND negative) **negatively predict next-day returns** (t ≈ −11 positive
jumps, −4 negative); reversal is sharp on the next trading day and transient (the
extreme-decile contrarian book earns a 0.6% risk-adjusted *loss* over a month —
i.e. the reversal is a one-day event). Robust in distress sub-periods. This is the
best single-stock estimate of "gap-fill": a large overnight gap on average partially
reverses the same day — direct counter-evidence to gap-and-go as a base case.

### 1.17 Plastun, Sibande, Gupta, Wohar (2020, NAJEF) — "Price gap anomaly in the US stock market: the whole story"
DJI / S&P 500 / Nasdaq indices 1928–2018. On gap days prices tend to continue in the
gap direction; "gaps get filled" is NOT supported; momentum after the gap is temporary.
Index-level only; no cost model; contradicts 1.16 at the single-stock level, which is
the level that matters — the two are reconciled by 1.8/1.16: index gaps are
macro-news (continue), single-stock attention gaps are retail pressure (reverse).

### 1.18 Cartea, Cucuringu, Jin, Wilson (2025) — "Volume Shocks and Overnight Returns"
SSRN 5156605. US stocks; unexpected intraday volume spikes predict **higher close-to-
open returns and NO intraday (open-to-close) effect**; ML (TabNet) long-short ≈ 18%/yr,
Sharpe ≈ 1.1 gross; no frictions modelled (a blog replication on Nasdaq biotech with
costs: Sharpe 1.5). Foundational: Gervais, Kaniel, Mingelgrin (2001, JF) "high-volume
return premium" (weekly; larger when the volume shock comes WITHOUT a price move);
Li, Yin, Zhao (2024, Eur. J. Finance): extreme abnormal-volume deciles earn positive
short-run returns that turn negative as volume mean-reverts. Selection (volume shock)
is causal at the close. Message: the volume premium is an OVERNIGHT premium — the
opposite of an intraday hold.

### 1.19 Ahn, Fan, Noh, Park (2024, R&R JFQA) — "Retail Ebb and Flow and the Overnight–Intraday Return Gap"
SSRN 4752520. Korea, exhaustive investor-type flows. Retail trading intensity CAUSES
the overnight > intraday gap (instrumented by nominal share price): retail net-buys at
the open and net-sells at the close every day. In a high-retail name the open is
systematically the worst price of the day to buy. Same message as 1.8 with
identification.

### 1.20 Kelley & Tetlock (2013, JF) — "How Wise Are Crowds? Insights from Retail Orders and Stock Returns"
US retail order flow 2003–2007 (a wholesaler's feed). Both aggressive and passive
retail net buying positively predict the NEXT MONTH's return with no reversal;
aggressive orders predict news. Contrast with 1.7: monthly, pre-2008, all retail vs
Robinhood attention spikes. The reconciliation in the later literature (Boehmer-Jones-
Zhang-Zhang 2021 and after) is horizon: retail imbalance is informative at weekly
horizons on average and toxic in attention spikes. Requires order-flow data we don't
have; not testable here.

### 1.21 Schultz (2024, JFQA) — "Short Squeezes and Their Consequences"; Svoboda, Kapounek, Albrecht (2026, NAJEF) — "Systematic signals of short squeezes"
Schultz: squeezes are common in the hardest-to-borrow names (utilisation >= 90%); they
cost shorts ~31 bps/month of missed excess return; the squeezed stock's next-quarter
excess return is only +0.94% — i.e. squeezes are a cost to shorts, not a durable long
return. Svoboda et al.: hand-collected rare events; squeeze probability rises with
short-interest ratio and attention spikes, falls with institutional ownership. Neither
gives an intraday rule; both say the conditioning variables (SI, utilisation, borrow
fee, attention) are T-1 observable. Zaynutdinova, Strong, Baig (AFA 2026, "Seeking
Gamma"): 669 gamma-squeeze events 2019–2023 identified from net delta volume / OI,
**+5.13% CAR in the following month**; GME's squeeze started in Fall 2020. Requires
options data.

### 1.22 Bradley, Hanousek, Jame, Xiao (2024, RFS) — "Place Your Bets? The Value of Investment Research on Reddit's WallStreetBets"
RFS 37(5) 1409–1459. WSB "due diligence" posts predicted returns and cash-flow news
BEFORE GameStop; predictability vanished after, as posts shifted to price-pressure /
attention names. Social-media catalyst signals are, post-2021, noise for the long side.
(Consistent with our own REFUTED LLM catalyst-quality filter for ORB longs.)

### 1.23 LULD / halts
- SEC DERA (staff WP, 2014): LULD pilot 2013–2014 vs the prior single-stock circuit
  breaker; LULD **reduces** the frequency and magnitude of large short-term price
  reversals (transitory volatility) for Tier 1 and Tier 2. Bands 5% (Tier 1) / 10%
  (Tier 2) around the 5-min reference price, doubled in the open/close windows;
  15-second limit state then 5-minute pause. Not a continuation study.
- Shanghai halts (Wu et al., arXiv 1309.1138; 203 stocks 2009–2011): after
  negative-event halts **63.4% of stocks reverse within 1 minute and 89.5% within 2
  minutes** of the reopen; positive events plateau; absolute return, volume and spread
  peak at the reopen and decay as a power law. Different mechanism (1-hour information
  halts) but the only quantified reopen study.
- **No peer-reviewed US paper on post-LULD-halt continuation vs reversal for
  2023–2025 was found.** The "halt-and-continue" folklore in the small-cap day-trading
  community is untested academically. Our 1-min data can detect halts as missing
  minutes with a reopen print; see H-B9.

### 1.24 Pump / ramp-and-dump
- "Detecting Pump&Dump Stock Market Manipulation from Online Forums" (arXiv 2301.11403, 2023): pump-and-dump in small caps identified from
  the price/volume shape and time-matched forum posts; classifier 85% accuracy / F1
  62%. No return table in the abstract.
- FINRA Reg. Notice 22-25 (2022) and the 2024–2025 SEC/Nasdaq actions: "ramp-and-
  dump" small-cap IPOs (< $25M raised, < $100M valuation, < 20M shares, $4–6 offer,
  foreign — mostly China-operations — issuers, 90%+ allocations to foreign nominee
  accounts, WhatsApp "pig-butchering" groups told to place limit orders at set
  prices). Price spikes "on the day of, or days immediately following, the listing",
  then collapses to at or below the offer price; by late 2024 the ramp moved to weeks
  or months post-IPO. Nasdaq's Sept 2025 $25M-float rule targets these.
- Crypto pump literature (Gandal et al.; Kamps & Kleinberg) finds Telegram pumps
  "modestly successful", decaying over time; not stocks.
- Implication: these names are in OUR universe (>= 5% range, > $5) and are
  long-side poison AND unshortable. A pre-IPO-age / float / jurisdiction exclusion is
  a deliberate rule, not a filter tweak (see H-B12).

### 1.25 "Does Overnight News Explain Overnight Returns?" (arXiv 2507.04481, 2025; authors not verified from the abstract page)
30 years US, 2.4M articles, supervised topic model selecting topics by contemporaneous
return explanation. News prevalence and news-response differences explain "a large
part" of the overnight > intraday gap; the model forecasts out-of-sample which stocks
do well overnight and badly intraday. Supports: catalyst presence predicts the
OVERNIGHT return, and its intraday continuation is a separate (weaker) question.

---

## 2. Cross-paper synthesis (what the literature actually agrees on)

1. **Selection is the edge, not the pattern.** The same 5-min ORB is −0.02R below
   average RV and +0.38R at 30x RV (1.1); every academic day-trader study finds the
   population loses and a < 1% minority persists (1.4, 1.5). Our own finding that the
   raw detectors have no edge on the whole market is the norm in this literature, not
   an anomaly.
2. **The edge in liquid names is single-digit bps.** 1.1 (2–15 bps/trade at R = 0.1
   ATR), 1.2 (break-even 2.2 c/share), 1.13 (1–5 bps per half hour), 1.10 (3.8–6.9
   bps/day). Every number is gross of spread and dies at 40 bps — the papers do not
   "work" in a 40-bps universe, they are silent about it.
3. **Overnight and intraday are different markets** (1.8, 1.9, 1.11, 1.12, 1.16,
   1.18, 1.19, 1.25). Gaps, volume shocks and news are OVERNIGHT-return phenomena.
   The single-stock intraday base rate after an attention gap is negative
   (1.8, 1.16), and the volume premium is realised close-to-open, not open-to-close
   (1.18). "Gap-and-go" has no academic support at the single-stock level; index gaps
   do continue (1.17), which is why the folklore exists.
4. **Retail flow has a daily rhythm**: buy at the open (extrapolative), buy the dip in
   the last half hour (1.10 Table 11, 1.19). Selling into the open of a high-attention
   name and buying the extreme intraday loser at 15:30 are the two rules the literature
   supports; the last one is the only one whose effect size grows with the move size
   (small caps 14.7 bps/day) and therefore might survive our cost in the tail.
5. **Intraday momentum in single stocks is institutional** (1.14); in retail-
   dominated names it is weak or reversed (1.10). Our universe is retail-dominated.
6. **Attention spikes are followed by negative returns** at 1–20 day horizons (1.7,
   1.22). Catalyst presence without institutional participation (RV without dollar
   volume from institutions) is a fade signal.

---

## 3. Why the strongest paper does not contradict our 2026-09-15 null result

Our study: 5-min ORB / HOD-break / flag on every US symbol-day 2025-01 → 2026-09 with
day range >= 5% and open >= $5 (~300K symbol-days, point-in-time incl. delisted),
after a 40-bps round-trip spread: no net edge. Zarattini-Barbon-Aziz report Sharpe
2.81. The two are not in tension, for five separate reasons, any one of which is
sufficient:

**3.1 Universe.** Theirs: ADV14 >= 1M shares, ATR14 > $0.50, price > $5, then the
top-20 by 09:30–09:35 relative volume across ~7,000 names. The names that drive the
book (Table 4) are DDD, FSLR, NVDA, TSLA, AMD, ADBE, RCL, W — $20–$500 stocks with
1–3 bps effective spreads and deep books. Ours: any symbol whose day range was >= 5%,
which is dominated by $5–20 small caps with 20–80 bps spreads. There is essentially no
overlap: a >= 5%-range day on NVDA is rare; a >= 5%-range day on a $7 biotech is Tuesday.

**3.2 The stop is the size of our spread.** R = 0.1 x ATR14. For their names ATR/price
≈ 2–4%, so R ≈ 20–40 bps. Their best bucket (RV > 30x) makes +0.38R ≈ 8–15 bps per
trade gross-of-spread; the RV >= 1 bucket makes +0.08R ≈ 2–3 bps. Against our 40-bps
cost the paper's own numbers predict **−1R to −2R per trade net** in a 40-bps universe.
Nothing in the paper suggests the edge scales with R; it scales with relative volume.

**3.3 Selection lookahead runs the other way.** Their selection is causal at 09:35.
Our universe is selected on the day's REALISED range (end-of-day information). That
conditioning does not create a breakout edge (it includes both trend days and
whipsaw days with >= 5% range), but it means our null is a statement about
"breakouts on days that turned out to be volatile", not about "breakouts on days
that were abnormally active at 09:35". Those are different populations; the paper's
Figure 4 says the second one is where the money is, and we have not measured it.

**3.4 Hold-to-close with no target, ~20% win rate, and 4x leverage.** Their P&L is a
right-tail product: most trades lose ~1R (20–40 bps), a few make 5–15R. Our tested
exits (2R target, HOD/flag stops of 1–5% of price) truncate that tail and pay the
spread on a much bigger R, which flips the arithmetic. The 41.6% IRR is 20 names x
4x leverage x ~2–10 bps per name per day — a frequency-and-leverage number that only
exists because the spread is ~1 bp.

**3.5 Cost model.** $0.0035/share commission, no spread, no slippage, no borrow, fills
at the stop level. The Brusco replication of the sister paper (same authors, same
construction) shows 97% of the PnL vanishing at 2 c/share. There is no published
replication of the stocks-in-play paper with a spread model; until there is, the
2.81 Sharpe is an upper bound on a 1-bp-spread universe, and an irrelevance in ours.

**What this implies for us.** The right test of the paper is NOT to re-run it on our
>= 5%-range cache. It is to rebuild their causal universe (ADV >= 1M, ATR > $0.50,
> $5) for 2024-01 → 2026-09 — the true out-of-sample window they never had — compute
09:35 relative volume for every eligible name, take the top-20, and run their exact
spec with a per-name spread model. If the RV-monotone per-trade edge (−0.02R / +0.08R
/ +0.38R) does not reproduce OOS in the liquid universe, the paper is dead; if it does,
the question becomes whether any of that survives at OUR spreads (it will not at 40
bps; it might at the 5–10 bps of $20–50 names with ADV >= 1M).

---

## 4. Hypothesis table

Effect sizes are priors from the papers, converted to bps of price where the paper's R
is known; "net" means after our 40-bps round trip unless stated. "Causal?" = every
input known at entry. "Testable now" = with the >= 5%-range 1-min cache + PIT daily
bars + SPY minutes; "needs universe" = requires 1-min bars for symbol-days outside the
cache (Alpaca REST can fetch them; ~1,500 names x 430 days ≈ 650K symbol-days for the
liquid universe).

| H-id | Rule (one sentence) | Prior effect size | Causal? | Data needed | Falsification on 2025-01 → 2026-09 (pre-committed) | Cost sensitivity |
|---|---|---|---|---|---|---|
| **H-B1** Zarattini SIP-ORB, exact spec | ADV14 >= 1M, ATR14 > $0.50, open > $5; at 09:35 rank by (09:30–09:35 vol / 14-day mean of same); keep RV >= 1, top-20; stop order at 5-min high/low in candle direction; stop 0.1 x ATR14; hold to 16:00; 20 names, 4x cap. | +0.08R/trade at RV >= 1, +0.38R at RV > 30x, R ≈ 20–40 bps → 2–15 bps/trade gross; portfolio Sharpe 2.81 in-sample at ~1 bp cost. | Yes (all inputs T-1 or 09:35; resting stop order). | **Needs universe**: 1-min bars for all ADV >= 1M names 2024-01 → 2026-09 (paper ends 2023-12, so this is the only clean OOS). Per-name spread model by price/ADV. | Reject if OOS gross mean R/trade at RV >= 1 is <= 0 OR the RV-bucket monotonicity (−/+/++) fails OR the top-20 daily-return Sharpe at the paper's cost is < 1.0. Keep only if it also clears the name's own measured spread. | Extreme: break-even spread ≈ 0.1–0.3R ≈ 3–10 bps. Dead at 40 bps by construction; possibly alive at 5–10 bps. Long-only variant loses half the book. |
| **H-B2** RV monotonicity in OUR universe | On >= 5%-range days, per-trade gross ORB/HOD P&L is monotone increasing in 09:35 relative volume (14-day same-window baseline), and the top RV decile clears 40 bps. | Paper: −0.02R → +0.08R → +0.38R across RV buckets (R = 0.1 ATR). In our units (R = range or consolidation low, 1–5% of price) the prior is that the SIGN pattern transfers, magnitude unknown. | Yes (09:35 RV; our `rv_profile` already computes a cousin). Note the cache itself is selected on realised range — report as conditional-on-volatile-day, not as a forecast. | Testable now (cache + daily bars for the 14-day OR-volume baseline; need the 09:30–09:35 minute volume for the prior 14 days of each cached symbol — fetch if not cached). | Reject if the top-decile-RV net mean R is <= 0 in BOTH 2025 and 2026, or the bucket ordering is not monotone in either year. | High. Because our R is 10x the paper's, the same 0.1–0.4R edge is 20–200 bps — only the top bucket has a chance. |
| **H-B3** Gap-fade (single-stock) | For attention gappers (gap >= +5% at the open, RV < 2x or no institutional dollar volume), short at the open and cover at the close (or: never buy the open). | 1.8: high-attention names open high and reverse intraday; 1.16: overnight jump returns negatively predict the next-day return (t ≈ −11); 1.7: −4.7%/20d after Robinhood herding. Intraday magnitude for a >= 5% gap in small caps: prior −1 to −3% open-to-close mean, wide. | Yes for the signal (gap and RV at 09:31–09:35); the short requires a locate (fail closed). | Testable now for the >= 5%-range subset (biased toward big movers — the fade AND the continuation tails are both over-represented); the unbiased version **needs** all gap >= 5% symbol-days regardless of realised range (PIT daily bars give the list; fetch 1-min). | Reject if mean open-to-close return of gap >= 5% names is >= 0 in either year, or if the RV-split (low RV fades, high RV does not) is not present. | Medium: the move is % of price, the cost is 40 bps; borrow fees and halts are the real cost — model both. |
| **H-B4** First-30-min → rest-of-day continuation | Buy at 10:00 the names whose 09:30–10:00 return is in the top decile (given >= 5% intraday move by 10:00), hold to close. | 1.14 index-level only; 1.13 1–5 bps/half-hour; 1.14b says institutional; 1.10 says single-stock last-half-hour REVERSES. Prior for retail small caps: ~0 gross, negative net. | Yes. | Testable now. | Reject (expected) if top-decile 10:00 → 16:00 mean net return <= 0 in both years. Keep only if > +50 bps net in both years. | Very high: the effect is bps-scale. |
| **H-B5** End-of-day reversal in the extreme-loser tail | At 15:30 buy names with ROD3 (prior close → 15:00) <= −8% (small caps), sell at 16:00 (or the close auction). | 1.10: L-S 3.78 VW / 6.86 EW bps/day; **smallest quintile 14.7 bps/day**; the effect is all on the loser side and grows with |ROD3|. Prior for the <= −8% tail: +40–100 bps mean last-half-hour return gross (extrapolation, untested). | Yes (ROD3 known at 15:00; 30-min window). | Testable now — the >= 5%-range cache contains exactly these days (big losers are in the cache by construction, no selection bias for a loser-side rule as long as the loss is realised by 15:00, which must be checked). | Reject if mean 15:30 → 16:00 return of ROD3 <= −8% names is < +40 bps in either year, or the effect is not monotone in |ROD3|. Also require: no worse in 2026 than 2025 (post-paper OOS). | Medium: 40 bps against a 40–100 bps prior — the whole test is whether the tail clears cost; use close-auction fills (MOC) to cut the exit spread. |
| **H-B6** Intraday residual reversal | Fade the last interval's factor residual, per 30-min interval, long-short. | 162% annualised gross (1.15) = liquidity provision. | Yes. | Testable in principle; pointless at 40 bps. | Not worth running; listed to record that breakout rules are the SHORT side of this book. | Fatal. |
| **H-B7** Volume-shock overnight premium | At the close, buy the top decile of daily abnormal volume (vs 20-day mean) among >= $5 names; sell at the next open. | 1.18: ~18%/yr Sharpe ~1.1 gross on the ML version; the sort version smaller; 1.18b: premium larger when the volume shock has NO price move. Intraday leg ≈ 0. | Yes (volume at 15:59 / close). | Testable now from PIT DAILY bars alone (close and next open). | Reject if the top-decile close-to-open mean is <= +20 bps net (close auction + open auction fills, ~15–25 bps total) in either year. | Medium: two auction fills, no intraday spread. Not a day-trading rule — but it is where the volume signal's money is. |
| **H-B8** Short-interest / squeeze conditioning | Gate longs on high SI ratio + attention spike; expect fatter right tail. | 1.21: squeeze probability rises with SI and attention; +5.13% CAR/month for gamma-squeeze events; squeezes are common in >= 90% utilisation names. No intraday effect size. | SI is bi-monthly (stale); borrow fee daily (vendor). | **Needs data**: short interest / utilisation / options OI — not in our stack. | Defer; specify when data exists: top-SI-quintile x top-RV-decile vs the rest, same rule. | Unknown. |
| **H-B9** Post-LULD-halt continuation | After an up-halt reopen (5-min pause, Tier 2), buy the first print above the reopen price, stop at the reopen low, hold N minutes / to close. | No US evidence. Shanghai: 63–90% reverse within 1–2 min after negative halts; SEC: LULD suppresses transitory reversals. Prior: 0 with a fat right tail. | Yes (halt detected from missing minutes; reopen print observable). | Testable now: halts appear as gaps in the 1-min bars of cached >= 5%-range days (which is where halts live). | Reject if mean return from reopen+1 min to +15 min and to close is <= 0 net in both years; also measure the reversal fraction at +1/+2/+5 min à la Shanghai. | High: reopen spreads are the widest of the day; use the paper's (1.23) reopen decay to set the entry delay. |
| **H-B10** Retail-open pressure fade (Berkman) | Never place a long entry in the first 5 minutes on names with prior-day extreme return or attention; if entering, wait for a pullback below the open. | 1.8: implicit cost of buying at the open > effective half spread; 1.10 Table 11: overnight return loads +1.43 on ROD3 for winners; FH return +0.61 for winners, +2.12 for losers (retail extrapolative buys in the first half hour). | Yes. | Testable now: open → 09:35 → 10:00 → close path of gap-up names in the cache; compare entry at 09:31 vs at first pullback. | Reject if entries at the open beat delayed entries net in both years (they should not). | Low: this is a timing rule inside an existing trade, it changes cost not signal. |
| **H-B11** Catalyst without volume is a fade | Among news-tagged gappers, split by RV; the no-volume news bucket has negative intraday return. | 1.7/1.22 attention → negative; our own ORB finding (news x PM$ combo is the only positive cell) is the same shape. | Yes (news feed timestamps must be <= entry). | Testable now with our Benzinga/Alpaca news CSVs joined to the cache. | Reject if the news-without-RV bucket is >= 0 net in both years. | Medium. |
| **H-B12** Ramp-and-dump exclusion | Exclude from any long rule: IPO age < 180 days AND float < 20M (or public float < $25M) AND foreign (esp. China-ops) issuer; also any name whose first-day range is > 100%. | FINRA/SEC: spikes on/after listing then collapse to <= offer; no academic return table. | Yes (IPO date and float are T-1). | Testable now: IPO age from the first PIT daily bar; float from the universe table. | Reject the exclusion if the excluded bucket's long-side mean net return is >= the kept bucket's in both years. | None — it removes cost. |
| **H-B13** Base rate for any new discretionary intraday rule | Treat the population base rate as the prior: ~97% of persistent retail day traders lose; the persistent-skill group earns ~28 bps/day net (Taiwan) with Sharpe 0.2. | 1.4, 1.5. | n/a | n/a | n/a — this is the prior that every H above must beat with a pre-registered rule. | n/a |

Testable now: H-B2, H-B3 (biased subset), H-B4, H-B5, H-B7, H-B9, H-B10, H-B11, H-B12.
Need a different universe: H-B1 (liquid ADV >= 1M names on ALL days), H-B3 unbiased
(all gap >= 5% days, quiet-range days included), H-B8 (short interest / options).

Ranking by (prior net effect x testability): **H-B5 > H-B1 (as an OOS replication in
the liquid universe) > H-B3 / H-B7** (both say the same thing: gaps and volume shocks
pay overnight and fade intraday, so the long-intraday side of our universe is the wrong
side). H-B2 is the cheapest sanity check and should run first because it decides
whether any of the ORB family deserves further work on our data.

---

## 5. Data-quality caveats specific to our stack

- The >= 5%-range cache is END-OF-DAY selected. It is fine for exit studies and for
  loser-side rules whose signal is realised before the window (H-B5, H-B9), and it is
  the wrong population for any entry-selection study (H-B1, H-B3): use PIT daily bars to
  define the causal list, then fetch the 1-min bars for the members not in the cache.
- Spread by price band matters more than any signal here: the same rule that is worth
  +0.08R at 1 bp is worth −1R at 40 bps. Every test must carry a per-name spread
  model (price, ADV, time of day), not a flat 40 bps.
- Shorts: half of 1.1's book and all of 1.16/1.8's implied trade are short. Our live
  account fails closed on locates; test long-only and short-only separately and never
  net them.
- Halts: 1-min bars show halts as missing minutes; the reopen print is the first bar
  after. Do not treat a resting order as fillable across a halt.
- The papers' OOS is our IN-sample: 1.1 ends 2023-12, 1.10 ends 2019-12, 1.16 ~2020.
  A 2025–2026 test is out-of-sample for every one of them.

## 6. Source list

- Zarattini, Barbon, Aziz (2024) SSRN 4729284 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284 (full text read)
- Zarattini & Aziz (2023) SSRN 4416622; replication https://github.com/giovannibrusco/zarattini-2023-orb-qqq
- Zarattini & Aziz (2023) VWAP SSRN 4631351 — https://concretumgroup.com/volume-weighted-average-price-vwap-the-holy-grail-for-day-trading-systems/
- QuantConnect replication — https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/
- Chague, De-Losso, Giovannetti (2020) SSRN 3423101; Quantpedia summary https://quantpedia.com/retail-day-trading-is-an-uphill-battle/
- Barber, Lee, Liu, Odean, "Cross-Section of Speculator Skill" (JFM 2014) — https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf ; "Do Day Traders Rationally Learn" — https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trading%20and%20Learning%20110217.pdf
- Jordan & Diltz (2003) FAJ 59(6); Garvey & Murphy (2005) SSRN 908615; Cheng et al. (2023) Heliyon 9(4)
- Barber, Huang, Odean, Schwarz (2022) JF 77(6) — SSRN 3715077
- Berkman, Koch, Tuttle, Zhang (2012) JFQA 47(4) — SSRN 1625495
- Lou, Polk, Skouras (2019) JFE 134(1) — https://personal.lse.ac.uk/polk/research/TugOfWar.pdf (full text read)
- Baltussen, Da, Soebhag (2024) SSRN 5039009 — https://academicweb.nd.edu/~zda/EOD.pdf (full text read)
- Bogousslavsky (2021) JFE 141(1) — SSRN 2869624
- Barardehi, Bogousslavsky, Muravyev (RFS 2026) — SSRN 4069509
- Heston, Korajczyk, Sadka (2010) JF — arXiv 1005.3535
- Gao, Han, Li, Zhou (2018) JFE 129(2) — SSRN 2440866; Chen, Haboub, Khan, Mahmud (2024) RQFA — https://link.springer.com/article/10.1007/s11156-024-01319-8
- Brogaard, Han, Kim (2024) SSRN 4731947
- Bahcivan, Dam, Gonenc (2023/2025) SSRN 4335622
- Plastun, Sibande, Gupta, Wohar (2020) NAJEF 52 — SSRN 3461283
- Cartea, Cucuringu, Jin, Wilson (2025) SSRN 5156605; Gervais, Kaniel, Mingelgrin (2001) JF; Li, Yin, Zhao (2024) Eur. J. Finance — https://www.tandfonline.com/doi/full/10.1080/1351847X.2024.2303092
- Ahn, Fan, Noh, Park (2024) SSRN 4752520
- Kelley & Tetlock (2013) JF 68(3)
- Schultz (2024) JFQA 59(1) — SSRN 4025226; Svoboda, Kapounek, Albrecht (2026) NAJEF 85 — SSRN 5334668; Zaynutdinova, Strong, Baig (AFA 2026) SSRN 5959235
- Bradley, Hanousek, Jame, Xiao (2024) RFS 37(5) — SSRN 3806065
- SEC DERA, "LULD Pilot Plan and Extraordinary Transitory Volatility" — https://www.sec.gov/files/marketstructure/research/dera_wp_luld_and_extraordinary_transitory_volatility.pdf (full text read); Wu et al., Shanghai halts — arXiv 1309.1138 (full text read)
- "Detecting Pump&Dump Stock Market Manipulation from Online Forums" — arXiv 2301.11403; FINRA Regulatory Notice 22-25 — https://www.finra.org/rules-guidance/notices/22-25
- "Does Overnight News Explain Overnight Returns?" — arXiv 2507.04481
