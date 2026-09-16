# Topic C — Overnight vs intraday returns, opening gaps, the open and close auctions, first-30 / last-30 minutes

Literature review for the OneMil systematic desk. Written 2026-09-16. Scope: the overnight/intraday
return decomposition and the strategies built on it; opening gaps and gap-fill; opening- and
closing-auction effects; first-30-minutes vs rest-of-day predictability in single stocks;
end-of-day momentum/reversal; intraday seasonality; retail order imbalance and Robinhood attention.
Foundational papers plus the 2022–2026 follow-ups.

**How the evidence was gathered and how much to trust each entry.** Every paper below carries a
verification tag:
- `[primary]` — I read the paper's own text (PDF text extracted locally) and the numbers are quoted from it.
- `[abstract]` — only the abstract / a publisher page was readable; numbers are from the abstract.
- `[secondary]` — numbers come from a search-engine or third-party summary (Alpha Architect, NY Fed blog, a
  practitioner post). Treat as approximately right; verify before building on them.
- `[memory]` — cited from prior knowledge because the source could not be opened this session; magnitude
  not verified.
SSRN, ScienceDirect, Wiley, Springer, ResearchGate and Cambridge PDFs are behind bot walls from this
node, so the mix leans on author-hosted PDFs and open mirrors.

**Our data (what "testable now" means).** Daily point-in-time OHLCV for every US symbol 2025–2026
including delisted (≈10K symbols × ~420 days) → overnight = open/prev_close − 1 and intraday =
close/open − 1 are available for the whole market; 1-minute bars for SPY and for every symbol-day with
≥5% range; Alpaca REST 1-min back to 2016 for any symbol on demand. No TAQ, no sub-penny retail
flags, no auction imbalance feeds, no Robintrack. A hypothesis is "testable now on daily bars" if it
needs only open/high/low/close/volume at the daily frequency.

**One data caveat that matters for everything in this topic.** The whole literature's overnight return
is measured from the previous official close (the closing auction print) to the official open (the
opening auction print) or, in some papers, to the 9:30 quote midpoint. Our daily-bar `open` must be
checked against the listing exchange's auction print: a consolidated first trade that is a few cents
off the auction price is a few bps of "overnight return" that never existed, and most effects below
are 2–15 bps per night. Check this on a handful of NYSE and Nasdaq names against Alpaca's auction
records (`get_auctions`) before any test is run. The same holds for the close.

---

## 0. Executive summary

1. **The index-level overnight drift is the best-known and the most decayed effect.** In S&P 500 futures
   the entire premium accrued in the 2:00–3:00 ET hour (European open) at ~3.6–3.7 %/yr (1.48 bps/day)
   over 1998–2019 (Boyarchenko–Larsen–Whelan RFS 2023; Bondarenko–Muravyev JFQA 2023). The NY Fed's own
   2026 update finds the window has averaged ~zero since 2021, the order-imbalance driver (dispersion
   of end-of-day signed volume) has compressed 55 %, and the NightShares ETFs built on it closed after
   14 months. Elm Wealth's 2025 update: SPY overnight-only $1 → $1.66 vs whole-day $1 → $2.07 over the
   last five years. **Prior for 2025–26: ~0–1 bps/night on SPY, not worth the spread.**
2. **The cross-sectional tug of war (Lou–Polk–Skouras JFE 2019) is enormous gross and lasts for years**:
   value-weight overnight-winner-minus-loser decile = +3.47 %/month overnight alpha (t = 16.8) and
   −3.02 %/month intraday alpha (t = −9.7), signal lagged up to 60 months still works. It is a clientele
   effect (retail at the open, institutions into the close), it is concentrated in retail/attention
   names, and it is **gross of a spread crossed twice a day** — Elm's individual-stock version (38 %/yr
   gross, t ≈ 17) "generated insufficient returns to overcome 1 bps per trade" over 2014–2022. Only
   large caps with 1–3 bps spreads can carry it; $5–20 small caps at ~40 bps cannot.
3. **The open is where retail overpays and the close is where it buys the dip.** Berkman–Koch–Tuttle–
   Zhang (JFQA 2012): prior-day attention stocks open high and give it back in the first hour, by more
   than the effective half-spread. Baltussen–Da–Soebhag (2025): the last 30 minutes reverse the day's
   cross-sectional move — long the day's losers 15:30→16:00 earns 3.78 bps/day VW (t = 10.7), 6.86 EW,
   14.7 bps in the smallest quintile (t = 27), entirely from the loser (long) side, driven by retail
   buy-the-dip and short-sellers covering; it reverts the next morning. Robust in every 3-year window
   since 1993 including 2020–2023.
4. **Market intraday momentum (first-30-min / rest-of-day sign predicts last-30-min) is real historically
   and dead-flat in the 0DTE era.** Gao et al. (JFE 2018) SPY 1993–2013: 6.67 %/yr, Sharpe 1.08, 4.3–4.5 %/yr
   after costs; Baltussen et al. (JFE 2021): equity futures 6.86 %/yr, Sharpe 1.73, gamma-hedging driven.
   A 2026 practitioner replication on 1,085 SPX sessions (Apr-2022 → Aug-2026) finds slope +0.006, t = 0.6
   unconditionally; only short-dealer-gamma days (15 % of sessions) show t = 3.1.
5. **Gap-fill is folklore with one honest academic answer and no single-stock replication.** Plastun et al.
   (2020, S&P/DJI/Nasdaq 1928–2018): on the gap day prices *continue* in the gap direction, no evidence gaps
   fill within five days, the continuation strategy's win rate fell from ~99 % (1930s–60s) to 61–67 %
   (1989–2018). Practitioner QQQ data 1999–2023: 0.5–1 % gaps fill same day 53–59 %, 2 %+ gaps 29–33 %;
   downs fill more than ups. For single $5–20 stocks nobody has published fill rates by size/cap — our
   daily bars can settle it in an afternoon.
6. **Closing auction**: 7.5 % of volume by 2018, ~10 % by 2019–2021; closing price sits at the pre-close bid
   or ask 68 % of the time; mean absolute deviation from the 4:00 midpoint 8 bps (20.6 bps small caps,
   63 bps at the 1 % tail) and it "reverts almost fully overnight" — 85 % by the next morning (Bogousslavsky–
   Muravyev JFM 2023); Jegadeesh–Wu (JFE 2022) say the temporary component takes 3–5 days to fully dissipate
   and that exploiting it is "significantly profitable". Opening auctions are illiquid with much larger
   price impact than closing auctions (Goyal–Jegadeesh–Wu JFQA 2026) — do not enter overnight positions MOO.
7. **Retail order imbalance predicts ~10 bps/week (Boehmer et al. JF 2021) but is not monetizable by retail
   itself**, and the sub-penny identification mis-signs 28 % of trades (Barber et al. JF 2024). Robinhood
   herding stocks earn −4.7 % over the following 20 days (Barber et al. JF 2022). We have no TAQ or
   Robintrack, so only attention proxies from bars are testable.
8. **Three hypotheses worth our time, all testable now on daily bars**: (H-C2) cross-sectional overnight
   continuation restricted to large caps; (H-C3) fading the open in prior-day attention stocks (our ≥5 %-range
   universe *is* the attention universe); (H-C9/H-C8) single-stock gap-fill and gap-continuation
   probabilities by gap size, cap and prior-day attention — the missing table in the literature, and the
   direct input to how our ORB/HOD books should treat gap-ups. Priors in §4.

---

## 1. Paper-by-paper

Format per paper: citation/URL · universe & selection (are the inputs known at decision time?) · span/freq ·
exact rule · costs · reported result · out-of-sample / post-publication · critiques.

### A. Index-level: the overnight drift

#### A1. Cliff, Cooper, Gulen — "Return Differences between Trading and Non-trading Hours: Like Night and Day" (SSRN 1004081, 2008; JAM 2011) `[primary]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1004081 (open mirror: https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/d0749895-bc80-4bf5-9b53-fed6eed60914.pdf)
- Universe: S&P 500 constituents (TAQ), 44 AMEX Internet-index stocks, 14 largest ETFs (SPY, QQQ, DIA…), S&P E-mini. 1993–2006, daily.
- Rule: hold close→open ("night"); compare with open→close ("day"). No selection — the point is the unconditional decomposition. Inputs trivially known.
- Result: S&P 500 stocks night return 2.82–4.76 bps/day (t = 3.9–17.8 depending on trade vs mid-quote and pooling), day return −2.85 to +0.22 bps, night-minus-day 2.61–7.61 bps. Internet stocks: night +16.8 to +18.4 bps, day −14.2 to −17.0 bps. Night volatility (124 bps sd for S&P stocks) *lower* than day (203 bps). Holds every weekday, pre/post ECNs, pre/post decimalization. **Much of the negative day return is the first hour**: −1.16 to −3.59 bps (t up to −13.4) for S&P stocks, also in 13/14 ETFs and the E-mini.
- Costs: not netted; the authors note it is a premium decomposition, not a strategy.
- OOS: continued through the 2010s at the index level (Glasserman et al. below: ON−ID spread 2.75 bps/day for S&P 500 stocks 2000–2022), then faded post-2021 (A3, A5).
- Critique: mid-quote vs trade-price night returns differ by ~2 bps — exactly the size of the effect, which is the spread-crossing problem in miniature.

#### A2. Bondarenko, Muravyev — "Market Return Around the Clock: A Puzzle" (JFQA 58(3), 2023) `[abstract + Cambridge page]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3596245 ; https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/market-return-around-the-clock-a-puzzle/089E33AC0B4D3B9A02CBA31EDF6505B3
- E-mini S&P 500 futures, 24-hour, ~2004–2020 (sample includes the 2020 COVID period, where the effect is stronger).
- Result: the 4 hours around the European open carry the entire average market return, Sharpe 1.6, "remain high after transaction costs"; the other 20 hours are a noisy zero. US cash hours essentially flat; 9:30–10:30 modestly positive. Mechanism: uncertainty resolution at the European open (VIX futures rise overnight and fall into the European open).
- Costs: futures — ~1 tick.
- OOS: see A5 — the window is ~zero since 2021.

#### A3. Boyarchenko, Larsen, Whelan — "The Overnight Drift" (RFS 36(9), 2023; NY Fed Staff Report 917, rev. Aug 2022) `[primary: staff report]`
- URL: https://www.newyorkfed.org/research/staff_reports/sr917 ; https://academic.oup.com/rfs/article-abstract/36/9/3502/7076616
- E-mini S&P 500, 1998–2020 (23 years), hourly/5-min.
- Result: 2:00–3:00 ET averages **3.7 %/yr = 1.48 bps/day**, positive in 20/23 years, significant in 17; the only hour that survives Bonferroni / Benjamini–Yekutieli across sub-samples. The 9:00–10:00 hour is large and negative only in recessions (2000–03, 2007–09), flat otherwise. Mechanism: inventory risk — end-of-day order imbalance (relative signed volume 15:15–16:15) predicts the overnight reversal; sell-offs reverse strongly, rallies barely (asymmetry); zero imbalance → ~zero reversal; in volume-time the reversal completes after ~60,000 contracts (≈ 3:00).
- **Exact strategy & cost line (their words):** the 2:00–3:00 long has a gross Sharpe of 1.1 which "accounting for bid-ask spreads … reduces to −0.5"; widening to 1:30–3:30 gives 1.3 gross, 0.3 net. So even in ES the pure-window strategy is a cost story.
- Critique: a 1.5 bps/day effect on a 1-tick instrument is inside the spread for anyone who is not already a liquidity provider.

#### A4. Haghani, Ragulin, Dewey (Elm Wealth) — "Night Moves: Is the Overnight Drift the Grandmother of All Market Anomalies?" (SSRN 4139328, 2022; JOIM, Markowitz Award 2024) `[secondary: Elm's own article]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4139328 ; https://elmwealth.com/night-moves-overnight-drift/
- SPY 1993–2022; S&P 500 single stocks 1995–2022; meme names, ARKK, GBTC.
- Rule (single-stock): each period rank S&P 500 stocks on trailing 2-year overnight-minus-intraday return; long top 20 % ("attention" stocks), short bottom 20 % ("neglected"), hold overnight only, flat intraday.
- Result: SPY open→close $1 → $1.21 in 30 years (below T-bills); the individual-stock L/S 38 %/yr gross, t ≈ 17, max drawdown 4 %, long leg 29 % vs short leg 6 %; weekend nights 1.5× weekday nights; AMC 2019–May 2022 +30,000 % overnight vs −99.6 % intraday.
- Costs: their own numbers — round-trip impact at 1 % of volume ≈ 40 bps; 1990s commissions ~100 bps/yr; borrow ~1 %/yr; and **"in 2014–2022 the strategy generated insufficient returns to overcome 1 bps per trade over 8 years"**. Performance peaked 2012–2015 and waned after the academic papers circulated (2008–2015). From March 2020 the short (neglected) leg got *larger*, reversing the historical pattern.
- Mechanism: retail market-buys at the open into shallow liquidity; institutions sell into the deep close.
- Elm's March-2025 update ("Still Working the Night Shift", https://elmwealth.com/night-shift/) `[secondary]`: 23 retail-favourite names over 5 years: overnight-only $1 → $31.70 (Sharpe 0.7; equal-weight portfolio Sharpe 2.0 overnight vs 0.0 daytime), daytime-only $1 → $0.80; **SPY 5-year overnight $1 → $1.66 vs whole-day $2.07** — the index drift is gone, the retail-name drift is not. Explicit cost line: "transaction costs, including market impact, that most investors face would erase most or all of the total expected gain." NightShares ETFs (NSPY/NIWM, launched June 2022) closed after ~1 year because of twice-daily full turnover.

#### A5. Boyarchenko et al. — "The Disappearing Overnight Drift" (Liberty Street Economics, NY Fed, July 2026) `[secondary: the blog itself]`
- URL: https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/
- Sample extended to December 2025. The 2:00–3:00 window "averaged close to zero" over Jan-2021 → Dec-2025 across ES, NQ and YM. Using E[R_ON] = imbalance × variance / risk capacity: the sd of end-of-day relative signed volume fell from 6.5 % to 2.9 % (−55 %) — the main channel; VIX mean 20.4 → 19.4 (no change); overnight volume share 15 % → 16 % (no change).
- Implication for us: any overnight-index hypothesis must be tested on 2021+ data only; pre-2021 evidence is not a prior any more.

#### A6. Glasserman, Krstovski, Laliberte, Mamaysky — "Does Overnight News Explain Overnight Returns?" (arXiv 2507.04481 / SSRN 5336382, July 2025) `[primary]`
- URL: https://arxiv.org/abs/2507.04481
- S&P 500 constituents, 1996–2022 (returns 2000/2001–2022), 2.4M full-text news articles; daily; supervised topic model.
- Result: cumulative ON − ID for S&P 500 stocks averages **2.75 bps/day (≈7.2 %/yr) since 2000**. Annual firm-level correlations 2000–2021: intraday→intraday +0.19, overnight→overnight +0.29, cross-period −0.26/−0.27, close-to-close −0.04 (none) — the Lou–Polk–Skouras tug of war at the annual horizon. News-forecast portfolios (Table 3, 2001–2022, EW, 25 stocks, annual selection): top overnight picks 11.0 bps/night vs 3.4 for the rest (diff 7.6, se 0.7); bottom intraday picks −6.3 bps/day vs +2.0 (diff −8.3, se 1.4). 2011–2022 (Table 4): 6.7 vs 4.0 overnight (diff 2.6–3.2), −3.6 vs +2.0 intraday. Removing the top/bottom picks makes the remaining ON−ID difference insignificant.
- Costs: authors' own verdict — "because of the extreme turnover required to trade the over-intra effect, our findings fall short of being a viable trading strategy."
- Critique: annual rebalancing keeps the *selection* cheap but the *holding* (long overnight, flat intraday) still crosses the spread twice a day; useful for timing already-planned entries/exits, not as a stand-alone book.

### B. Cross-section: overnight vs intraday continuation and reversal

#### B1. Lou, Polk, Skouras — "A Tug of War: Overnight versus Intraday Expected Returns" (JFE 134, 2019, 192–213) `[primary]`
- URL: https://personal.lse.ac.uk/polk/research/TugOfWar.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650
- All US common stocks (CRSP + TAQ/TRTH opens), 1993–2013 (open prices from TAQ), monthly rebalanced, value-weight deciles; robust to excluding small caps and to open/close price definitions (their fn. 2).
- Rule: at month end rank on the past month's cumulative **overnight** return; long top decile, short bottom, measure next month's overnight and intraday components separately. Mirror sort on past intraday return. Inputs known at decision time (prior-month opens/closes).
- Result: overnight-sorted VW L/S: **+3.47 %/month overnight 3-factor alpha (t = 16.83), −3.02 %/month intraday alpha (t = −9.74)**. Intraday-sorted VW L/S: +2.41 %/month intraday alpha (t = 7.70), −1.77 %/month overnight (t = −7.89). Signals lagged 60 months still give joint t > 20. 14 anomalies decomposed: momentum (price, industry, earnings, time-series) and short-term reversal earn their premia **overnight**; value, profitability, investment, beta, idiosyncratic vol, issuance, accruals, turnover earn them **intraday** with an opposite-signed overnight premium. Overnight Sharpe of momentum 0.77, more than twice its close-to-close Sharpe. Smoothed ON−ID spread of a strategy forecasts its close-to-close performance. Institutions trade against momentum intraday (13-F + TAQ), generating the intraday→overnight tug of war.
- Costs: acknowledged, not netted — "transaction costs will make the actual profitability … much less attractive"; the conclusion suggests it "may be profitable after transaction costs for execution-savvy short-term investors", "particularly … equity index futures" (time-series momentum), and otherwise as an order-timing rule (open vs close) for institutions.
- Post-publication: Elm (A4) is the practitioner replication with costs — waning since ~2015. Glasserman (A6) confirms the annual-horizon correlations through 2022.
- Critique: 3.47 %/month ≈ 16 bps/night for the VW decile spread; the long leg alone is ~half that. With a spread crossed twice a night, only stocks with quoted spreads ≤ 3 bps can carry the long leg; the decile extremes are exactly the retail/attention names with the widest spreads. Also: the overnight return sort mechanically loads on the bid-ask bounce unless opens are mid-quotes (Aboody et al. address this; see B4).

#### B2. Hendershott, Livdan, Rösch — "Asset Pricing: A Tale of Night and Day" (JFE 138, 2020, 635–662) `[primary]`
- URL: http://faculty.haas.berkeley.edu/hender/CAPMday-night.pdf ; https://www.ssrn.com/abstract=3117663
- All US common stocks (CRSP opens), 1992–2016; international; industry, B/M, cash-flow/discount-rate beta portfolios; Treasury futures. Betas = rolling 12-month daily.
- Result: **night SML slope +14 bps per unit beta (17.5 h close→open), day SML slope −15 bps per unit beta**, close-to-close flat. Highest-beta decile: day −8 bps, night +20 bps. Individual-stock Fama–MacBeth: day slope −7.7 bps (t = −5.5), night +6.4 bps (t = 7.8). Night-minus-day implied premium 14.1 bps (EW 25.6 bps). Implied risk-free rate is negative at night. Treasury futures show the mirror pattern.
- Rule (their two strategies): (i) long high-beta / short low-beta **overnight**, flip to short high / long low **intraday**, weights = beta minus mean beta → 0.10 %/day, sd 0.79 %, annualized 25.2 %, Sharpe 2.03; (ii) top-decile vs bottom-decile portfolio version → 0.44 %/day, annualized 108 %, Sharpe 3.78. **Gross.**
- Costs: not netted. Both strategies turn the whole book twice a day; strategy (ii) is 44 bps/day gross on a 2-leg book that pays ~4 half-spreads/day — survives only in names with ≤ 5 bps spreads.
- OOS: no 2017+ evidence in the paper. Our test on 2025–26 daily bars is cheap (see H-C6).

#### B3. Bogousslavsky — "The Cross-Section of Intraday and Overnight Returns" (JFE 141(1), 2021, 172–194) `[abstract]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2869624 ; https://econpapers.repec.org/article/eeejfinec/v_3a141_3ay_3a2021_3ai_3a1_3ap_3a172-194.htm
- ~30 years of US intraday returns, half-hour partitions.
- Result: anomalies accrue over the day very differently; **size and illiquidity premia are realized in the last 30 minutes**; a mispricing (Stambaugh–Yu–Yuan) factor earns positive intraday returns but "performs poorly at the end of the day", increasingly so over time — arbitrageurs cut positions before the close because of overnight margin and lending costs. This is the arbitrage-capital side of the last-half-hour price pressure that Baltussen–Da–Soebhag (C5) later refine.
- Critique: half-hour returns on individual stocks from trade prices embed bid-ask bounce; the abstract-level result is about factor portfolios, where it nets out.

#### B4. Berkman, Koch, Tuttle, Zhang — "Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open" (JFQA 47(4), 2012, 715–741) `[abstract; magnitudes secondary]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1625495 ; https://bearworks.missouristate.edu/articles-cob/576/
- US stocks, TAQ, 1996–2008 (from memory), daily.
- Rule/finding: stocks that attracted retail attention the previous day (proxied by squared returns, abnormal volume, net retail buying) have **high opening prices → positive overnight return followed by an intraday reversal**; the reversal is larger for hard-to-value / costly-to-arbitrage names and in high-sentiment periods; "the additional implicit transaction costs for retail traders who buy high-attention stocks near the open frequently exceed the effective half spread."
- Trading interpretation: short the open (or delay planned buys past the first hour) in prior-day attention stocks. Costs: for the *retail* buyer the cost is the effect; for a fader, the short must be borrowed in exactly the names that are hard to borrow.
- OOS: Baltussen–Da–Soebhag (C5) re-confirm retail extrapolative buying at the open using Robinhood 2018–19 data and BJZZ 2010–19 flags; Jones–Pyun–Wang (2024) `[memory]` same. Elm (A4) is the same effect at the index-of-attention-stocks level.

#### B5. Aboody, Even-Tov, Lehavy, Trueman — "Overnight Returns and Firm-Specific Investor Sentiment" (JFQA 53(2), 2018, 485–505) `[primary]`
- URL: https://anderson-review.ucla.edu/wp-content/uploads/2021/03/Aboody-et-al_overnight_returns_and_firmspecific_investor_sentiment_JFQA2018.pdf
- US common stocks, weekly and monthly overnight-return deciles, >1,000 weeks (≈1992–2013); bid-ask-spread checks included.
- Result: **weekly persistence** — decile-10 minus decile-1 of week-w overnight return has week w+1 overnight return spread of 1.76 pp (−90 bps vs +86 bps), then 1.48, 1.33, 1.21 pp at w+2..w+4; stronger in hard-to-value firms. **Longer-term reversal** — long decile 1 / short decile 10 of the month's overnight return earns +0.62 %/month 4-factor alpha (7.4 %/yr) over the next 12 months; 4.4–9.7 %/yr in the hardest-to-value quartiles, insignificant in most easiest-to-value quartiles.
- Interpretation: overnight return = firm-specific sentiment. The persistence is the same object as B1's overnight continuation; the 12-month reversal is the sentiment unwinding.
- Costs: the 12-month reversal leg is a monthly-rebalanced, low-turnover portfolio — **the one overnight-derived signal that is cheap to trade**. Not netted in the paper.

#### B6. Akbas, Boehmer, Jiang, Koch — "Overnight Returns, Daytime Reversals, and Future Stock Returns" (JFE 145(3), 2022, 850–875) `[abstract]`
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0304405X21004116 ; https://ideas.repec.org/a/eee/jfinec/v145y2022i3p850-875.html
- US stocks, monthly.
- Rule: count within the month the days with (positive overnight return AND negative intraday return); a higher frequency = a more intense daily tug of war (noise traders overnight vs arbitrageurs intraday). **Higher intensity predicts higher next-month close-to-close returns**: daytime arbitrageurs over-correct the persistent overnight pressure (they under-estimate the chance of positive overnight news arriving).
- Magnitude: not readable from the abstract; `[memory]` the top-minus-bottom decile spread is on the order of 0.5–1 %/month. Verify before use.
- Costs: monthly rebalance → cheap. Inputs: prior-month opens/closes only → **testable now on daily bars** (H-C14).

#### B7. Laarits, Sammon — "The Retail Habitat" (NYU/HBS working paper, Nov 2022) `[primary]`
- URL: https://as.nyu.edu/content/dam/nyu-as/econ/documents/fall-2022/Laarits%20Seminar%20Paper.pdf
- Retail-initiated trade share (BJZZ) quintiles: ~2 % of trades retail in Q1 vs ~20 % in Q5; persistent for a year. High-retail stocks: more intangibles, longer duration, higher mispricing scores, prices respond ~half as much to earnings surprises, no earnings-announcer premium, wider spreads especially around earnings; long Q5 / short Q1 outperformed since 2020 and fell less in the GFC.
- Why it is here: it defines the universe in which every overnight/attention effect above lives, and explains why institutions do not arbitrage it (cost + hard-to-value). Our $5–20, ≥5 %-range universe is the Q5 habitat.

### C. Intraday time-series and cross-section: first 30, last 30, periodicity

#### C1. Gao, Han, Li, Zhou — "Market Intraday Momentum" (JFE 129(2), 2018, 394–414) `[primary: SSRN version]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866 (open mirror: https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/ee7dac49-530b-4950-b5d0-e0b5eee08f2e.pdf)
- SPY, TAQ, Feb-1993 → Dec-2013, 13 half-hour returns/day; r1 = prev close → 10:00, r12 = 15:00–15:30, r13 = 15:30–16:00.
- Rule: at 15:30 go long SPY for the last half-hour if r1 > 0, short if r1 < 0 (variant: require r1 and r12 same sign, else flat). Inputs known at 15:30.
- Result: r1 (and r12) predict r13 in and out of sample; timing strategy **6.67 %/yr, sd 6.19 %, Sharpe 1.08, success rate 54.4 %** vs buy-and-hold 6.04 %/yr Sharpe 0.29 and "always long the last half-hour" −1.11 %/yr; certainty-equivalent gain 6.02 %/yr (γ = 5); stronger on volatile, high-volume, recession and macro-news days (FOMC-minute days 20 %/yr). Holds for 10 other liquid ETFs.
- Costs (their Table 10, post-decimalization, buy at 15:30 ask / sell at bid, close at the auction print): first-half-hour signal 6.93 % → **4.46 %/yr net** (post-2001), 7.96 % → 6.52 % net (post-2005, proportional spread ≈ 1.2 bps); two-signal version 5.50 % → 4.74 %.
- Post-publication: Limkriangkrai–Chai–Zheng (PBFJ 2023) `[primary]` replicate SPY 1996–2013 and report the US effect "remains robust during the COVID-19 crisis"; APAC ETFs mixed (China/Japan yes, HK/Singapore no). The 0DTE-era check (C3) says it is gone unconditionally in 2022–26.

#### C2. Baltussen, Da, Lammers, Martens — "Hedging Demand and Market Intraday Momentum" (JFE 142(1), 2021, 377–403) `[primary]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3760365 ; https://academicweb.nd.edu/~zda/intramom.pdf
- 62 futures (17 equity index, 16 bond, 21 commodity, 8 FX), Dec-1974 → May-2020, intraday.
- Rule: r_ROD = return from prior close to 30 min before close; at that point go long the last 30 min if r_ROD > 0, short if < 0. (η(r_ONFH) uses only the first half-hour; η(r_ONFH, r_ROD) trades only when both agree.)
- Result (Table 6, 1/N within asset class, annualized): equity index futures η(r_ROD) **6.86 %, sd 3.96 %, Sharpe 1.73, success 55 %**; η(r_ONFH) 4.21 % / Sharpe 1.07; both-agree 5.47 % / Sharpe 1.60 / success 61 %; always-long-last-30 0.44 % / Sharpe 0.11. Bonds Sharpe 1.62, commodities 1.42, FX 0.87. r_ROD beats r_ONFH out of sample. **Reverts over the next days.** Driver: gamma hedging — the effect is present only when option market makers' net gamma exposure is negative and scales with it; leveraged-ETF hedging demand on an index scales its momentum cross-sectionally and over time; strongest in Feb–May 2020.
- Costs: "we do not consider transaction costs … exploiting the effect in the S&P 500 futures yields a positive net Sharpe ratio when we assume transaction cost equal to a tick."
- Critique: the mechanism is a *conditional* one (dealer short gamma); in the 0DTE era dealer gamma flips sign intraday, which is exactly what C3 finds.

#### C3. "Is intraday momentum still alive? 1,085 SPX sessions of the 0DTE era" (firmtape, dev.to, 2026) `[secondary: practitioner]`
- URL: https://dev.to/firmtape/intraday-momentum-is-dead-in-the-0dte-era-we-measured-it-on-1085-spx-sessions-43g0
- SPX cash index, 14-Apr-2022 → 20-Aug-2026; daily regression of the last-30-min return on the open→15:30 return.
- Result: slope +0.009 (t 0.4) 2022, +0.002 (0.1) 2023, −0.016 (−0.7) 2024, +0.016 (1.0) 2025, −0.012 (−0.6) 2026, **all +0.006 (t 0.6)**. Conditioning on the 0DTE dealer book signed at 15:30: short-gamma sessions (15 % of days) slope +0.055 (t 3.1).
- Critique: index level, not tradable prices; one author; but it is the only 2025–26 number in this topic and it agrees with C2's mechanism. Our SPY 1-min bars reproduce this in minutes (H-C5).

#### C4. Heston, Korajczyk, Sadka — "Intraday Patterns in the Cross-Section of Stock Returns" (JF 65(4), 2010) and Haendler, Heston, Korajczyk, Sadka — "The Intra-day Stock Return Periodicity Puzzle" (2025) `[primary (2010 draft); secondary (2025)]`
- URLs: https://www.bauer.uh.edu/departments/finance/documents/Heston-Korajczyk-Sadka-jf-2010-01-07.pdf ; https://www.kellogg.northwestern.edu/academics-research/research/detail/2025/the-intra-day-stock-return-periodicity-puzzle/
- US stocks, TAQ, 2001–2005 (2010 paper), 13 half-hour intervals; Fama–MacBeth of interval-k return on lag-j interval returns.
- Result: negative autocorrelation at short lags (bid-ask bounce, sub-hour liquidity imbalances) but **positive continuation at lags that are multiples of 13 half-hours (same time tomorrow, the day after…), significant for ≥ 40 trading days**; smallest daily-lag t-stat over the first week 9.62. Decile L/S "same half-hour tomorrow" earns **3.01 bps per half-hour** on day 1; the opening half-hour daily-lag decile spread earns >11 bps and the closing one >8 bps (vs −8 / −11 bps for the non-daily lags); best daily winners +1.66 bps, worst daily losers −1.35 bps per half-hour. Effective half-spread in the sample 1.7 bps, so "timing trades can reduce execution costs by the equivalent of the effective spread" — the authors frame it as an execution-timing tool, not a stand-alone book.
- 2025 update `[secondary]`: the periodicity persists out of sample; trading frictions and trader-type proxies explain all of the open and mid-day periodicity and up to 30 % of the close periodicity; open periodicity ≈ VWAP algos, close periodicity ≈ market-on-close flow.
- Critique for us: at 1–3 bps per half-hour the signal is below the spread in anything but the top-liquidity names; it is an execution rule (when to place an order we already want) rather than a source of trades.

#### C5. Baltussen, Da, Soebhag — "End-of-Day Reversal" (SSRN 5039009; Dec 2024, rev. Apr/May 2025) `[primary]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009 ; https://academicweb.nd.edu/~zda/EOD.pdf ; Alpha Architect summary https://alphaarchitect.com/end-of-trading/
- US common stocks in TAQ ≥ 126 days history, 1993–2019/2020 (27 years), second-level prices; excludes stocks below the 10th NYSE size percentile, $5 price filter in some specs; returns cross-sectionally winsorized 1/99.
- Definitions: ROD3 = prev close → 15:00 (skips 15:00–15:30 deliberately to avoid bounce); LH = 15:30 → 16:00.
- Rule: at 15:30 sort on ROD3; **long the bottom quintile (intraday losers), short the top**, hold 15:30→16:00, flat otherwise. Inputs known at 15:30.
- Result: **VW L/S 3.78 bps/day (t = 10.69), ≈ 9.5 %/yr; EW 6.86 bps/day, ≈ 17.3 %/yr**; six-factor alpha 3.71 bps (t 10.6); decile version 6.38 bps (t 17.3); smallest size quintile 14.71 bps/day (t 27.2), largest 3.41 bps (t 10.6); present in every 3-year window, in the most liquid/largest names, in mid-quote returns, when the last 5 minutes are skipped, and after controlling for lagged-LH seasonality (C4), gamma hedging (individual and index options, LETFs), Stambaugh–Yu–Yuan mispricing, earnings and news days. **Entirely from the negative-ROD3 side**: ROD3 × 1[ROD3<0] t = −6.97, ROD3 alone t = 0.23. **Transitory**: adding the next overnight to LH kills the coefficient (−0.12, t −0.47); adding through next close flips it (0.58, t 1.42). The bottom ROD3 decile held only 15:30–16:00 compounded ~+400 % over 27 years; the top decile ~0. Mechanism: retail buy-the-dip in LH (three retail measures: small trades 1993–2000, BJZZ 2010–19, Robinhood holdings 2018–19 — all show LH buying of extreme-ROD3 names, much stronger for losers), plus new short positions in LH fall >3× more for losers than winners (overnight risk management). Retail is *extrapolative* at the open and *contrarian* at the close.
- Costs: "the strategy as presented might not be exploitable by many investors after accounting for transaction costs"; proprietary desks and execution-timing users can capture it.
- Reconciliation with C1/C2: individual stocks show time-series momentum, but cross-stock autocorrelation dominates, so the cross-section shows reversal (their decomposition: XS strategy −3.21 bps, TS +2.79 bps, cross-covariance −5.97 bps/day).
- Post-publication: none yet beyond the 2025 revision. Our ≥5 %-range 1-min universe is the extreme-ROD3 subsample (H-C4).

#### C6. Brogaard, Han, Kim — "Intraday Residual Reversal in the U.S. Stock Market" (SSRN 4731947, 2024) `[title only — could not be opened]`
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4731947
- Listed because it is the 2024 cross-sectional intraday reversal paper using residual (factor-adjusted) intraday returns; magnitudes unverified. Treat C5 as the reference until this is read.

### D. The close and the open as mechanisms

#### D1. Bogousslavsky, Muravyev — "Who Trades at the Close? Implications for Price Discovery and Liquidity" (JFM 66, 2023) `[primary: June-2021 draft]`
- URL: https://www.ssrn.com/abstract=3485840 ; https://www.sciencedirect.com/science/article/abs/pii/S1386418123000502
- US stocks, 2010–2018, TAQ + auction prints.
- Result: closing auction = **7.48 % of daily dollar volume in 2018 (3.11 % in 2010)**, $15.2B/day; growth driven by ETF/passive ownership (diff-in-diff: +20 % closing volume after S&P 500 addition, −15 % after deletion); spikes on index-rebalance, month-end and option-expiry days, lower on earnings days. Closing price ≠ 4:00 midpoint in 98.2 % of auctions but **mean |deviation| 8.1 bps vs mean half-spread 7.6 bps; matches the pre-close bid or ask 68.5 % of the time; exceeds the half-spread in 23.4 %; > 63 bps in 1 % of cases; 20.6 bps small caps vs 2.66 bps large caps**; +0.81 bps per +1 % auction turnover; NYSE 1.2 bps larger than Nasdaq (D-orders). **Deviations "reverse almost fully overnight, even adjusted for the half spread" — ~85 % by the next morning; for names with after-hours liquidity one-third to one-half of the reversal is done within 30 minutes of the close.** Little price discovery at the auction (variance ratios, WPC). Side effect: opening liquidity deteriorated over 2010–18 for S&P 500 stocks — first-15-minute turnover −22 %, effective spread +10 bps, depth −63 %, more adverse selection at the open.
- Trading reading: fade the closing print against the 3:59 midpoint (short if it printed at/above the ask on volume, long if at/below the bid), exit after hours or at the open. Gross ≈ the deviation beyond the half-spread (a few bps on average, 20–60 bps in the tail); the exit crosses the open's (worse) spread.

#### D2. Jegadeesh, Wu — "Closing Auctions: Nasdaq versus NYSE" (JFE 143(3), 2022, 1120–1139) `[abstract]`
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0304405X21005092 ; https://ideas.repec.org/a/eee/jfinec/v143y2022i3p1120-1139.html
- Closing volume peaked ~10 % of total in 2019. Auctions attract uninformed/passive flow; the cost of trading in the auction is generally lower than in continuous trading; NYSE offers more depth. **The temporary component of auction price impact takes 3–5 days to fully dissipate**, and "trading strategies that exploit this price impact and its reversals are significantly profitable" (magnitudes not in the abstract).
- Reconcile with D1: D1 measures the *deviation from the pre-close midpoint* (mostly gone by the open); D2 measures the *imbalance-driven price impact* (multi-day). Both say: the close's move is partly noise you can fade.

#### D3. Goyal, Jegadeesh, Wu — "Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets: A Benchmark for Cost of Trading on Anomalies" (JFQA, 2026) `[abstract]`
- URL: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-impact-in-closing-auctions-opening-auctions-and-continuous-markets-a-benchmark-for-cost-of-trading-on-anomalies/0F72910A79C5B42CF6E85F55164CE846
- US stocks Jan-2012 → Dec-2021. Square-root impact model beats linear. **Price impact is lower in the closing auction than in continuous trading (except Nasdaq microcaps); opening auctions are illiquid with substantially larger impact.** Annualized execution cost of L/S anomaly portfolios 17–41 bps (all stocks), 9–21 bps ex-microcaps.
- Rule for us: overnight books should be *entered* at the close (MOC/LOC) and *exited* in continuous trading after the first minutes or at the close, never MOO. Any overnight hypothesis costed at "spread × 2" is optimistic on the open side and pessimistic on the close side.

#### D4. Hu, Murphy — "Vestigial Tails? Floor Brokers at the Close in Modern Electronic Markets" (Management Science, 2024/25) `[secondary]`
- URL: https://pubsonline.informs.org/doi/10.1287/mnsc.2023.00884
- Closing-auction price changes reverse more on NYSE than Nasdaq; NYSE's late floor-broker (D-order) entries create larger last-minute abnormal imbalances, especially where floor-broker fees inhibit auction competition. Magnitudes not verified.

### E. Opening gaps and gap-fill

#### E1. Plastun, Sibande, Gupta, Wohar — "Price Gap Anomaly in the US Stock Market: The Whole Story" (NAJEF 52, 2020) `[primary]`
- URL: https://repository.up.ac.za/handle/2263/78336 ; https://www.sciencedirect.com/science/article/abs/pii/S1062940820300747
- DJI, S&P 500 (1928–2018), Nasdaq; daily; gap = open ≠ prior close (index level, so gaps are small and frequent).
- Result: **on the gap day prices continue in the gap direction; "no evidence was found that price gaps in the US stock market were filled within five days"; no seasonality; the anomaly is a one-day momentum and it has decayed.** Trading simulation (buy at the open of an up-gap day / sell at the open of a down-gap day, close at the close): S&P 500 win rate 99 % in 1929–38 and 1949–68, 83–94 % 1969–78, 61–69 % 1979–2018; 2009–2018: up-gaps 106 trades, 67 % successful, +5.9 %/yr; down-gaps 107 trades, 61 %, +5.6 %/yr (z ≈ 4). Gross of costs.
- Critique: index-level gaps of a few bps are a different animal from a $7 stock gapping 15 %; the "no fill" finding cannot be transferred to single stocks. The test design (gap-day continuation vs. a random-entry benchmark) is the right one to copy.

#### E2. Practitioner QQQ gap-fill table (epicctrader.com, data through 14-Feb-2023) `[secondary]`
- URL: https://epicctrader.com/gap-fills/
- QQQ, 10-Mar-1999 → 14-Feb-2023, 6,005 sessions. Gaps > 0.5 % on 39.9 % of days, > 1 % on 17.0 %, > 2 % on 4.6 %. **Same-day full fill: down-gaps 59.4 % (0.5–1 %), 46.7 % (1–2 %), 28.9 % (2 %+); up-gaps 53.5 %, 44.7 %, 32.9 %.** Within two days: down 74 / 57 / 38 %, up 63 / 53 / 43 %. Half-fill same day: 65–78 %. Down-gaps fill more than up-gaps (long-run uptrend).
- Use: a prior for the shape (fill probability falls steeply with gap size; half-fills are common). No costs, no strategy P&L, ETF only.

#### E3. "Price Gaps and Volatility: Do Weekend Gaps Tend to Close?" (JRFM 18(3):132, 2025) `[secondary: search summary]`
- URL: https://www.mdpi.com/1911-8074/18/3/132
- DJIA, Nasdaq, DAX, 2013–2023, weekend gaps. Question posed: are moves into the gap "genuine gap-closing" or just volatility? The summary indicates the fills are largely explained by volatility, not by a gap-seeking force. Magnitudes not verified.

#### E4. What is missing in the literature
No refereed paper reports **single-stock** opening-gap fill probability by gap size × market cap × prior-day attention with 2022+ data. Grant–Wolf–Yu (2005, index futures) found intraday reversals after large positive opening changes `[cited in E1]`; Berkman et al. (B4) is the closest single-stock result (attention gap-ups fade in the first hour). This is a gap our daily bars fill directly (H-C8, H-C9).

### F. Retail order flow and attention

#### F1. Boehmer, Jones, Zhang, Zhang — "Tracking Retail Investor Activity" (JF 76(5), 2021, 2249–2305) `[abstract]`
- URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13033 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2822105
- US stocks, TAQ 2010–2015 (from memory); retail marketable orders identified by sub-penny price improvement (buys at x.xx01–x.xx40, sells at x.xx60–x.xx99); weekly order imbalance.
- Result: **stocks with net retail buying outperform those with net selling by ≈ 10 bps over the following week**, persisting for several weeks; < half of the predictive power is order-flow persistence; the rest is not contrarian liquidity provision or news sentiment — "suggestive, but only suggestive" evidence of information.
- Costs: 10 bps/week on a weekly-rebalanced L/S is inside the spread for small caps; the paper does not net costs.
- Post-publication: F2 (identification errors), F3 (the paradox), Battalio–Jennings–Salgam–Wu 2024 `[memory]` (wholesaler execution changes since 2020 alter the sub-penny signature).

#### F2. Barber, Huang, Jorion, Odean, Schwarz — "A (Sub)penny for Your Thoughts: Tracking Retail Investor Activity in TAQ" (JF 79(4), 2024) `[abstract]`
- URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13334 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4202874
- 85,000 own trades in six retail accounts, Dec-2021 → Jun-2022. The BJZZ algorithm identifies 35 % of their trades as retail, **mis-signs 28 % of the identified trades**, and gives uninformative imbalance measures for 30 % of stocks; signing by quote-midpoint keeps identification and cuts signing error to 5 %.
- Implication: every 2010–2019 retail-imbalance result is measured with substantial noise; effects survive but magnitudes are attenuated.

#### F3. Barber, Lin, Odean — "Resolving a Paradox: Retail Trades Positively Predict Returns but Are Not Profitable" (JFQA, 2023) `[abstract]`
- URL: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/resolving-a-paradox-retail-trades-positively-predict-returns-but-are-not-profitable/6AAA9078F50C2597F44D73FA6A8E3F0D
- Equal-weighted tests hide that retail buying concentrates in attention stocks that then underperform: **extreme-quintile retail-imbalance L/S earns −14.8 %/yr in heavy-retail-trading stocks vs +6.6 %/yr in other stocks**; smaller trades are worse. So "follow retail" works only outside the retail habitat, and "fade retail" works inside it — the same partition as B7.

#### F4. Barber, Huang, Odean, Schwarz — "Attention-Induced Trading and Returns: Evidence from Robinhood Users" (JF 77(6), 2022, 3141–3190) `[abstract; magnitude secondary]`
- URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13183 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3715077
- Robintrack hourly user-holdings, May-2018 → Aug-2020. Herding = large increases in the number of Robinhood holders. **Top daily herding stocks: average 20-day abnormal return −4.7 %**; Robinhood outages disproportionately cut trading in high-attention stocks; the app's design and its inexperienced users both contribute.
- Costs: at −470 bps over 20 days the signal is cost-insensitive; the binding constraint is borrow availability/fees in exactly these names. Data ended with Robintrack in Aug-2020; attention must be proxied (volume spikes, range, gap) — H-C11.

#### F5. Related retail-timing evidence used above (not separately reviewed)
- Robinhood "Examining high-frequency patterns in Robinhood users' trading behavior" (IRFA 2025) `[title only]` — retail activity concentrated near the open.
- Jones, Pyun, Wang (2024) `[memory, cited by C5]` — retail investors are extrapolative at the open.
- Barardehi, Bernhardt, Da, Warachka (2022) `[memory]` — retail liquidity provision and the intraday premium.

---

## 2. Cross-cutting critiques (apply before believing any number above)

1. **Spread crossed twice a day.** Every overnight-only or last-30-minutes-only result is gross. The honest netting is: buy at the close (auction ≈ midpoint ± half-spread; impact low per D3), sell at the open (auction ≈ high impact per D3, or continuous at 9:31 with the widest spread of the day per D1). For US large caps with 1–3 bps quoted spreads, a round trip costs 2–6 bps + impact; for $5–20 small caps at ~40 bps, 40–80 bps. Against gross overnight effects of 2–16 bps/night, large caps are marginal and small caps are dead — which is why B1, B2 and A4 all end with "order timing for trades you were making anyway".
2. **Open-price definition.** Trade-price vs mid-quote opens change the S&P night return from 4.76 to 2.82 bps (A1). Our daily-bar open must be validated against auction prints (see the caveat at the top).
3. **Post-2021 regime.** The index drift is gone (A5), market intraday momentum is flat (C3), closing-auction share and 0DTE gamma have changed the last 30 minutes. Only C5 (end-of-day cross-sectional reversal, through 2020 with every 3-year window positive) and the retail-attention effects (A4 update, B4/C5) have any evidence near our window. Everything else needs its 2025–26 number measured before it is a prior.
4. **Retail flow identification noise** (F2) attenuates every retail-imbalance magnitude; attention *proxies* from prices/volume (B4, C5) are cleaner for us because we have them.
5. **Survivorship.** Most academic samples are CRSP/TAQ with delistings included; practitioner tables (E2) and Elm's stock lists (A4) are survivor-selected. Our point-in-time daily panel is the right one; do not seed any gap test from a current symbol list.
6. **Short side.** Fading the open in attention names and shorting the intraday winners at 15:30 both require borrow in hard-to-borrow $5–20 names. Long-only versions (long the ROD3 losers; long the post-sell-off overnight) are what is realistic, and C5 says the long side is where the effect lives.

---

## 3. What our data can and cannot do

| Need | Have? | Notes |
|---|---|---|
| Overnight/intraday decomposition, whole market, 2025–26 | Yes (daily OHLC, PIT incl. delisted) | Validate `open` vs auction print first. |
| Overnight beta, attention proxies (|ret|, volume ratio, range, gap) | Yes (daily) | Rolling betas from daily close-to-close. |
| Gap size, gap fill (low ≤ prev close / high ≥ prev close), gap-day continuation | Yes (daily H/L) | Fill = touch, not close; both are computable. |
| SPY first-30 / last-30 minute returns | Yes (SPY 1-min) | C1/C2/C3 replication is immediate. |
| Single-stock last-30-min reversal on extreme ROD3 days | Partial (1-min only for ≥5 % range symbol-days) | That subsample *is* the extreme-ROD3 sample; the rest via Alpaca REST. |
| Closing-print deviation vs 15:59 midpoint | Approximate (1-min last trade vs close; no quotes) | Need Alpaca quotes/auctions for a clean version. |
| Retail order imbalance (sub-penny), Robintrack | No | Only attention proxies. |
| Premarket/overnight news per ticker | Yes (ORB news pipeline, Alpaca/Benzinga) | Lets us test A6-style news × overnight splits. |
| Dealer gamma / 0DTE positioning | No | C3's conditional result cannot be replicated. |

---

## 4. Hypothesis table

Conventions: effect sizes are **gross** per trade unless stated; "large" = US large cap, quoted spread ~1–3 bps;
"small" = $5–20 names, ~40 bps. Cost sensitivity = gross prior minus one round trip (close entry + open
exit or one intraday round trip). "Testable now" = daily bars only. Falsification is on our 2025-01 →
2026-08 point-in-time panel unless noted; t-stats Newey–West; "months green" out of 20.

| H-id | One-sentence rule | Prior effect (gross) | Inputs known at decision time | Data needed | Falsification criterion on 2025–26 | Cost sensitivity | Testable now on daily bars? |
|---|---|---|---|---|---|---|---|
| **H-C1** Index overnight long | Buy SPY at the close, sell at the open, every day. | Historic 3–5 bps/night (A1), 2.75 bps ON−ID (A6); **post-2021 ≈ 0** (A5, Elm 5-yr SPY overnight $1.66 vs $2.07 whole-day). Prior for 2025–26: 0–1 bps/night, Sharpe < 0.4. | Yes (none needed). | SPY daily O/C. | Dead if mean overnight SPY return ≤ 0 or t < 2 over 2025–26; also compare to intraday mean. | SPY spread ~1 bp → net ≈ prior − 1–2 bps → ≤ 0. | **Yes.** |
| **H-C2** Cross-sectional overnight continuation (B1) | At each month-end (or weekly, per B5) rank on trailing 21-day cumulative overnight return; long top decile close→open only; optional short bottom decile. | VW decile L/S 3.47 %/mo overnight alpha ≈ 16 bps/night (t 16.8), long leg ~8 bps/night; weekly decile spread 1.76 pp/week (B5); Elm: insufficient vs 1 bp/trade since 2014. Prior for 2025–26 large caps: long leg 3–6 bps/night. | Yes (prior opens/closes). | Daily O/C, market cap for VW and cap buckets. | Dead if VW top-minus-bottom decile overnight spread < 5 bps/night or t < 2.5 or < 12/20 months green; long-leg-only version must beat the SPY overnight by ≥ 3 bps/night. Run separately for ≥ $10B and $5–20 small caps. | Large: 4–8 bps gross − 2–6 bps cost → marginal (0–4 bps). Small: 8–16 bps − 40–80 bps → dead. | **Yes.** |
| **H-C3** Fade the open in prior-day attention stocks (B4, A1, C5) | For symbol-days whose prior day had top-decile |return| × volume ratio (our ≥ 5 %-range flags), expect open > subsequent prices: sell/short at 9:30–9:35, cover at 10:30 (or avoid buying before 10:30). | S&P stocks first hour −1 to −3.6 bps (A1); attention names' reversal > effective half-spread (B4); extreme-attention days: prior in the −30 to −80 bps open→10:30 range for $5–20 names (order-of-magnitude, not measured in any paper). | Yes (prior-day bars, opening print). | Daily O/C (open→close part); 1-min bars for the first-hour part (have them for ≥ 5 %-range days). | Dead if mean open→10:30 return on prior-day top-decile attention names is > −20 bps (small) / > −5 bps (large) or t > −2; also require positive overnight return on the same names (else it is not the B4 mechanism). | Short side pays ~40 bps + borrow in small caps → needs ≥ 60 bps gross; the *no-chase* version (delay entries to 10:30) is free. | **Partly** (overnight vs open-to-close on daily bars now; first-hour needs 1-min). |
| **H-C4** End-of-day reversal, long side (C5) | At 15:30 rank on prev-close→15:00 return; buy the bottom decile at 15:30, sell MOC at 16:00. | VW L/S 3.78 bps/day (t 10.7), EW 6.86, smallest quintile 14.7 bps; long (loser) side is ~all of it; reverts next morning. Prior for our extreme-loser subsample: +10–20 bps per trade. | Yes (15:00 price, prior close). | 1-min bars 15:00–16:00 (have for ≥ 5 %-range days; Alpaca REST for a control sample); official close. | Dead if bottom-decile ROD3 names' 15:30→16:00 mean < +5 bps or t < 2, or if the next-day overnight does not give back ≥ half (mechanism check). | Enter with limits at 15:30 (~½ spread), exit MOC (~½ spread, low impact per D3): large 1–3 bps → fine; small ~40 bps → needs the ≥ 15 bps decile tail only. | No (needs intraday). |
| **H-C5** Market intraday momentum (C1, C2, C3) | At 15:30 long SPY last 30 min if prev-close→15:30 return > 0, short if < 0. | 6.7–6.9 %/yr, Sharpe 1.1–1.7 (1974–2020); **2022–26 slope +0.006, t 0.6** (C3). Prior: ≈ 0 unconditional. | Yes. | SPY 1-min. | Dead if the regression slope of r(15:30–16:00) on r(prev close–15:30) has t < 2 on 2025–26, or the timing strategy's Sharpe < 0.5 gross. | SPY 1 bp/side → nearly cost-free; still dead if the slope is 0. | No (SPY 1-min, which we have). |
| **H-C6** Overnight beta tilt (B2) | Long top-beta decile / short bottom-beta decile close→open; flat or reversed intraday. | Night SML +14 bps/β, top-decile night +20 bps, day −8 bps (1992–2016); portfolio strategy 44 bps/day gross. Prior for 2025–26 large caps: 5–10 bps/night decile spread. | Yes (rolling beta from daily bars). | Daily O/C. | Dead if the night SML slope (decile portfolios) ≤ 0 or t < 2 on 2025–26; also require the day slope < 0 (else the mechanism has changed). | Two legs, two crossings/night: large 4–12 bps → survives only in the largest names; small: dead. | **Yes.** |
| **H-C7** Fade the closing print (D1, D2) | If close deviates from the 15:59 midpoint by > half-spread (proxy: close vs last 1-min bar), take the opposite side in after-hours or at 9:31, exit by 10:00. | Mean |deviation| 8 bps (20.6 small caps), 85 % reverts by next morning; 1 % tail 63 bps. Prior: +5–15 bps per trade on the > half-spread subset. | Yes (close print vs 15:59). | 1-min bars to 16:00 + official close (approximate); quotes for a clean version. | Dead if regressing open(t+1)/close(t) − 1 on close/mid(15:59) − 1 gives slope > −0.5 or t > −2 on the ≥ 5 %-range sample. | Exit at 9:31 crosses the widest spread of the day: large 2–4 bps → OK on the tail; small 40 bps → only the > 40 bps tail. | No (needs 1-min close bars; partial with our subset). |
| **H-C8** Gap-day continuation vs reversal in single stocks (E1 vs B4) | On gap-up days (open/prev close − 1 > x %), test whether close > open (continuation, E1) or close < open (attention reversal, B4) by gap size, cap, prior-day attention and news. | Index: continuation 61–67 % hit rate 1989–2018 (E1). Single-stock attention gap-ups: reversal (B4). Prior: continuation for large-cap news gaps, reversal for small-cap no-news gaps; magnitude unknown. | Yes (open print). | Daily OHLC; news flags (have). | Report the 2×2 (cap × news) table of P(close > open) and mean open→close; a cell is "live" only if hit rate ≠ 50 % with t ≥ 3 and mean |open→close| ≥ 2 × cost. | Open entry at 9:30 pays the open's impact (D3); large 3–5 bps, small 40 bps. | **Yes.** |
| **H-C9** Single-stock gap-fill probability (E2, E4) | For gap-ups of size g in cap bucket c, P(low ≤ prev close same day) and P(half-fill); the missing table. | QQQ: 53–59 % same-day fill for 0.5–1 % gaps, 29–33 % for 2 %+; single stocks unknown; prior for 5–15 % small-cap gaps: 25–40 % same-day full fill, 50–65 % half-fill. | Yes (open print). | Daily OHLC. | This is a measurement, not a bet: publish the table; a fade-the-gap rule is live only if E[payoff] = P(fill) × gap − (1 − P(fill)) × adverse move > 2 × cost in that cell. | Small caps: 40 bps vs 5–15 % gaps → cost-insensitive on the bet, borrow-sensitive on the short. | **Yes.** |
| **H-C10** Retail imbalance follows (F1–F3) | Long stocks with top-decile weekly retail buy imbalance, short bottom decile, hold one week. | +10 bps/week (F1) but −14.8 %/yr inside heavy-retail stocks (F3); signing noise 28 % (F2). | Needs TAQ sub-penny flags — **not available**. | TAQ. | Not testable; keep as context for H-C3/H-C11. | 10 bps/week vs 40 bps small → dead; large marginal. | No (no data). |
| **H-C11** Robinhood-herding fade via attention proxy (F4) | Each day flag the top-N names by (volume ratio × positive return × range); short (or avoid longs) for 5–20 days. | −4.7 % over 20 days after top Robinhood purchases (2018–20). Proxy prior: −1 to −3 % over 20 days for small-cap attention spikes. | Yes (prior-day bars). | Daily OHLCV. | Dead if 20-day mean return after top-decile attention days is > −1 % or t > −2; require monotonicity across attention deciles. | Cost-insensitive (hundreds of bps); borrow-constrained. | **Yes.** |
| **H-C12** Same-half-hour periodicity (C4) | For a stock that moved strongly in half-hour k yesterday, expect the same sign in half-hour k today; use as entry timing for planned trades. | 3 bps per half-hour decile spread; 11 bps in the opening half-hour (2001–05); persists OOS per 2025 update. | Yes. | Full-cross-section 1-min bars — only the ≥ 5 %-range subset now. | Dead if the lag-13 (same-time-next-day) Fama–MacBeth coefficient has t < 2 on our subset. | 3 bps < any spread → execution-timing only, never a stand-alone trade. | No (needs 1-min; partial). |
| **H-C13** News × overnight split (A6) | Among names with own-ticker overnight/premarket news, compare overnight vs intraday returns to no-news names; hold close→open only on persistent-news exposure. | Top news-forecast overnight picks 11.0 bps/night vs 3.4 (2001–22), 6.7 vs 4.0 (2011–22); authors call it non-viable as a strategy. | Yes (news timestamps ≤ 9:30). | Daily O/C + our news pipeline. | Dead if news-exposed names' mean overnight return − no-news names' < 2 bps/night or t < 2. | Same double-crossing as H-C2 → large only. | **Yes** (daily bars + existing news CSVs). |
| **H-C14** Tug-of-war intensity (B6) | Monthly: count days with (overnight > 0 and intraday < 0); long top decile / short bottom decile next month close-to-close. | Sign known (positive), magnitude unverified `[memory: ~0.5–1 %/mo]`. | Yes (prior-month opens/closes). | Daily O/C. | Dead if decile spread < 30 bps/month or t < 2 over the 20 months; otherwise the cheapest signal in this review. | Monthly turnover → cost-insensitive even in small caps (one crossing/month ≈ 40 bps vs 50–100 bps/mo). | **Yes.** |
| **H-C15** Post-sell-off overnight reversal on SPY (A3) | If today's SPY intraday return is in the bottom quintile (proxy for negative close imbalance), buy at the close, sell at the open. | Asymmetric reversal after sell-offs (A3); window ≈ 0 since 2021 (A5). Prior: +2–4 bps/night conditional, ≈ 0 unconditional. | Yes (close-location proxy). | SPY daily O/C (better: 1-min close-location). | Dead if the conditional overnight mean after bottom-quintile days is < 2 bps or t < 2. | SPY 1 bp → tradable only if ≥ 3 bps conditional. | **Yes.** |

---

## 5. Test order and design notes

1. **Data validation first**: on 50 NYSE + 50 Nasdaq names over 20 days compare our daily `open`/`close` with Alpaca auction prints; measure the median |gap| in bps. If > 1 bp, rebuild opens from the auction print before any overnight test.
2. **Run the daily-bar batch in one script** (H-C1, C2, C6, C8, C9, C11, C13, C14, C15): they share one panel (overnight, intraday, gap, range, volume ratio, cap bucket, news flag). Report per-hypothesis: mean bps, NW t, months green, cap-bucket split, cost-netted mean at 2 bps (large) and 40 bps (small).
3. **Pre-commit the falsification rows above** before looking at results; write the numbers into this file under each H-id.
4. **Then the 1-min batch** (H-C4, C5, C7, C12) on SPY + the ≥ 5 %-range subset, with an Alpaca-REST control sample of 200 random symbol-days so the subset's selection (range ≥ 5 %) is not mistaken for the effect.
5. **Interpretation rule from CLAUDE.md applies**: these are relative tools. Nothing here is a P&L forecast; a hypothesis that survives §4 earns a shadow window, not a book.

---

## 6. Strongest three (what to run first)

- **H-C2, large caps only** — the cross-sectional overnight continuation. Largest, longest-lived, best-documented cross-sectional effect in the topic (t = 16.8, 60-month persistence, confirmed at the annual horizon through 2022). Prior 3–6 bps/night long-leg in ≥ $10B names after a 2–4 bps round trip; the small-cap version is dead on cost by construction. Testable today.
- **H-C3, the no-chase version** — prior-day attention names open high and fade in the first hour (A1, B4, C5's extrapolative-retail-at-the-open evidence, Elm's attention-stock update). For us this is not a new book but a rule about our existing ones: do not buy the 9:30–10:00 print in a name that was on yesterday's movers list; prior −20 to −80 bps open→10:30 in small caps, free to implement. The overnight-vs-intraday split is testable today; the first-hour piece uses 1-min bars we already hold.
- **H-C9 + H-C8, the gap table** — no refereed paper gives single-stock gap-fill / gap-continuation probabilities by gap size × cap × news; the index evidence (continuation, no five-day fill) and the QQQ table (fills fall from ~55 % to ~30 % as gaps go from 1 % to 2 %+) are the only priors. It is a measurement our PIT daily bars make in an afternoon, and it feeds directly into how ORB/HOD treat gap-ups.

---

## Sources

- Lou, Polk, Skouras (2019) JFE — https://personal.lse.ac.uk/polk/research/TugOfWar.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650
- Hendershott, Livdan, Rösch (2020) JFE — http://faculty.haas.berkeley.edu/hender/CAPMday-night.pdf ; https://www.ssrn.com/abstract=3117663
- Bogousslavsky (2021) JFE — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2869624 ; https://econpapers.repec.org/article/eeejfinec/v_3a141_3ay_3a2021_3ai_3a1_3ap_3a172-194.htm
- Cliff, Cooper, Gulen (2008/2011) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1004081 ; https://link.springer.com/article/10.1057/jam.2011.2
- Bondarenko, Muravyev (2023) JFQA — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3596245
- Boyarchenko, Larsen, Whelan (2023) RFS / NY Fed SR 917 — https://www.newyorkfed.org/research/staff_reports/sr917 ; https://academic.oup.com/rfs/article-abstract/36/9/3502/7076616
- NY Fed Liberty Street (July 2026) "The Disappearing Overnight Drift" — https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/
- Haghani, Ragulin, Dewey (2022) "Night Moves" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4139328 ; https://elmwealth.com/night-moves-overnight-drift/ ; update https://elmwealth.com/night-shift/
- Glasserman, Krstovski, Laliberte, Mamaysky (2025) — https://arxiv.org/abs/2507.04481 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5336382
- Berkman, Koch, Tuttle, Zhang (2012) JFQA — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1625495
- Aboody, Even-Tov, Lehavy, Trueman (2018) JFQA — https://anderson-review.ucla.edu/wp-content/uploads/2021/03/Aboody-et-al_overnight_returns_and_firmspecific_investor_sentiment_JFQA2018.pdf
- Akbas, Boehmer, Jiang, Koch (2022) JFE — https://www.sciencedirect.com/science/article/abs/pii/S0304405X21004116
- Laarits, Sammon (2022) "The Retail Habitat" — https://as.nyu.edu/content/dam/nyu-as/econ/documents/fall-2022/Laarits%20Seminar%20Paper.pdf
- Gao, Han, Li, Zhou (2018) JFE — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866
- Baltussen, Da, Lammers, Martens (2021) JFE — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3760365 ; https://academicweb.nd.edu/~zda/intramom.pdf
- Limkriangkrai, Chai, Zheng (2023) PBFJ "Market intraday momentum: APAC evidence" — https://researchmgt.monash.edu/ws/files/519509174/494419119_oa.pdf
- firmtape (2026) 0DTE-era check — https://dev.to/firmtape/intraday-momentum-is-dead-in-the-0dte-era-we-measured-it-on-1085-spx-sessions-43g0
- Heston, Korajczyk, Sadka (2010) JF — https://www.bauer.uh.edu/departments/finance/documents/Heston-Korajczyk-Sadka-jf-2010-01-07.pdf ; Haendler et al. (2025) — https://www.kellogg.northwestern.edu/academics-research/research/detail/2025/the-intra-day-stock-return-periodicity-puzzle/
- Baltussen, Da, Soebhag (2024/25) "End-of-Day Reversal" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009 ; https://academicweb.nd.edu/~zda/EOD.pdf ; https://alphaarchitect.com/end-of-trading/
- Brogaard, Han, Kim (2024) "Intraday Residual Reversal" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4731947
- Bogousslavsky, Muravyev (2023) JFM "Who Trades at the Close?" — https://www.ssrn.com/abstract=3485840 ; https://www.sciencedirect.com/science/article/abs/pii/S1386418123000502
- Jegadeesh, Wu (2022) JFE — https://www.sciencedirect.com/science/article/abs/pii/S0304405X21005092
- Goyal, Jegadeesh, Wu (2026) JFQA — https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-impact-in-closing-auctions-opening-auctions-and-continuous-markets-a-benchmark-for-cost-of-trading-on-anomalies/0F72910A79C5B42CF6E85F55164CE846
- Hu, Murphy "Vestigial Tails?" Management Science — https://pubsonline.informs.org/doi/10.1287/mnsc.2023.00884
- Plastun, Sibande, Gupta, Wohar (2020) NAJEF — https://repository.up.ac.za/handle/2263/78336 ; https://www.sciencedirect.com/science/article/abs/pii/S1062940820300747
- QQQ gap-fill table — https://epicctrader.com/gap-fills/
- JRFM (2025) weekend gaps — https://www.mdpi.com/1911-8074/18/3/132
- Boehmer, Jones, Zhang, Zhang (2021) JF — https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13033 ; https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:5:p:2249-2305
- Barber, Huang, Jorion, Odean, Schwarz (2024) JF — https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13334
- Barber, Lin, Odean (2023) JFQA — https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/resolving-a-paradox-retail-trades-positively-predict-returns-but-are-not-profitable/6AAA9078F50C2597F44D73FA6A8E3F0D
- Barber, Huang, Odean, Schwarz (2022) JF — https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13183 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3715077
