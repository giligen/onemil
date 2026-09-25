# E4 — The Honest Floor and the Math

*Route E4 of the passive-income review. Account: OneMil live Alpaca, equity $65,083, 2026-09-25.
Written as a partner, not a salesman — every number below is disappointing relative to $1-3K/mo, and
that disappointment is the finding.*

## 0. Correction before anything else

The request that triggered this study says *"i believe the only edge is with HOD-break."* That belief
is backwards versus the project's own record. Per `docs/CLAUDE_HISTORY.md` and the 9/22-9/24 commits:
**HOD-break is CLOSED** — zero gross edge across 12,135 signals, a 35-cell exit lab that couldn't beat
the live rule by more than noise (±0.04R), and the order-flow follow-up (cells 1,393-1,395) found the
one positive-looking cell was entirely a cost artifact. **ORB (opening-range breakout) is the one line
with a real, if modest, out-of-sample edge**: like-for-like live config (catalyst veto OFF), 2025-01
through 2026-09, +0.105R/fill, t=3.31, n=473, independently rebuilt. Out of regime it's noisier
(2023-24H1 +0.089R n=106, 2024H2 -0.007R n=59) but the sign holds. Bull flag is unproven out of sample.
Nothing else in nine months of research survived. This report treats that as the ceiling on what the
trading stack can deliver *today*, and asks a separate question: what does the other $65K do while
that edge compounds slowly?

## 1. What $65K earns with zero research

All total-return figures below are **my own computation** from Alpaca daily bars, dividend/split-adjusted
(`adjustment=all`), 2022-01-03 → 2026-09-23 (2026-05-04 for JEPQ, which launched then; 2022-12-28 for
BOXX, which launched then). Script: `research/passive_income/` scratch run, cross-checked against the raw
(price-only) series to confirm dividends were actually baked into the adjustment (`dividends_appear_baked_in`
was `true` for every ticker except BOXX, which pays no material distribution by design — its "yield" is
pure NAV growth from the box-spread roll, so raw and adjusted returns are the same). 2022 drawdown = worst
peak-to-trough within calendar 2022 on the adjusted (total-return) series. Worst/best month = worst/best
calendar-month adjusted return anywhere in the 4.7-yr sample.

| Ticker | What it is | CAGR (total return) | 2022 return | 2022 max DD | Worst month (all-time in sample) | Best month |
|---|---|---|---|---|---|---|
| **BIL** | 1-3mo T-bill ETF | 3.87%/yr | **+1.38%** | -0.04% | -0.01% (Feb-22) | +0.50% |
| **BOXX** | Box-spread ETF (T-bill substitute, favorable tax) | 4.67%/yr | +0.07%* | ~0.0% | +0.18% (never negative in sample) | +0.52% |
| **SHY** | 1-3yr Treasury ETF | 1.97%/yr | -3.76% | -4.92% | -1.41% (Mar-22) | +1.64% |
| **SPY** | S&P 500, dividends reinvested | 12.08%/yr | -18.65% | -24.5% | -9.24% (Sep-22) | +10.51% |
| **JEPI** | JPM equity premium income (S&P, collar+call overlay) | 6.59%/yr | -3.06% | -13.7% | -6.35% (Sep-22) | +7.51% |
| **JEPQ** | JPM Nasdaq premium income (covered call) | 16.35%/yr† | -12.89%† | -20.1% | -8.86% (Sep-22) | +8.18% |
| **QYLD** | Global X Nasdaq covered call (100% notional, monthly ATM) | 8.64%/yr | -19.41% | -24.4% | -7.24% (Sep-22) | +7.14% |
| **XYLD** | Global X S&P covered call | 7.21%/yr | -12.47% | -18.7% | -6.61% (Sep-22) | +5.98% |

\* BOXX return for 2022 is a 3-day partial year (launched 12/28/22), not informative.
† JEPQ's period starts May 2022 mid-crash and is entirely inside the 2023-2026 AI/Nasdaq bull run — its
16.35%/yr is the most regime-flattered number in this table and should be discounted hard for a forward view.

**Current distribution yields** (cited, not computed — verify at fund-provider sites before sizing):
JEPI ~8.1% (stockanalysis.com, Sep-26), JEPQ 30-day SEC yield 12.87% / trailing distribution yield ~11.1%
(J.P. Morgan factsheet, stockanalysis.com, Sep-26), QYLD ~11.5% (stockanalysis.com, Sep-26), XYLD
typically runs a couple points under QYLD (same Global X monthly-ATM mechanic on the lower-vol S&P
instead of Nasdaq) — no single current-quarter figure was retrievable in this session; treat ~9-10% as
an estimate to verify.

**Read the table as an adversary, not a pitch:**
- Every covered-call ETF's *worst month* clusters on the exact same date, Sept-2022 — they are not
  diversifiers from equities, they are equities with the top shaved off and the bottom mostly intact.
  Best month capped at 6-8% vs SPY's 10.5%; worst month only ~2-3pp better than SPY's -9.2%. That is a
  short-volatility payoff, not an income payoff — the "yield" is compensation for selling away the tail,
  not a bond-like coupon.
- QYLD's 2022 calendar return (-19.4%) was *worse* than SPY's (-18.65%) despite the monthly premium,
  because Nasdaq fell faster than the premium could cushion. High distribution yield does not mean low
  risk; QYLD/XYLD/JEPQ are single-stock-index-concentrated equity risk with negative convexity, full stop.
- The only lines in this table that are actually "constant" — worst month inside ±1.4%, no calendar-year
  loss (BOXX has none in-sample; BIL's only close call is +1.38% in the worst year on record for bonds) —
  are BIL/SHY/BOXX, and they pay 2-4.7%/yr. That is the true floor.
- None of these require options approval on the account (the fund does the option-writing internally),
  which matters given `options_approved_level: None` on OneMil.

## 2. The compounding math — what $1K/$2K/$3K/mo actually demands

On $65,083, a *constant* monthly income target is an annualized-return target:

| Target | Annual $ | Required annualized return on $65,083 |
|---|---|---|
| $1,000/mo | $12,000 | **18.4%/yr** |
| $2,000/mo | $24,000 | **36.9%/yr** |
| $3,000/mo | $36,000 | **55.3%/yr** |

Every one of those clears the entire table in §1, including SPY's *best-of-the-decade* 12.08%/yr run
(2022-2026 covers the fastest AI-driven bull market in a generation and it still isn't $1K/mo on $65K).
Compare against documented systematic strategies people actually run at scale, for context on what "good"
looks like institutionally:

| Strategy class | Annualized return | Sharpe | Worst drawdown | Notes / source |
|---|---|---|---|---|
| CBOE S&P 500 PutWrite (PUT) — sell monthly ATM puts vs T-bills | ~6.7%/yr since 2007 | 0.49 (2007-), 0.65 (longer, per CBOE-commissioned research) | -32.7% (vs SPX -50.9%) | 2022 return -7.7% vs SPX -18.1% — genuine cushioning, still equity-correlated. [CBOE PUT factsheet](https://cdn.cboe.com/resources/indices/factsheet/CboeGlobalIndices_PUT-Index.pdf), [Bondarenko 2019](https://cdn.cboe.com/resources/education/research_publications/PutWriteCBOE19_v14_by_Prof_Oleg_Bondarenko_as_of_June_14.pdf) |
| Managed futures / trend CTA, investable proxy (DBMF, since May-2019) | ~9.5-12%/yr since inception | not independently verified this session (vendor-cited ~2.2 trailing, not trustworthy as a long-run figure) | -20.4% since inception | 2022 **+21.5%** (crisis alpha), 2023 **-8.9%** (whipsaw), 2024 +7.3%, 2025 +13.8% — the point is the *sign flips year to year*, this is not constant income. [ETF Trends 5-yr review](https://www.etftrends.com/5-years-dbmf-proves-managed-futures/) |
| Quality factor premium (QMJ), 1964-2023 | +4.7%/yr *excess* over market, vol 9.9% | 0.47 | not cited this session | Academic long-run premium, long-only factor ETFs realize a fraction of this after fees/tracking error. [AlphaArchitect summary](https://alphaarchitect.com/cross-section-of-returns/) |
| Equity market-neutral hedge funds (HFRI/Barclay universe) | low-single to mid-single digits %/yr historically | ~0.5-1.0 in favorable eras (not pinned down to a current figure this session) | short, sharp, correlated tail events: Aug-2007 "quant quake," 2020 "quant bust" | ~1/5th equity volatility in normal times, but **not retail-accessible at $65K** — LP structures, institutional minimums. Directionally cited only; no single hard number retrieved. |

**The honest reading:** even the best-regarded "boring but real" systematic strategies in the world —
put-writing, trend-following, quality factor, market-neutral — run at **mid-single-digit to low-double-digit
percent per year**, with genuine crash risk or multi-year flat stretches, run by full-time teams with
institutional cost structures. None of them clear 18%/yr, let alone 37% or 55%, as a sustained,
low-variance number. A retail account targeting $1-3K/mo *constant* on $65K is not asking for "boring
income," it is asking for a return profile that doesn't exist as a documented, repeatable, low-tail
strategy anywhere in the literature searched for this report. The one thing in the whole project that
gets close on a per-unit-risk basis is ORB, at ~+0.08-0.11R/fill — and it is rate-limited by trade
frequency (`≥3 fills/week` per the live cadence bar), not by return-per-trade, which is a fundamentally
different (and slower, but survivable) way to compound than "find a bigger annualized number."

## 3. What "passive and constant" can honestly mean at $65K

Two honest shapes exist, and they do not blend into a third:

1. **Actually constant, actually passive: BIL/SHY/BOXX.** ~$210-255/mo at current yields (4.3-4.7%/yr
   on BOXX/BIL), worst month ever seen in-sample was -1.4bp to -1.4% (SHY, during the fastest rate-hike
   cycle in 40 years). This is real "set and forget" — but it is roughly **a fifth to a tenth of the
   $1-3K/mo target**, and it will not get you there by itself except over decades (§4).
2. **Higher current income, NOT constant: JEPI/JEPQ/QYLD/XYLD.** $430-700/mo in current cash distributions
   on $65K blended, but the account's *mark-to-market value* moves with equities minus a capped upside —
   worst month realized in-sample was -6.4% to -8.9% (Sept 2022 alone), i.e. **-$4,200 to -$5,800 in a
   single month** on the full $65K if concentrated there. The distribution check still arrives in a
   down month (the premium income is largely uncorrelated to direction), but the account balance does not
   feel "constant" — it feels like owning stock with a coupon stapled on.

There is no third bucket in the honestly-documented, retail-accessible universe that pays $1-3K/mo
*constant* on $65K without either (a) far more capital, (b) leverage/negative-skew products (short
vol, single-name covered calls, 0DTE income strategies — all excluded here as "not boring"), or (c) an
actual trading edge, which is what the other eight months of this project were for and which produced
exactly one surviving line (ORB).

## 4. Capital growth vs. income — when does each target become a normal problem

At a plain 8-12%/yr (the range spanned by SPY's realized CAGR and a diversified equity-income blend,
*not* the CTA/PUT/quality-factor numbers above, which mostly sit lower), the capital required for each
target to be a "no-edge-needed" outcome:

| Target | 8%/yr needs | 10%/yr needs | 12%/yr needs |
|---|---|---|---|
| $1,000/mo | $150,000 | $120,000 | $100,000 |
| $2,000/mo | $300,000 | $240,000 | $200,000 |
| $3,000/mo | $450,000 | $360,000 | $300,000 |

From today's $65,083, with **zero new deposits**, pure compounding at 10%/yr:
- $1,000/mo ($120K) → ~6.2 years
- $2,000/mo ($240K) → ~13.4 years
- $3,000/mo ($360K) → ~18.3 years

That is the boring-anchor timeline, and it is slow **precisely because it asks a fixed-income/equity-beta
portfolio to do the whole job alone**. The one piece of this project that shortens that timeline without
adding tail risk is the ORB edge: it doesn't need a bigger annualized number, it needs more *fills* at the
same edge-per-fill, which is a frequency/capacity problem (more setups, more capital deployed per setup as
the ramp clears), not a "find a strategy with a 40%/yr Sharpe-1 return" problem — because that strategy
does not exist in the documented literature reviewed here.

## Bottom line

- **Zero-research floor on $65K:** ~$210-255/mo, genuinely constant, from BIL/SHY/BOXX. That's the true
  passive number — a fifth of the $1K/mo ask.
- **Higher-yield equity-income sleeve (JEPI/JEPQ/QYLD/XYLD):** $430-700/mo in cash distributions, but
  the account can mark down $4-6K in a single bad month (realized: Sept 2022) — not constant, not
  low-tail, just a different, more honest way to describe "owning stocks."
- **No documented, retail-accessible strategy — T-bills, covered calls, trend CTAs, put-writing, quality
  factor, market-neutral funds — clears 18%/yr (the $1K/mo bar) as a sustained, low-variance number.**
  $2-3K/mo constant is not a "find a better ETF" problem; it is either a 4-7x capital problem or an edge
  problem.
- **The edge problem already has one answer in this project: ORB**, ~+0.08-0.11R/fill, ~$500-800/mo in a
  hot market at current sizing, gated by fill frequency not by return-per-trade — slower than the owner
  wants but the only line here with an actual out-of-sample, cost-charged, independently-rebuilt positive
  number behind it. HOD-break, which the request named as "the edge," has none.

## Sources

- Own computation: Alpaca daily bars, `adjustment=all`, 2022-01-03 to 2026-09-23, SPY/JEPI/JEPQ/QYLD/XYLD/BIL/SHY/BOXX.
- [CBOE S&P 500 PutWrite Index factsheet](https://cdn.cboe.com/resources/indices/factsheet/CboeGlobalIndices_PUT-Index.pdf)
- [Bondarenko, "Historical Performance of Put-Writing Strategies," CBOE-commissioned, 2019](https://cdn.cboe.com/resources/education/research_publications/PutWriteCBOE19_v14_by_Prof_Oleg_Bondarenko_as_of_June_14.pdf)
- [CBOE S&P 500 PutWrite Index — Wikipedia](https://en.wikipedia.org/wiki/CBOE_S%26P_500_PutWrite_Index)
- [iMGP DBi Managed Futures Strategy ETF (DBMF) — ETF Trends, 5-year review](https://www.etftrends.com/5-years-dbmf-proves-managed-futures/)
- [Société Générale Prime Services indices](https://wholesale.banking.societegenerale.com/en/prime-services-indices/)
- [AlphaArchitect — Quality, Factor Momentum, and the Cross-Section of Returns (QMJ 1964-2023)](https://alphaarchitect.com/cross-section-of-returns/)
- [stockanalysis.com — JEPI dividend history](https://stockanalysis.com/etf/jepi/dividend/), [JEPQ](https://stockanalysis.com/etf/jepq/dividend/), [QYLD](https://stockanalysis.com/etf/qyld/dividend/)
- [J.P. Morgan JEPQ fund story / distribution notice](https://am.jpmorgan.com/content/dam/jpm-am-aem/americas/us/en/literature/fund-story/STO-JEPQ.pdf)
- Internal: `docs/CLAUDE_HISTORY.md`, commits `7d75aef` (ORB +0.105R/fill live-config rebuild), `0e97abd` (HOD order-flow FAIL), `8e93e9d`/prior HOD-break closure record.
