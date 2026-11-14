# Web survey 2026-11-14 — short-horizon executable equity edges (scout, ~20 tool calls, thin result)

Honest summary: the 2025-11+ literature surfaced NOTHING that is a new executable single-stock edge. Most useful finds are
cost/constraint facts and one pre-registered NULL-ish replication. Everything below is untrusted web data; SSRN returned 403
on every fetch, so SSRN items are search-snippet only (marked S). Items marked (U) are blog/practitioner, unverified.

## Ranked top-10 table (rank 5 = could PREREG next week on our data) — rewritten after round 2
| # | Item | Date | Headline | Rank |
|---|------|------|----------|------|
| 1 | Concretum Group (U, paywalled): "Identifying Stocks to Fade" / "When Short Sellers Create Overnight Alpha" / "Should You Trade Thin Stocks?" | 2026-05..06 | intraday-loser/overnight-winner in less-covered US names; search snippet: 2012-26 CAGR 37.1%, Sharpe ~1.6, maxDD ~35%; cost/borrow/trade count UNKNOWN; says borrow fees "punishing", locates not guaranteed, liquidity-dependent | 4 (matches our two live cells: overnight leg + gap-fade; use as hypothesis, not evidence) |
| 2 | Nguyen & Truong, pre-earnings lottery run-up, JBFA 53(1) 2026 | 2026-02 | high-minus-low earnings-MAX = +85 bp risk-adj in 10 days pre-announcement; prior-EA max return predicts; pseudo-date placebo in paper; construction details/t/costs not retrievable (Wiley 403) | 3 (long-only, daily bars; earnings dates from EDGAR 8-K Item 2.02) |
| 3 | Muravyev-Pollet-Pearson, borrow fee kills options-signal alpha (JFE 2025; Gies 2026-01) | 2026-01 | ~10% of stocks have fee >= 2%; predictability ~0 net of fee | 3 (mandatory borrow-fee cost line in gap-fade-short) |
| 4 | Short-seller flow / call-auction divergence (FRL 2026, S0.1544612326004563, search snippet) | 2026 | buy-side divergence in OPENING call auction predicts overnight return (high-low 3.05%/month, Chinese-style call auction) and fully reverses intraday; stronger under tight short constraints; market not US | 2 |
| 5 | Boyarchenko-Larsen-Whelan, Disappearing Overnight Drift (NY Fed Liberty St 2026-07-01) | 2026-07 | E-mini 02:00-03:00 ET drift ~370 bp/yr 1998-2020 (>60% of 590 bp close-close) -> ~0 in 2021-25 (n=1,245 d); attributed to lower closing-imbalance dispersion; futures, no t, no costs | 2 (overnight-edge decay warning for the MOC->MOO cell) |
| 6 | Vidovich & Zerjal EOD reversal (arXiv 2611.09286) | 2026-11-06 | 3.05 bp/day t=3.0 2023-26; full index 3.40->0.74; no cost claim | 2 |
| 7 | Daniel & Klos, Inefficiencies in Securities Lending Market (FoFI 2026, Mar) ; JFE 2026 "Institutional mandates, lending, short constraints" (S0304405X26001200) | 2026 | passive funds underweight high-fee stocks; shorting stocks added to Russell 2000 ~25% costlier; PDFs unreadable by fetch | 2 |
| 8 | Chen-Chen-Stivers-Sun, overnight vs intraday (SSRN 7443340, 2026-09-10) | 2026-09 | high INFO-Z stocks: stronger overnight momentum; high price/52wk-high: weaker intraday momentum; abstract only | 2 |
| 9 | Lim Blue Ocean overnight adverse selection (SSRN 6610883); SEC tick relief to Nov 2027 | 2026 | overnight spreads wider (~1/3 adverse selection); half-penny tick not live until Nov 2027 | 1 (cost facts) |
| 10 | Battalio et al. Illusion of Retail (SSRN 4579159, Jan 2026); Iwanaga-Sakemoto (SSRN 5807282, Aug 2026): index overnight return negatively predicts first half-hour | 2026 | BJZZ retail ID <1/3 recall; index-level only | 1 |

## Details (every item: signal, window, number, cost, borrow, Alpaca feasibility, data)
**1. EOD reversal after 2019** — Vidovich & Zerjal, arXiv q-fin.ST 2611.09286, 2026-11-06. PIT S&P500 minus ~80 heavy names.
Intraday losers beat winners over last half hour (to last trade before the close auction). Dollar-vol-weighted 3.13 bp/day
(t 7.5) 2016-19; 3.05 bp (t 3.0) 2023-26; 2020-22 ~0; full index 3.40->0.74 bp. Authors make NO net-of-cost claim; private
prereg; window-dependent (2023 carries it). Matches our own closed S&P finding (3 bp/day). Alpaca: MOC at 15:30 look-back
needs intraday bars only, but 3 bp < any cost. Rank 2.
**2. Borrow fee kills options-signal alpha** — Muravyev, Pollet, Pearson (Gies, JFE Oct 2025; press 2026-01-07). Infers borrow
fee from put-call parity; signal returns shrink toward zero net of the fee. Only ~10% of stocks have fee >= 2%. Relevance:
any short-side cell must charge the Alpaca locate fee per name-day; we have it from the locate sweep. Rank 3.
**3. Lottery demand pre-earnings** — Nguyen & Truong, JBFA 53(1):32-52, 2026. Extreme positive past earnings-window return
(earnings-MAX) predicts risk-adjusted return in the 10 days before the next announcement; high-low ~85 bp (abstract
listing; t-stat not seen). Same authors' 2018 JFM paper found plain MAX disappears conditional on earnings. Cost: unknown,
likely none. Long-only, daily holds, no borrow. Data: daily bars + earnings dates (EDGAR 8-K/Alpaca). Our PEAD line failed,
but this is a PRE-event window, different. Rank 3 (85 bp / 10d minus cost needs small-cap spread; multiplicity risk).
**4. Overnight session microstructure** — Lim (SSRN 6610883) and Eaton-Shkilko-Werner "Nocturnal Trading" (AFA). Overnight
effective spreads higher than RTH; ~2/3 of the premium is wider quotes not adverse selection (30 symbols x 24 sessions,
Sep25-Mar26). BMLL (Oct 2026, U): ~$3.5bn/night (52%) of overnight volume not on consolidated tape on Bruce+Blue Ocean. NYSE
Arca overnight session has no opening auction (NYSE FAQ). No documented overnight->09:30 price predictability found (Q3 open).
**5. News pricing** — arXiv 2608.14014 (v2 2026-10-18, AI4F @ ICAIF 2026). Event study, not a strategy. Needs LLM-tagged paid
news; cost flat per side, no borrow/locate (authors admit). Skip unless we buy news.
**6. Retail imbalance data quality** — SSRN 4579159 (2026-01-29, search snippet). BJZZ algorithm <1/3 recall on true retail
trades. Original BJZZ: ~10 bp next-week spread. Any retail-order-imbalance feature built from sub-penny TAQ is suspect.
**7. Microstructure** — SEC extended temporary relief for Rules 600(b)(89)(i)(F), 610(c), 612 to Nov 2027 and proposed
rescinding Rule 611 (secondary sources; confirm vs SEC release; SIFMA letter file S7-2026-10 asked for extension). So the
$0.005 tick (SEC: 74% of share volume tick-constrained; Nasdaq: tick costs 4-100 bp below $25) is NOT live; odd-lot quotes
beat NBBO by up to 26% for high-price groups (NBER, Review of Finance 2025) — little for $1-20 names.
**8. Overnight/intraday heterogeneity** — SSRN 7443340 (S). Known Lou-Polk-Skouras tug-of-war family; 93 pages; no costs seen.
Related: 2025 arXiv 2507.04481 (overnight news), FRL 2025 "11.6M price observations" (paywalled, 403).
**9-10 (U)** see table. Also: Tapeboard daily borrow-fee boards (2026-11-11 top: JZ 563.7% annualized) — useful live data
for HTB cost, not research. Short squeeze (Cambridge JFQA 2022): fee>=25% names cost 29-37 bp/month in squeezes; fee 10-25%
bucket mean excess return -0.85% vs 3.4-3.6% total shorting cost — old but directly relevant to the gap-fade short.

## Not found (be honest)
No 2025-11+ paper on: shorting high-attention gap-up small caps net of borrow; overnight-session return predictability;
small-cap options premia net of spreads (best net-of-cost evidence: BSIC 2023, delta-hedged index straddles +1.17% gross ->
-9.07% net, U); closing/opening auction imbalance -> next-day returns in single stocks (only old Finnish/MiFID II work).
## Next-week PREREG suggestions
(a) pre-earnings 10-day window conditioned on earnings-MAX (item 3), long, MOC/MOO, with measured spreads; (b) add the
borrow-fee-charged lens (item 2) to the gap-fade-short cell as a mandatory cost line.

## Round 2 additions (calls 30-58)
**R2-1 Concretum Group series (U, unverified, paywalled).** "Identifying Stocks to Fade" (signals "especially within the less
covered segments of the US market", meaningfully predictive of intraday returns), "When Short Sellers Create Overnight
Alpha" (2026-05-30; "Intraday Losers, Overnight Winners"; thesis: aggressive small-cap shorts cover late in the session),
"Should You Trade Thin Stocks?" (2026-06-20; performance differs widely between most/least liquid names; asks if an
illiquidity premium exists). A search snippet attributes 2012-2026 CAGR 37.1%, Sharpe ~1.6, maxDD ~35% to the May post;
I could NOT verify cost, borrow, trade count or that this figure belongs to that post. Concretum itself warns locates are
far from guaranteed and borrow fees can erase the edge. Fit: this is literally our overnight-leg (MOC buy intraday losers ->
MOO sell) and conflicts/complements the gap-fade short. Needs: Alpaca daily/minute bars; no short. PREREG rank 4 as a
hypothesis; first define "short-seller flow" proxy (late-session reversal / close location) on our cache.
**R2-2 Nguyen & Truong construction (item 5 of coordinator).** Retrievable: stocks sorted by EA_MAXRET (max daily return
around PAST earnings announcements); high-low risk-adjusted spread 85 bp in the 10 days before the current announcement;
pseudo-announcement-date placebo shows premium specific to the pending-announcement window; same authors' 2018 JFM paper
used 1973-2015 US stocks. NOT retrievable (Wiley 403, Massey PDF binary, no pdftotext on node): exact MAX window, number of
past quarters, universe/price filters, weights, t-stats, costs. Implementable with free data: earnings dates from EDGAR
8-K Item 2.02 filing dates (timing before/after open must be inferred from acceptance timestamp), daily bars from cache.db;
long-only, buy D-10 sell D-1 (MOC both ends). Risks: 85 bp/10d is gross, small caps with big EA moves have wide spreads;
likely microcap-driven; multiplicity. Rank 3.
**R2-3 Disappearing overnight drift (NY Fed, 2026-07-01).** Futures not stocks; no t, no cost. Useful as a prior that
overnight premia decayed once published and tracks closing-imbalance dispersion. Rank 2.
**R2-4 Call-auction divergence (FRL 2026).** Non-US call-auction market (14:57-15:00 closing call suggests China); opening
divergence -> overnight return +3.05%/month spread that reverses intraday; stronger under short constraints. Not transferable
without US opening-auction imbalance data (Nasdaq NOII feed is paid). Rank 2.
**R2-5 Borrow-cost short strategy (IndexBox summary of unnamed study via Yahoo, 2026-03, U).** Jan 2010-mid 2025; shorting
10-50% annualized-fee stocks outperformed on risk-adjusted basis; >50% fee not profitable net; ~6-month average hold; no
numbers; source study not identified. Consistent with Cambridge JFQA 2022 "Short squeezes and their consequences".
Implication for our HTB gap-fade: very high fee names (median HTB 8.6 bp/day in our sweep = ~22% annualized) sit in the
profitable-ish bucket but our hold is 1 day so fee is small; squeeze tail is the risk. Rank 2.
**R2-6 Retail options around earnings (De Silva-Smith-So "Losing is Optional", MIT IDE/UCLA, 2022-24 vintage, search
snippet).** Retail net long straddles, loses after half-spread; ORATS (U, Feb 2026): straddle buyers ~+45% in the recent
season vs negative 12-quarter history; BSIC 2023: index straddles +1.17% gross -> -9.07% net. No 2026 peer-reviewed
net-of-spread small-cap earnings-straddle study found. Rank 1.
**R2-7 Practitioner gap-fade stats (U).** "Why I Short Small Caps": 3,685 gaps >50% since 2008, 70.8% closed red (win rate
only). Stonkscapital: short every gap-up at open, cover at close, n=3,636, no net figures. StratLab "All About Gaps":
633,571 gap-up obs, 3,106 stocks (long side). Quantified Strategies FXI overnight short: 765 trades, 0.1% avg, PF 1.2 (cost
leaves no room). Oct 2026 preprint (preprints.org 202610.1262, 403): 256-strategy mega-cap search winner +432 bp/yr backtest
-> -84 bp OOS (U, snippet) — supports our multiplicity discipline. Rank 1-2.
## Still not found after round 2
No 2026 paper on: LULD/halt returns (only Tapeboard halt logs, ~30-50 halts/day), small-cap call overwriting net of spreads,
US closing-cross imbalance -> next-day single-stock return, 24/5 session return predictability. Tools gaps: SSRN, Wiley,
ScienceDirect return 403; no PDF text extractor on node (pdftotext/pypdf absent), so lancs FoFI 2026 and Massey PDFs unread.
