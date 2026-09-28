# Refuter: PREREG_1626 result (cells 1,626–1,629, leveraged single-stock ETF decay pair)

Written 2026-09-28 by the adversarial refuter. Scripts: `review/refute_1626.py`, `review/refute_1626_divs.py`.
Evidence CSVs are in `review/refute_*.csv`. New bar caches are `review/bars_adjall.parquet` and
`review/bars_split_legs.parquet`: 128 legs each, LOST 0/128 per adjustment. The builder's `bars.parquet` was not touched.

## Verdict: the FAIL stands (not refuted)
None of the defects below turns a failing criterion into a pass. Every correction either leaves the number
inside its noise or moves it down. Four of the seven frozen VAL criteria fail, each for its own reason:

| criterion (frozen) | builder | worst-case after refuter corrections | robust? |
|---|---|---|---|
| day-clustered t ≥ 2.5 @ 15 % rail | 0.91 | 0.75 (total-return bars) … 1.04 (best leave-one-pair-out); day-mean t 0.48; rebuild 1.26; oldest-leg pairs 0.91–1.00; winsorised p1/p99 0.87 | FAIL in every variant |
| ≥ 10 pairs ETB on both legs | 0 | **67/68 legs are `shortable=False` at Alpaca** (only NBIG is shortable+ETB) → 0 executable pairs | FAIL, structural |
| worst month ≥ −3 % | −41.85 % (pair-month) | portfolio level (equal-weight live pairs): −6.1 % (2026-04), −4.6 % (2026-05), −4.0 % (2025-05), −3.5 % (2025-11): 4 of 18 months breach | FAIL at both levels |
| realised drag within 30 % of theory (top vol) | 0.38 | the theory column is half the pair's drag (below), so the true ratio is ≈ 0.19–0.22. The frozen item cannot be met on a random walk | FAIL, mis-specified |
| mean ≥ +4 bps @ 15 % | 5.19 | 4.23 on total-return bars (marginal) | passes, barely |
| ≥ 60 % pairs positive | 0.76 | 0.71 on total-return bars | passes |
| TRAIN same sign, t ≥ 1 | −2.72 (t −0.63) | TRAIN is contaminated (see §A, §H). On clean data the sign is undetermined | FAIL / untestable |

My independent replica of the engine reproduces the builder exactly: TRAIN −2.72, VAL +5.19 bps/day, t 0.92,
worst pair-month −42.5 %.

## A. Pair matching: 3 wrong pairs and one hindsight-labelling hazard
* **AI_2.0x is a basket, not C3.ai.** AIBU/AIBD are the "Direxion Daily AI and Big Data Bull/Bear 2X" funds, which
  track an index. Their realised beta to C3.ai is 0.1–0.8 in every quarter against a label of 2, with 34 tracking-break
  leg-days. This pair supplies **59 % of TRAIN pair-days (218/371)** and 374 VAL pair-days. The PREREG forbids it
  ("not allowed: non-single-stock underlyings"). Builder and rebuild both missed it (builder "AI", rebuild "DATA").
  Removing it: VAL 5.89 (t 0.98) on split bars, 4.93 (t 0.83) on total-return bars.
* **COIN_2.0x short leg CONI was a −1x fund until 2025Q1.** Its realised beta was −1.00 in 2024Q3, 2024Q4 and 2025Q1.
  The 11 TRAIN days of this pair are therefore a net-short-$1 directional book (+21.5 bps/day), not a neutral pair.
  The same hazard appears elsewhere: FBL was 1.5x in 2023H1, NVD and TSDD were −1.5x in 2023Q4 (outside the builder's
  windows), and NVDQ (β −1.28) and TSLQ (β −1.37) were not −2x across TRAIN. **Taking the leverage from today's
  name is hindsight.** Leverage has to be verified per quarter from returns.
* **SK_2.0x uses a leveraged ETF as its "underlying".** In Alpaca, `SK` is the "Corgi SK hynix 2x Daily ETF";
  SK hynix itself is not US-listed. The pair's vol, theory and 1,627 gate therefore run on 2× the vol (4× the variance).
* **STSM bad print.** STSM shows −64.8 % on 2026-03-20 and +183 % on 2026-03-23, while TSM moved ±2.8 %. That gives pair-days
  of +3,557 and −3,486 bps (net +71). This is a price-scale artefact that inflates the variance.
* **Recycled tickers.** CBRG (543-day bar gap) and SPAX (434-day gap) are reused tickers. The partner legs start later,
  so the P&L is unaffected, but those legs' "first bar" dates are not the fund launches.
* **The alternate-leg choice drops the history.** The builder picked one leg per side "tradable+active first, then
  alphabetical". That discards the older products (TSLL 2022-08, NVDL 2022-12, MSTX 2024-08) and leaves TRAIN with 4 pairs.

## B. Rebalance convention and delta drift: neutral only at the rebalance instant
The convention is implemented as documented (replica match). Between rebalances, |D_long − D_short| in VAL has mean
0.24, p50 0.16, p90 0.57, p99 1.15 and max 2.41. The net underlying delta is −gap as a share of gross, so the book
carries on average 24 % of gross (p90 57 %, max 241 %) as a delta against the trend. It is short gamma: the loss months
are the paths where the underlying trends inside the 5-session window.

## C. Mechanism and theory: the frozen calibration is wrong twice
1. **The theory is coded for the long leg only.** The code computes ½(L²−L)σ² with L = +2, which is σ². The −2x leg's
   drag is ½(4+2)σ² = 3σ², so the pair's log-drag per $2 gross is 2σ². In VAL the builder's theory averages 29.5 bps
   against a correct 59.0. Every realised/theory ratio in RESULT_1626 (0.64 / 0.16 / 0.44, headline 0.38) is
   overstated 2×.
2. **Log-drag is not the expected P&L of a short.** For constant-share shorts over a window, P&L ≈ 4[Σr² − (Σr)²]
   per $1 per leg. Under a martingale, E(Σr)² = ΣE r², so the **expected gross is zero**: the drag is a median effect,
   and the right tail of the long leg pays for it. Decomposing the P&L on the underlying's own path gives, in bps/day
   of $2 over 32 clean VAL pairs, 1,149 windows and 5,693 pair-days:
   * actual gross: **11.87**
   * identity: 4.64 (drag +58.55, trend −53.90)
   * residual: 7.23, which is fund fees, financing and omitted distributions (see F). Tracking and stale-close effects
     are unquantified.
   * pooled 5-session variance ratio: **0.921 (bootstrap 95 % CI 0.83–1.01)**, which cannot be told apart from a
     random walk.

   The frozen item "realised within 30 % of ½(L²−L)σ²" is unattainable on any random-walk population, and the
   mid-tercile dip is noise around a zero-mean identity. The honest calibration for the record is VR = 0.92 plus the
   fee/financing residual. That residual is exactly what an ETF lender prices into the borrow fee.

## D. Borrow
* The asset flags are today's snapshot, which is hindsight in favour of feasibility, and they still give **0 executable
  pairs**: 67 of 68 legs have `shortable=False` at Alpaca. A paper book on the ORB account is impossible as frozen.
* **Break-even rail**: the gross after cost is 11.10 bps/day, which breaks even at **28 %/yr per leg** on split bars and
  ≈ 25.6 %/yr on total-return bars.
* No rail rescues the t. Even at the 5 % rail it is 1.61, below 2.5. Hard-to-borrow fees on these products are commonly
  10–50 %/yr or more, which would put the book near or below zero.

## E. Survivorship
* All 68 legs have bars up to at least 2026-09-18, and the selection puts "active" first. **The population is 100 %
  survivors by construction.** RESULT's "0/34 closed … kept in the population" is therefore not a check.
* The asset snapshot lists only 17 inactive leveraged names, 6 of them on a common-stock underlying (for example LMNX,
  the LMND 2x long). Closed single-stock funds are essentially missing from the source, so closures cannot be counted
  from it.
* Launches: 30 of 34 pairs listed after 2025-03-31, with the median first bar on 2026-01-22. The median number of live
  pairs per VAL day is 16, and 8 in 2025.
* Closures follow a destroyed leg or an AUM collapse, which is exactly the trend-tail path that loses here. The survivor
  bias therefore flatters the book and cannot rescue a FAIL.

## F. Distributions and expense ratios
* Expense ratios and financing sit in NAV, so they are correctly in the close series.
* **Distributions are not.** The bars are `adjustment=split`, and a short pays the distribution. 18 of 68 legs paid one.
  The large ones were year-end capital-gains distributions, mostly in December 2025, and each shows up in split bars
  as a fake short gain:

  | leg | total distributions |
  |---|---|
  | FBL | 50.6 % on 2023-12-27, 53.6 % in total |
  | NVD | 41.5 % |
  | TSDD | 37.4 % |
  | PLTG | 16.7 % |
  | RKLX | 14.8 % |
  | RGTU | 13.3 % |
  | TSMG | 11.7 % |
  | NVDG | 11.6 % |
  | AMDG | 11.1 % |
  | QBTX | 10.5 % |

* Re-scored on total-return (`adjustment=all`) bars:
  * VAL: **4.23 bps/day (t 0.75)**, against 5.19 (t 0.92)
  * TRAIN: −4.63 (t −1.15), against −2.72

  The builder overstates the result by ≈ 1.0 bps/day in VAL and 1.9 in TRAIN. The details are in `refute_variants.csv`
  and `refute_leg_distributions.csv`.

## G. Trend-path tail
The worst 10 pair-months are genuine trend or whipsaw paths, with no tracking breaks on those legs:

| pair-month | pair loss | underlying move / note |
|---|---|---|
| QBTS 2025-10 | −42.5 % | whipsaw inside windows, gap up to 1.98 |
| RKLB 2026-05 | −34.9 % | RKLB +74 % |
| CRCL 2026-03 | −34.5 % | |
| IONQ 2026-04 | −33.1 % | +57 % |
| NBIS 2026-04 | −28.1 % | +33 % |
| QBTS 2026-04 | −25.8 % | +41 % |
| SNDK 2026-08 | −22.4 % | +29 % |
| CRWV 2026-04 | −21.0 % | +44 % |
| IREN 2026-05 | −19.8 % | +40 % |
| IONQ 2025-09 | −17.9 % | +44 % |

* The losses cluster in April–May 2026, a speculative-tech momentum burst. The tail is one factor shared across pairs,
  so 34 pairs do not diversify it.
* Removing the top 5 % of pair-days gives −22.1 bps/day (t −6.7). Removing the bottom 5 % gives +30.6 (t 6.3). The
  mean is decided entirely by the 5 % tails.

## H. The TRAIN leg
* The builder's TRAIN is 59 % the AI basket, plus the CONI −1x days, NVDA (71 days) and TSLA (71 days).
* Oldest-leg pairs give 1,387 TRAIN pair-days at −10.0 / −12.7 bps/day (t −1.4 / −1.7), with a worst month of −71.8 %.
  But NVDQ and TSLQ were not 2x over TRAIN.
* The only clean TRAIN pair is MSTX/SMST: 151 days at +11.5.
* On clean data, TRAIN cannot establish a sign.

## Consequence
FAIL stands, and the cell is closed. The "calibration on record" in RESULT_1626 should be replaced by:
* VR 0.92 (CI 0.83–1.01);
* the corrected theory of 2σ² per $2;
* the total-return numbers;
* the AI basket removed;
* executability recorded as structurally impossible at Alpaca (67/68 legs not shortable).
