# RESULT cell 1,710 -- 52-week-high momentum, STAGE 1 (frozen rule, PREREG_52WH_20261029 + Amendment 1)
**Stage 1 is rejection-only: a pass here is not a pass.** Here it is a FAIL on VAL, which is final per the amendment (sleeve dead, no purchase).
Code `research/sleeves/fiftytwo_wh.py` (shared pure rule fns + driver; `--evaluate`), table `monthly_1710.csv` (+`_floor0`), caches `cache/`.
TEST 2023+ never read (date predicate pushed into the parquet scan; `load_panel` raises past 2022-12-31). No orders, no data bought.

## Setup (as run)
* Panel `momentum_weekly/panel_2016_2026.parquet` (Alpaca, split+dividend ADJUSTED, survivorship-EXPOSED), 10,738 symbols 2016-2022.
* Screen: 45.9 % of symbols excluded (wrapper/ETF 26.1 %, name pattern pref/warrant/unit/right/note/contra 9.4 %, punctuation 6.2 %,
  asset-map unknown 3.7 %, not in map 0.5 %). Eligible ~1,550-2,100 names/month. Price >= $5 on the ADJUSTED price (see caveat 2), 60d median $vol >= $5M.
* ratio = close_t / max(high of the 252 sessions BEFORE t), signal = last session of month, top-20 equal weight, entry = next session's open,
  exit = first session's open of the following month (open-to-open). Cost 15 bps RT (30 sens). Benchmarks: universe EW (gross), SPY (adjusted open-to-open).
* TRAIN is effectively 2017-02..2019-12 (n 35): the 252-session lookback eats 2016. VAL 2020-01..2022-11 (n 35; Dec-2022 entry needs a 2023 exit open = sealed, dropped).

## Results (RT 15 bps; month = entry month)
| split | strat net %/mo | univ EW | SPY | ex vs univ [yr-block CI] | ex vs SPY [CI] | Sharpe | maxDD | ex-top-5 % ex | green (null mean/p95) |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | +0.87 | +0.98 | +1.23 | -0.11 [-0.43,+0.20] | -0.37 [-0.54,-0.25] | 0.79 | 17.7 % | -0.22 | 71 % (65/74) pctile 83 |
| VAL | +0.02 | +0.87 | +1.00 | **-0.85 [-2.21,+0.06]** | -0.98 [-2.85,+0.50] | 0.01 | **26.2 %** | -1.26 | **46 %** (57/66) pctile 1 |
RT 30 bps: VAL ex-univ -1.00 [-2.36,-0.09], Sharpe -0.09, DD 28.2 %, green 46 %; TRAIN ex-univ -0.26 [-0.58,+0.05], DD 18.1 %.
Published 30 x 6-month (cost RT15/6 per month): TRAIN ex-univ +0.18 [-0.18,+0.54], Sharpe 0.93, DD 18.2 %; VAL -0.11 [-0.53,+0.55], Sharpe 0.42, DD 25.0 %, green 54 %.
No price floor (adjusted-price floor off): VAL ex-univ -1.04 [-2.37,-0.22], DD 27.6 %, green 46 % -- same verdict.

## Pass bar on VAL, line by line
1. Mean monthly excess vs universe > 0 with bootstrap CI > 0: **FAIL** (-0.85 %/mo, CI [-2.21,+0.06]; point estimate negative, CI upper bound barely above 0).
2. >= 55 % green months AND above the count-matched random-20 null (1,000 draws; cadence C4 = null + 10 pp): **FAIL** (46 % vs null 57 %; 1st percentile of the null).
3. Max DD <= 20 %: **FAIL** (26.2 %; TRAIN 17.7 % passes). 4. Price-scale: see below. **Verdict VAL: FAIL on 1, 2 and 3.**
TRAIN (informative only): excess -0.11 %/mo (CI spans 0), green 71 % vs null 65 % (+6 pp, below the +10 pp bar) -- the long leg only tracks its own universe.

## Price-scale / tail check
* Year-cell |t| max 2.85 (VAL), 0.71 (TRAIN): none > 6. Name-months beyond +500 % / -80 %: TRAIN 0; VAL 2 FLAGGED, both 2022-10:
  BHVN -88.9 % (Pfizer/Biohaven spin-off: symbol re-used for the NEW Biohaven Ltd at ~$13 -- an unadjusted corporate-action ARTIFACT)
  and RLMD -83.0 % (not verified; treated as possibly real). Resolution test: set BHVN to 0 % -> VAL ex-univ -0.73, green 49 %, DD 24.5 %;
  drop both -> -0.59, green 49 %, DD 24.5 %. Verdict unchanged on every criterion, so the flag does not block the (rejection) conclusion.
* Max single-name month +152 % (TRAIN) / +130 % (VAL). 2,314 name-periods had no exit open (fallback = last close); count logged as WARNING.

## Adversarial caveats
1. Survivorship INFLATES a long book; VAL still fails, which is why the verdict is final. A TRAIN "pass" would have meant nothing.
2. The $5 floor uses today's split-adjusted price, which wrongly drops names that split up later (winners) -- biases against the book;
   the no-floor run shows no help (-1.04). Dividend-adjusted open-to-open returns are total returns for strategy, universe and SPY alike.
3. Only 35 months per split and a 3-4 block year bootstrap: the CI is coarse; the VAL fail rests on three criteria, not the CI alone.
4. Excess vs the universe is net-of-cost strategy vs gross passive EW (conservative against the book by ~15 bps/mo).
5. VAL is the 2020-22 regime (COVID crash/rebound, 2021 junk rally, 2022 bear): momentum crashes there; this says nothing about 2023+ (sealed).
6. Asset map is present-day (exclusion only); names absent from it (0.5 %) are dropped, delisted unknowns (3.7 %) too.
7. TOM/MOM overlap and shared-tail correlation not computed: moot after the VAL fail.
8. Multiplicity: this programme-cell count +1 for the frozen cell, +2 sensitivities (30x6, no-floor); no tuning performed.

## Stage 2 price (Databento metadata.get_cost / get_billable_size, no download, nothing billed)
* **Stage 2 as written is NOT purchasable:** `EQUS.SUMMARY` starts 2024-07-01 (the 2009-2022 request returns 422 data_start_before_available_start);
  no Databento US-equity dataset starts before 2018-05-01 (XNAS.ITCH, XNYS.PILLAR), so TRAIN 2010-19 cannot exist there.
* Closest point-in-time substitute, single-venue (NOT consolidated; XNAS.ITCH / XNYS.PILLAR only, no ARCA/AMEX): ohlcv-1d ALL_SYMBOLS to 2023-01-01:
  XNAS.ITCH from 2018-05-01 $15.68 (561 MB); XNYS.PILLAR from 2019-01-01 $10.21 (366 MB); definition schema $63.02 and $50.43 -> ~$139 total.
* Not recommended: the sleeve is dead on Stage 1, so no spend is needed.
