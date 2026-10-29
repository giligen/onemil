# RESULT cell 1,711 -- ETF time-series momentum (PREREG_TSM_20261029), built by Sonnet agent, NOT yet independently rebuilt

Rule: `research/sleeves/tsm_etf.py` (run `nice -n 19 python3 research/sleeves/tsm_etf.py`); ledger `monthly_1711.csv`.
Data: local `panel_2016_2026.parquet`, all 12 ETFs present, 1,762 sessions 2016-01-04..2022-12-30, 0 NaN bars. TEST never
read (parquet filter `bar_date <= 2022-12-31`). Fills: first-session open to next first-session open; 5 bps round trip on
drift-adjusted turnover (mean one-way turnover 12.3 %/mo, cost 0.3 bp/mo). Benchmark = 1/12 basket, monthly rebalanced, same cost.
Deviation: VAL entry month 2022-12 DROPPED (its exit open is 2023-01-03, sealed) -> VAL = 2020-01..2022-11, n 35. TRAIN 2017-02..2019-12, n 35.

## Headline (monthly, net of cost)
| split | TSM mean | basket mean | excess | year-block CI | iid t | Sharpe TSM/bkt | maxDD TSM/bkt | green TSM/bkt | ex-top5% mean |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | +0.38 % | +0.76 % | -0.38 % | [-0.90, -0.05] | -1.59 | 0.84 / 1.19 | 8.4 % / 7.9 % | 71 % / 80 % | +0.24 % |
| VAL | +0.20 % | +0.39 % | -0.19 % | [-1.02, +0.63] | -0.41 | 0.36 / 0.33 | 8.9 % / 19.5 % | 57 % / 63 % | +0.03 % |

Count-matched null (random subset, same size each month, 1,000 draws): green share TRAIN 75.3 % (p95 82.9), VAL 59.4 % (p95 65.7);
actual 71.4 % / 57.1 % -> p(null >= actual) 0.89 / 0.88. Null mean return VAL +0.24 %/mo vs TSM +0.20 %. The signal adds nothing over a random same-size book.

## Frozen VAL pass bar, line by line
* A1 excess > 0 with CI > 0: FAIL (-0.19 %/mo, CI [-1.02, +0.63]).
* A2 (OR clause) maxDD <= half basket DD: PASS (8.9 % <= 9.7 %) AND mean >= basket mean - 0.1 %/mo: FAIL (+0.20 vs +0.29 floor). A fails.
* B >= 50 % green months: PASS (57 %; but below basket 63 % and the null 59 %).
* C maxDD <= 15 %: PASS (8.9 %).
**VERDICT: FAIL on criterion A (only the drawdown clause half-passes). Sleeve #2 REJECTED at Stage 1; no rebuild, no paper.**

## Sensitivities (VAL | TRAIN; mean net / excess / maxDD / green)
* 4 %/yr hurdle: VAL +0.24 % / -0.16 % / 7.2 % / 54 % -> A still FAIL (mean floor +0.29 %); TRAIN +0.39 % / -0.37 % / 5.3 %.
* 1/N_longs weight (fully invested): VAL +0.15 % / -0.24 % / 21.1 % (worse than basket 19.5 %) / 57 %; TRAIN maxDD 16.1 % > 15 %.
* 15 bps round trip: identical to 5 bps to 0.01 % (turnover too low for cost to matter).

## Per-year excess vs basket (mean monthly)
2017 -0.15, 2018 -0.05, 2019 -0.90, 2020 -1.02, 2021 -0.11, 2022 +0.63 %. Only 2022 (rate-shock bear: avg 3.4 longs of 12) pays;
the whole VAL drawdown benefit is the 2020-03 crash month (TSM -6.3 % vs basket -11.6 %) and 2022. Average longs 8.5 (TRAIN) / 7.3 (VAL); no all-cash month.

## Overlap (monthly net return correlation, sealed months cut at load)
* TOM (`research/index_overnight/tom_1651.csv`, mean of SPY/QQQ/IWM): TRAIN -0.03, VAL +0.05 (n 35 each). Position overlap ~0 by construction (TOM exits on TSM's entry day; live TOM ledger has only 16 rows).
* MOM (`research/momentum_weekly/1700c_monthly.csv` net_ret, entry-month alignment): TRAIN +0.68, VAL +0.62 -- HIGH; both are long-only 12-month momentum, so TSM is not a diversifier to MOM in equity-led months.
* Live-account correlation: not computed (no monthly live series).

## Panel adjustment / price scale
Dividend-adjusted (total-return style): SPY 2016-01-04 close in panel 170.68 vs ~201 raw as I remember it (not verifiable offline), so
distributions are inside the returns (benefits bond/REIT/HYG legs; no hurdle-vs-yield issue left). No split-like jumps: max |daily ret| 12.1 %
(EFA), 0 sessions > 25 % on any ETF. Adjustment is as-of the panel build date, so old closes carry later dividends -- mom12 uses only ratios, harmless.

## Adversarial caveats
* Benchmark is a 2017-22 bull-ish 12-ETF basket; excess is negative mostly in 2019-20 (V-shaped recoveries whipsawed by 12-1 style signal). Only 6 years, 3 per split: year-block CI over 3 blocks is crude; iid t is -1.6 / -0.4, nowhere near significance either way.
* VAL fails by 0.09 %/mo on the mean floor; with 35 months the SE (~0.45 %/mo) makes the pass/fail margin noise-level, but the frozen bar is the bar -- no re-tuning.
* Single-lookback, single-frequency frozen variant; cells counted: 1 main + 3 sensitivities + 1,000-draw null (programme cell 1,711).
* Open-to-open fills assume the 09:46 live execution matches the official open (ETF slippage small, not measured here). Nothing was run on TEST.
* Builder not independent of the spec; this agent only implemented the PREREG. Rejection needs no rebuild per the ladder (rebuild is for passes).
