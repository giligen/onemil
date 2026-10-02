# RESULT -- cell 1,700c: monthly momentum decile, Alpaca free history, vs SPY

PREREG_1700c.md (FROZEN). Window 2016-01-01..2026-09-30, 129 nominal monthly rebalances, 128 complete holding periods (last rebalance 2026-09-01 is open beyond the fetch boundary, excluded from stats).

## By-year: book vs SPY (net of cost; price return, dividends not separately modeled -- see Data note)

| Year | Months | Book | SPY | Excess |
|---|---|---|---|---|
| 2016 | 12 | +0.0% | +14.1% | -14.1% |
| 2017 | 12 | +17.2% | +21.3% | -4.1% |
| 2018 | 12 | -11.4% | -6.8% | -4.5% |
| 2019 | 12 | +33.2% | +34.0% | -0.8% |
| 2020 | 12 | +57.7% | +18.2% | +39.5% |
| 2021 | 12 | -1.6% | +28.6% | -30.2% |
| 2022 | 12 | -10.1% | -18.0% | +7.8% |
| 2023 | 12 | +10.1% | +24.7% | -14.7% |
| 2024 | 12 | +25.9% | +26.4% | -0.6% |
| 2025 | 12 | +19.1% | +17.7% | +1.4% |
| 2026 | 8 | +12.5% | +11.7% | +0.8% |

## Reads (halfA/halfB/whole/covid_2020; annualised net-of-cost unless marked)

| window | n_periods | ann_return_net | ann_return_spy | excess_ann_return | ann_alpha_spy | t_alpha | sharpe_net | max_dd | max_dd_spy | null_pct_return | null_pct_excess | turnover_avg | cost_drag_annual | green_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halfA | 53 | 0.2346 | 0.1679 | 0.0517 | 0.0400 | 0.4483 | 0.9048 | -0.2802 | -0.2291 | 100.0000 | 100.0000 | 0.2653 | 0.0089 | 0.6604 |
| halfB | 62 | 0.0691 | 0.1324 | -0.0559 | -0.0687 | -0.8781 | 0.3903 | -0.2675 | -0.2331 | 98.0000 | 98.0000 | 0.3053 | 0.0107 | 0.5968 |
| whole | 115 | 0.1424 | 0.1506 | -0.0077 | -0.0190 | -0.3249 | 0.6392 | -0.3485 | -0.2331 | 100.0000 | 100.0000 | 0.2847 | 0.0098 | 0.6261 |
| covid_2020 | 3 | -0.3941 | -0.3795 | -0.0237 | 0.1347 | 0.3148 | -0.4428 | -0.2264 | -0.1636 | 99.9000 | 99.9000 | 0.3112 | 0.0109 | 0.3333 |

**Pass bar (PREREG): excess>0 both halves AND alpha_t(whole)>=2.0 AND maxDD<=1.25xSPY AND null_pct_excess>=95 both halves -> FAIL**
- excess>0: halfA=True (+5.2%), halfB=False (-5.6%)
- alpha t-stat whole window = -0.32 (need >=2.0): False
- max DD book=-34.9% vs 1.25x SPY=29.1% (SPY maxDD=-23.3%): False
- null percentile of excess: halfA=100.0 (>=95: True), halfB=98.0 (>=95: True)

## Survivorship statement (Databento PIT panel, read-only, 2024-07-> coverage only)

- 2024: 11970 Databento PIT symbols, 2877 missing from the Alpaca panel (24.0%)
- 2025: 13435 Databento PIT symbols, 2535 missing from the Alpaca panel (18.9%)
- 2026: 14196 Databento PIT symbols, 1573 missing from the Alpaca panel (11.1%)
- Inactive Alpaca assets with >=1 bar served: 2469 / 19144 (12.9%)

## Delisted-bars answer (step 1 probe, 1700c_delisted_probe.csv)
- SIVB (2022-01-03..2023-03-10): served=True, n_bars=297
- TWTR (2022-01-03..2022-10-27): served=True, n_bars=206
- FRC (2022-01-03..2023-05-01): served=True, n_bars=332

## Completeness: requested=32965 with_bars=15789 LOST=468 | active_requested=14388 active_with_bars=13546 active_coverage=94.1% gate(>=95%)=FAIL -> VOID

## Methodology notes / caveats
- Universe common-stock filter is Alpaca asset-NAME pattern only (ETF/ETN/fund/trust/warrant/unit/preferred/right), applied for the full window and both active+inactive assets; Databento security_type cross-check (PREREG's "where known") was NOT run this pass for time -- a scoping cut, not a silent gap, logged here.
- adjustment=ALL folds dividends into back-adjusted price (total-return-like), not a pure split-only price return -- see module docstring.
- A ticker string reused by two different Alpaca asset records (one active, one inactive) is excluded from the universe if EITHER record's name matches the exclusion pattern.
- No variant (decile width, skip, universe floor) was tried after seeing these numbers, per PREREG.
