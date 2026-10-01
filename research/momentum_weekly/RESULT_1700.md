# RESULT — cell 1,700: weekly-rebalanced 12-month momentum sleeve, phase 1

PREREG_1700.md (FROZEN). Owner ask: "every Monday buy the stocks that did the best P&L over the past year; every Monday rebalance." Phase 1 only — data on disk, zero fetches.
Window 2025-07-01..2026-09-30, 60 weekly rebalances (halfA 2025-07-01..2025-12-31, halfB 2026-01-01..2026-09-30).
Exclusions (full raw panel, 16217 symbols): 7 test-ticker symbols (`^Z[A-Z]ZZT$`), 5292 wrapper-tagged symbols (orb_asset_class_map_20260711.csv). That map's "wrapper" tag also strips plain index ETFs (SPY, IWM) as holdings — SPY kept only as the benchmark, pulled before exclusion; QQQ is tagged "stock" in the same file and stays tradable.
Cross-checks: cache.db daily_bars (read-only) 211/300 sampled closes within 0.5%; overnight_high panel $-volume (dvol20) agreement 90.9% of 3260676 matched rows within 10%.

## Reads (27 = 3 variants x 3 N x 3 windows; annualised net-of-cost unless marked)

| variant | N | window | n_weeks | ann_return_net | ann_alpha_spy | sharpe_net | max_dd | turnover_avg | cost_drag_annual | null_pct_return | null_pct_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 10 | halfA | 26 | -0.588 | -1.482 | -0.857 | -0.397 | 0.231 | 0.033 | 0.000 | 6.500 |
| M1 | 10 | halfB | 34 | -0.779 | -1.661 | -2.696 | -0.625 | 0.224 | 0.034 | 0.000 | 0.000 |
| M1 | 10 | whole | 60 | -0.711 | -1.502 | -1.704 | -0.774 | 0.227 | 0.034 | 0.000 | 0.000 |
| M1 | 20 | halfA | 26 | -0.398 | -0.981 | -0.617 | -0.362 | 0.227 | 0.032 | 0.000 | 3.900 |
| M1 | 20 | halfB | 34 | -0.041 | -0.187 | 0.230 | -0.387 | 0.187 | 0.029 | 5.000 | 12.200 |
| M1 | 20 | whole | 60 | -0.216 | -0.491 | -0.101 | -0.387 | 0.204 | 0.030 | 0.000 | 2.000 |
| M1 | 50 | halfA | 26 | -0.423 | -1.047 | -1.074 | -0.348 | 0.215 | 0.030 | 0.000 | 0.000 |
| M1 | 50 | halfB | 34 | 0.250 | -0.029 | 0.698 | -0.291 | 0.175 | 0.027 | 58.100 | 12.300 |
| M1 | 50 | whole | 60 | -0.106 | -0.437 | -0.037 | -0.348 | 0.192 | 0.029 | 0.000 | 0.100 |
| M2 | 10 | halfA | 26 | -0.573 | -1.667 | -1.011 | -0.424 | 0.238 | 0.034 | 0.000 | 4.300 |
| M2 | 10 | halfB | 34 | -0.200 | -0.439 | -0.278 | -0.380 | 0.179 | 0.028 | 2.900 | 11.700 |
| M2 | 10 | whole | 60 | -0.391 | -0.862 | -0.654 | -0.525 | 0.205 | 0.031 | 0.000 | 2.400 |
| M2 | 20 | halfA | 26 | -0.056 | -0.694 | 0.150 | -0.311 | 0.202 | 0.028 | 8.400 | 19.600 |
| M2 | 20 | halfB | 34 | 0.150 | 0.060 | 0.539 | -0.332 | 0.185 | 0.029 | 36.000 | 22.500 |
| M2 | 20 | whole | 60 | 0.055 | -0.170 | 0.339 | -0.332 | 0.193 | 0.028 | 12.400 | 10.200 |
| M2 | 50 | halfA | 26 | -0.207 | -0.766 | -0.331 | -0.321 | 0.208 | 0.029 | 0.000 | 1.400 |
| M2 | 50 | halfB | 34 | 0.123 | -0.126 | 0.510 | -0.258 | 0.163 | 0.025 | 17.600 | 6.500 |
| M2 | 50 | whole | 60 | -0.034 | -0.356 | 0.091 | -0.321 | 0.182 | 0.027 | 0.000 | 0.200 |
| M3 | 10 | halfA | 26 | -0.709 | -1.495 | -1.982 | -0.597 | 0.238 | 0.034 | 0.000 | 0.100 |
| M3 | 10 | halfB | 34 | -0.569 | -0.894 | -1.339 | -0.517 | 0.332 | 0.051 | 0.000 | 2.000 |
| M3 | 10 | whole | 60 | -0.637 | -1.121 | -1.635 | -0.745 | 0.292 | 0.044 | 0.000 | 0.000 |
| M3 | 20 | halfA | 26 | -0.838 | -2.294 | -4.412 | -0.624 | 0.273 | 0.039 | 0.000 | 0.000 |
| M3 | 20 | halfB | 34 | -0.073 | -0.078 | 0.176 | -0.452 | 0.266 | 0.041 | 3.600 | 13.200 |
| M3 | 20 | whole | 60 | -0.565 | -0.977 | -1.161 | -0.640 | 0.269 | 0.040 | 0.000 | 0.000 |
| M3 | 50 | halfA | 26 | -0.602 | -1.359 | -2.771 | -0.414 | 0.262 | 0.038 | 0.000 | 0.000 |
| M3 | 50 | halfB | 34 | -0.151 | -0.347 | -0.276 | -0.324 | 0.224 | 0.035 | 0.000 | 0.000 |
| M3 | 50 | whole | 60 | -0.388 | -0.749 | -1.255 | -0.441 | 0.240 | 0.036 | 0.000 | 0.000 |

## Pass bar (phase 1): alpha>=+8%/yr AND Sharpe>=1.0 AND null pct(return)>=95, both halves; cost drag<4%/yr; max DD<=20%

- M1 N=10: fail; whole-window alpha -1.502, Sharpe -1.70, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M1 N=20: fail; whole-window alpha -0.491, Sharpe -0.10, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M1 N=50: fail; whole-window alpha -0.437, Sharpe -0.04, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M2 N=10: fail; whole-window alpha -0.862, Sharpe -0.65, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M2 N=20: fail; whole-window alpha -0.170, Sharpe 0.34, null-pct(ret) 12 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M2 N=50: fail; whole-window alpha -0.356, Sharpe 0.09, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M3 N=10: fail; whole-window alpha -1.121, Sharpe -1.64, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M3 N=20: fail; whole-window alpha -0.977, Sharpe -1.16, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- M3 N=50: fail; whole-window alpha -0.749, Sharpe -1.26, null-pct(ret) 0 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read

## Read the negative as an adversary — mechanism check (not a bug, but a universe-definition gap)
Spot-checked week 1 (signal 2025-07-03, entry 2025-07-07, M1 N=10) directly against the raw panel: top-10 by
trailing-252d return were NUTX +200%, CYN +179%, QBTS+ +116%, ARQQ +110%, QUBT +49%, TNXP +47%, DNA +38%,
CERO +33%, KZIA +30%, NUKK +25% — a basket of small/micro-cap momentum/hype names (quantum-computing and
biotech-binary names prominent), not an alignment or sign error: realized gross return that week (-4.5%)
matches the pipeline's own number exactly. Mean weekly GROSS return (before the ~0.1-0.3%/wk cost) is itself
deeply negative for low N (M1 N=10 gross -1.94%/wk, M3 N=10 gross -1.58%/wk) — cost is not the driver.
One of the 10 names above, QBTS+, is a WARRANT, not a common stock or ETF. The exclusion recipe this cell was
given (test-ticker regex + orb_asset_class_map 'wrapper' tag) does not filter warrant/unit/rights suffixes;
a conservative '+'/'=' suffix count finds ~186 such symbols (~1.2%) in the 2025-26 raw panel, undercounting
pure-letter W/U/R-suffixed warrants. This likely inflates the extreme-negative reads at low N (warrants carry
leveraged, option-like trailing returns that mechanically dominate a raw-return top-10 and then revert hard).
Per the PREREG multiplicity rule the universe was NOT changed after seeing this — reported as specified, with
this flagged as the first fix to pre-register (a common-stock security_type / suffix filter) before any re-run.

## MDE and capital note

MDE for a 65-week series is ~2 Sharpe units of noise (PREREG figure) — thin by construction; see null_*_std columns in 1700_reads.csv for the empirical per-cell null spread.
Capital note: the sleeve holds $65,000 overnight all week (median position $1,300-$6,500 across N=50..10). Same equity base as ORB/HOD overnight exposure on this account — a market-wide gap-down morning hits all three simultaneously (shared tail, not diversified away by running multiple books).

Overall phase-1 verdict: no cell passes the pre-registered bar on both halves.
