# RESULT -- cell 1,700b: weekly momentum sleeve, literature-definition universe fix

PREREG_1700b.md (FROZEN). Fixes cell 1,700 (RESULT_1700.md): the naive top-N universe selected micro-cap hype names and a warrant (QBTS+). This cell restricts to domestic common stock + a size/liquidity floor + conventional portfolio widths, pre-declared before any number was read.
Window 2025-07-01..2026-09-30: 60 weekly, 14 monthly rebalances (halfA 2025-07-01..2025-12-31, halfB 2026-01-01..2026-09-30).
Common-stock filter (Databento point-in-time security_type, 27 monthly files, unique symbols last-seen in window): {'Q': 6636, 'C': 5664, 'O': 1141, 'W': 876, 'P': 600, 'U': 526, 'A': 458, 'R': 270, 'S': 129, 'L': 48, 'V': 12, 'T': 1} -> **5664 kept as C (common)**; excluded by class: Q=ETF, P=preferred, W=warrant, U=unit, R=rights, L=LP, V=royalty-trust, O/A=foreign-ordinary/ADR, S=mixed REIT/closed-end-fund. 0 panel rows unmatched to a (symbol,month) definition -> excluded (undercount only). SPAC-name filter: 1095 symbols with "acquisition" in name (orb_asset_class_map, covering 12802/16217 raw symbols). 7 test-ticker symbols (`^Z[A-Z]ZZT$`). Price >= $10, adv20 >= $20M, both at the signal date.
Cross-checks: cache.db daily_bars (read-only) 219/300 sampled closes within 0.5%; overnight_high panel $-volume (dvol20) agreement 95.0% of 1742669 matched rows within 10%.

## Reads (12 = 4 books x 3 windows; annualised net-of-cost unless marked)

| book | freq | window | n_periods | median_pool_n | median_port_n | ann_return_net | ann_alpha_spy | sharpe_net | max_dd | turnover_avg | cost_drag_annual | null_pct_return | null_pct_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decile_monthly | monthly | halfA | 5 | 1535.000 | 153.500 | 0.488 | -0.274 | 1.938 | -0.052 | 0.350 | 0.009 | 100.000 | 9.200 |
| decile_monthly | monthly | halfB | 8 | 1633.500 | 163.000 | 0.243 | 0.093 | 0.836 | -0.170 | 0.267 | 0.009 | 70.700 | 1.400 |
| decile_monthly | monthly | whole | 13 | 1595.000 | 159.500 | 0.332 | 0.088 | 1.188 | -0.170 | 0.302 | 0.009 | 99.500 | 1.200 |
| decile_weekly | weekly | halfA | 26 | 1554.000 | 155.000 | 0.328 | -0.065 | 1.320 | -0.134 | 0.159 | 0.021 | 100.000 | 68.300 |
| decile_weekly | weekly | halfB | 34 | 1633.000 | 163.000 | 0.234 | -0.017 | 0.901 | -0.190 | 0.125 | 0.019 | 65.400 | 4.100 |
| decile_weekly | weekly | whole | 60 | 1603.500 | 160.500 | 0.274 | -0.025 | 1.073 | -0.190 | 0.139 | 0.020 | 98.400 | 11.900 |
| top20_weekly | weekly | halfA | 26 | 1554.000 | 20.000 | -0.035 | -0.466 | 0.137 | -0.278 | 0.187 | 0.026 | 10.200 | 17.800 |
| top20_weekly | weekly | halfB | 34 | 1633.000 | 20.000 | 0.431 | 0.133 | 1.104 | -0.256 | 0.196 | 0.030 | 90.500 | 46.800 |
| top20_weekly | weekly | whole | 60 | 1603.500 | 20.000 | 0.206 | -0.095 | 0.649 | -0.291 | 0.192 | 0.028 | 61.700 | 23.100 |
| top50_weekly | weekly | halfA | 26 | 1554.000 | 50.000 | 0.223 | -0.275 | 0.709 | -0.226 | 0.198 | 0.028 | 77.700 | 27.200 |
| top50_weekly | weekly | halfB | 34 | 1633.000 | 50.000 | 0.407 | 0.072 | 1.140 | -0.214 | 0.136 | 0.021 | 95.700 | 35.400 |
| top50_weekly | weekly | whole | 60 | 1603.500 | 50.000 | 0.324 | -0.050 | 0.952 | -0.226 | 0.163 | 0.024 | 95.700 | 24.900 |

## Pass bar (identical to 1700): alpha>=+8%/yr AND Sharpe>=1.0 AND null pct(return)>=95, both halves; cost drag<4%/yr; max DD<=20%

- decile_weekly: fail; whole-window alpha -0.025, Sharpe 1.07, null-pct(ret) 98, median pool 1604, median N 160 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- top50_weekly: fail; whole-window alpha -0.050, Sharpe 0.95, null-pct(ret) 96, median pool 1604, median N 50 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- top20_weekly: fail; whole-window alpha -0.095, Sharpe 0.65, null-pct(ret) 62, median pool 1604, median N 20 -> fail -> non-positive whole-window point estimate, no phase-2 case on this read
- decile_monthly: fail; whole-window alpha 0.088, Sharpe 1.19, null-pct(ret) 100, median pool 1595, median N 160 -> fail -> positive whole-window point estimate, phase 2 (free Alpaca history) worth running

## MDE and capital note

MDE for a 65-week series is ~2 Sharpe units of noise (PREREG figure); the monthly book has only ~14 periods and is far thinner still -- read it as directional, not conclusive. See null_*_std columns in 1700b_reads.csv for the empirical per-cell null spread.
Capital note: the sleeve holds $65,000 overnight all week/month. Same equity base as ORB/HOD overnight exposure on this account -- shared tail on a gap-down morning.

Overall phase-1 verdict: no book passes the pre-registered bar on both halves.
