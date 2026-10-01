# PREREG — cell 1,700: weekly-rebalanced 12-month momentum sleeve (FROZEN 2026-10-01 19:20 UTC)

Owner 10/1: "Every Monday buy the stocks that did the best P&L over the past year. Every Monday rebalance based on this."

## Rule family (fixed)
Universe at each Monday: US common stocks and ETFs priced ≥ $5 with 20-day average dollar volume ≥ $5M, from the
point-in-time daily panel on disk (Databento EQUS.SUMMARY parquet under data/research/databento, delisted included;
cross-checked with data/cache.db daily_bars read-only). Signal: trailing 252-trading-day total return (price return
on the data available) measured at Friday's close. Variants: M1 = 12-0 (the full year), M2 = 12-1 (skip the last 21
days — the academic convention against short-term reversal), M3 = 6-1. Portfolio: top N by the signal, N ∈ {10, 20,
50}, equal weight, bought at Monday's open (market-on-open), held one week, rebalanced the next Monday at the open
(only the changed names trade; weights drift within the week). Costs: 5 bps per side + half the measured spread
proxy (daily high-low/close × 0.1, capped at 20 bps) on every traded dollar; overnight gaps are real (daily bars).

## Windows
Phase 1 (this cell, data on disk): rankings possible from 2025-07 (the panel starts 2024-07-01) → 2025-07..2026-09,
≈ 65 Mondays; halves 2025-07..2025-12 and 2026-01..2026-09. Phase 2 (only on the owner's word — a free Alpaca daily
fetch of ~3,000 names back to 2016, survivorship stated): the long history incl. the 2020 crash.

## Reads (per variant × N; both halves and whole)
Weekly return series; annualised return, volatility, Sharpe (weekly, annualised), max drawdown, worst week, best
week, green-week share; turnover per week (share of the portfolio replaced) and the cost drag in %/yr; beta to SPY
and the alpha vs SPY and vs the equal-weight universe; the count-matched null (1,000 random top-N draws from the
universe each Monday → the percentile of the realised annualised return and Sharpe); concentration (median position
size in $ at $65K; names below $5M dollar volume that would move on our size); the capital note: the sleeve holds
$65K overnight all week — it does not collide with intraday books on buying power under margin, but it is the same
equity at risk; the shared tail with ORB/HOD on gap-down mornings.

## Pass bar (phase 1, a sleeve on its own bar)
Annualised alpha vs SPY ≥ +8 % with weekly Sharpe ≥ 1.0 on BOTH halves, null percentile ≥ 95 both, cost drag
< 4 %/yr, max drawdown ≤ 20 % in the window. A pass → independent rebuild from prose → paper sleeve (its own script
and ledger, like the turn-of-month sleeve) at $20K notional first. A null states the MDE for a 65-week series
(≈ 2 Sharpe units of noise — thin) and recommends phase 2 if the point estimate is positive.

## Multiplicity
3 variants × 3 N × 3 windows = 27 reads + nulls. Not allowed: changing N, the skip, the costs or the universe after
seeing numbers; survivorship from a current-listing universe (the panel's delisted names must be in).

## Output
`research/momentum_weekly/RESULT_1700.md` (≤ 100 lines), `1700_weekly.csv`, `1700_reads.csv`, `1700_momentum.py`,
`1700_momentum.log`. The agent returns ≤ 150 words.
