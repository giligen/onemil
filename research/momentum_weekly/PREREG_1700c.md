# PREREG — cell 1,700c: monthly momentum decile on free 2016→ history, does it beat SPY (FROZEN 2026-10-02 06:05 UTC)

Owner 10/2: "get the data, run it on history, show me it beats SPY". Phase 2 of PREREG_1700 / 1,700b: the ONLY book
carried forward is the monthly-rebalanced top decile (the weekly books had negative alpha; the naive top-10 loses).

## Data (free only)
Alpaca daily bars (adjusted for splits; `adjustment=all` if available, else state it) for EVERY us_equity asset the
Alpaca assets endpoint returns with status active AND inactive (delisted names included where Alpaca still serves
their bars — tested first on 3 known delisted tickers and reported), 2015-07-01..2026-09-30, plus SPY. Written once
to `research/momentum_weekly/panel_2016_2026.parquet` (atomic write, resumable, LOST count + completeness gate: ≥ 95 %
of active assets with bars, else VOID). Survivorship statement required: per year, the share of names in the Databento
point-in-time panel (2024-07→) that are missing from the Alpaca panel, and the number of inactive assets with bars.

## Book (fixed, as 1,700b's `decile_monthly`)
Universe at each rebalance (first trading day of the month, at its open): price ≥ $10 at the prior close, 20-day
average dollar volume ≥ $20M, data history ≥ 252 days, not an ETF/ETN/fund/trust/warrant/unit/preferred/right (by
asset name pattern for inactive assets; by Databento security_type 'C' where known), not `^Z[A-Z]ZZT$`. Signal: 12-1
momentum (return from t−252 to t−21 trading days). Portfolio: top decile of the signal, equal weight, held one month;
costs 5 bps per side + half the spread proxy (daily high−low/close × 0.1, capped at 20 bps) on every traded dollar;
dividends ignored on both the book and SPY (price returns for both — stated).

## Windows and reads
2016-01..2026-09 (≈ 129 rebalances); halves H1 = 2016-01..2021-06, H2 = 2021-07..2026-09; 2020-02..2020-04 shown.
Per window: annualised net return, SPY's, excess return, alpha and beta vs SPY (monthly OLS, t on alpha), Sharpe,
max drawdown (book and SPY), worst month, turnover/month, cost drag/yr, green-month share, the count-matched null
(1,000 random same-size draws from the same monthly universe → percentile of the book's annualised return and of its
excess over SPY), and the equity curve table by year (book vs SPY).

## Pass bar
Beats SPY = net annualised excess return > 0 in BOTH halves AND alpha t ≥ 2.0 on the whole window AND max drawdown
≤ 1.25 × SPY's AND null percentile of the excess ≥ 95 % in both halves. A pass → independent rebuild from this prose →
paper sleeve at $20K notional with its own script and ledger. A fail is reported with the by-year table; no variant
(decile width, skip, universe floor) is tried after seeing numbers.

## Output
`research/momentum_weekly/1700c_fetch.py`, `1700c_momentum.py` (reuse `1700b_momentum.py`'s functions where the
panel format allows), `1700c_monthly.csv`, `1700c_reads.csv`, `1700c_by_year.csv`, `RESULT_1700c.md` (≤ 100 lines,
the by-year book-vs-SPY table first), `1700c.log`. The agent returns ≤ 150 words.
