# PREREG — cell 1,700e: a bear-market exit for the A1 momentum book (FROZEN 2026-10-02 11:35 UTC)

Owner 10/2: "can we discover the bearish market and exit?" Book under test = cell 1,700d's best cell, "A1": large caps
(prior close ≥ $10, ADV20 ≥ $200M), top 20 by 12-1 momentum, equal weight, weekly rebalance (Monday open), costs as
1,700c. Unfiltered reference: 24.8 %/yr vs SPY 15.0 %, excess +7.2 / +9.6 by half, alpha t 0.6, max DD −63 %.
Expectation written down first: a market-trend filter should cut the 2020 and 2022 drawdowns; it cannot fix 2021
(SPY +29 % while the book lost 12 %: a factor reversal inside a bull market, not a bear market).

## Regime filters (fixed, the literature's parameters, nothing tuned)
Evaluated at each Monday rebalance on data through the prior Friday close; when OFF the book is 100 % cash
(0 % return), and re-entry buys the current top 20 at the next ON rebalance:
* F0 none (reference).
* F1 SPY close > SPY 200-day simple moving average.
* F2 absolute momentum: SPY 12-month (252-day) return > 0.
* F3 Faber: SPY close > 10-month SMA, evaluated at month-end only (held for the month).
* F4 book drawdown kill: the book is OFF after a 20 % drawdown from its own peak until SPY closes back above its
  200-day SMA.
* F5 F1 with a 2 % hysteresis band (ON above SMA × 1.02, OFF below SMA × 0.98) — the whipsaw-reduced version.
Robustness: each filter also on the 12-1 top-10 and top-50 books (same universe).

## Reads (whole 2016-01..2026-09, H1 2016-01..2021-06, H2 2021-07..2026-09)
As 1,700c/d: annualised net return, SPY, excess, alpha and beta (OLS, t), Sharpe, max DD (book, SPY), worst period,
turnover, cost drag, green-period share, the count-matched null percentile of the excess (300 draws, stated);
plus: share of weeks in cash, number of switches, the by-year table for every filter on A1, and the 2020-02..04,
2021 and 2022 sub-windows for F0 vs each filter.

## Pass bar (unchanged from 1,700c/d)
Net excess over SPY > 0 in BOTH halves AND alpha t ≥ 2.0 AND max DD ≤ 1.25 × SPY's AND null percentile ≥ 95 % in
both halves. 6 filters × 3 books = 18 cells; Bonferroni line stated. A pass → independent rebuild → paper at $20K.
A fail is reported with the by-year table; no MA length or threshold is tuned after seeing numbers.

## Output
`1700e_regime.py` (reuse 1700d_grid.py's loaders, book construction, costs, stats, null), `1700e_cells.csv`,
`1700e_by_year.csv`, `RESULT_1700e.md` (≤ 90 lines). The agent returns ≤ 150 words.
