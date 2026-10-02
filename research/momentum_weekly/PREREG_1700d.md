# PREREG — cell 1,700d: the owner's "top 10 works" — parameter grid on the 10-year free panel (FROZEN 2026-10-02 11:05 UTC)

Owner 10/2: "I know as a fact that buy the top 10 works very well. Play with the params, rebalancing weekly on the
past 52 weeks, maybe 13 months instead of 52 weeks." Our reads so far: the naive top-10 weekly on a broad universe
loses (1,700, 60 Mondays: micro-cap hype + warrants); the monthly decile on 10 years does not beat SPY (1,700c).
The owner's experience is probably a large-cap list — so the grid includes one. Programme count: 1,700d.

## Data
`research/momentum_weekly/panel_2016_2026.parquet` (Alpaca daily, 15,789 names incl. 2,469 delisted, 2015-07..
2026-09-30; survivorship stated per 1,700c) + SPY. No new data.

## Grid (fixed before any number; every cell reported, none dropped)
* Universe at each rebalance: U1 liquid common stocks (prior close ≥ $10, ADV20 ≥ $20M — 1,700c's); U2 large caps
  (prior close ≥ $10, ADV20 ≥ $200M — ≈ the 400–500 most-traded names, our proxy for an index list without
  point-in-time constituents); both exclude ETF/ETN/fund/trust/warrant/unit/preferred/right names and `^Z[A-Z]ZZT$`,
  and require ≥ 273 trading days of history.
* N ∈ {10, 20, 50}, equal weight.
* Lookback: L1 52 weeks (t−252..t), L2 12-1 (t−252..t−21), L3 13 months (t−273..t), L4 13-1 (t−273..t−21).
* Rebalance: weekly (Monday open to Monday open) and monthly (first trading day's open).
* Costs: 5 bps per side + half the spread proxy (daily high−low/close × 0.1, capped at 20 bps) on traded dollars;
  price returns for the book and SPY (dividends ignored on both).
2 × 3 × 4 × 2 = 48 cells × 3 windows (whole 2016-01..2026-09, H1 2016-01..2021-06, H2 2021-07..2026-09).

## Reads per cell
Annualised net return, SPY's, excess, alpha and beta vs SPY (weekly/monthly OLS, t on alpha), Sharpe, max drawdown
(book, SPY), worst period, turnover per rebalance, cost drag/yr, green-period share, count-matched null (1,000 random
same-N draws from the same universe at each rebalance → percentile of the excess over SPY), top-5-name
contribution share (lottery check), by-year table for the best cell.

## Pass bar (as 1,700c — a cell "works" only if)
Net excess over SPY > 0 in BOTH halves, alpha t ≥ 2.0 on the whole window, max drawdown ≤ 1.25 × SPY's, null
percentile of the excess ≥ 95 % in both halves. Multiplicity: with 48 cells, ≥ 2–3 would pass a 5 % bar by chance
on one half; the both-halves requirement is the guard, and the best cell is re-read with the Bonferroni line stated.
A pass → independent rebuild from this prose → paper sleeve at $20K. No parameter outside the grid is tried after
seeing numbers; the grid is the owner's ask, written down before the run.

## Output
`1700d_grid.py` (reuse 1700c_momentum.py's loaders/costs), `1700d_cells.csv` (48 × 3 rows), `1700d_best_by_year.csv`,
`RESULT_1700d.md` (≤ 90 lines: the 48-cell table sorted by H2 excess with both-halves flags, the pass list, the
best cell's by-year table), `1700d.log`. The agent returns ≤ 150 words.

## Amendment 1 (2026-10-02 11:10 UTC, before any number was read)
Owner: "maybe 6 mo", "weekly rebalance sounds much better". Added lookbacks L5 = 6 months (t−126..t) and L6 = 6-1 (t−126..t−21): 72 cells × 3 windows. Weekly cells are listed first in the RESULT. Pass bar unchanged; the Bonferroni line now counts 72.
