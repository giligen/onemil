# PREREG — cell 1,700r: residual (beta-adjusted) momentum (FROZEN 2026-10-02 17:50 UTC, before any number)

Why: ~100 cells of stops, sizing, caps, gates, hedges and timing all failed to cut the sleeve's drawdown without
paying for it one-for-one, and the reconciliation found the sleeve's alpha after beta ≈ 0 (t 0.6): the book is a
high-beta theme bet, so its drawdowns ARE the market/theme leg. Every repair so far acted on the portfolio; none
changed WHAT is ranked. Mechanism under test (Blitz–Huij–Martens 2011): ranking on the part of a stock's return its
factors do not explain selects names that rose for their own reasons, carries less market/theme beta, and has
historically had about half the crash risk of total-return momentum at a similar return.

## Signal (everything else = the reconciled sleeve: U2, Monday open, all N names reset to 1/N weekly, costs as 1,700c)
For each name and day: beta(s) from an OLS with intercept of the name's daily returns on the factor returns over the
trailing W days ending at t−1; residual_d = r_d − Σ beta × factor_d (the intercept is NOT removed — it is the thing
ranked). Score = Σ residual over days t−252..t−21 (or t−126..t−21) ÷ the standard deviation of those residuals.
A name needs ≥ 80 % of the W window present. Causal: nothing after the Friday close before the rebalance.

## Family (16 cells, declared now; judged as a family, never by its best cell)
factor set {SPY only, SPY + QQQ + IWM} × lookback {12-1, 6-1} × N {20, 30} × beta window W {252, 504}.
REF = the reconciled sleeve (must reproduce 27.18 % / −44.5 % / $507,823 before any cell is read).

## Reads per cell (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026)
CAGR, max DD, CAGR/DD ratio, end $, Sharpe, worst year, years beating SPY /10, rolling-5-year share, the five 1,700j
episode depths, beta to SPY and annualised alpha with its t (weekly regression), weekly-return correlation with REF,
share of holdings shared with REF, turnover and cost drag, paired weekly difference vs REF (mean, t, ex-top-5 %).

## Pass rule
A cell "improves" if max DD is ≥ 8 points better than REF AND CAGR ≥ 22 % AND the ratio is ≥ REF's + 0.15.
The family is REAL only if ≥ 12 of 16 cells improve AND the ratio beats REF's in BOTH halves in ≥ 12 of 16 AND the
median cell (by ratio) cuts ≥ 3 of the 5 episodes. If real: recommend the MEDIAN cell, also report a 50/50 blend of
it with REF, and it goes to an independent rebuild before any paper flag. If 6–11 improve: "partial" — report the
axis that separates passing from failing cells, no recommendation. Otherwise FAIL. Cell count on this line: +16.

## Output
`1700r_residual.py` (reuse the 1700j/1700l daily engine with a custom score matrix; load the panel as
1700p_neighbours.py does — dictionary-encoded symbols — and keep return matrices float32), `1700r_cells.csv`,
`RESULT_1700r.md` (≤ 70 lines, caveats section written as an adversary). ONE process through
`bash scripts/research_run.sh -m 2500M`. The agent returns ≤ 150 words.
