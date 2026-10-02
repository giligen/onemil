# PREREG — cell 1,700n: a diversifier and a hedge for the momentum sleeve (FROZEN 2026-10-02 17:25 UTC, before any number)

Owner 10/2: "get to a similar $500K over 10 y … with smaller DD"; 54 cells of stops, sizing, caps and pause signals
all failed because they act INSIDE the factor (CAGR/DD stays ≈ 0.6). This cell acts OUTSIDE it: add a second return
stream that is positive on its own and uncorrelated in the sleeve's drawdown episodes, or hedge the market leg.
Reference = the reconciled sleeve (1,700j REF: CAGR 27.2 %, max DD −44.5 %, $508K from $50K, daily basis).

## Part A — asset-class momentum diversifier (ETFs from the same free panel; none are in the stock sleeve)
ETF universe (fixed list; those present in the panel on each date): SPY QQQ IWM EFA EEM VNQ TLT IEF SHY LQD HYG TIP GLD
SLV DBC USO XLE XLF XLK XLV XLI XLP XLY XLU XLB. Signal 12-1 return ÷ 252-day vol; absolute filter: a slot whose ETF
has 12-1 ≤ SHY's goes to SHY. Books: A1 top 3 monthly; A2 top 3 weekly; A3 top 5 monthly. Costs 2 bps + spread proxy.
Blends with the stock sleeve, rebalanced monthly to the target mix: 70/30, 50/50, 30/70 (sleeve/diversifier) for A1
(and for A2/A3 if A1's stand-alone max DD > −25 %).

## Part B — market hedge on the sleeve (short SPY; borrow 0.5 %/yr; cash earns 0)
B1 static 30 % short SPY; B2 static 50 %; B3 beta-matched (trailing 126-day beta × 0.5); B4 conditional: 50 % short
only while SPY < its 200-day SMA; B5 conditional on the sleeve's own trailing 63-day vol above its 3-year median.

## Reads (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026)
Per cell: CAGR, max DD, CAGR/DD ratio, end $, Sharpe, worst year, years beating SPY /10, rolling-5-year share,
correlation of weekly returns with the sleeve (whole and inside the five 1,700j episodes), each episode's depth,
the diversifier's own return inside each episode, turnover and cost.

## Pass rule
A blend or hedge is recommended only if max DD improves by ≥ 10 points vs REF AND CAGR ≥ 20 % AND the CAGR/DD ratio
rises by ≥ 0.15 AND ≥ 3 of 5 episodes are cut AND both halves keep CAGR ≥ 15 %. Also reported for the owner's
choice: the point on each frontier that maximises end $ subject to max DD ≤ −30 %. 3 + 9 + 5 = 17 cells; nothing
outside the grid after seeing numbers.

## Output
`1700n_diversify.py` (reuse 1700j_frontier.py's daily engine for the sleeve), `1700n_cells.csv`, `RESULT_1700n.md`
(≤ 80 lines). ONE process through scripts/research_run.sh (-m 2500M). The agent returns ≤ 150 words.
