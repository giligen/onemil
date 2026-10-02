# PREREG — cell 1,700j: the drawdown frontier of the momentum sleeve (FROZEN 2026-10-02 14:05 UTC)

Owner 10/2: paper sleeve from Monday, live in 2–3 weeks, "I'm 100 % sure we can reduce the drawdowns." Book = the
risk-adjusted top 20 weekly (1,700g V2; CAGR 27 %, max DD −42 %, 5/10 years). Already tested and recorded: market
exits (all hurt), 20 % vol target (DD −30 %, CAGR 12 %), 20 %-from-entry name stop (DD −34 %, CAGR 18 %), 50/50 SPY
(DD −33 %, CAGR 22 %), N = 50 (DD −36 %, CAGR 22 %). This cell maps the frontier instead of hunting one knob.

## Step 1 — drawdown anatomy (before any repair is read)
The five deepest peak-to-trough episodes of V2 2017–2026: peak date, trough date, recovery date, depth, length, SPY's
drawdown over the same dates, the book's holdings at the peak (names, implied theme), and how the loss was taken
(the names' own declines vs rotation into new names that then fell). Written first; the repairs are read against it.

## Step 2 — repairs (fixed grid, 14 cells; each on V2, weekly, costs as 1,700c, no leverage)
* T1 trailing name stop 15 % / T2 20 % / T3 25 % from the name's highest close since entry (exit to cash, re-entry
  only when it re-qualifies at a later rebalance).
* S1 vol target 30 % / S2 35 % (milder than the tested 20 %), exposure = min(1, target ÷ trailing 63-day realised vol).
* D1 drawdown-responsive size: half size while the book is > 15 % below its own high, full size at a new high.
* N1 N = 30; N2 N = 40.
* C1 single-name cap by weekly re-equalisation (no name above 8 %: the weekly rebalance already re-equalises
  entrants only; C1 re-equalises every name every week — the turnover cost of discipline).
* J1 T2 + S1; J2 T2 + N1; J3 T2 + D1; J4 T2 + S1 + N1.
* Reference V2 unchanged.

## Reads
Per cell: CAGR, max DD, the five-episode depths (does the repair cut the SAME episodes?), worst year, years beating
SPY /10, rolling 5-year windows beating SPY (share of 56), Sharpe, turnover, cost drag, weeks at reduced size,
and the frontier chart as a table (CAGR vs max DD) — the owner picks the point.

## Pass rule (a repair is recommended only if)
Max DD improves by ≥ 10 points vs V2 AND CAGR gives up ≤ 5 points AND the rolling-5-year share stays ≥ 90 % AND the
cut applies to at least 3 of the 5 episodes (not one lucky episode). Nothing outside the grid after seeing numbers.

## Output
`1700j_frontier.py` (reuse the 1700g/1700i machinery), `1700j_episodes.csv`, `1700j_cells.csv`, `RESULT_1700j.md`
(≤ 90 lines: anatomy table first, then the frontier table). The agent returns ≤ 150 words.
