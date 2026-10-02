# PREREG — cell 1,700l: mid-week actions on the selected names (FROZEN 2026-10-02 15:40 UTC, before any number)

Owner 10/2: "anything mid week on the selected stocks that can help?" Reference = the reconciled sleeve (risk-adjusted
top 20, all names reset to 1/20 every Monday open; CAGR 27.2 %, max DD −44.5 % on the daily basis, 1,700j REF).
Already measured and NOT repeated: daily trailing name stops to cash (15/20/25 % — cost 3.5–4.5 pts CAGR, cut 0–9 pts
of DD), volatility targeting, drawdown sizing, N, caps (1,700e/i/j/k, 54 cells, all fail).

## Cells (fixed; each on the reference, costs as 1,700c, daily bars, decisions on the prior close, trades at the open)
* M1 stop-and-REPLACE: a name closing 15 % below its highest close since entry is sold at the next open and replaced
  the same morning by the highest-ranked name not held (Monday's ranking); the slot is never cash. M1b at 20 %.
* M2 mid-week re-rank: the full ranking and 1/20 reset run on Monday AND Thursday (twice weekly).
* M3 earnings handling (EDGAR 8-K item 2.02 calendar, research/edgar_desk/events_raw.csv, acceptance UTC → ET):
  M3a a held name is sold at the open of its event session and re-bought at the next Monday reset if it still ranks;
  M3b a name with an event session inside the coming week is not ENTERED on Monday (kept names stay). Coverage of the
  calendar on held names reported; VOID below 80 %.
* M4 rebalance timing: M4a Monday CLOSE instead of the open; M4b Wednesday open; M4c Friday close (signal through
  Thursday) — the day/time-of-week placebo, also the execution question for the live sleeve.
* M5 gap-down exit: a held name opening ≥ 8 % below the prior close is sold at that open, slot replaced as M1.
* M6 mid-week add on strength: a held name making a new 20-day closing high mid-week gets +2.5 % weight taken pro rata
  from the others until Monday (the "press the winner" rule).

## Reads
CAGR, max DD, end $ from $50K, Sharpe, worst year, years beating SPY /10, rolling-5-year share, turnover and cost
drag, the five 1,700j episodes' depths, paired weekly return difference vs REF (mean, t, ex-top-5 %).

## Pass rule
A mid-week action is recommended only if the CAGR-to-max-DD ratio rises by ≥ 0.10 over REF's (0.61) with CAGR
≥ 25 % AND the paired weekly difference is ≥ 0 ex-top-5 % AND both halves (2017–2021, 2022–2026) agree in sign.
M4 is also read as an execution choice: a timing within ±1 pt of CAGR and DD of the Monday open is "equivalent".
10 cells; nothing outside the grid after seeing numbers.

## Output
`1700l_midweek.py` (reuse 1700j_frontier.py's daily engine), `1700l_cells.csv`, `RESULT_1700l.md` (≤ 80 lines).
ONE process, nice 19, < 2 GB. The agent returns ≤ 150 words.
