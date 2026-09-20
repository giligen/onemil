# PREREG — exec_cost entry cells (1,289 / 1,290)

Pre-registered before running the analysis. Book: `analysis_results/orb_bplus_book.csv`
(filled rows only, `entered==1`). TEST >= 2026-06-01 is SEALED — report TRAIN (2025) and
VAL (2026-01..05) only.

## CELL 1,289 — post-then-cross entry

For each fill, fetch the SIP trade tape (Alpaca `get_trades` / `StockTradesRequest`) and
the NBBO quotes for the 60 seconds after the trigger (trigger = the first 1-min bar whose
high > range_high; reuse the trigger-instant NBBO already fetched at
`research/fuckup_audit/P_cost/spreads.parquet` — `entry_fill_ts`/`entry_bid`/`entry_ask` —
which is defined identically: "the fill instant = the first TRADE in the breakout minute
with price > range_high", NBBO = last quote at or before it).

Rule: rest a BUY limit at the NBBO midpoint at the trigger for 30 s; it FILLS if a trade
prints at or below the limit within the 30 s (obtainable: a trade at our price after our
order); otherwise cross at the ask at t+30 s.

Old cost = half-spread at the trigger. New cost = 0 on passive fills (report the
spread-earned share) and the half-spread at t+30 s on the rest.

Report: passive fill rate, Δ cost per trade in R and $, and the DECIDING TABLE: outcome
(net R) of trades that filled passively vs those that crossed at t+30 s vs the book's
original fill — if the passive fills are the worse trades and the crossed ones are the
runners entered later at a worse price, adverse selection; report the net Δ$ of the whole
rule on both splits with day-clustered t.

**Pass bar**: net Δ$ > 0 both splits and no adverse selection (passive-fill outcomes not
worse than crossed-fill outcomes by more than 0.05 R).

## CELL 1,290 — spread-in-R gate

Half-spread at the trigger ÷ R (R = range_high − range_low) per fill from the same quote
data. Veto (NO refill) fills with ratio > 0.10; report also 0.05 and 0.15 as a
monotonicity check (3 sub-cells, counted).

**Pass bar**: net $ up and MDD not worse in both splits; the vetoed set's net must be
negative in both splits.

## Data source note

`research/orb_multiwindow/` (named in the task brief as the source of a Stage-Q NBBO
walk) contains no NBBO/quote file — checked directly. The actual reusable artifact is
`research/fuckup_audit/P_cost/spreads.parquet` (Stage P, honest ORB book, SIP consolidated
quotes, "last quote at or before the decision" honesty rail), which already carries the
trigger-instant NBBO (`entry_bid`/`entry_ask`/`entry_fill_ts`) for this same book. CELL
1,290 uses only this file. CELL 1,289 additionally fetches a fresh 30 s trade+quote window
starting at `entry_fill_ts` via the same `StockHistoricalDataClient` / `DataFeed.SIP` path
used by `fetch_spreads.py`, since post-trigger ticks were not previously pulled.
