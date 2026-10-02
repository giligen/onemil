# PREREG — cell 1,700o: is the conditional market hedge real? Neighbours of 1,700n B4 (FROZEN 2026-10-02 17:50 UTC)

1,700n: 0/17 pass, but ONE cell improved both return and drawdown: B4 = short SPY at 50 % of the sleeve's value only
while SPY closes below its 200-day SMA (on 8 % of days): CAGR 28.9 % vs 27.2 %, max DD −40.3 % vs −44.5 %, $579K vs
$508K. One cell of 17 on 8 % of days is a candidate, not a finding. This cell reads its NEIGHBOURS, fixed now:
* Trend rule: SPY < 150-day SMA / 200-day SMA / 10-month SMA (month-end evaluation) / SPY 252-day return < 0.
* Hedge ratio: 25 % / 50 % / 75 % / 100 % of the sleeve's value.
* Instrument: short SPY / short QQQ.
4 × 4 × 2 = 32 cells (B4 itself is one of them). Borrow 0.5 %/yr, costs as 1,700n, evaluated on the prior close.
Reads per cell: CAGR, max DD, ratio, end $, days hedged, the hedge leg's own P&L per hedged spell (how many spells,
how many profitable), each of the five 1,700j episodes' depth, halves 2017–2021 / 2022–2026.
A conditional hedge is REAL only if: ≥ 24 of 32 neighbours improve BOTH CAGR and max DD vs REF, the hedge leg is
profitable in ≥ 60 % of spells, and both halves improve the ratio. Otherwise B4 is recorded as a favourable draw.
If real, the recommended cell is the family's MEDIAN neighbour (never the best), and it goes to the paper sleeve as
a flag (default off) with the owner's word. Output: 1700o_hedge.py, 1700o_cells.csv, RESULT_1700o.md (≤ 60 lines).
