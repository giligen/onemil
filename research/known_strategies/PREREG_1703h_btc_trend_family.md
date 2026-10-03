# PREREG — cell 1,703h: is the 20-day BTC trend rule a plateau or a peak? (FROZEN 2026-10-03, before any number)

Cells 1,703e/g judged ONE lookback (20 days) because the literature used it. Before anyone acts on it personally
(owner holds IBIT; IBIT-trend 32 % / −26 % vs hold 24 % / −53 % since 2024-01), the neighbour family must be read:
a rule whose edge sits only at its published parameter is a fit, a plateau is a mechanism.

## Fixed specification
Data: BTC-USD daily closes 2014-09 → 2026-09 from yfinance (free), spliced with the Alpaca OHLC in
`1703g_btc_ohlc.parquet` where both exist (price-scale check: close-ratio median within 0.5 % on the overlap, else NOT
reportable). Rule family, each a separate cell, long BTC when the condition held at the prior close, else cash, 10 bp per switch:
- momentum sign over N days, N ∈ {10, 15, 20, 30, 50, 100}  (6 cells; N = 20 is the 1,703e rule)
- close above its N-day SMA, N ∈ {50, 100, 200}  (3 cells)
- "always long" and "always long 50 % of capital" as references (not cells).
Windows: 2014-09 → 2026-09 (full), 2018-01 → 2026-09 (1,703e window), and the three bear years alone (2018, 2022, 2025).
Cells: +9 (1,703h-1…9).

## Reads
CAGR, max DD, CAGR/DD, time in market, switches/yr, worst year, each bear year's return; rank of N = 20 among the
momentum family on CAGR/DD in each window; the spread of CAGR/DD across N ∈ {15, 20, 30} (plateau width).
Stack read with the guarded sleeve (GREF, `research/momentum_weekly/1700u_curves_daily.csv` column `guard`, 2017-01 →):
weekly-return correlation of the best plateau cell and the 50/50 portfolio's CAGR / max DD / ratio vs GREF alone (0.75).

## Pre-committed rule
PLATEAU if every N ∈ {15, 20, 30} has CAGR/DD within 0.15 of N = 20 in both windows AND each beats always-long on
CAGR/DD; PEAK otherwise. A PEAK means the 1,703e/g numbers are a fit and the owner is told not to act on the 20-day rule.
A PLATEAU does not change the 1,703e stack verdict (stand-alone bar unmet) — it only licenses the personal-holding use.
The SMA cells answer whether the simpler "above the 200-day" form (the one most people run) does the same job.

## Output
`research/known_strategies/1703h_family.py`, `1703h_cells.csv`, `1703h_byyear.csv`, `RESULT_1703h.md` ≤ 45 lines
(price-scale check first, the family table, the plateau/peak verdict by the rule above, adversary caveats).
Through `bash scripts/research_run.sh -m 2000M`. Agent returns ≤ 150 words.
