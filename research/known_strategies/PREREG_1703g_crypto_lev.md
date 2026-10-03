# PREREG — cell 1,703g: leveraged BTC trend with stops, executable on Alpaca (FROZEN 2026-10-03, before any number)

Owner 10/3: "we use alpaca! how can we get a multiplier on crypto with shallower stops". The executable instruments on
Alpaca are US-listed ETFs (commission-free, spread 1–5 bp): IBIT (1×), BITX / BITU (2× daily reset, since 2023-06/2024-04).
Alpaca crypto spot carries no leverage for a US retail account and costs 15–25 bp per trade — 40 switches a year = 6–10 %/yr,
so spot is NOT an execution candidate for a 20-day trend rule. 3× single-asset crypto ETFs do not exist in the US.

## Fixed specification (one grid, every cell counted)
Base rule = cell 1,703e BTC leg: long when the 20-day return at close t−1 > 0, else cash; evaluated daily on the BTC
calendar series (`1703e_crypto.parquet`); a position change executes at the next US-session open of the ETF (the ETF
trades 09:30–16:00 ET only; the BTC signal moves 24/7 — weekend and overnight gaps are part of the result, not an error).
Leverage L ∈ {1, 2} simulated as a daily-reset product on BTC's open-to-open return: ETF return = L·r − (ER + (L−1)·(T-bill + 50 bp))/365,
ER = 25 bp (IBIT) / 185 bp (BITX); the simulation is checked against the real BITX series on its overlap (close-to-close
correlation ≥ 0.99 and the tracking difference reported; if the check fails the 2× cells are NOT reportable).
Stop S ∈ {none, 2.0, 1.0, 0.5} × 20-day ATR of BTC, measured from the entry price, evaluated on the BTC DAILY LOW (a
stop hit during the US session fills at the stop; a gap through the stop fills at the next ETF open — the worse of the two);
after a stop the rule stays flat until the 20-day signal turns negative and positive again (no immediate re-entry).
Cost 10 bp per switch (spread + slippage on the ETF). Windows 2018-01 → 2026-09 and 2022-01 → 2026-09 (BTC series;
pre-ETF years are the simulated product). Cells: 2 L × 4 S = 8 (+8 to the programme count; 1,703g-1…8).

## Reads
CAGR, max DD, CAGR/DD, worst month, months green, switches/yr, time in market, ex-top-5 % of months, and the two halves
(2018–21 / 2022–26) same-signed. Reference rows: BTC hold 1×, simulated 2× hold, the 1,703e BTC 1× trend (37.7 % / −58.1 %).

## Pre-committed expectations (written before the run)
Variance drag of a daily-reset 2× at BTC's 48–64 % vol ≈ L²σ²/2 − Lσ²/2 = σ² ≈ 23–40 %/yr of log growth: 2× hold is
expected to compound WORSE than 1× hold over 2018–26 (Kelly-optimal leverage ≈ μ/σ² ≈ 1.0–1.3). The trend rule
removes part of the variance (cash ~48 % of days), so 2×-trend may beat 1×-trend on CAGR with a deeper DD.
Stops: a 0.5-ATR stop on a 3–4 %/day asset is hit by noise most weeks; expected to lower CAGR and DD together with
CAGR/DD falling (the R-must-exceed-noise rule). Pass bar for "recommend to the owner as a personal rule": CAGR/DD ≥ the
1× trend's 0.65 AND max DD better than −45 % AND both halves same-signed; otherwise the answer is "leverage via position
size on IBIT, no stop beyond the 20-day rule".

## Output
`research/known_strategies/1703g_lev.py`, `1703g_cells.csv`, `1703g_monthly.csv` (monthly returns of every cell),
`RESULT_1703g.md` ≤ 50 lines with the BITX tracking check first. Through `bash scripts/research_run.sh -m 2500M`.
Agent returns ≤ 150 words. Independent re-read by a second agent before the owner sees a number: recompute the 1×/none
and 2×/2.0-ATR cells from this prose only, compare month by month.
