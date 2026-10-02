# PREREG — cell 1,700q: do fear gauges predict the next week? (FROZEN 2026-10-02 17:35 UTC, before any number)

Owner 10/2: "there's no statistical predictor for SPY's next 5d direction based on fear-and-greed, VIX, etc?"
Already tested and NOT repeated: trend gates on SPY (200-day, 12-month, Faber, hysteresis — 1,700e), the book's own
momentum, dispersion and the crash guard (1,700i). Never tested here: volatility and sentiment gauges.

## Data (free only; a LOST count and a coverage line per series, VOID below 80 % of weeks)
* VIX, VIX3M, VIX9D, VVIX daily closes — CBOE history CSVs (cdn.cboe.com/api/global/us_indices/daily_prices/
  <NAME>_History.csv); fallback FRED VIXCLS / yfinance ^VIX.
* SPY, TLT, HYG, IEF daily — panel_2016_2026.parquet for 2016+; SPY and ^VIX 1993+ via yfinance (adjusted) for the
  long-history read. Price-scale check: yfinance SPY vs the panel's SPY on the overlap (daily return corr ≥ 0.999).
* Universe breadth from the panel (U2 names): share above the 50-day average, net new 52-week highs minus lows.
* The sleeve's daily equity: 1700j_frontier.py's `simulate()` REF (must reproduce 27.18 % / −44.5 % / $507,823).
* CNN Fear & Greed actual index if a free history is reachable (report coverage; no scraping behind a login).

## Predictors (11; every value known at the Friday close, percentile ranks on a TRAILING 252-day window only)
P1 VIX level percentile · P2 VIX 5-day change · P3 term structure VIX ÷ VIX3M · P4 VVIX percentile ·
P5 VIX9D ÷ VIX · P6 HYG minus IEF 20-day return · P7 share of U2 above the 50-day average · P8 net 52-week
highs minus lows (share of U2) · P9 SPY 5-day return · P10 SPY minus TLT 20-day return ·
P11 fear-and-greed proxy = mean trailing percentile of {SPY vs its 125-day average, P8, P7, P6, −P1, P10}
(CNN's recipe without the put/call leg). P12 = CNN's actual index, only if coverage ≥ 80 %.

## Targets (non-overlapping weeks: Monday open → next Monday open, the sleeve's own holding week)
T1 SPY return · T2 the sleeve's return · T3 sleeve minus SPY. Windows: 2017-01..2026-09 (all predictors, ~505 weeks);
1993..2026 for T1 with P1, P2, P9 only (~1,700 weeks). Halves 2017–2021 / 2022–2026 (long window: 1993–2009 / 2010–2026).

## Reads per predictor × target
Quintile table (mean next-week return, share of up weeks, n), top-minus-bottom spread with t, Spearman rank
correlation with t, both halves, the spread ex the top and bottom 5 % of weeks. Base rate first: share of up weeks
unconditionally (the "always up" forecast). MDE stated beside every spread (2.8 × SE).

## Joint model (the "direction" question asked directly)
Logistic regression of the sign of T1 (and of T2) on P1–P11, standardised, walk-forward: fit on all weeks through
year Y−1 (first fit on 2017–2019), predict year Y; out-of-sample 2020–2026. Reads: OOS hit rate vs the always-up
base rate (binomial z), OOS mean return on predicted-up vs predicted-down weeks with t, calibration by decile.

## Pass rule (36 tests + 2 models; Bonferroni)
A predictor is "real" only if |t| of the quintile spread ≥ 3.2 AND the sign agrees in both halves AND the spread keeps
its sign ex-5 % tails AND the quintile means are monotone (|Spearman of the five means| ≥ 0.8). The joint model is
"real" only if the OOS hit rate beats always-up by ≥ 4 points with z ≥ 2 AND predicted-down weeks have a negative
mean. ONLY for what passes: the tradable test on the sleeve — cash (and half size) during the flagged state, threshold
on an expanding window, costs as 1,700c — read as CAGR / max DD / end $ vs REF; recommended only if both improve and
its 1,700o-style neighbours (threshold ± 2 steps) agree in ≥ 75 % of cells. Nothing outside this list after numbers.

## Output
`1700q_fear.py`, `1700q_cells.csv`, `1700q_series.parquet` (the predictor panel, gitignored), `RESULT_1700q.md`
(≤ 80 lines: base rate, the 36-row table, the model, the verdict). ONE process through
`bash scripts/research_run.sh -m 2500M`; load the panel with a column/symbol filter, never whole. Agent returns ≤ 150 words.
