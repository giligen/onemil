# PREREG — cell 1,700g: beat SPY in MOST years — risk-adjusted and volatility-scaled momentum (FROZEN 2026-10-02 12:05 UTC)

Owner 10/2: "I do want to beat SPY on most of the years." The objective is the year-by-year hit rate, not the CAGR.
Reference: our A1 (large caps, top 20, 12-1, weekly) beats SPY in 6 of 10 years 2017–2026 (loses 2017, 2018, 2021,
2023); the MTUM index fund (125 names, risk-adjusted momentum, semi-annual) also 6 of 10. Both fail "most years".

## Variants (fixed; the literature's constructions, nothing tuned)
Universe U2 (prior close ≥ $10, ADV20 ≥ $200M, point-in-time, delisted included), weekly Monday rebalance, equal
weight, costs as 1,700c (5 bps + half spread proxy), no leverage (exposure ≤ 100 %, the rest cash at 0 %):
* V1 A1 reference (top 20 by 12-1 return).
* V2 risk-adjusted ranking: top 20 by (12-1 return) ÷ (252-day daily-return volatility) — MSCI's construction.
* V3 volatility-scaled exposure: V1's book with exposure = min(1, 20 % ÷ trailing 126-day realised vol of the book's
  own daily return) — Barroso & Santa-Clara's "momentum has its moments" scaling, long-only cap at 100 %.
* V4 V2 + V3.
* V5 V2 with 50 names.
* V6 intermediate momentum: top 20 by the return from t−252 to t−126 (the 12-7 construction).
Also each variant on N = 50 where not already (V1, V3, V6) → 9 cells.

## Reads (whole 2017-01..2026-09 compounding from $50K, H1 2017–2021, H2 2022–2026)
By year: book %, SPY %, book $, SPY $; years beating SPY (of 10); both-halves excess; alpha t; Sharpe; max DD (book,
SPY); worst year; weeks in cash; turnover; the count-matched null percentile of the hit rate (1,000 random same-N
books: how often ≥ k of 10 years beat SPY by chance).

## Pass bar
Beats SPY in ≥ 7 of 10 calendar years AND excess > 0 in both halves AND max DD ≤ 1.25 × SPY's AND the hit-rate null
percentile ≥ 95 %. A pass → independent rebuild → paper sleeve at $20K. A fail reports the full by-year table; no
parameter (vol target, window, N) is tuned after seeing numbers. Multiplicity: 9 cells, stated.

## Output
`1700g_vol.py` (reuse 1700d/1700e machinery), `1700g_cells.csv`, `1700g_by_year.csv`, `RESULT_1700g.md` (≤ 80 lines:
the by-year tables with $ first, then the 9-cell summary with the pass flags). The agent returns ≤ 150 words.
