# RESULT 1,700u -- term-structure gate on the GUARDED sleeve (PREREG_1700u.md, FROZEN)

GREF reproduced BEFORE any cell: CAGR 29.34% / max DD -38.3% / end $596,394 (target 29.34 / -38.3 / 596,394). Guard ON in all cells; 2017-01..2026-09, $50K; gate at Friday close, Monday open; cash 0; costs on traded notional. gated week: book sold at Monday open (engine cost), cash 0, re-bought at next ungated Monday open; half = 1/(2N) per name.
Shift test (data deleted after a Friday, 10 Fridays x 4 gauges x 5 thr): 0 mismatches. Halves ratio GREF 0.67 / 0.99. GREF 3 deepest episodes: 2021-02-16..2021-05-11 -38.3%; 2020-02-14..2020-03-19 -37.1%; 2025-02-13..2025-04-07 -33.6%.
| cell | CAGR | DD | ratio | end $K | gated | sp | GREF wk mean gated/ungated H1 | H2 (%) | I | R | G<U |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 252|p10|cash | 32.4% | -37.1% | 0.87 | 747 | 17% | 42 | -0.77/+0.82 | +0.21/+0.78 | . | Y | Y |
| 252|p10|half | 31.2% | -37.1% | 0.84 | 684 | 17% | 42 | -0.77/+0.82 | +0.21/+0.78 | . | Y | Y |
| 252|p15|cash | 31.6% | -37.1% | 0.85 | 704 | 22% | 49 | -0.51/+0.86 | +0.23/+0.80 | . | Y | Y |
| 252|p15|half | 30.8% | -37.1% | 0.83 | 667 | 22% | 49 | -0.51/+0.86 | +0.23/+0.80 | . | Y | Y |
| 252|p20|cash | 31.0% | -37.1% | 0.84 | 675 | 28% | 54 | -0.57/+1.01 | +0.45/+0.76 | . | . | Y |
| 252|p20|half | 30.7% | -37.1% | 0.83 | 659 | 28% | 54 | -0.57/+1.01 | +0.45/+0.76 | . | Y | Y |
| 252|p25|cash | 28.7% | -37.1% | 0.77 | 569 | 33% | 62 | -0.38/+1.01 | +0.49/+0.76 | . | . | Y |
| 252|p25|half | 29.6% | -37.1% | 0.80 | 608 | 33% | 62 | -0.38/+1.01 | +0.49/+0.76 | . | . | Y |
| 252|p30|cash | 30.4% | -37.1% | 0.82 | 646 | 37% | 71 | -0.25/+1.04 | +0.20/+0.95 | . | Y | Y |
| 252|p30|half | 30.5% | -37.1% | 0.82 | 652 | 37% | 71 | -0.25/+1.04 | +0.20/+0.95 | . | Y | Y |
| 504|p10|cash | 33.0% | -37.1% | 0.89 | 782 | 15% | 37 | -0.57/+0.81 | +0.03/+0.77 | . | Y | Y |
| 504|p10|half | 31.5% | -37.1% | 0.85 | 700 | 15% | 37 | -0.57/+0.81 | +0.03/+0.77 | . | Y | Y |
| 504|p15|cash | 29.7% | -37.1% | 0.80 | 613 | 22% | 45 | -0.53/+0.88 | +0.60/+0.69 | . | . | Y |
| 504|p15|half | 29.9% | -37.1% | 0.81 | 624 | 22% | 45 | -0.53/+0.88 | +0.60/+0.69 | . | Y | Y |
| 504|p20|cash | 32.0% | -39.5% | 0.81 | 729 | 27% | 53 | -0.45/+0.94 | +0.23/+0.82 | . | Y | Y |
| 504|p20|half | 31.2% | -37.2% | 0.84 | 686 | 27% | 53 | -0.45/+0.94 | +0.23/+0.82 | . | Y | Y |
| 504|p25|cash | 27.1% | -37.7% | 0.72 | 506 | 33% | 63 | -0.18/+0.93 | +0.44/+0.79 | . | . | Y |
| 504|p25|half | 28.9% | -37.1% | 0.78 | 576 | 33% | 63 | -0.18/+0.93 | +0.44/+0.79 | . | . | Y |
| 504|p30|cash | 25.3% | -40.9% | 0.62 | 439 | 38% | 66 | +0.09/+0.85 | +0.30/+0.89 | . | . | Y |
| 504|p30|half | 28.0% | -37.1% | 0.75 | 539 | 38% | 66 | +0.09/+0.85 | +0.30/+0.89 | . | Y | Y |

PASS COUNTS: improve (CAGR >= GREF and DD >= 3 pts better) 0/20 (need 15); ratio beats GREF both halves 13/20 (15); gated-wk mean < ungated both halves 20/20 (15); median-cell episodes >= 3 pts shallower 2/3 (2); top-year share 17% in 2021 (<= 40%: ok); SPY 2008-2016 gated < ungated 10/10 (8).
MEDIAN cell (11th of 20 by ratio) VIXratio|w252|p30|cash: CAGR 30.41% / DD -37.1% / end $646,275 / gated 37% of weeks in 71 spells; ratio 0.82 vs GREF 0.77; episode depths (GREF -> cell): -38.3% -> -25.7%; -37.1% -> -37.1%; -33.6% -> -26.1%; paired weekly diff -0.021% (t -0.2, ex-top5% -0.419%).
Median cell by-year (gated share of the year | share of all gated weeks): 2017: 40%|10% 2018: 9%|3% 2019: 48%|13% 2020: 27%|7% 2021: 62%|17% 2022: 15%|4% 2023: 60%|16% 2024: 23%|6% 2025: 48%|13% 2026: 44%|9%
SPY 2008-2016 (weekly Mon open -> next Mon open, VIX/VIX3M percentile, same definition, min 126 obs; NOTE the CBOE VIX3M_History.csv fetched here starts 2009-09-18 (not 2007-12) so the first valid week is 2010-03-22 and n_valid is 354 of 469, i.e. the 2008-09 crisis is NOT in the read; gated vs ungated mean %, t, n gated/valid, first valid week):
  w252 p10: +0.05 vs +0.30 t -1.2, n 74/354, from 2010-03-22 -> below
  w252 p15: +0.05 vs +0.33 t -1.3, n 98/354, from 2010-03-22 -> below
  w252 p20: +0.04 vs +0.35 t -1.5, n 117/354, from 2010-03-22 -> below
  w252 p25: +0.11 vs +0.33 t -1.0, n 128/354, from 2010-03-22 -> below
  w252 p30: +0.13 vs +0.34 t -1.0, n 147/354, from 2010-03-22 -> below
  w504 p10: -0.08 vs +0.35 t -1.8, n 80/354, from 2010-03-22 -> below
  w504 p15: -0.01 vs +0.35 t -1.6, n 104/354, from 2010-03-22 -> below
  w504 p20: +0.10 vs +0.32 t -1.0, n 118/354, from 2010-03-22 -> below
  w504 p25: +0.14 vs +0.31 t -0.8, n 133/354, from 2010-03-22 -> below
  w504 p30: +0.16 vs +0.32 t -0.8, n 149/354, from 2010-03-22 -> below
VERDICT: CLOSED.
Adversary caveats: (1) 20 neighbours of one lead, not independent; family count +20. (2) Costs are the engine band model, not NBBO; each gated week sells and re-buys the book. (3) Gated-week means are GREF counterfactual weekly returns, whole-sample selection. (4) SPY 2008-2016 is a buy-and-hold proxy, not the sleeve, and is dominated by the 2008-09 crisis (few gated weeks at low thresholds: check n). (5) The median-cell pick is by ratio rank, a few spells decide each cell. (6) Gate with half-size scales book to 1/(2N) but residual cash earns 0.
