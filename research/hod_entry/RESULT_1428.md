# RESULT — cell 1,428: gapper-universe causal arming (LADDER.md row 1,428)

Population: 7487 symbol-days requested, 0 LOST (0.0 %), 3 unsimulable (< K+2 minute bars).

| holdout | fills (rate) | mean net R | stop-slip R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | median spread bps | R % of price | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 343 (66.1 %) | -0.067 | -0.166 | -1.11 | -0.180 | 11.4 | 100.0 % / 0.0 pp | 15 | 2.02 % | report-only (TRAIN-H2) |
| VAL | 444 (73.8 %) | +0.109 | +0.017 | 0.49 | +0.008 | 16.8 | 100.0 % / 0.0 pp | 17 | 1.88 % | FAIL: mean >= +0.15, t >= 2 |

* VAL verdict: FAIL: mean >= +0.15, t >= 2.
* Cost: measured half-spread at fill + B0 exit leg (causal_arming); nbbo.csv is the $20+ book's spread table and rarely covers this pool, so exit cost falls back to the fill-instant half-spread for most fills.
* Spread bps is derived from cost_R * R (approx, assumes entry and exit half-spreads are close) — not an independent NBBO measurement for this pool; read as indicative.
* Small caps: R must exceed the spread — see the R %-of-price and spread-bps columns above per CLAUDE.md.
* Population: point-in-time Databento EQUS.SUMMARY daily bars, gap >= 5 %, open $3-30, prior-day volume >= 500K, 2025-07-01..2026-05-31, test tickers excluded; TEST not read.