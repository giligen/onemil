# PREREG — Sleeve #3: low-volatility anomaly (Baker–Bradley–Wurgler 2011 / Blitz–van Vliet 2007), frozen 2026-10-29 (Fable)

Sleeves #1 (52WH, 1,710) and #2 (ETF TSM, 1,711) rejected on frozen bars. #3 by the survey rule. Cell 1,712.
Stage 1 is rejection-only on the survivorship-exposed Alpaca panel `research/momentum_weekly/panel_2016_2026.parquet`
(same loader and name-pattern exclusions as `fiftytwo_wh.py`); a pass here is evidence for a PIT run, never a pass.

## Rule (`research/sleeves/lowvol.py`)
* Universe each rebalance: names in the panel with ≥ 252 sessions of history, price ≥ $5, 60-day median dollar volume
  ≥ $5M, name-pattern exclusions (wrappers, test tickers, preferreds/units/warrants), top 1,000 by dollar volume.
* Signal on the last session of each calendar QUARTER at the close: realised volatility = std of daily log returns over
  the prior 252 sessions. Long the 30 LOWEST-vol names, equal weight, hold 3 months, full rebalance quarterly. Orders
  at the next session's official open; cost 15 bps round trip (sensitivity 30); turnover is low (report it).
* Sensitivity (reported, not the sleeve): 60-session vol; monthly rebalance; top 50.
* Trigger day: first session of Jan/Apr/Jul/Oct — distinct from MOM (Monday) and TOM (month-end); report overlap/correlation
  with MOM, TOM, TSM (monthly_1711.csv) and SPY.

## Evaluation
TRAIN 2017-01..2019-12, VAL 2020-01..2022-12 (the quarter whose exit open falls in 2023 is dropped, as in 1,711);
TEST 2023+ SEALED. Benchmarks: equal-weight top-1,000 universe and SPY. Per split: mean monthly return, excess vs the
universe, Sharpe, max DD, ex-top-5 % months, green-month share vs a count-matched random-30 null (1,000 draws),
year-block bootstrap CI, beta to SPY (the published anomaly is a risk-adjusted one: report CAPM alpha too).
Pass on VAL (frozen): CAPM alpha vs SPY > 0 with CI > 0 AND max DD ≤ 0.75 × SPY's DD in the window AND ≥ 50 % green
months AND mean return ≥ universe mean − 0.2 %/mo (we do not pay for alpha with a lower mean).

## Ladder
Stage-1 pass → PIT data question (Databento US-equity daily from 2018-05: TRAIN 2019-05..2020-12 would be too short —
if Stage 1 passes, the PIT rerun uses 2019-05..2022-12 as ONE split and the sealed 2023+ as TEST; owner GO on the spend
first) → rebuild → paper $10K → live $5K after 2 cycles realized > 0. Kill: 4 quarterly cycles realized < 0, or one
cycle < −5 %. Nothing to the owner before the rebuild.
