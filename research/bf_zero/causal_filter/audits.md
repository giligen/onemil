# Causal-filter — standing audits

## A5 cohort assertion

`selectfeat.assert_no_cohort` passes: `cohort` (end-of-day information) is not among the scored features, and appears only as the diagnostic `cache %` column of the anatomy tables. The label is each signal's own R under the spec exits, never the cohort.

## A1 availability — coverage % per split, then per time band

| split   |   gap_pct |   prev_range_pct |   dist_20d_high_pct |   bar_vol_x |   above_vwap |   spy_5m_ret |   spy_range3 |   dist_open_pct |   rv_clock |   rv_profile |   drive_min |   n_prior |   is_wrapper |   coh_by_t |   entry_m |   price |   has_news |
|:--------|----------:|-----------------:|--------------------:|------------:|-------------:|-------------:|-------------:|----------------:|-----------:|-------------:|------------:|----------:|-------------:|-----------:|----------:|--------:|-----------:|
| TEST    |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       99.7 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| TRAIN   |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       87.9 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| VAL     |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       99.6 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |

| hb          |   gap_pct |   prev_range_pct |   dist_20d_high_pct |   bar_vol_x |   above_vwap |   spy_5m_ret |   spy_range3 |   dist_open_pct |   rv_clock |   rv_profile |   drive_min |   n_prior |   is_wrapper |   coh_by_t |   entry_m |   price |   has_news |
|:------------|----------:|-----------------:|--------------------:|------------:|-------------:|-------------:|-------------:|----------------:|-----------:|-------------:|------------:|----------:|-------------:|-----------:|----------:|--------:|-----------:|
| 09:30-09:45 |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       99.1 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| 09:45-10:00 |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       98.7 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| 10:00-11:00 |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       94.5 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| 11:00-13:00 |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       97.4 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |
| 13:00+      |       100 |              100 |                 100 |         100 |          100 |          100 |          100 |             100 |       72.3 |          100 |         100 |       100 |          100 |        100 |       100 |     100 |        100 |

### missingness bias — mean R of covered vs uncovered rows (TRAIN)

| feature           |   cov_pct |   meanR_covered |   meanR_missing |   n_missing |
|:------------------|----------:|----------------:|----------------:|------------:|
| gap_pct           |     100   |           0     |         nan     |           0 |
| prev_range_pct    |     100   |           0     |         nan     |           0 |
| dist_20d_high_pct |     100   |           0     |         nan     |           0 |
| bar_vol_x         |     100   |           0     |         nan     |           0 |
| above_vwap        |     100   |           0     |         nan     |           0 |
| spy_5m_ret        |     100   |           0     |         nan     |           0 |
| spy_range3        |     100   |           0     |         nan     |           0 |
| dist_open_pct     |     100   |           0     |         nan     |           0 |
| rv_clock          |      87.9 |          -0.004 |           0.031 |         891 |
| rv_profile        |     100   |           0     |         nan     |           0 |
| drive_min         |     100   |           0     |         nan     |           0 |
| n_prior           |     100   |           0     |         nan     |           0 |
| is_wrapper        |     100   |           0     |         nan     |           0 |
| coh_by_t          |     100   |           0     |         nan     |           0 |
| entry_m           |     100   |           0     |         nan     |           0 |
| price             |     100   |           0     |         nan     |           0 |
| has_news          |     100   |           0     |         nan     |           0 |

## A2 causality trace

| feature           | computed_from                                                                                                                                                      |
|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gap_pct           | open of the signal day (09:30 bar) / prev daily close — both known at 09:31                                                                                        |
| prev_range_pct    | previous daily bar high/low/close — known before the open                                                                                                          |
| dist_20d_high_pct | entry level / rolling 20d high SHIFTED by one day — prior sessions only                                                                                            |
| bar_vol_x         | volume of the signal bar / mean volume of bars strictly before it                                                                                                  |
| above_vwap        | entry level vs the VWAP through bar i-1 (strictly before the fill bar)                                                                                             |
| spy_5m_ret        | SPY close at the signal minute vs 5 minutes earlier                                                                                                                |
| spy_range3        | SPY daily range, 3-day mean, SHIFTED one day                                                                                                                       |
| dist_open_pct     | entry level / the 09:30 open — the spec s own causal floor                                                                                                         |
| rv_clock          | cumulative volume to the signal bar / the same-clock mean over the prior 20 HELD days (shift(1) then rolling) — no same-day information                            |
| rv_profile        | cumulative volume to the signal bar / (ADV20 x the market-wide clock fraction) — ADV20 is prior sessions, the fraction is a fixed constant in trading/hod_break.py |
| drive_min         | minute the day first traded 5% above its open, strictly before the signal bar                                                                                      |
| n_prior           | count of prior held days in the volume profile — prior sessions only                                                                                               |
| is_wrapper        | static asset-class map (instrument type, not a price)                                                                                                              |
| coh_by_t          | count of same-anchor siblings that signalled STRICTLY EARLIER the same day                                                                                         |
| entry_m           | the signal minute itself                                                                                                                                           |
| price             | the capped fill price, known at the fill                                                                                                                           |
| has_news          | articles published before 09:30 ET on the signal day                                                                                                               |

## A3 price-scale check (200 sampled keys)

daily-file close vs the last RTH 1-min close on the SAME symbol-day: n 200, median |diff| 0.065%, p95 0.424%, share > 1% 1.0%, share > 5% (a split) 0.0%.

## A4 obtainability (capped limit = level x 1.006)

NBBO quoted at the fill instant for 99.9% of signals; of those, **15.3% had an ask ABOVE the cap** and would NOT have filled. Those rows are removed before the book in the `measured` arm.

| split   |    n |   no_fill_pct |   meanR_fillable |   meanR_nofill |
|:--------|-----:|--------------:|-----------------:|---------------:|
| TEST    | 3518 |          15   |           -0.07  |         -0.081 |
| TRAIN   | 7380 |          15.7 |            0.013 |         -0.072 |
| VAL     | 4741 |          15   |           -0.04  |         -0.072 |
