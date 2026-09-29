# ORB-candidate x HOD-break outcome diagnostic

**Question**: are ORB candidates later in the day good HOD-break trades? Does the ORB signal
(candidate / selected / winner-loser) predict the HOD-break outcome on the same symbol-day?

**Base book**: `research/hod_entry/model_1478_L3_predictions.csv` (9,911 fills, day/symbol/fill_min/split/outcome_R).
**ORB side**: `analysis_results/orb_features_20260925_2054.csv` — the cumulative rebuild (most recent timestamp,
carries the `entered`/`win` columns), header self-describes as "ORB_5_vanilla, broad universe" backtest population,
2025-01-02..2026-09-25, 434 unique days. This is NOT confirmed to be the exact live B+ selection-chain config;
treat as the best-available ORB candidate/entered/outcome proxy in the repo. No ORB CSV found anywhere under
`research/orb_*` or `analysis_results/` carries an entry- or exit-*time-of-day* field (checked
`orb_latency_bt/{population,results,availability}.csv`, `orb_static_lock_trades.csv`, `orb_bplus_book.csv`,
`orb_monthly_static_lock.csv`) — only a `date` column. **The causal ordering check (c) — ORB exit before the HOD
fill minute — is NOT COMPUTABLE from available files.**

## Coverage

- HOD window: 2025-07-01 .. 2026-05-29 (230 unique days), 9,911 fills.
- ORB file window: 2025-01-02 .. 2026-09-25 (434 unique days).
- Overlap: all 230 HOD days fall inside the ORB file's date range → **9,911 / 9,911 (100%) of HOD fills are on a
  day with ORB feature coverage.** (Coverage is by day only; symbol-day match is the join key below.)

## (a) HOD fills on an ORB-candidate symbol-day vs not

| split | group | n | mean R | day-clust t | ex-top5% |
|---|---|---:|---:|---:|---:|
| TRAIN | IS candidate | 97 | -0.150 | -1.01 | -0.265 |
| TRAIN | NOT candidate | 4,301 | -0.167 | -4.22 | -0.280 |
| VAL | IS candidate | 136 | **-0.036** | -0.30 | -0.145 |
| VAL | NOT candidate | 5,377 | -0.174 | -4.40 | -0.287 |

Paired diff (candidate − not): TRAIN +0.017R, VAL **+0.138R**, both directions less-negative on ORB-candidate
days, in both splits.

## (b) HOD fills on an ORB entered/selected symbol-day vs not

| split | group | n | mean R | day-clust t | ex-top5% |
|---|---|---:|---:|---:|---:|
| TRAIN | ORB entered | 91 | -0.169 | -1.06 | -0.293 |
| TRAIN | not entered | 4,307 | -0.167 | -4.21 | -0.279 |
| VAL | ORB entered | 125 | -0.085 | -0.69 | -0.189 |
| VAL | not entered | 5,388 | -0.173 | -4.38 | -0.285 |

Paired diff (entered − not): TRAIN -0.002R (flat), VAL +0.088R. Weaker than (a); the plain "was there an ORB
candidate that day" split carries more of the effect than the selection-chain filter does.

## (c) HOD fills on an ORB winner vs loser symbol-day (entered names only) — causality NOT verified

| split | group | n | mean R | day-clust t | ex-top5% |
|---|---|---:|---:|---:|---:|
| TRAIN | ORB winner same-day | 61 | +0.192 | 0.98 | +0.100 |
| TRAIN | ORB loser same-day | 30 | **-0.903** | **-6.95** | -1.079 |
| VAL | ORB winner same-day | 87 | +0.112 | 0.80 | +0.023 |
| VAL | ORB loser same-day | 38 | **-0.537** | **-2.71** | -0.677 |

n = 216 HOD fills sit on an ORB-entered symbol-day (61+30+87+38); for every one of them we could not establish
whether the ORB trade's exit preceded the HOD fill's `fill_min`, because no ORB CSV in the repo carries an
intraday timestamp. `fill_min` itself spans 09:36–14:01 ET (median ~10:30 ET) in the base book, i.e. many HOD
fills happen early enough that an ORB result plausibly is NOT yet known — so treating (c) as a same-day-known
predictor would risk exactly the look-ahead failure mode flagged in `feedback_population_filter_look_ahead.md`.

## Bottom line

(a)/(b): a same-day ORB candidate/entered flag is associated with a smaller (less negative) HOD outcome_R, most
visible in VAL under (a) (t = -0.30 vs -4.40, i.e. indistinguishable from zero vs clearly negative) — directionally
consistent TRAIN and VAL, but neither group is *positive* with a clean t-stat; this is "less bad," not an edge.

(c): the winner/loser split is the largest effect (loser same-day t -6.95 TRAIN / -2.71 VAL) but it is NOT usable
as a forward signal without exit-time data, because we cannot confirm the ORB outcome was known before the HOD
fill — this is the highest-priority follow-up (get ORB entry/exit minute timestamps, e.g. from
`research/orb_latency_bt` chase-rule replay logs or the trades DB, then redo (c) causally).
