# PREREG — HOD-break winner-likelihood score: ONE pre-registered multivariate model, decile buckets. Cell 1,355 (+ placebo)

Owner 2026-09-21: "buckets good enough statistically that we can change the size or pass". Univariate cells
(12 causal filters, 4 rank cells) found nothing; each univariate cut also costs a multiplicity slot. This pass asks
the question once, multivariately, with the answer frozen before any holdout is scored.

## Population and outcome
`features.csv` rows (the causal HOD-break population; the same `meas, obtainable` R used by `cells.py`).
Outcome for fitting: win = R > 0. Outcome for judging: mean net R per bucket.

## Features (fixed list; nothing added after seeing results)
dist_open_pct, rv_adv, bar_vol_x, above_vwap, gap_pct, prev_range_pct, adv20, dist_20d_high_pct, spy_5m_ret,
spy_range3, pm_covered, pole_gain, retrace, flag_len, drive_min, pull_len, n_prior, ck_min, rv_clock, is_wrapper,
entry_m (minute of day), rank_sig (from `rank_cells2.py`), and news_before_signal = 1 if `cache.db::news_cache`
or `news_history` has an article for the symbol with a timestamp between 16:00 ET the prior day and the signal
minute (causal; 0 if the tables do not cover the day — report coverage; if coverage < 80 % the feature is
dropped and said so). No price/outcome-derived fields, no date fields.

## Model (fixed)
L2-regularized logistic regression (C = 1.0) on standardized features, fit ONCE on TRAIN-H1 (2025-01..06).
Score = predicted P(win). Buckets = deciles of score, with decile edges computed on TRAIN-H1 and applied unchanged
to TRAIN-H2 (2025-07..12) and VAL (2026-01..05). Placebo: the same fit with day-shuffled outcomes, reported beside
it (its top decile must NOT be positive, else the lift is leakage).

## Report (`SCORE_REPORT.md`)
Per split: decile table (n, mean net R, iid t, day-clustered t, win rate), Spearman of decile vs mean R, top-2-
decile fills/week at the live config, bottom-3-decile mean R, coefficient table with signs, AUC on each split, the
placebo's decile table, and the cadence block for the top-2-decile book on VAL.

## Pass bar (all)
1. Top-2-decile mean net R > 0 with day-clustered t ≥ 2 on TRAIN-H2 AND on VAL;  2. deciles monotone
(Spearman ≥ 0.6) on both holdouts;  3. bottom-3-decile mean R < 0 on both;  4. top-2-decile ≥ 3 fills/week;
5. placebo top decile ≤ 0 on both holdouts;  6. coefficient signs of the three largest |β| agree with a stated
mechanism (the report must name it).
Pass → independent rebuild (Haiku, from this prose) → live PREREG: the dry run gates on score ≥ the decile-8 edge
and reports both cohorts; sizing by bucket only after 40 dry fills in the kept cohort.
Fail → the report states the lift the test could have detected (top-2-decile SE) and the HOD loop moves to the
failed-break short. Programme count 1,356 (cell + placebo).
