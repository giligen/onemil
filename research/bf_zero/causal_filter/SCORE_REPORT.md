# SCORE_REPORT — HOD-break winner-likelihood score (cell 1,355 + placebo)

Implements `PREREG_SCORE.md` exactly. Script: `score_model.py`. Population: `features.csv` rows,
split TRAIN/VAL only (TEST never read — this study does not ask for it). R = the `net_meas`, obtainable
arm used by `cells.py`, after `trading.hod_break.run_book` (the ONE book rule). Model fit ONCE on
TRAIN-H1 (2025-01..06); decile edges cut on TRAIN-H1 scores, applied unchanged to TRAIN-H2 (2025-07..12)
and VAL (2026-01..05).

## Feature reconciliation (decided before any fitting, per PREREG's fixed list)
| PREREG name | Status | Resolution |
|---|---|---|
| dist_open_pct, bar_vol_x, above_vwap, gap_pct, prev_range_pct, adv20, dist_20d_high_pct, spy_5m_ret, spy_range3, drive_min, n_prior, rv_clock, is_wrapper, entry_m | in features.csv | used as-is |
| rv_adv | not a features.csv column | mapped to `rv_profile` (the only `rv_*` field the causal-filter build produced; also the feature named in `selection.json`'s 5 survivors) |
| pm_covered, pole_gain, retrace, flag_len, pull_len, ck_min | not in `build_features.py` output at all (grepped, zero hits) | **dropped** — these are bull-flag pattern fields, not computable for the causal HOD-break population; not silently substituted |
| rank_sig | computed | reused `rank_cells2.compute_rank_cand` verbatim (rank by `dist_open_pct` desc among same-day signals with `entry_m <=` the signal's, min-rank ties) |
| news_before_signal | computed | see below |

Final feature vector (17): dist_open_pct, rv_profile, bar_vol_x, above_vwap, gap_pct, prev_range_pct,
adv20, dist_20d_high_pct, spy_5m_ret, spy_range3, drive_min, n_prior, rv_clock, is_wrapper, entry_m,
rank_sig, news_before_signal.

## news_before_signal
`news_cache` has date-only granularity (no publication time) — per the pre-decided caveat,
`news_before_signal = 1` iff the symbol has a `news_cache` or `news_history` row dated the prior
trading day or the day before that (same-day news excluded as unknowable at the signal minute).
**Coverage 100.0%** (every signal's 2-day lookback window falls inside the combined date span of
`news_cache` [2025-12-26, 2026-09-21] ∪ `news_history` [2025-01-02, 2026-05-19]) — feature kept (≥80% rule).
**Prevalence 2.2%** of rows have news in the lookback window.

## Rows and NaN-feature drop (book-selected population)
| split | book n | NaN-feature drop | kept |
|---|---|---|---|
| TRAIN-H1 | 1017 | 86 (8.5%) | 931 (322 wins) |
| TRAIN-H2 | 1201 | 7 (0.6%) | 1194 |
| VAL | 1028 | 3 (0.3%) | 1025 |

## Decile tables — real model (score = P(win), edges cut on TRAIN-H1)
**TRAIN-H1** (fit split, shown for reference only — not a holdout)
| decile | n | meanR | t_iid | t_clust | WR% | fills/wk |
|---|---|---|---|---|---|---|
| 1 | 94 | -0.417 | -3.02 | -3.19 | 27.7 | 3.62 |
| 2 | 93 | -0.394 | -2.87 | -2.79 | 30.1 | 3.58 |
| 3 | 93 | -0.225 | -1.53 | -1.57 | 33.3 | 3.58 |
| 4 | 93 | -0.408 | -2.95 | -2.92 | 31.2 | 3.58 |
| 5 | 93 | -0.215 | -1.57 | -1.59 | 33.3 | 3.58 |
| 6 | 93 | -0.212 | -1.42 | -1.45 | 33.3 | 3.58 |
| 7 | 93 | -0.177 | -1.17 | -1.29 | 35.5 | 3.58 |
| 8 | 93 | -0.082 | -0.54 | -0.50 | 39.8 | 3.58 |
| 9 | 93 | -0.316 | -2.21 | -2.28 | 31.2 | 3.58 |
| 10 | 93 | 0.233 | 1.42 | 1.46 | 50.5 | 3.58 |

Spearman(decile, meanR) = 0.77, AUC = 0.558.

**TRAIN-H2** (holdout 1)
| decile | n | meanR | t_iid | t_clust | WR% | fills/wk |
|---|---|---|---|---|---|---|
| 1 | 5 | 0.281 | 0.47 | 0.52 | 40.0 | 0.19 |
| 2 | 27 | -0.097 | -0.34 | -0.32 | 40.7 | 1.00 |
| 3 | 53 | -0.298 | -1.52 | -1.50 | 35.8 | 1.96 |
| 4 | 70 | -0.547 | -3.51 | -3.85 | 22.9 | 2.59 |
| 5 | 87 | -0.161 | -1.05 | -1.08 | 35.6 | 3.22 |
| 6 | 153 | -0.057 | -0.49 | -0.47 | 39.9 | 5.67 |
| 7 | 152 | -0.278 | -2.57 | -2.48 | 34.9 | 5.63 |
| 8 | 192 | -0.158 | -1.48 | -1.48 | 37.0 | 7.11 |
| 9 | 230 | -0.232 | -2.41 | -2.15 | 34.3 | 8.52 |
| 10 | 225 | -0.252 | -2.59 | -2.25 | 36.4 | 8.33 |

Spearman(decile, meanR) = **-0.248**, AUC = 0.506.

**VAL** (holdout 2)
| decile | n | meanR | t_iid | t_clust | WR% | fills/wk |
|---|---|---|---|---|---|---|
| 1 | 44 | -0.480 | -2.32 | -3.63 | 29.5 | 1.91 |
| 2 | 11 | -0.108 | -0.24 | -0.23 | 36.4 | 0.48 |
| 3 | 40 | -0.235 | -1.08 | -0.99 | 35.0 | 1.74 |
| 4 | 71 | -0.442 | -2.78 | -2.85 | 29.6 | 3.09 |
| 5 | 78 | -0.006 | -0.04 | -0.04 | 42.3 | 3.39 |
| 6 | 115 | 0.153 | 1.13 | 1.22 | 48.7 | 5.00 |
| 7 | 104 | -0.354 | -2.63 | -2.36 | 33.7 | 4.52 |
| 8 | 132 | -0.293 | -2.21 | -2.30 | 34.8 | 5.74 |
| 9 | 155 | -0.086 | -0.75 | -0.75 | 40.0 | 6.74 |
| 10 | 275 | -0.161 | -1.82 | -1.51 | 41.1 | 11.96 |

Spearman(decile, meanR) = **0.309**, AUC = 0.520.

## Top-2-decile (9+10) and bottom-3-decile (1-3) summary
| split | top-2 n | top-2 meanR | top-2 SE | top-2 t_clust | top-2 fills/wk | bottom-3 meanR |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 455 | **-0.242** | 0.068 | -3.01 | 16.85 | -0.200 |
| VAL | 430 | **-0.134** | 0.070 | -1.55 | 18.70 | **-0.334** |

(top-2 t_clust computed from the pooled decile-9+10 rows, day-clustered.)

## Coefficient table (standardized features; sorted by \|coef\|)
| feature | coef |
|---|---|
| above_vwap | -0.180 |
| rank_sig | +0.175 |
| spy_range3 | -0.162 |
| rv_clock | +0.162 |
| n_prior | +0.125 |
| prev_range_pct | +0.122 |
| spy_5m_ret | +0.071 |
| gap_pct | -0.065 |
| entry_m | -0.058 |
| drive_min | +0.052 |
| bar_vol_x | +0.048 |
| dist_open_pct | +0.037 |
| rv_profile | -0.033 |
| is_wrapper | +0.030 |
| adv20 | -0.026 |
| news_before_signal | -0.014 |
| dist_20d_high_pct | +0.010 |

**Stated mechanism for the three largest \|β\|:** `above_vwap` (-0.18) — a HOD-break already above VWAP
at signal time has less room left to run before hitting exhausted/faded buyers, so the model reads it as
*late*, not *strong* (a late-break penalty). `rank_sig` (+0.175, higher rank number = weaker mover) —
positive sign means WEAKER-momentum signals score higher win probability, the opposite of a "ride the
leader" mechanism; this contradicts the plain reading of rank_sig as an attention-rank filter and is not
cleanly explainable by a momentum mechanism. `spy_range3` (-0.162) — a choppier 3-bar SPY range lowers
win probability, consistent with a "calm tape helps single-stock breakouts hold" mechanism. Two of the
three signs are plausible; `rank_sig`'s sign runs counter to the mechanism it was designed to test —
criterion 6 is **not cleanly met**.

## AUC by split
TRAIN-H1 (fit) 0.558, TRAIN-H2 0.506, VAL 0.520 — barely above chance on both holdouts.

## Placebo (day-shuffled outcomes, fit on TRAIN-H1 noise, judged with REAL net R)
Day-shuffle: each day's (win, net) pool replaced by resampling (with replacement) from a different,
randomly-mapped day's pool — destroys the row-level feature↔outcome link, keeps day-level marginals.

| split | placebo top-decile n | placebo top-decile meanR |
|---|---|---|
| TRAIN-H1 (fit) | 93 | -0.141 |
| TRAIN-H2 | 41 | -0.192 |
| VAL | 58 | -0.262 |

Placebo top decile is ≤ 0 on both holdouts — no evidence of leakage-driven lift.

## Cadence block — VAL top-2-decile book
`{n: 430, tpw: 18.7, meanR: -0.134, t: -1.91, WR: 40.7%, wkR: -2.5, green_share: 0.35, worst_week: -18.9,
ex5%: -0.248, ex1%: -0.159, cap3: -0.134, MDE(2·SE): 0.14}` — well short of the cadence bar (needs
meanR > 0, ≥55% green weeks; this book clears only the ≥3 fills/week bar).

## Pass bar — verdict
| # | Criterion | TRAIN-H2 | VAL | Pass? |
|---|---|---|---|---|
| 1 | top-2-decile meanR > 0, day-clustered t ≥ 2 | -0.242, t=-3.01 | -0.134, t=-1.55 | **FAIL** (both negative) |
| 2 | deciles monotone, Spearman ≥ 0.6 | -0.248 | 0.309 | **FAIL** (neither meets bar; TRAIN-H2 is negative) |
| 3 | bottom-3-decile meanR < 0 | -0.200 | -0.334 | PASS |
| 4 | top-2-decile ≥ 3 fills/week | 16.85/wk | 18.70/wk | PASS |
| 5 | placebo top decile ≤ 0 | -0.192 | -0.262 | PASS |
| 6 | 3 largest \|β\| signs agree with a stated mechanism | — | — | **borderline/FAIL** (rank_sig sign contradicts its own mechanism) |

**Overall: FAIL** (criteria 1 and 2 fail outright on both holdouts; a single multivariate score built
from this feature set does not separate winners from losers out-of-sample).

### Detectable lift (criteria 1's own SE, since the test failed)
Top-2-decile SE: TRAIN-H2 = 0.068 R/trade (n=455), VAL = 0.070 R/trade (n=430). At these SEs the test
could detect a top-2-decile edge of roughly ±0.14R (2·SE) at 2-sigma; the observed top-2-decile means
(-0.24R TRAIN-H2, -0.13R VAL) are within/below that band on the negative side — this is a genuine null,
not an underpowered one.

Per PREREG: **Fail → the HOD loop moves to the failed-break short** (next frame). Programme count now
1,356 (cell 1,355 + placebo).
