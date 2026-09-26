
## L3 (amendment 2): the break EXTENDS (day high after bar j >= level x 1.05)

Weeks spanned: {'TRAIN': 27, 'VAL': 22}

### Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)

| label | model | holdout | n_kept | n_dropped | base_rate | kept_mean | dropped_mean | t_kept | ex_top5 | fills_wk | kept_bigday_rate | kept_cacheonly_share | auc | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3 | HGB | TRAIN-H2 | 62 | 122 | 0.2500 | 0.4892 | -0.7179 | 2.4370 | 0.4132 | 2.2963 | 0.7419 | 0.2903 | 0.7127 | False |
| L3 | HGB | VAL | 99 | 117 | 0.2685 | -0.2521 | -0.3240 | -1.9424 | -0.3703 | 4.5000 | 0.3232 | 0.1717 | 0.5803 | False |
| L3 | LR | TRAIN-H2 | 62 | 122 | 0.2500 | 0.2058 | -0.5739 | 1.0113 | 0.1153 | 2.2963 | 0.5645 | 0.2097 | nan | False |
| L3 | LR | VAL | 56 | 160 | 0.2685 | 0.0397 | -0.4068 | 0.2061 | -0.0691 | 2.5455 | 0.4286 | 0.1429 | 0.6025 | False |

### Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, tick_window_has_bar_j)
* L3: VAL AUC = 0.4891 (void if > 0.55) -> decoy_void = False

### Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)
* L3: VAL AUC = 0.5495 (pass <= 0.53); placebo kept mean = -0.3138 vs whole-book VAL mean (diff tol 0.05); placebo_ok = False

### Top-10 permutation importances (HGB, VAL, ROC AUC drop) -- best params {'max_depth': 3, 'learning_rate': 0.1, 'max_iter': 600, 'min_samples_leaf': 50}, CV AUC 0.7127
  - R_pct: 0.0281
  - atr14_pct: 0.0252
  - pre_break_print_count: 0.0195
  - pullback_depth_pct: 0.0181
  - level_vs_prior_high_pct: 0.0171
  - dist_from_open_pct: 0.0146
  - n_universe_j: 0.0124
  - symbol_persistence: 0.0114
  - pre_break_mean_trade_size: 0.0105
  - adv20: 0.0104

### VAL decile table (HGB probability, report-only; decile 10 = highest prob)
| decile | n | mean_prob | mean_R | bigday_rate |
|---|---|---|---|---|
| 1 | 22 | 0.0005 | -0.6930 | 0.1818 |
| 2 | 22 | 0.0023 | 0.0531 | 0.1818 |
| 3 | 21 | 0.0048 | -0.3341 | 0.2381 |
| 4 | 22 | 0.0109 | -0.7840 | 0.0909 |
| 5 | 21 | 0.0214 | -0.0083 | 0.3333 |
| 6 | 22 | 0.0506 | -0.1480 | 0.3636 |
| 7 | 21 | 0.0928 | 0.0395 | 0.3333 |
| 8 | 22 | 0.1746 | -0.5874 | 0.3636 |
| 9 | 21 | 0.4221 | -0.0529 | 0.3333 |
| 10 | 22 | 0.8156 | -0.3584 | 0.2727 |

### Cache-only share check (base 0.195, tol +/-0.05)
* L3/HGB/TRAIN-H2: kept cache-only share = 0.2903 -> within tol = False
* L3/HGB/VAL: kept cache-only share = 0.1717 -> within tol = True
* L3/LR/TRAIN-H2: kept cache-only share = 0.2097 -> within tol = True
* L3/LR/VAL: kept cache-only share = 0.1429 -> within tol = False
