# Cell 1,478-L3-v2: re-run under Amendment 3 (arm bar = last bar CLOSED before fill_min)

Built from /home/ec2-user/onemil/research/hod_entry/features_1478_A_v2.csv (build_features_1478_A.py --arm-bar-closed), features B+C unchanged.

## Side-by-side vs the original (leaky) L3 run (RESULT_1478.md, PREREG Amendment 2)

| | original (leaky arm bar) | v2 (corrected arm bar) |
|---|---|---|
| VAL AUC (HGB) | 0.7232 | 0.7151 |
| kept VAL mean net R | -0.0908 | -0.0951 |
| kept VAL t (day-clustered) | (not reported in the original) | -1.8317 |
| kept TRAIN-H2 mean net R | 0.1446 (orig) | 0.3150 |
| kept TRAIN-H2 t | 2.0710 (orig) | 4.8633 |
| TRAIN-H2 CV AUC | 0.7251 (orig) | 0.7223 |
| decoy VAL AUC | 0.5795 (void, >0.55) | 0.5795 (void=True) |
| placebo VAL AUC | 0.4827 | 0.4881 |

**Verdict: the original AUC 0.72 does survive the arm-bar correction** (v2 VAL AUC 0.7151 vs original 0.7232; v2 kept VAL mean -0.0951 vs original -0.0908).

## L3 (amendment 2): the break EXTENDS (day high after bar j >= level x 1.05)

Weeks spanned: {'TRAIN': 27, 'VAL': 22}

### Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)

| label | model | holdout | n_kept | n_dropped | base_rate | kept_mean | dropped_mean | t_kept | ex_top5 | fills_wk | kept_bigday_rate | kept_cacheonly_share | auc | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3 | HGB | TRAIN-H2 | 1466 | 2932 | 0.2624 | 0.3150 | -0.4077 | 4.8633 | 0.2272 | 33.1852 | 0.6467 | 0.2940 | 0.7223 | False |
| L3 | HGB | VAL | 2048 | 3465 | 0.2757 | -0.0951 | -0.2151 | -1.8317 | -0.2032 | 44.2727 | 0.4458 | 0.2568 | 0.7151 | False |
| L3 | LR | TRAIN-H2 | 1466 | 2932 | 0.2624 | -0.0781 | -0.2112 | -1.1657 | -0.1862 | 34.0741 | 0.4543 | 0.2688 | nan | False |
| L3 | LR | VAL | 2244 | 3269 | 0.2757 | -0.0582 | -0.2476 | -1.1345 | -0.1647 | 45.2727 | 0.4367 | 0.2411 | 0.7128 | False |

### Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, tick_window_has_bar_j)
* L3: VAL AUC = 0.5795 (void if > 0.55) -> decoy_void = True

### Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)
* L3: VAL AUC = 0.4881 (pass <= 0.53); placebo kept mean = -0.1916 vs whole-book VAL mean (diff tol 0.05); placebo_ok = True

### Top-10 permutation importances (HGB, VAL, ROC AUC drop) -- best params {'max_depth': 5, 'learning_rate': 0.03, 'max_iter': 200, 'min_samples_leaf': 50}, CV AUC 0.7223
  - symbol_persistence: 0.0244
  - breadth_share_j: 0.0208
  - R_pct: 0.0114
  - atr14_pct: 0.0102
  - range_to_j_pct: 0.0085
  - sector_peers_pool_j: 0.0066
  - prior_range_pct: 0.0064
  - pullback_depth_pct: 0.0035
  - n_universe_j: 0.0029
  - breadth_5d: 0.0019

### VAL decile table (HGB probability, report-only; decile 10 = highest prob)
| decile | n | mean_prob | mean_R | bigday_rate |
|---|---|---|---|---|
| 1 | 552 | 0.0680 | -0.2288 | 0.0779 |
| 2 | 551 | 0.1146 | -0.1888 | 0.1361 |
| 3 | 551 | 0.1485 | -0.2424 | 0.1579 |
| 4 | 551 | 0.1800 | -0.2342 | 0.1561 |
| 5 | 552 | 0.2157 | -0.2153 | 0.2192 |
| 6 | 551 | 0.2593 | -0.2529 | 0.2505 |
| 7 | 551 | 0.3136 | -0.1122 | 0.3194 |
| 8 | 551 | 0.3814 | -0.0817 | 0.4156 |
| 9 | 551 | 0.4619 | -0.1597 | 0.4519 |
| 10 | 552 | 0.5858 | 0.0103 | 0.5725 |

### Cache-only share check (base 0.195, tol +/-0.05)
* L3/HGB/TRAIN-H2: kept cache-only share = 0.2940 -> within tol = False
* L3/HGB/VAL: kept cache-only share = 0.2568 -> within tol = False
* L3/LR/TRAIN-H2: kept cache-only share = 0.2688 -> within tol = False
* L3/LR/VAL: kept cache-only share = 0.2411 -> within tol = True
