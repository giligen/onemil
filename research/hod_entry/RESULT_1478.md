# RESULT 1,478 -- supervised big-day predictor at the arm bar

## Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)

| label | model | holdout | n_kept | n_dropped | base_rate | kept_mean | dropped_mean | t_kept | ex_top5 | fills_wk | kept_bigday_rate | kept_cacheonly_share | auc | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | HGB | TRAIN-H2 | 1466 | 2932 | 0.5691 | -0.0604 | -0.2200 | -1.0016 | -0.1674 | 36.3704 | 0.9898 | 0.3458 | 0.8704 | False |
| L1 | HGB | VAL | 2095 | 3418 | 0.5625 | -0.1245 | -0.1987 | -2.5489 | -0.2349 | 46.5000 | 0.9413 | 0.2912 | 0.8783 | False |
| L1 | LR | TRAIN-H2 | 1466 | 2932 | 0.5691 | -0.1027 | -0.1988 | -1.7716 | -0.2119 | 37.2593 | 0.9543 | 0.3431 | nan | False |
| L1 | LR | VAL | 2168 | 3345 | 0.5625 | -0.1350 | -0.1936 | -2.7960 | -0.2452 | 46.2273 | 0.9350 | 0.2966 | 0.8829 | False |
| L2 | HGB | TRAIN-H2 | 1466 | 2932 | 0.3072 | -0.0021 | -0.2491 | -0.0381 | -0.1059 | 33.4815 | 0.7988 | 0.2626 | 0.9404 | False |
| L2 | HGB | VAL | 2253 | 3260 | 0.3419 | -0.0375 | -0.2624 | -0.6492 | -0.1437 | 45.2273 | 0.7288 | 0.2015 | 0.9417 | False |
| L2 | LR | TRAIN-H2 | 1466 | 2932 | 0.3072 | -0.0665 | -0.2170 | -1.0955 | -0.1737 | 32.9259 | 0.6255 | 0.2565 | nan | False |
| L2 | LR | VAL | 2240 | 3273 | 0.3419 | -0.0250 | -0.2701 | -0.4525 | -0.1303 | 44.7727 | 0.6098 | 0.1996 | 0.8210 | False |

## Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, tick_window_has_bar_j)
* L1: VAL AUC = 0.6199 (void if > 0.55) -> decoy_void = True
* L2: VAL AUC = 0.7916 (void if > 0.55) -> decoy_void = True

## Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)
* L1: VAL AUC = 0.5071 (pass <= 0.53); placebo kept mean = -0.2060 vs whole-book VAL mean (diff tol 0.05); placebo_ok = True
* L2: VAL AUC = 0.5059 (pass <= 0.53); placebo kept mean = -0.2423 vs whole-book VAL mean (diff tol 0.05); placebo_ok = False

## Top-10 permutation importances (HGB, VAL, ROC AUC drop)
### L1 (best params {'max_depth': 3, 'learning_rate': 0.03, 'max_iter': 600, 'min_samples_leaf': 50}, CV AUC 0.8704)
  - range_to_j_pct: 0.2146
  - symbol_persistence: 0.0177
  - n_universe_j: 0.0059
  - vwap_dist_pct: 0.0058
  - dist_from_open_pct: 0.0049
  - breadth_share_j: 0.0041
  - atr14_pct: 0.0029
  - high20: 0.0016
  - prior_range_pct: 0.0014
  - R_pct: 0.0014
### L2 (best params {'max_depth': 5, 'learning_rate': 0.03, 'max_iter': 200, 'min_samples_leaf': 200}, CV AUC 0.9404)
  - prev_day_volume: 0.3206
  - range_to_j_pct: 0.1084
  - symbol_persistence: 0.0050
  - vwap_dist_pct: 0.0035
  - atr14_pct: 0.0026
  - breadth_share_j: 0.0018
  - R_pct: 0.0012
  - n_universe_j: 0.0011
  - prior_range_pct: 0.0010
  - time_of_day_min: 0.0005

## VAL decile table (HGB probability, report-only; decile 10 = highest prob)
### L1
| decile | n | mean_prob | mean_R | bigday_rate |
|---|---|---|---|---|
| 1 | 552 | 0.1151 | -0.2833 | 0.1087 |
| 2 | 551 | 0.2205 | -0.2127 | 0.1996 |
| 3 | 551 | 0.3182 | -0.2229 | 0.2722 |
| 4 | 551 | 0.4251 | -0.2530 | 0.3176 |
| 5 | 552 | 0.5536 | -0.1187 | 0.4402 |
| 6 | 551 | 0.6973 | -0.1091 | 0.5826 |
| 7 | 551 | 0.8467 | -0.0884 | 0.7568 |
| 8 | 551 | 0.9635 | -0.1274 | 0.9474 |
| 9 | 551 | 0.9983 | -0.2822 | 1.0000 |
| 10 | 552 | 0.9995 | -0.0079 | 1.0000 |
### L2
| decile | n | mean_prob | mean_R | bigday_rate |
|---|---|---|---|---|
| 1 | 552 | 0.0015 | -0.2805 | 0.0000 |
| 2 | 551 | 0.0018 | -0.3825 | 0.0000 |
| 3 | 551 | 0.0021 | -0.3513 | 0.0000 |
| 4 | 551 | 0.0515 | -0.1636 | 0.0454 |
| 5 | 552 | 0.2216 | -0.2285 | 0.1504 |
| 6 | 551 | 0.3467 | -0.1465 | 0.2740 |
| 7 | 551 | 0.4955 | -0.1504 | 0.4211 |
| 8 | 551 | 0.6937 | 0.0656 | 0.6352 |
| 9 | 551 | 0.9055 | 0.0069 | 0.8929 |
| 10 | 552 | 0.9902 | -0.0746 | 1.0000 |

## Cache-only share check (base 0.195, tol +/-0.05)
* L1/HGB/TRAIN-H2: kept cache-only share = 0.3458 -> within tol = False
* L1/HGB/VAL: kept cache-only share = 0.2912 -> within tol = False
* L1/LR/TRAIN-H2: kept cache-only share = 0.3431 -> within tol = False
* L1/LR/VAL: kept cache-only share = 0.2966 -> within tol = False
* L2/HGB/TRAIN-H2: kept cache-only share = 0.2626 -> within tol = False
* L2/HGB/VAL: kept cache-only share = 0.2015 -> within tol = True
* L2/LR/TRAIN-H2: kept cache-only share = 0.2565 -> within tol = False
* L2/LR/VAL: kept cache-only share = 0.1996 -> within tol = True

Elapsed: 682s

## Verdict (2026-09-26)
* **Decoy VOID on both labels.** Metadata-only (store_served_1438, rth_bar_count_1438,
  tick_window_has_bar_j) predicts L1 at VAL AUC 0.62 and L2 at 0.79, both over the 0.55 void line:
  which store originally served a fill's bars in cell 1,438 still correlates with whether that day
  was a big-range day, even after every bar-derived feature was rebuilt from ONE fresh SIP store
  (per the amendment). decoy_void = True for both labels; per the amendment this cell is VOID.
* **Corroborating leak signal, independent of AUC.** The real models' KEPT set's cache-only share
  (store_served_1438 mean) breaches the +/-5pp tolerance around the base 19.5% on L1 (both models,
  both holdouts: 29-35%) and on L2 TRAIN-H2 (26%) -- the real feature set, though it excludes the
  decoy columns by name, still selects disproportionately from the old sparse-cache cohort.
* **Fails the profitability bar regardless of the leak.** Every kept-set mean net R is NEGATIVE
  (-0.002 to -0.135 R, vs the +0.15 bar) on both holdouts, both labels, both models. The models DO
  rank-order real signal (kept bigday_rate 0.73-0.99 vs base rates 0.34-0.57; deciles are
  monotonic; the label-shuffled placebo AUC is ~0.51 on both labels, confirming the real models'
  0.88/0.94 VAL AUC is not a training-pipeline artifact) -- permutation importance shows this comes
  overwhelmingly from range_to_j_pct (L1: how far the day has already moved by arm bar j) and
  prev_day_volume (L2: which re-derives one of L2's own AND-clauses, "prior volume >= 1M", from a
  different vendor than the Databento label -- correlated but not a bar-j look-ahead). Even
  perfect-precision selection of a big-range day is a ceiling of only +0.17 R (cell 1,457); at
  73-99% precision here the HOD-break entry/exit mechanics themselves absorb the rest.
* **Placebo:** L1 passes both placebo conditions; L2's placebo AUC is fine (0.506) but its kept
  mean (-0.242) misses the whole-book VAL mean (-0.171) by 0.072 R, over the 0.05 tolerance --
  noted, not chased further (small-sample selection noise on a random top-tercile draw).
* Net: no cell here reaches the pass bar on ANY criterion that matters (decoy void, cache-only
  share, AND kept-mean profitability all fail); this round's answer to "predict the big day" is a
  measured AUC (0.88 / 0.94 VAL, largely explained by two causal-but-tautological features) that
  does not convert to a tradeable edge under this book's cost. Per PREREG_1478's own protocol, an
  independent reimplementation from this prose (never reading cell_1478.py) is required before any
  of these numbers are relayed to the owner as a finding, and the refuters' first lens should be
  the decoy leak and the cache-only-share breach, not the headline AUC.
## Judge's interim note (2026-09-26 ~14:30 UTC)
* Decoy model VOID on both labels (VAL AUC 0.62 / 0.79): the old store identity predicts the big day even after every bar
  feature was rebuilt from one fresh SIP store — the look-ahead cohort leaks through correlated features; the real
  models' kept sets over-select the cache-only cohort (29–35 % vs 19.5 %).
* L1 / L2: high AUC (0.88 / 0.94) is the label's own construction (range already realised at bar j; prior-day volume
  and price re-derive two of L2's three terms); every kept set is negative on VAL (−0.03 to −0.14 R); no decile
  reaches +0.15. FAIL as specified.
* 1,479 pyramid: ΔR +0.02 / +0.04, t 0.7 / 1.8 — FAIL. 1,480 failed-break short: −0.53 / −0.68 R, t −8.6 / −12.6 —
  FAIL decisively, and the strength of that loss is the lead behind `PREREG_1481.md` (buy the retest).
* Open: the independent rebuild's L2 kept set is +0.36 R on VAL (Jaccard 0.5–0.7 vs the builder) — a reconciliation is
  running (`review/1478_reconciliation.md`); the corrective future-only label L3 is running. No verdict until both land.

## L3 (amendment 2): the break EXTENDS (day high after bar j >= level x 1.05)

Weeks spanned: {'TRAIN': 27, 'VAL': 22}

### Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)

| label | model | holdout | n_kept | n_dropped | base_rate | kept_mean | dropped_mean | t_kept | ex_top5 | fills_wk | kept_bigday_rate | kept_cacheonly_share | auc | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3 | HGB | TRAIN-H2 | 1466 | 2932 | 0.2624 | 0.1446 | -0.3225 | 2.0710 | 0.0480 | 33.3333 | 0.5675 | 0.2926 | 0.7251 | False |
| L3 | HGB | VAL | 2116 | 3397 | 0.2757 | -0.0908 | -0.2202 | -1.6229 | -0.1994 | 45.0909 | 0.4551 | 0.2566 | 0.7232 | False |
| L3 | LR | TRAIN-H2 | 1466 | 2932 | 0.2624 | -0.0304 | -0.2350 | -0.4615 | -0.1361 | 34.3333 | 0.4673 | 0.2742 | nan | False |
| L3 | LR | VAL | 2240 | 3273 | 0.2757 | -0.0486 | -0.2540 | -0.9445 | -0.1548 | 45.7273 | 0.4379 | 0.2442 | 0.7130 | False |

### Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, tick_window_has_bar_j)
* L3: VAL AUC = 0.5795 (void if > 0.55) -> decoy_void = True

### Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)
* L3: VAL AUC = 0.4827 (pass <= 0.53); placebo kept mean = -0.1687 vs whole-book VAL mean (diff tol 0.05); placebo_ok = True

### Top-10 permutation importances (HGB, VAL, ROC AUC drop) -- best params {'max_depth': 5, 'learning_rate': 0.03, 'max_iter': 200, 'min_samples_leaf': 200}, CV AUC 0.7251
  - breadth_share_j: 0.0203
  - symbol_persistence: 0.0157
  - R_pct: 0.0077
  - atr14_pct: 0.0052
  - prior_range_pct: 0.0045
  - range_to_j_pct: 0.0043
  - vwap_dist_pct: 0.0042
  - sector_peers_pool_j: 0.0036
  - dist_from_open_pct: 0.0031
  - spy_ret_open_to_j: 0.0028

### VAL decile table (HGB probability, report-only; decile 10 = highest prob)
| decile | n | mean_prob | mean_R | bigday_rate |
|---|---|---|---|---|
| 1 | 552 | 0.0760 | -0.2084 | 0.0833 |
| 2 | 551 | 0.1206 | -0.2691 | 0.1198 |
| 3 | 551 | 0.1551 | -0.2467 | 0.1397 |
| 4 | 551 | 0.1895 | -0.1932 | 0.1688 |
| 5 | 552 | 0.2280 | -0.2286 | 0.1975 |
| 6 | 551 | 0.2725 | -0.1761 | 0.2505 |
| 7 | 551 | 0.3310 | -0.1298 | 0.3376 |
| 8 | 551 | 0.4056 | -0.2123 | 0.4047 |
| 9 | 551 | 0.4924 | -0.0977 | 0.4682 |
| 10 | 552 | 0.6104 | 0.0564 | 0.5870 |

### Cache-only share check (base 0.195, tol +/-0.05)
* L3/HGB/TRAIN-H2: kept cache-only share = 0.2926 -> within tol = False
* L3/HGB/VAL: kept cache-only share = 0.2566 -> within tol = False
* L3/LR/TRAIN-H2: kept cache-only share = 0.2742 -> within tol = False
* L3/LR/VAL: kept cache-only share = 0.2442 -> within tol = True

## Reconciliation of the rebuild's sign flip (2026-09-26, `review/1478_reconciliation.md`)
The independent rebuild had mapped cell 1,457's `flag_1457` (the full-day range ≥ 10 % CEILING flag — the L1 label
itself) to a column it took for the causal ATR %, and separately computed an ATR from the fill day's own unshifted bars;
both leaked the label. With the two columns removed and the same grid/seed refit, its mover-day kept set is −0.07 R (HGB)
/ −0.02 R (LR) on VAL — the builder's −0.04 / −0.03 within noise. The builder is right; the sign flip was a label leak in
the check itself, caught by the reconciliation. Verdict unchanged: FAIL (decoy VOID, kept sets negative). L3 pending.

## Judge's verdict on L3 (2026-09-26 16:40 UTC) — the owner's question, measured
L3 = "the break extends ≥ 5 % beyond the level after the arm bar" (base rate 26 % / 28 %). Decoy AUC 0.58 (void by the
letter of the amendment: the old store identity still carries ≈ 0.03 of AUC). Real models: VAL AUC 0.72 (HGB) / 0.71
(LR), placebo 0.48 — the extension IS partly predictable at the arm bar, and the kept top tercile raises the extension
rate from 28 % to 46 %. It does not convert into money under the break entry: kept mean +0.14 R on TRAIN-H2 (t 2.1)
and −0.09 R on VAL (t −1.6). FAIL. Reading: a fill at the ask above the level loses even on many days the break later
extends, because 87 % of breaks first dip back under the level (cell 1,480) and the consolidation-low stop sits 1.6 %
below. The predictability lives in the extension; the loss lives in the entry. That points at exactly one joint,
pre-declared in `PREREG_1481.md` amendment 1 before any retest number is read.
