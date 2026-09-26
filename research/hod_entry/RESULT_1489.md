# RESULT — cells 1,489 (BUY) / 1,490 (SHORT): retest-instant classifier

Best HGB params (5-fold CV AUC 0.643 inside TRAIN): {'max_depth': 5, 'learning_rate': 0.03, 'max_iter': 600, 'min_samples_leaf': 200}

TRAIN AUC (in-sample) 0.954; VAL AUC 0.515; placebo VAL AUC 0.495; decoy VAL AUC 0.543 (ok, <=0.55)

threshold_buy (TRAIN top-tercile p_hgb) = 0.4038; threshold_short (TRAIN bottom-tercile p_hgb) = 0.2237

Base rate Y=1: TRAIN 0.342, VAL 0.355


## Per cell x holdout

| cell | holdout | n_pop | n_kept | kept_mean | t_kept | ex_top5 | dropped_mean | auc | fills_wk | cacheonly_share_kept | cacheonly_share_pop | paired_dR_vs_base | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1489 | TRAIN | 3957 | 1319 | 1.2940 | 32.6033 | 1.2568 | -0.8166 | 0.9538 | 48.8519 | 0.2464 | 0.1918 | 0.3034 | True |
| 1489 | VAL | 5016 | 2070 | -0.0642 | -2.6075 | -0.1734 | -0.1062 | 0.5150 | 94.0909 | 0.1783 | 0.1740 | 0.2249 | False |
| 1490 | TRAIN | 253 | 66 | -0.0970 | -0.2491 | -0.2309 | -0.5446 | 0.9443 | 13.2000 | 0.1212 | 0.1739 |  | False |
| 1490 | VAL | 0 | 0 | NaN | NaN | NaN | NaN | NaN | 0.0000 | NaN | NaN |  | False |

## Notes on 1,490 (SHORT)

- TRAIN: match rate 253/3957 (6.4%)
- VAL: match rate 0/5016 (0.0%) -- NOT SCORABLE, no matched short outcomes in this holdout

## Top-10 VAL permutation importances (HGB, AUC drop)

| feature | importance |
|---|---|
| arm_range_to_j_pct | 0.0060 |
| brk_high_pct_of_level | 0.0052 |
| dip_speed_min_high_to_tr | 0.0045 |
| ctx_spy_ret_fill_to_tr | 0.0043 |
| arm_atr14_pct | 0.0036 |
| arm_float_shares | 0.0036 |
| arm_spy_ret_open_to_j | 0.0025 |
| arm_spread_bps_at_arm | 0.0025 |
| arm_sic2 | 0.0022 |
| arm_trigger_print_size | 0.0021 |

## Caveats

- 1,490 SHORT has ZERO matched short outcomes in VAL: rebuild_1479_1480.csv's short leg was built only on the TRAIN-period sample (2025-07-01..2025-12-31); all 406 matched rows (253 shortable & non-SSR) fall in TRAIN, VAL match count = 0. Cell 1,490 is therefore NOT SCORABLE against the frozen pass bar (VAL kept mean requires VAL rows) and FAILS by construction/data-coverage, not by an estimated negative edge. TRAIN-only diagnostics are reported for information, never as a pass.
- Match rate for 1,490's population overall: 406/8973 fills (4.5%), as flagged in FEATURES_1489.md; of those, 253 are shortable & non-SSR (the primary book population).
- `ctx_n_prior_retests_same_level` has zero variance in this population (always 0); included per spec but contributes no signal.
- `arm_arm_m` / `arm_arm_minute` (redundant identity index) and `tape_coverage` / `tape_source` (constant / near-constant provenance flags) excluded from the real feature set as non-market metadata; `store_served_1438` excluded as a duplicate of `decoy_store_served_1438`.
- Breadth-at-retest-minute (PREREG item 4) is not implemented as specified (FEATURES_1489.md): `arm_breadth_share_j`/`arm_breadth_count_j` (breadth at the ARM bar, median 0.85 min earlier) stand in as the causal proxy, disclosed there.
- `ctx_spy_ret_fill_to_tr` is NaN for 70% of rows (cache.db SPY series ends 2026-03-20, bars_fills_1478.db carries no SPY rows) -- HGB handles this natively; LR median-imputes it, which is a real information loss for the simple model on the majority of 2026 rows.
- Day-clustered t computed on per-day mean returns (weighting every day equally), per the programme's standing convention.
- `dip_bid_stepped_down_thru_level_5s`, `c1490_shortable`, `c1490_ssr` were stored as True/False/NaN objects in the CSV and were cast to 1.0/0.0/NaN before modeling.

## Judge (main session, 2026-09-26 18:20 UTC) — 1,489 FAIL (no out-of-sample signal), 1,490 VOID (no VAL coverage)

* 1,489 BUY: VAL AUC 0.515 (placebo 0.495, decoy 0.543 — no leak into the label); kept top tercile −0.064 R (t −2.6),
  dropped −0.106 R; TRAIN in-sample AUC 0.95 is memorisation (day-ungrouped CV 0.64). The independent rebuild from the
  prose lands at the same place (VAL AUC 0.517); the kept-set Jaccard 0.685 is what two chance-level models produce
  on borderline rows, not a feature-definition difference (`review/1489_compare.md`). The paired +0.22 R vs the base is
  the retest mechanism, identical on kept and dropped rows — the classifier adds nothing. FAIL on every bar.
* 1,490 SHORT: the 1,480 short leg exists for 406 fills, all in 2025H2 — 0 VAL rows. VOID by coverage; with AUC at
  chance the bottom tercile would carry no information either. Closed with the buy side.
* Refuter 1 found a real look-ahead in the feature matrix that does NOT change the verdict (a leak can only inflate an
  AUC that is already at chance) but matters elsewhere: `build_features_1478_A.py` selects bar j as the last bar with
  m < fill_min, and fill_min is FRACTIONAL (minute + seconds), so bar j = the FILL bar in 98.4 % of rows — its range,
  close and volume are realised after the fill. The 1,478 "arm-bar" features therefore include the break bar; the
  1,478-L3 claim "extension ≥ 5 % predictable at the arm bar, AUC 0.72" is UNVERIFIED until re-run with bar j =
  the last bar with m ≤ floor(fill_min) − 1 (PREREG_1478 amendment 3). Every negative 1,478 number stands.
* Refuter 2: on the retest book itself the cache-only cohort (the store-identity look-ahead cohort) earns +0.15 R and
  the real-SIP cohort −0.14 R on VAL (kept and dropped alike) — the real-SIP retest long is −0.14 R, worse than the
  −0.09 R headline. Carried to the 1,481 record and to the 1,493 judging (real-SIP-only read of the selected cell).
Programme count 1,490 (1,493–1,547 running).
