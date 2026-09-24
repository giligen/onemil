# REPORT -- breakout thermometer (research/thermo/PREREG.md, cells 1,420-1,422)

## Cell 1,420 -- ORB live-config fills x T_ORB
n_fills=638
- Causal: n_hot=244 n_cold=385 mean_hot=0.0880 mean_cold=0.0957 hot-cold=-0.0077 t_cluster=-0.14 t_iid=-0.16
- Oracle (same-day T, look-ahead): n_hot=249 n_cold=380 mean_hot=0.1069 mean_cold=0.0833 hot-cold=0.0236 t_cluster=0.42 t_iid=0.48
- Stale (window ends 60 trading days earlier): n_hot=204 n_cold=400 mean_hot=0.1290 mean_cold=0.0712 hot-cold=0.0578 t_cluster=0.99 t_iid=1.11
- 2023-24 leg: n_hot=70 n_cold=86 mean_hot=0.0172 mean_cold=0.0855 hot-cold=-0.0682 t_cluster=-0.73 t_iid=-0.72
- 2025-26 leg: n_hot=174 n_cold=299 mean_hot=0.1165 mean_cold=0.0986 hot-cold=0.0179 t_cluster=0.26 t_iid=0.31
- hot-cohort ex-top-5% mean R: -0.0096
- fills/week in hot weeks: 3.588
- share of days hot: all=0.421 2023-24=0.484 2025-26=0.356
- quintile table (T bucket -> mean R, count): [{'q': 0, 'mean': 0.11459603873481006, 'count': 129}, {'q': 1, 'mean': 0.12383313654711296, 'count': 126}, {'q': 2, 'mean': 0.08286164586333482, 'count': 130}, {'q': 3, 'mean': 0.045074484806896346, 'count': 125}, {'q': 4, 'mean': 0.09376414735082901, 'count': 128}]
- **VERDICT: FAIL** legs={'pooled_hot_minus_cold_ge_0.15': False, 'pooled_t_cluster_ge_2': False, '2023_24_hot_minus_cold_gt_0': False, '2023_24_each_cohort_ge_15': True, '2025_26_hot_minus_cold_gt_0': True, '2025_26_each_cohort_ge_15': True, 'ex_top5_hot_gt_0': False}

## Cell 1,421 -- HOD B0 signals x T_HOD (TRAIN-H2 + VAL; TRAIN-H1 = burn-in; TEST dropped on read)
n_scored=8248
- Causal: n_hot=2902 n_cold=5244 mean_hot=-0.2667 mean_cold=-0.2902 hot-cold=0.0234 t_cluster=0.41 t_iid=0.74
- Oracle (same-day T, look-ahead): n_hot=2980 n_cold=5166 mean_hot=-0.2340 mean_cold=-0.3094 hot-cold=0.0754 t_cluster=1.32 t_iid=2.38
- Stale (window ends 60 trading days earlier): n_hot=3354 n_cold=4669 mean_hot=-0.2869 mean_cold=-0.2837 hot-cold=-0.0031 t_cluster=-0.05 t_iid=-0.10
- TRAIN-H2 leg: n_hot=2002 n_cold=1449 mean_hot=-0.2350 mean_cold=-0.2738 hot-cold=0.0388 t_cluster=0.43 t_iid=0.82
- VAL leg: n_hot=900 n_cold=3795 mean_hot=-0.3374 mean_cold=-0.2964 hot-cold=-0.0410 t_cluster=-0.50 t_iid=-0.80
- hot-cohort VAL ex-top-5% mean net_R: -0.4568
- hot fills/week (VAL, first-12/day 4-concurrent slot sim): 11.409
- quintile table (T bucket -> mean net_R, count): [{'q': 0, 'mean': -0.29721997221560204, 'count': 1675}, {'q': 1, 'mean': -0.29680982761867986, 'count': 1588}, {'q': 2, 'mean': -0.2702025740490688, 'count': 1626}, {'q': 3, 'mean': -0.2912111575874376, 'count': 1628}, {'q': 4, 'mean': -0.2535312056681834, 'count': 1629}]
- **VERDICT: FAIL** legs={'train_h2_hot_minus_cold_ge_0.10': False, 'val_hot_minus_cold_ge_0.10': False, 'val_t_cluster_ge_2': False, 'val_hot_mean_net_r_ge_0': False, 'fills_per_week_hot_ge_3': True}

## Cell 1,422 -- bull-flag P1 trades x T_ORB (cross-book, REPORT-ONLY, no pass bar)
n=71
- n_hot=28 n_cold=43 mean_hot=0.1145 mean_cold=0.8537 hot-cold=-0.7391 t_cluster=-1.91 t_iid=-1.99
- ex-top-5% mean R: 0.3646
- Caveat (research/bf_2024/REPORT.md): the 2024H2 file's raw n includes QBTS.WS, a warrant kept only by symbol-list match; the live rule excludes it (corrected n=26, +0.104 R). Not re-filtered here -- report-only, no PREREG basis to change the population.

## Data
- ORB population: 20280 rows, 933 days, 2023-01-03..2026-09-23
- ORB book (fills, entered==1): 638 of 897 rows
- HOD (TRAIN+VAL, TEST dropped): 12135 rows, 344 days, 2025-01-10..2026-05-29

## Adversary lenses (PREREG.md)
Oracle (look-ahead) hot-cold must exceed the causal hot-cold, else the causal number is noise. Stale (window ending 60 trading days earlier) reported beside. Per-period legs rule out a pure 2024->2025 level shift carrying the pooled result alone.

## Main-session review (2026-09-24 ~22:10 UTC)
* **Consistency gate:** first run STOPPED correctly — `runB_true` was a catalyst-veto-ON book. Amended before any
  thermometer number (PREREG amendment): the veto-ON rebuild of the same features CSV reproduces `runB_true` 2025
  (84 / $6,662 vs 85 / $6,561; 2026 Jan–May 40 / $6,018 vs 42 / $6,398), so the machinery is sound; scored on veto OFF.
* **Verdicts (frozen bars): 1,420 FAIL, 1,421 FAIL.** No thermometer effect in either book: ORB hot − cold −0.008 R
  (t −0.14; SE ≈ 0.055, so an effect ≥ ~0.15 R/fill is excluded at 80 % power); even the look-ahead oracle is only
  +0.024 (ORB) / +0.075 (HOD) — the recent win rate of the breakout population carries almost no information about the
  next day's book, causal or not. HOD hot cohort −0.27 R. Consequences as pre-committed: no regime sizing on ORB; HOD
  time-uninformative at this horizon. 1,422 (BF, report-only, n 71, a warrant not re-filtered) hot − cold −0.74 R: no
  claim.
* **The correction this study surfaced (bigger than its verdicts):** at the LIVE config (veto OFF) ORB 2025 is +0.106
  R/fill on 211 fills (t 2.45) and 2026-01..09-23 +0.105 on 262 (t 2.29) — pooled +0.105, t 3.31, $18,653 at $375,
  ex-top-5 % ≈ 0 (tail-carried). The "+0.27 in 2025" used in `research/orb_2023/REPORT.md` / `orb_2024/REPORT.md` was
  the veto-ON book. Per fill, ORB out of regime (+0.089 2023-24H1, −0.007 2024H2) is close to in regime; what the
  regime changes is FREQUENCY (1.4–2.3 fills/wk vs 4.3–7.3).
