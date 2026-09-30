# RESULT -- cell 1,677: the model-gated take-profit

PREREG: research/hod_entry/PREREG_1677.md, FROZEN 2026-09-30 07:40 UTC. Population n=5506 (1,663 join, floored r_pct>=1.5%). 180 paired reads (5k x 2m x 3tau x 3variants x 2scorings) + placebo (10 seeds/cell) + X6 comparator, all built.

## Pass bar: 0/90 (variant,k,m,tau) combos pass BOTH scorings (dR>=+0.05R, day_t>=2.5, ex-top5%>0, beats placebo by >=+0.03R)

Nothing clears the pass bar on both scorings.

## Best cell per variant (by min(mean_dR) across the two scorings; reported whether or not it passes)

### FULL: k=60 m=1.0 tau_low=0.3
| scoring | n_pool | n_fired | share_fired | mean_dR | iid_t | day_t | ex_top5_dR | MDE | giveback_saved (n) | continuation_forgone (n) | mean_cost_R | placebo dR/t | X6 mean_dR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2->VAL | 248 | 43 | 0.173 | +0.3029 | 1.61 | 2.26 | +0.1448 | 0.5267 | +1.394 (21) | +0.738 (22) | 0.0260 | -0.0430/-0.37 | -0.1676 |
| VAL->TRAIN-H2 | 162 | 42 | 0.259 | +0.1345 | 0.69 | 0.12 | -0.0388 | 0.5458 | +1.499 (17) | +0.793 (25) | 0.0298 | -0.0722/-1.27 | -0.1103 |

### P50: k=60 m=1.0 tau_low=0.3
| scoring | n_pool | n_fired | share_fired | mean_dR | iid_t | day_t | ex_top5_dR | MDE | giveback_saved (n) | continuation_forgone (n) | mean_cost_R | placebo dR/t | X6 mean_dR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2->VAL | 248 | 43 | 0.173 | +0.1514 | 1.61 | 2.26 | +0.0724 | 0.2634 | +0.697 (21) | +0.369 (22) | 0.0130 | -0.0215/-0.37 | -0.0838 |
| VAL->TRAIN-H2 | 162 | 42 | 0.259 | +0.0673 | 0.69 | 0.12 | -0.0194 | 0.2729 | +0.749 (17) | +0.396 (25) | 0.0149 | -0.0361/-1.27 | -0.0552 |

### TS: k=60 m=1.0 tau_low=0.3
| scoring | n_pool | n_fired | share_fired | mean_dR | iid_t | day_t | ex_top5_dR | MDE | giveback_saved (n) | continuation_forgone (n) | mean_cost_R | placebo dR/t | X6 mean_dR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2->VAL | 248 | 43 | 0.173 | +0.3029 | 1.61 | 2.26 | +0.1448 | 0.5267 | +1.394 (21) | +0.738 (22) | 0.0260 | -0.0430/-0.37 | -0.1676 |
| VAL->TRAIN-H2 | 162 | 42 | 0.259 | +0.1345 | 0.69 | 0.12 | -0.0388 | 0.5458 | +1.499 (17) | +0.793 (25) | 0.0298 | -0.0722/-1.27 | -0.1103 |

## Adequacy
Reused, never re-fit: f1668.walk_k/dR_cut (the FULL exit mechanic + 6bps cost), f1676's stat kit, gfix.build_group_cols_fix (the exact G7-without-close_R+ALL feature group). The 10 success models (5k x 2 scorings) were re-run ONCE with 1676's identical fit code + seed because neither the models nor per-fill scores survived 1676's original run (see script docstring) -- disclosed, not hidden. X6 is the exit-lab's trailing-stop MECHANISM re-implemented on this line's own bars/cost (different population, so its own plumbing was not reused). TS checks only the 5 fixed checkpoints (no refit) rather than literally every minute -- disclosed. Both scorings always reported separately; MDE beside every t; no pooled-only numbers. A pass here still requires independent reimplementation before the owner sees a number, per PREREG.

Files: 1677_take_profit.py, 1677_reads.csv, 1677_per_fill.csv, 1677_take_profit.log, models under research/hod_entry/models/1676_G7noclose_k*_ALL_success_*.joblib.
