# RESULT 1,673 -- short the predicted failure

Population: 5506 fills (1669/1670 join, primary r_pct>=1.5%, halves=split). Fired (P>=tau) across all 5 rules x 3 tau x 2 scorings: 18609. SSR-skipped shorts (entry >=10% below prior close): 148. No-entry-bar (fired but no next bar to short into): 0. Day ended before 15:55 ET (forced last-bar EOD): 10. Missing prior-close (SSR check skipped, treated as shortable): 0.

Costs: short entry 7bps, cover 6bps (stop/target), EOD cover 11bps; locate/borrow fee ignored on paper -- STATED, not modeled. Fill convention: exact stop/target touch price (matches the existing long-book convention); gapped_through_share in 1673_per_short.csv/reads reports how often the exit bar's own open had already passed the level (that share is priced optimistically by this convention).

Placebo: 1669/1670 did not save label-shuffled per-fill probabilities (only placebo AUC), so the placebo check is NOT AVAILABLE for this cell.

## Pass bar (short mean net R >= +0.10R, day t >= 2.5, ex-top-5% > 0, >=3 shorts/wk, BOTH scorings)
0/45 (label,k,tau,variant) cells pass on both scorings (placebo not checked -- any pass below is provisional).

## Best cell per label (by min(day_t) across both scorings), short-only stats
**FF10@1 best: tau=0.6 variant=H**
  * TRAIN->VAL: n=77 mean_net_R=0.065 iid_t=1.00 day_t=1.09 ex_top5=0.023 mde=0.182 hit_rate=0.662 avg_hold_min=6.5 worst_day_R=-2.685 shorts/wk=3.67 gapped_through=0.195
  * VAL->TRAIN-H2 (swap): n=91 mean_net_R=0.062 iid_t=1.09 day_t=1.59 ex_top5=0.014 mde=0.158 hit_rate=0.637 avg_hold_min=7.4 worst_day_R=-2.253 shorts/wk=3.48 gapped_through=0.143
**FF5@1 best: tau=0.6 variant=T2**
  * TRAIN->VAL: n=22 mean_net_R=-0.031 iid_t=-0.12 day_t=0.02 ex_top5=-0.192 mde=0.694 hit_rate=0.455 avg_hold_min=69.6 worst_day_R=-3.123 shorts/wk=1.05 gapped_through=0.045
  * VAL->TRAIN-H2 (swap): n=35 mean_net_R=0.429 iid_t=2.13 day_t=2.26 ex_top5=0.342 mde=0.563 hit_rate=0.629 avg_hold_min=53.3 worst_day_R=-1.086 shorts/wk=1.34 gapped_through=0.257
**PSTOP@10 best: tau=0.6 variant=T2**
  * TRAIN->VAL: n=1313 mean_net_R=-0.156 iid_t=-4.83 day_t=-1.52 ex_top5=-0.283 mde=0.090 hit_rate=0.360 avg_hold_min=116.2 worst_day_R=-25.190 shorts/wk=62.52 gapped_through=0.154
  * VAL->TRAIN-H2 (swap): n=809 mean_net_R=-0.018 iid_t=-0.41 day_t=-0.20 ex_top5=-0.139 mde=0.120 hit_rate=0.426 avg_hold_min=107.1 worst_day_R=-18.787 shorts/wk=30.95 gapped_through=0.156
**PSTOP@2 best: tau=0.8 variant=H**
  * TRAIN->VAL: n=967 mean_net_R=-0.018 iid_t=-1.01 day_t=-0.12 ex_top5=-0.078 mde=0.050 hit_rate=0.405 avg_hold_min=13.0 worst_day_R=-9.200 shorts/wk=46.05 gapped_through=0.180
  * VAL->TRAIN-H2 (swap): n=556 mean_net_R=-0.036 iid_t=-1.52 day_t=0.66 ex_top5=-0.092 mde=0.067 hit_rate=0.435 avg_hold_min=13.7 worst_day_R=-9.255 shorts/wk=21.27 gapped_through=0.239
**PSTOP@5 best: tau=0.7 variant=T2**
  * TRAIN->VAL: n=1290 mean_net_R=-0.123 iid_t=-3.67 day_t=-0.85 ex_top5=-0.248 mde=0.094 hit_rate=0.365 avg_hold_min=104.5 worst_day_R=-24.089 shorts/wk=61.43 gapped_through=0.148
  * VAL->TRAIN-H2 (swap): n=797 mean_net_R=-0.013 iid_t=-0.30 day_t=-0.54 ex_top5=-0.131 mde=0.123 hit_rate=0.415 avg_hold_min=101.6 worst_day_R=-16.645 shorts/wk=30.49 gapped_through=0.154

## Portfolio read (paired dR of adding the short overlay to the base long book)
**FF10@1 best overlay: tau=0.6 variant=T2**
  * TRAIN->VAL: portfolio_mean_dR=0.0043 day_t=1.45 ex_top5=-0.0132 worst_day_R=-0.117 n_eligible=3142
  * VAL->TRAIN-H2 (swap): portfolio_mean_dR=0.0116 day_t=3.05 ex_top5=-0.0198 worst_day_R=-0.191 n_eligible=2330
**FF5@1 best overlay: tau=0.8 variant=base**
  * TRAIN->VAL: portfolio_mean_dR=-0.0002 day_t=0.13 ex_top5=-0.0011 worst_day_R=-0.032 n_eligible=3142
  * VAL->TRAIN-H2 (swap): portfolio_mean_dR=0.0019 day_t=1.77 ex_top5=-0.0010 worst_day_R=-0.062 n_eligible=2330
**PSTOP@10 best overlay: tau=0.6 variant=T2**
  * TRAIN->VAL: portfolio_mean_dR=-0.0785 day_t=-1.22 ex_top5=-0.1909 worst_day_R=-0.610 n_eligible=2605
  * VAL->TRAIN-H2 (swap): portfolio_mean_dR=-0.0073 day_t=0.18 ex_top5=-0.1118 worst_day_R=-0.752 n_eligible=1942
**PSTOP@2 best overlay: tau=0.8 variant=T2**
  * TRAIN->VAL: portfolio_mean_dR=-0.0294 day_t=-0.13 ex_top5=-0.1318 worst_day_R=-0.357 n_eligible=3100
  * VAL->TRAIN-H2 (swap): portfolio_mean_dR=-0.0138 day_t=-0.56 ex_top5=-0.1087 worst_day_R=-0.525 n_eligible=2292
**PSTOP@5 best overlay: tau=0.7 variant=T2**
  * TRAIN->VAL: portfolio_mean_dR=-0.0547 day_t=-1.08 ex_top5=-0.1623 worst_day_R=-0.455 n_eligible=2897
  * VAL->TRAIN-H2 (swap): portfolio_mean_dR=-0.0048 day_t=-0.19 ex_top5=-0.1078 worst_day_R=-0.717 n_eligible=2167

Full grid (all 90 (label,k,tau,variant,scoring) cells): `1673_reads.csv`. Row-level actual shorts: `1673_per_short.csv`.

## Not allowed items honored
No re-fitting or re-scoring of any model (P(stop) values are read, not recomputed); tau/k/variant fixed by the PREREG before any number was seen; no bar after the decision bar used for the decision (entry/exit walk starts strictly after fill+k); SSR rail applied to every fired signal; both scorings reported throughout, never pooled-only.
