# RESULT -- cell 1,676: candle shapes, in depth (G1-G7, 1/5/10/15-min frames)

PREREG: research/hod_entry/PREREG_1676.md, FROZEN 2026-09-30 06:50 UTC, amendment 1 (G7) 07:05 UTC -- verified against the file on disk before G7 was built, not taken on a relayed message alone. Population n=5506 (primary r_pct>=1.5%); halves: {'VAL': np.int64(3157), 'TRAIN-H2': np.int64(2349)}.

## Scope disclosure
- G6 R3 uses forward-from-k labels (fail_after_k / success_next15_from_k) at k=5, not the base ARM-relative fail_any/fail10/success15, to avoid the k=15 window leaking bars past an already-resolved outcome into the feature.
- G7 TA-Lib (61 flags) runs the full k-grid at w=5 and all widths at k in {1,5,15,30}; continuous shape features and climax/effort run the full (k,w) grid. Bounds the shared 2-CPU node; disclosed, not dropped silently.
- "Short" money geometry follows 1,673 (stop=short_entry+1R, target=long stop); "add" is the incremental R of a later same-direction entry riding to the SAME realized exit.

## R3 -- AUC table (best rows by group, both scorings; full table in 1676_reads.csv)

| group | label | scoring | n_test | AUC | placebo AUC | gap |
|---|---|---|---|---|---|---|
| ALL-shape+1670 | fail10 | VAL->TRAIN-H2 | 2348 | 0.795 | 0.508 | 0.287 |
| ALL-shape | fail10 | VAL->TRAIN-H2 | 2348 | 0.783 | 0.492 | 0.290 |
| ALL-shape | success15 | TRAIN-H2->VAL | 3157 | 0.768 | 0.523 | 0.245 |
| ALL-shape+1670 | success15 | TRAIN-H2->VAL | 3157 | 0.763 | 0.523 | 0.240 |
| G6 | success15 | TRAIN-H2->VAL | 3157 | 0.755 | 0.555 | 0.199 |
| G6 | fail10 | VAL->TRAIN-H2 | 2348 | 0.751 | 0.486 | 0.264 |
| ALL-shape+1670 | fail_any | VAL->TRAIN-H2 | 2348 | 0.695 | 0.531 | 0.164 |
| ALL-shape | fail_any | VAL->TRAIN-H2 | 2348 | 0.691 | 0.524 | 0.166 |
| G6 | fail_any | VAL->TRAIN-H2 | 2348 | 0.664 | 0.504 | 0.161 |
| G2 | fail10 | VAL->TRAIN-H2 | 2348 | 0.569 | 0.513 | 0.055 |
| G1 | fail10 | VAL->TRAIN-H2 | 2348 | 0.558 | 0.500 | 0.058 |
| G4 | fail10 | TRAIN-H2->VAL | 3157 | 0.548 | 0.526 | 0.022 |
| G3 | fail10 | VAL->TRAIN-H2 | 2348 | 0.540 | 0.532 | 0.008 |
| G2 | success15 | VAL->TRAIN-H2 | 2349 | 0.524 | 0.518 | 0.005 |
| G4 | success15 | VAL->TRAIN-H2 | 2349 | 0.523 | 0.497 | 0.026 |
| G3 | fail_any | VAL->TRAIN-H2 | 2348 | 0.521 | 0.516 | 0.005 |
| G1 | fail_any | VAL->TRAIN-H2 | 2348 | 0.517 | 0.489 | 0.028 |
| G1 | success15 | TRAIN-H2->VAL | 3157 | 0.516 | 0.497 | 0.019 |
| G3 | success15 | TRAIN-H2->VAL | 3157 | 0.514 | 0.522 | -0.009 |
| G5 | fail_any | TRAIN-H2->VAL | 3157 | 0.511 | 0.507 | 0.004 |

G7 rolling -- top 15 (group, label, k) by AUC:

| group | label | k | scoring | n_test | AUC | placebo AUC |
|---|---|---|---|---|---|---|
| G7_k60+ALL | g7success_next15_from_60 | 60 | TRAIN-H2->VAL | 3139 | 0.974 | nan |
| G7_k60 | g7success_next15_from_60 | 60 | TRAIN-H2->VAL | 3139 | 0.974 | nan |
| G7_k45+ALL | g7success_next15_from_45 | 45 | VAL->TRAIN-H2 | 2340 | 0.969 | nan |
| G7_k45 | g7success_next15_from_45 | 45 | VAL->TRAIN-H2 | 2340 | 0.967 | nan |
| G7_k45+ALL | g7success_next15_from_45 | 45 | TRAIN-H2->VAL | 3152 | 0.964 | nan |
| G7_k60 | g7success_next15_from_60 | 60 | VAL->TRAIN-H2 | 2330 | 0.963 | nan |
| G7_k60+ALL | g7success_next15_from_60 | 60 | VAL->TRAIN-H2 | 2330 | 0.962 | nan |
| G7_k45 | g7success_next15_from_45 | 45 | TRAIN-H2->VAL | 3152 | 0.961 | nan |
| G7_k30+ALL | g7success_next15_from_30 | 30 | TRAIN-H2->VAL | 3156 | 0.938 | 0.666 |
| G7_k30 | g7success_next15_from_30 | 30 | TRAIN-H2->VAL | 3156 | 0.938 | 0.665 |
| G7_k30+ALL | g7success_next15_from_30 | 30 | VAL->TRAIN-H2 | 2348 | 0.936 | 0.642 |
| G7_k30 | g7success_next15_from_30 | 30 | VAL->TRAIN-H2 | 2348 | 0.935 | 0.640 |
| G7_k20+ALL | g7success_next15_from_20 | 20 | TRAIN-H2->VAL | 3157 | 0.920 | nan |
| G7_k20+ALL | g7success_next15_from_20 | 20 | VAL->TRAIN-H2 | 2348 | 0.918 | nan |
| G7_k20 | g7success_next15_from_20 | 20 | TRAIN-H2->VAL | 3157 | 0.917 | nan |

Does shape add to path+volume (ALL-shape vs ALL-shape+1670, mean AUC both scorings):
- fail_any: ALL-shape 0.675 -> +1670-ALL 0.681 (delta +0.006)
- fail10: ALL-shape 0.760 -> +1670-ALL 0.772 (delta +0.012)
- success15: ALL-shape 0.767 -> +1670-ALL 0.761 (delta -0.006)

## R1 -- top-vs-rest cells with |day-clustered t| >= 2 in either half

| feature | k | TRAIN-H2 n/mean/t | VAL n/mean/t | same sign |
|---|---|---|---|---|
| g1_15m_uwick | 5 | 457/0.0483/0.11 | 616/-0.1169/-3.17 | False |
| g2_1m_n10_higher_lows | 3 | 321/0.0276/-0.21 | 342/0.1793/2.65 | True |
| g2_1m_n10_higher_lows | 5 | 321/0.0276/-0.21 | 342/0.1793/2.65 | True |
| g5_day_range_atr | 5 | 461/-0.0601/-0.04 | 623/-0.1514/-2.29 | True |
| g2_1m_n10_vol_slope | 3 | 755/0.0810/1.38 | 953/0.1213/2.26 | True |
| g2_1m_n10_vol_slope | 5 | 453/0.0991/1.07 | 549/0.1556/2.14 | True |
| g2_10m_n10_higher_lows | 5 | 129/0.2987/1.28 | 194/-0.1327/-2.09 | False |
| g1_10m_range_atr | 3 | 751/0.1191/2.95 | 1044/0.0469/1.22 | True |
| g3_5m_climax_mins_since | 3 | 5/-1.0809/-15.08 | 2/0.5522/0.76 | False |
| g2_5m_n30_net_progress | 3 | 570/-0.0786/-2.02 | 684/0.0684/0.43 | False |
| g3_5m_climax_mins_since | 5 | 3/-1.0692/-17.87 | 1/-0.8666/nan | True |

## R2 -- pattern x context cells (>=50 fires), top 20 by |t| either half

| pattern | frame | context | half | n | mean net_R | day_t | lift vs book |
|---|---|---|---|---|---|---|---|
| CDLSHORTLINE | 1m | 1 | TRAIN-H2 | 354 | -0.0975 | -3.16 | -0.0764 |
| CDLDOJI | 1m | 1 | TRAIN-H2 | 297 | -0.1934 | -2.92 | -0.1724 |
| CDLHARAMI | 1m | 1 | TRAIN-H2 | 104 | -0.2708 | -2.84 | -0.2498 |
| CDLHIKKAKE | 1m | 1 | VAL | 238 | -0.1529 | -2.71 | -0.1234 |
| CDLBELTHOLD | 5m | 1 | TRAIN-H2 | 85 | 0.3479 | 2.31 | 0.3689 |
| CDLMARUBOZU | 5m | 1 | TRAIN-H2 | 35 | 0.4890 | 2.14 | 0.5101 |
| CDLLONGLEGGEDDOJI | 1m | 1 | VAL | 177 | -0.1053 | -2.04 | -0.0758 |
| CDLCLOSINGMARUBOZU | 5m | 1 | TRAIN-H2 | 71 | 0.2646 | 1.83 | 0.2857 |
| CDLDOJI | 1m | 1 | VAL | 371 | -0.1319 | -1.76 | -0.1023 |
| CDLCLOSINGMARUBOZU | 1m | 1 | VAL | 787 | -0.0197 | -1.66 | 0.0098 |
| CDLLONGLINE | 5m | 1 | TRAIN-H2 | 78 | 0.2642 | 1.65 | 0.2852 |
| CDLSHORTLINE | 5m | 1 | VAL | 34 | -0.2319 | -1.38 | -0.2023 |
| CDLLONGLINE | 10m | 1 | VAL | 31 | -0.1071 | -1.30 | -0.0776 |
| CDLLONGLEGGEDDOJI | 1m | 1 | TRAIN-H2 | 146 | -0.1048 | -1.29 | -0.0838 |
| CDLMARUBOZU | 1m | 1 | TRAIN-H2 | 394 | -0.0098 | -1.28 | 0.0113 |
| CDLSTALLEDPATTERN | 1m | 1 | VAL | 37 | 0.2356 | 1.26 | 0.2651 |
| CDLBELTHOLD | 1m | 1 | TRAIN-H2 | 610 | -0.0335 | -1.25 | -0.0125 |
| CDLHARAMICROSS | 1m | 1 | TRAIN-H2 | 47 | -0.3307 | -1.25 | -0.3096 |
| CDLDOJI | 5m | 1 | TRAIN-H2 | 23 | 0.4561 | 1.24 | 0.4771 |
| CDLLONGLEGGEDDOJI | 5m | 1 | TRAIN-H2 | 23 | 0.4561 | 1.24 | 0.4771 |

## R4 -- money reads

Best pre-entry cut (selected on TRAIN-H2 max|day_t|, mechanical, reported both halves):
- g3_5m_climax_mins_since k=5.0 half=TRAIN-H2: n=3 dR=-1.0692 t=-17.87
- g3_5m_climax_mins_since k=5.0 half=VAL: n=1 dR=-0.8666 t=nan

Shape-gated entry (ALL-shape model, P(success15)>=tau):
| tau | scoring | half | n | dR vs book | day_t | ex_top5 | fpw |
|---|---|---|---|---|---|---|---|
| 0.5 | TRAIN-H2->VAL | VAL | 364 | 0.7410 | 7.15 | 0.6424 | 17.7 |
| 0.6 | TRAIN-H2->VAL | VAL | 290 | 0.8354 | 7.85 | 0.7427 | 14.5 |
| 0.7 | TRAIN-H2->VAL | VAL | 231 | 0.8634 | 8.06 | 0.7719 | 11.6 |
| 0.5 | VAL->TRAIN-H2 | TRAIN-H2 | 311 | 0.8309 | 8.54 | 0.7472 | 12.2 |
| 0.6 | VAL->TRAIN-H2 | TRAIN-H2 | 251 | 0.9745 | 9.77 | 0.8982 | 9.9 |
| 0.7 | VAL->TRAIN-H2 | TRAIN-H2 | 203 | 1.0234 | 10.02 | 0.9472 | 8.0 |

G7 money reads (cut/short/add gated by the k-model), top 15 by |t|:
| k | action | tau | scoring | half | n | dR vs book | day_t | ex_top5 |
|---|---|---|---|---|---|---|---|---|
| 60 | add | 0.6 | VAL->TRAIN-H2 | TRAIN-H2 | 707 | -0.6084 | -10.43 | -0.7379 |
| 60 | add | 0.7 | VAL->TRAIN-H2 | TRAIN-H2 | 693 | -0.6057 | -10.09 | -0.7341 |
| 60 | add | 0.6 | TRAIN-H2->VAL | VAL | 977 | -0.4247 | -8.86 | -0.6680 |
| 60 | add | 0.7 | TRAIN-H2->VAL | VAL | 947 | -0.4372 | -8.61 | -0.6809 |
| 45 | add | 0.6 | VAL->TRAIN-H2 | TRAIN-H2 | 676 | -0.4063 | -8.43 | -0.5297 |
| 45 | add | 0.7 | VAL->TRAIN-H2 | TRAIN-H2 | 655 | -0.4158 | -8.23 | -0.5383 |
| 45 | add | 0.6 | TRAIN-H2->VAL | VAL | 907 | -0.3312 | -6.71 | -0.5433 |
| 45 | add | 0.7 | TRAIN-H2->VAL | VAL | 884 | -0.3357 | -6.71 | -0.5477 |
| 30 | add | 0.7 | VAL->TRAIN-H2 | TRAIN-H2 | 551 | -0.3223 | -6.28 | -0.4278 |
| 30 | add | 0.6 | TRAIN-H2->VAL | VAL | 923 | -0.2115 | -5.97 | -0.3901 |
| 30 | add | 0.7 | TRAIN-H2->VAL | VAL | 889 | -0.2146 | -5.84 | -0.3917 |
| 30 | add | 0.6 | VAL->TRAIN-H2 | TRAIN-H2 | 580 | -0.2961 | -5.67 | -0.4023 |
| 20 | add | 0.6 | TRAIN-H2->VAL | VAL | 788 | -0.2415 | -4.87 | -0.3959 |
| 15 | add | 0.7 | VAL->TRAIN-H2 | TRAIN-H2 | 499 | -0.1196 | -4.07 | -0.2184 |
| 20 | add | 0.7 | TRAIN-H2->VAL | VAL | 764 | -0.2240 | -4.00 | -0.3778 |

## Verdicts vs the pass bar (dR>=+0.05R, t>=2.5 day-clustered BOTH halves/scorings, ex_top5>0 both, >=3 fills/week)

Mechanically passing cuts/gates: ['shape-gated entry tau=0.5', 'shape-gated entry tau=0.6', 'shape-gated entry tau=0.7'] --
**INVALIDATED, see self-caught correction below. Revised verdict: NOTHING passes.**

## Self-caught correction (adversarial re-read of this cell's own numbers, before any owner-facing claim)
1. **G6/ALL-shape leakage.** run_R3's `arm_groups` loop paired G6 (post-entry shape over bars fill+1..fill+15) with
   the ARM/entry-relative labels fail_any/fail10/success15 -- the scope-disclosure above CLAIMS G6 uses
   forward-from-k labels instead, but the code never actually routes G6 to those columns (`fail_after_5` /
   `success_next15_from_5` exist in 1676_features.csv but are unused by run_R3). Consequence: success15 = "MFE>=1R
   within 15 min of entry" is computed over the SAME bars (fill+1..fill+15) as g6_k15_*'s CLV/red-share/wick
   features -- near-tautological. Importances confirm it: G6's and ALL-shape's top features for success15 are
   exclusively g6_k15_5m_clv_mean / g6_k15_5m_red_share / g6_k15_1m_clv_mean (1676_reads.csv, R3 rows). This is why
   G6/ALL-shape/ALL-shape+1670 show AUC 0.74-0.80 on success15/fail10 while the pure pre-entry groups (G1-G5, no
   leakage) sit at 0.51-0.57 (chance). The "shape-gated entry" R4 read (dR +0.74 to +1.02 R, t 7-10) inherits this
   leakage directly (it gates on the ALL-shape P(success15) model) and is an artifact, not an edge. Fix for any
   re-run: route G6 k=15/k=5 through fail_after_k/success_next15_from_k in run_R3, matching G7's convention.
2. **G7's high AUC is a distance-to-target artifact, not shape.** Permutation importances for every G7_k{1,5,15}
   model (1676_reads.csv) show g7_k{k}_w5_close_R at 7-50x the weight of every other feature combined (e.g. k=15
   TRAIN-H2->VAL: close_R 0.268 vs next-best 0.002). close_R = (close_at_k - entry)/R, i.e. current distance to the
   FIXED entry+1R target -- being already close to a fixed level mechanically raises the odds of touching it in the
   next 15 minutes as k (and so elapsed/remaining time) grows, independent of candle shape. AUC climbing 0.65 (k=1)
   -> 0.79 (k=5) -> 0.90 (k=15) -> ~0.97 (k=60) tracks this mechanical effect, not a shape signal. Actual shape
   features (wick/body/CLV-without-close_R, TA-Lib flags) each carry <1% importance throughout.
3. **Revised "does shape add to path+volume": NO**, once close_R/G6-window leakage are accounted for -- the
   ALL-shape vs ALL-shape+1670 deltas (+0.006 to +0.012 AUC) were already small and are not attributable to shape.
4. **What is NOT contaminated by (1)-(2) and still reads null:** R1 (pure ARM-instant, pre-entry G1-G5) -- no cell
   clears t>=2.5 in both halves; the two same-signed |t|>=2 cells (g2_1m_n10_higher_lows, g2_1m_n10_vol_slope) are
   t~1.1-1.4 in TRAIN-H2 and only clear 2.5 in VAL, failing the both-halves rule. R2 pattern x context cells --
   nothing clears 2.5 in both halves for the same pattern. G7's money reads for "add" (causally clean: gate uses
   only bars up to k, action enters at k+1) are large and NEGATIVE throughout (dR -0.12 to -0.61 R), correctly
   excluded from the passing list on sign alone, and are a plausible genuine (if unwanted) finding: buying more
   after a model says "likely to run" means buying late/high.
**Bottom line: closing the seven holes + G7 found no real candle-shape edge on this population, pre- or
post-entry; the one apparent pass was self-caught leakage. This needs an independent rebuild (per the PREREG pass
bar) before anyone treats even the null as final -- an independent implementation should re-run R3 with G6 routed
to forward-from-k labels and re-check whether G7 AUC collapses toward 0.5-0.6 once close_R is excluded or bucketed
coarsely.**

## Adequacy
This is the first-pass build closing the seven holes + the G7 amendment; nothing here is an owner-facing claim yet -- per PREREG, any pass requires an independent reimplementation from this prose before it is reported. Multiplicity is large (R1 ~1,300 + R2 ~100 + R3 ~64 + R4 ~12 base, plus ~1,100 more from G7); the both-halves/both-scorings rule and the sign line are the protection, not any single t. MDE at this n (~2,700/half) reported per-cell in 1676_reads.csv.

Files: 1676_features.csv, 1676_reads.csv, 1676_patterns.csv, 1676_shapes.py, 1676_shapes.log.
