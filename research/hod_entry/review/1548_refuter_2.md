# Refuter 2 — cells 1,548–1,549, lens: statistics and artifacts

Recomputed from `cell_1548_fills.csv` only (script `review/1548_refuter_2_chk.py`, day-clustered t,
day-block bootstrap 2,000 seed 1). Model AUCs from `model_1478_L3_v2_predictions.csv`.

## Verdict: FAIL stands (not refuted). Three interpretive claims in the result are wrong.

### VAL kept, every cut is <= 0 (both cells)
| cell | cut | n | mean R | t | ex-top5 | ex-top1 | drop best 2 days | cap +3R |
|---|---|---|---|---|---|---|---|---|
| 1548 | all | 1641 | -0.025 | -0.29 | -0.33 | -0.09 | -0.08 | -0.41 |
| 1548 | real-SIP | 1249 | -0.029 | -0.31 | -0.33 | -0.09 | -0.09 | -0.41 |
| 1548 | cache-only | 392 | -0.013 | -0.09 | -0.32 | -0.06 | -0.11 | -0.42 |
| 1548 | live cap 12/day, 4 conc. | 1065 | -0.001 | -0.01 | -0.29 | -0.06 | -0.05 | -0.40 |
| 1549 | all | 2048 | -0.113 | -2.15 | -0.25 | -0.14 | -0.15 | -0.11 |
| 1549 | real-SIP | 1522 | -0.157 | -2.99 | -0.30 | -0.19 | -0.19 | -0.16 |
| 1549 | live cap | 1034 | -0.145 | -2.63 | -0.28 | -0.17 | -0.17 | -0.15 |
No positive VAL kept number exists to be tail- or day-carried. Fills/week under the cap 48 (1548) / 47 (1549);
the cap binds hard (1641 -> 1065), and the builder's mean/t are on the UNCAPPED set (only fills_wk is capped) — the capped
book is no better. Monthly: every VAL month negative/flat (1548 2026-05 +0.09 the lone positive), every TRAIN month positive.

### Defect 1 — "regime, not edge" is not supported: TRAIN is in-sample to the MODEL, not only to d/s/W
`cell_1478.py` refits the HGB on all of TRAIN-H2 (`final.fit(Xv, yv)`) and the kept flag on TRAIN rows is that model's
own training-row prediction. In-sample TRAIN AUC = 0.920 (real-SIP 0.928) vs VAL 0.715; kept precision TRAIN 64.7 %
vs VAL 44.6 %. The TRAIN +0.49 R (t 4–7) on both cells and the TRAIN kept-dropped gap of +0.86/+0.88 R are memorisation,
not a 2025H2 regime. The TRAIN->VAL collapse is exactly what an overfit fit predicts; the data cannot distinguish
"regime" from "overfit" and the overfit explanation is mechanically certain. The PREREG itself says TRAIN is not
evidence. Do not relay "worked in 2025H2, regime" to the owner; say "in-sample only; out of sample negative".

### Defect 2 — "kept > dropped holds on VAL, the model's lift is real" is overstated
VAL kept-minus-dropped (day-block bootstrap):
- 1548 all +0.055 R, 95 % CI [-0.09, +0.20], P(<=0) 0.22; 1549 all +0.032, CI [-0.05, +0.11], P(<=0) 0.22 — NOT significant.
- real-SIP: 1548 +0.171 [0.02, 0.33] P 0.017; 1549 +0.081 [-0.00, 0.16] P 0.026 — modest, marginal.
- cache-only: 1548 -0.80 [-1.22, -0.37]; 1549 -0.39 [-0.61, -0.17] — REVERSED.
The AUC 0.715 (real-SIP 0.728) lift on the LABEL is real; its translation into R under these wrappers is small and
lives only in real-SIP rows.

### Defect 3 — cache-only rows are an artifact cohort; the only positive VAL numbers live there
VAL DROPPED cache-only rows earn +0.78 R (1548, t 5.3) and +0.40 R (1549, t 6.5) while real-SIP dropped rows earn
-0.20/-0.24. TRAIN cache-only kept 1549 +0.68 vs real-SIP +0.41. Every positive out-of-sample number in this programme's
fills is cache-only; cache-only precision of the kept flag is higher (0.513 vs 0.422 VAL) consistent with the 0.58 decoy
AUC. Any future cell on this population must be read on real-SIP rows only.

### Null percentile is not a selection null
The null draws n kept fills' BASE outcome from the kept set. For 1549 VAL n = 2,048 = the whole kept set, so the null is
the base mean (-0.095) with resampling jitter; percentile 0 just says wide-stop < base on the same fills. For 1548 the
null compares a different trade structure (sweep) on a subset against the base trade; percentile 100 says sweep (-0.025)
> base (-0.095), i.e. beating a negative baseline. The checked "null >= 99" box on 1548 is not evidence of edge.

### SWEEP is a lottery structure
Target payoff is a constant +5.79 R (d, s fixed); TRAIN mean +0.49 with winner-capped (+3 R) -0.13 — even the in-sample
number dies under a cap. VAL target share 13.2 % kept vs 5.0 % dropped; stop share 79.6 %.

### d, s, W
Fitted on TRAIN-H2 extenders (builder excludes 80–83 no-touch rows; rebuild does not, +8 % on d/s). Neither parameter set
turns VAL positive (rebuild -0.006 / -0.088). Verdict insensitive.
