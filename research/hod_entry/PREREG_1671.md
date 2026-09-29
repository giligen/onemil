# PREREG — cell 1,671: raw-sequence classifier (MiniRocket) on the same labels, against the feature model (FROZEN 2026-09-29 20:10 UTC)

Owner, 2026-09-29 20:00 UTC: can a fine-tuned time-series foundation model learn "failure within N minutes" from our
labelled fills? Decision (act-as-owner): before any GPU fine-tune, the cheap sequence baseline on raw bars must beat the
feature model on identical labels and splits. If it does not, a foundation model is not started.

## Population, labels, splits, cost
Exactly 1,670's: the 1,663 join, primary r_pct ≥ 1.5 %, halves = `split`, trades open at k, label = stop-out after k,
k ∈ {0, 1, 2, 5, 10}; the completed bar store (own day 100 %, SPY backfilled). Cost as 1,668–1,670.

## Inputs to the sequence model (no hand-crafted features)
For each fill and k: the multivariate minute sequence from bar fill−30 to bar fill+k (padded at the day's open), channels
= open, high, low, close expressed in R units relative to the fill price ((x − fill) / (fill − stop)), volume / the
day's mean bar volume so far, and SPY close in bps relative to its value at the fill bar. Sequences of unequal length
are left-padded with the first value (flagged with a mask channel).

## Model
sktime MiniRocket (multivariate, default 10,000 kernels, random_state 0) → RidgeClassifierCV (default alphas) — no
tuning, no epochs, CPU. TRAIN-H2 → VAL and the swap; within-day label-shuffle placebo; the same GBM-on-features result
from 1,670 is the comparator on the identical (fill, k) rows.

## Reads
R1 out-of-sample AUC per k and scoring for MiniRocket vs the 1,670 ALL-features GBM vs a stacked model (GBM on the
   1,670 features + MiniRocket's decision score as one extra feature); placebo AUC.
R2 the money: the cut at FIXED τ ∈ {0.5, 0.6, 0.7, 0.8} on the calibrated probability (isotonic on the training half),
   paired ΔR vs holding from k onward, iid and day-clustered t, ex-top-5 % ΔR, MDE, the 1,669 decomposition and the
   break-even vs achieved precision. Both scorings.

## Pass bar
A (k, τ) cut ships to paper only under 1,670's bar (paired ΔR ≥ +0.05 R, t ≥ 2.5, ex-top-5 % > 0 on BOTH scorings,
placebo ≤ 0.55 AUC) → independent rebuild before the owner sees a number. The foundation-model question is decided by
R1: a GPU fine-tune cell is opened only if MiniRocket or the stack beats the GBM by ≥ 0.03 AUC on both scorings at
some k AND the achieved precision at that k is within 0.10 of the break-even precision. Otherwise the answer to the
owner is "no, and here is why", with the table.

## Multiplicity
R1: 5 k × 3 models × 2 scorings = 30 AUC reads; R2: 5 × 4 × 2 = 40 paired reads. Programme count: > 2,400.

## Not allowed
Tuning kernels, alphas or τ; sequences that extend past bar fill+k; any exit-type information; pooled-only numbers.

## Output
`research/hod_entry/RESULT_1671.md` (≤ 120 lines), `1671_reads.csv`, `1671_scores.csv` (fill_id × k: MiniRocket score,
GBM P, stacked P), `1671_minirocket.py`, `1671_minirocket.log`. The agent returns ≤ 150 words.
