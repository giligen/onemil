# PREREG — cell 1,669: fast failures — anatomy, prediction at minute 1–2, and the value decomposition of cutting (FROZEN 2026-09-29 19:45 UTC)

Owner, 2026-09-29 19:40 UTC, on 1,668's "failure is predictable but not tradable": "sounds strange, check yourself, maybe
move some of the params/duration… focus on a specific sub-set of failure, e.g. those that stop out in sub 5 min or even
sub 2 min… find an interesting bucket/population of failures and use them for both the model and the candle story."

## What 1,668 established and what it did not
1,668 cut at k ∈ {3, 5, 10, 15} on P(any later stop-out) with τ fitted on the training half: AUC 0.63–0.67 out of sample,
paired ΔR −0.05…+0.02 R. It did NOT (a) look at k = 1–2, (b) condition on how much of the stop is still ahead when the
cut fires, (c) separate FAST failures from slow bleeds, (d) decompose the cut's value into saved loss, forgone gain and
cost. This cell does those four things, nothing else.

## Population and cost
1,663_features join, primary r_pct ≥ 1.5 % (n 5,506), unfloored beside; halves = `split`; bars from the completed
store (own day 100 %); cost as 1,668 (entry 7 bps in net_R, stop 6, target 0, EOD 11, early cut 6).

## Part 1 — anatomy of failure (descriptive, both halves)
For every fill that exits at the stop: minutes from fill to stop (share ≤ 1, ≤ 2, ≤ 5, ≤ 10, ≤ 30, > 30; median), MFE
before the stop in R, whether the price re-crossed the level before stopping, the fill bar's CLV and range/ATR, the
first post-fill bar's CLV, return and volume ratio. Same table for target exits and EOD exits. Then the TA-Lib 61 fire
rates on bars fill−2..fill+1 for the three exit classes, split by failure speed (≤ 5 min vs > 5 min). This is the
"candle story" read: which bar shapes precede a fast failure, with counts.

## Part 2 — the fast-failure model
Labels: FF2 = stop-out within 2 min of the fill bar; FF5 = within 5 min; FF10 = within 10 min. Decision instants: the
close of the fill bar (k = 0) and the close of bar fill+1 (k = 1). Features at k = 0: arm-time features (r_pct, ATR %,
level vs open / VWAP, cumulative volume / ADV20, minutes since open, level age), the fill bar's CLV, body share, range/ATR,
volume ratio, close vs fill price in R, the 61 TA-Lib flags on bars through the fill bar; at k = 1 add the same for bar
fill+1 and progress-per-unit-volume. Model: sklearn HistGradientBoostingClassifier, max_iter 200, defaults, n_jobs 1;
TRAIN-H2 → VAL and the swap; within-day label-shuffle placebo; permutation importances (n_repeats 5) on the scoring half.
Reads per label × k × scoring: AUC, and precision/recall at P ≥ 0.5, 0.6, 0.7, 0.8 (fixed thresholds, not fitted).

## Part 3 — the value decomposition of a cut (the "check yourself" read)
Cut rule C(label, k, τ): sell at the open of bar fill+k+1 when P ≥ τ, τ ∈ {0.5, 0.6, 0.7, 0.8} FIXED (no fitting), and
the variant C′ that also requires the current loss ≤ 0.25 R (most of the stop still ahead). For each rule and half:
paired ΔR vs the base, iid and day-clustered t, ex-top-5 % ΔR, MDE, share cut, AND the decomposition per cut trade:
saved loss on true positives (stop R − cut R), forgone gain on false positives (their base R − cut R), cost (6 bps in
R), each as a mean and as a total contribution to ΔR, so the sentence "predictable but not tradable" is replaced by
numbers: how much a right cut saves, how much a wrong cut forgoes, and the precision needed to break even
(break-even precision = (forgone + cost) / (saved + forgone + cost), reported beside the achieved precision).

## Pass bar
A cut rule ships to paper only if paired ΔR ≥ +0.05 R, t ≥ 2.5, ex-top-5 % ΔR > 0 on BOTH out-of-sample scorings
(VAL and the swap), placebo ≈ 0. Any pass → independent rebuild from this prose before the owner sees a number. A null
states the break-even precision vs achieved precision and the MDE first.

## Multiplicity
Part 2: 3 labels × 2 k × 2 scorings × 5 reads = 60. Part 3: 3 × 2 × 4 τ × 2 variants × 2 scorings = 96 paired reads.
Programme count on the HOD line: > 2,100.

## Not allowed
Fitting τ or k; using bars after the decision instant; conditioning on the exit type in any feature; pooled-only
numbers; reporting the training-half ΔR as a result.

## Output
`research/hod_entry/RESULT_1669.md` (≤ 160 lines: anatomy tables, candle fire-rate table by failure speed, model AUC /
precision table, the decomposition table, verdicts, adequacy), `1669_anatomy.csv`, `1669_reads.csv`, `1669_per_fill.csv`,
`1669_fast_failure.py`, `1669_fast_failure.log`. The agent returns ≤ 150 words.
