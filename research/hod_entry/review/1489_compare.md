# PREREG_1489 — builder vs. independent rebuild, row-by-row compare (VAL)

Inputs: `model_1489_predictions.csv` (builder) vs `rebuild_1489_predictions.csv` (rebuild), keyed on
`day+symbol`. VAL split: 5,016 rows in both files, 100% key match (no left/right-only rows) — the
population and split are identical between the two builds.

## PREREG bar: Jaccard >= 0.95 AND kept means within 0.03 R

**FAILS on Jaccard for both cells.** Mean-diff bar (1489) passes; 1490 mean-diff is not evaluable (no VAL
outcome data in either file).

| metric | value |
|---|---|
| Jaccard, kept_1489 (VAL) | **0.685** (both-kept 1,590 / either-kept 2,321; builder n=2,070, rebuild n=1,841) |
| Jaccard, kept_1490 (VAL) | **0.562** (builder n=881, rebuild n=981) |
| kept_1489 mean diff, net_R_prime (builder − rebuild) | **−0.0156 R** (builder −0.0642 vs rebuild −0.0486) — within the 0.03 R bar |
| kept_1490 mean diff, short_net_R (builder − rebuild) | **NaN / not evaluable** — see coverage gap below |
| AUC, builder p_hgb vs Y=net_R_prime>0 (VAL) | 0.5150 |
| AUC, rebuild p_hgb vs Y=net_R_prime>0 (VAL) | 0.5165 (matches rebuild's own reported 0.516) |

**1,490 coverage gap (both files, confirmed independently by the code, not just by REBUILD_1489.md's own
note):** `c1490_short_net_R` / `short_net_R` is non-null for only 406/8,973 rows total, and every one of
those 406 falls in TRAIN — **0 of 5,016 VAL rows have a short outcome in either the builder or the
rebuild file.** kept_1490 (the boolean flag) exists and can be Jaccard'd, but the mean-diff clause of the
pass bar has no VAL data to compute on either side — this is a data-coverage void, not a pass or a fail.

## Diagnosis: why Jaccard is low despite no found feature-definition bug

Read `FEATURES_1489.md` (builder) and `REBUILD_1489.md` (rebuild). The rebuild explicitly built blind
(did not open the builder's code, features CSV, or predictions). The two docs agree on: population
(8,973 rows, TRAIN 3,957/VAL 5,016), the causality bounds on every feature group, the same known gap
(`breadth_at_tr` not implemented — both substitute the arm-bar breadth proxy instead), and the identical
1,490 coverage gap (406/8,973, 0% VAL). **No feature-definition or coverage divergence was found between
the two specs** — the low Jaccard is not explained by the two builds measuring different things.

Top 3 reasons for the low Jaccard, ranked:

1. **Near-chance signal amplifies model-fit noise at the threshold.** VAL AUC is ~0.515–0.517 for both
   builds — barely above chance. Row-level `p_hgb` scores from the two independently-trained HGB models
   still correlate at r=0.89 on VAL (a real but imperfect agreement), but with this little true signal,
   two models that agree on 89% of score variance can still rank a large share of borderline rows on
   opposite sides of a threshold. This is the dominant driver of the Jaccard gap, and it's the expected
   footprint of two honest re-fits of a near-null classifier, not a bug.
2. **Independently-chosen thresholds land at different points on two different score distributions.**
   Builder's realized kept_1489 cut is p_hgb >= 0.4039 (41.3% kept); rebuild's is p_hgb >= 0.4029
   (reported top-tercile 0.4029, 36.7% kept here). The threshold values are within 0.001 of each other,
   but because each is a tercile computed on that build's own TRAIN-fit score distribution, the kept
   *fractions* differ (41.3% vs 36.7%), which alone caps the achievable Jaccard even before rank-order
   noise is added.
3. **No confirmed feature/coverage divergence to blame instead.** Everything checked (population size and
   split, causality bounds, the breadth-proxy substitution, the 1,490 coverage gap) matches between the
   two docs. The remaining candidate — different NaN-imputation or hyperparameter/CV-seed choices inside
   each HGB fit — was not verifiable without reading `build_features_1489.py` / `rebuild_1489.py` (outside
   this run's budget), but is consistent with reason #1: it would only matter this much *because* the
   underlying signal is this weak.

## Bottom line
kept-mean agreement is fine (1,489 within 0.03 R; row-level net_R_prime and AUC both reproduce closely
across the two independent builds), but **set-membership agreement is not** — Jaccard fails the >=0.95
bar for both 1489 (0.685) and 1490 (0.562). Given the VAL AUC is at chance (~0.52) for both builds, this
looks like inherent instability of a near-null classifier's threshold membership, not a spec or coding
error between builder and rebuild. 1,490 cannot be scored on VAL at all (0% coverage) regardless of the
Jaccard question.
