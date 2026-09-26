# REBUILD_1548 — independent rebuild of PREREG_1548.md (cells 1,548–1,549)

Built from `PREREG_1548.md` prose only. Did not open `cell_1548.py`, `test_cell_1548.py`,
`cell_1548_fills.csv` or `RESULT_1548.md`. Code: `rebuild_1548.py`. Fills: `rebuild_1548_fills.csv`.

## Population
`model_1478_L3_v2_predictions.csv`, `hgb_kept_L3 == True`: TRAIN-H2 1,466 / VAL 2,048 — matches
the PREREG's stated counts exactly. Non-kept (dropped) rows used only for the calibration line:
TRAIN-H2 2,932 / VAL 3,465.

## Part A — anatomy (kept population, both holdouts, all rows + real-SIP rows)
Drawdown = max % the price traded below `level` from the fill bar to the +5% touch (extenders) or
15:55 (non-extenders). Minutes-to-touch is extender-only (non-extenders never touch by definition).

| split | scope | group | n | dd p50% | dd p75% | mt p50 | mt p75 | breached consol low | base stopped before touch |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | all | extender | 948 | 1.02 | 2.14 | 70 | 150 | 26.9% | 26.9% |
| TRAIN | all | non_extender | 518 | 6.33 | 10.35 | – | – | 87.6% | – |
| TRAIN | real_sip | extender | 639 | 1.07 | 2.15 | 79 | 153.5 | 27.4% | 27.5% |
| TRAIN | real_sip | non_extender | 396 | 6.20 | 10.22 | – | – | 87.1% | – |
| VAL | all | extender | 913 | 1.45 | 2.74 | 62 | 143 | 31.2% | 32.3% |
| VAL | all | non_extender | 1135 | 4.54 | 6.94 | – | – | 84.2% | – |
| VAL | real_sip | extender | 643 | 1.47 | 2.74 | 64.5 | 143 | 30.5% | 31.7% |
| VAL | real_sip | non_extender | 879 | 4.54 | 6.82 | – | – | 84.5% | – |

Extenders dip shallow (median ~1–1.5% of level) and take ~1–2.5 hours to touch; non-extenders drift
much deeper (median 4.5–6.3%) with no touch by 15:55. Only ~27–32% of extenders breach the
consolidation low before touching — most of the "kept non-extenders −0.81R" loss in the PREREG's
deduction is NOT from getting stopped at the consolidation low; it is base-trade decay/EOD exit
while price never comes back. Full quantile table (p10/25/50/75/90, exit mix) is in the script's
`summarize_anatomy` output (not reproduced here for length; rerun `rebuild_1548.py` to print it).

## Part B params (TRAIN-H2, all-rows extenders, floors/caps per prose)
d = 1.021% (median dd), s = 2.137% (p75 dd), W = 120.0 min (p75 mt = 150, capped at 120).

## Part B — the two cells (VAL is the read; TRAIN-H2 and dropped-tercile are context)

| cell | split | pop | n_pop | n_fill | fill share | mean R | mean % | day-t | ex-top5% | winner-cap | fills/wk | real-SIP mean | real-SIP t | null pctile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1548 SWEEP | TRAIN | kept | 1,466 | 996 | 0.679 | **+0.482** | +0.543% | 3.62 | +0.222 | -0.069 | 36.9 | +0.490 | 3.18 | 99.9 |
| 1548 SWEEP | TRAIN | dropped | 2,932 | 2,043 | 0.697 | -0.358 | -0.404% | -7.56 | -0.596 | -0.416 | 75.7 | -0.406 | -8.48 | – |
| 1548 SWEEP | VAL | kept | 2,048 | 1,596 | 0.779 | **-0.006** | -0.007% | -0.07 | -0.291 | -0.351 | 72.5 | -0.043 | -0.49 | 99.4 |
| 1548 SWEEP | VAL | dropped | 3,465 | 2,231 | 0.644 | -0.058 | -0.066% | -0.97 | -0.346 | -0.204 | 101.4 | -0.185 | -3.25 | – |
| 1549 WIDE | TRAIN | kept | 1,466 | 1,466 | 1.000 | **+0.480** | +1.051% | 7.31 | +0.384 | +0.480 | 54.3 | +0.412 | 5.34 | 100.0 |
| 1549 WIDE | TRAIN | dropped | 2,932 | 2,932 | 1.000 | -0.372 | -0.817% | -10.86 | -0.507 | -0.372 | 108.6 | -0.445 | -13.31 | – |
| 1549 WIDE | VAL | kept | 2,048 | 2,048 | 1.000 | **-0.088** | -0.194% | -1.67 | -0.213 | -0.088 | 93.1 | -0.136 | -2.63 | 62.1 |
| 1549 WIDE | VAL | dropped | 3,465 | 3,465 | 1.000 | -0.130 | -0.283% | -3.17 | -0.257 | -0.130 | 157.5 | -0.219 | -5.18 | – |

Kept > dropped on both holdouts, both cells (calibration line holds: e.g. TRAIN 1549 +0.48 vs -0.37;
VAL 1549 -0.088 vs -0.130). Day concentration (VAL, kept): dropping the best 2 days barely moves
1548 (mean ex-best-2 = -0.062, still negative) and 1549 (ex-best-2 = -0.128) — VAL's near-zero/negative
mean is not a tail artifact, it is a genuinely flat-to-negative book.

## Pass bar (frozen, VAL) — BOTH CELLS FAIL
Required: mean R ≥ +0.15 (%≥+0.15), day-t ≥ 2.5, ex-top-5% > 0, winner-capped positive, ≥3 fills/wk,
null pctile ≥ 99, real-SIP mean ≥ +0.10 with t ≥ 2, TRAIN-H2 same sign t ≥ 1, kept > dropped both
holdouts, median R ≥ 0.5% of price.

* **1548 SWEEP**: VAL mean R -0.006 (need +0.15) — FAIL. t -0.07 (need 2.5) — FAIL. real-SIP mean
  -0.043 (need +0.10) — FAIL. TRAIN-H2 sign/t: pass (+0.48, t 3.6). Kept>dropped: pass.
* **1549 WIDE**: VAL mean R -0.088 (need +0.15) — FAIL. t -1.67 — FAIL. real-SIP mean -0.136 — FAIL
  (wrong sign). TRAIN-H2: pass (+0.48, t 7.3). Kept>dropped: pass.

Both cells clear the TRAIN-H2 and kept-vs-dropped checks but fail every VAL number that matters.
This independently reproduces the owner's stated read exactly: **TRAIN-H2 (2025H2) is strongly
positive on both trade structures (sweep entry AND base-entry/wide-stop), VAL (2026H1) is flat to
negative on both — the touch-then-path physics that the extension model predicts is a 2025H2 regime,
not a tradable edge.** Because the pattern holds under two structurally different trade wrappers
(passive sweep-fill entry with a tight stop vs. the original ask fill with the same wider stop/target),
it is not an artifact of either specific execution scheme.

## Verdict
FAIL on VAL for both 1,548 and 1,549 → per PREREG_1548's own consequence clause: **the extension
predictor is closed as a money signal on this population** (predictable at the arm bar, AUC 0.715,
but not tradable through either trade wrapper tested here). Anatomy is on record above. No live
change follows (`entry_mode: sweep_limit` does NOT ship).

## Caveats (read as an adversary)
* Smoke run (200 rows) caught a real bug — d/s were computed in % but first used as raw fractions in
  `1 - d`, driving stop/limit prices negative. Fixed before the full run; full run verified against
  the smoke run's shape (0% fill share pre-fix vs 68-78% post-fix).
* Real-SIP-only VAL n is smaller (1,214–1,962) but same sign/magnitude as all-rows VAL — not a
  store-identity artifact.
* Count-matched null pool = kept population's `outcome_R` (base trade, pre-existing) restricted to
  the filled cohort's days, 1,000 draws, seed 1548. VAL null percentiles (99.4 / 62.1) look strong for
  1548 only because the baseline `outcome_R` on the kept population is itself very negative
  (~-0.10R per the PREREG); beating a negative baseline is not the same as clearing the absolute
  +0.15R / t 2.5 pass bar, which both cells miss.
* Did not implement the halt refuter or the decoy-cohort-split refuter explicitly (budget); day-drop
  and real-SIP-split refuters are done above and do not change the verdict.
* This file was NOT diffed against the original builder's `cell_1548_fills.csv` (deliberately not
  opened, per task). A Jaccard/row-match check against that file is a separate step for whoever holds
  both outputs.
