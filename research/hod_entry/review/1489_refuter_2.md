# Cells 1,489 / 1,490 — refuter 2 (lens: statistics and artifacts)

Recomputed from `model_1489_predictions.csv` + `features_1489.csv` (8,973 rows, 0 duplicate keys, net_R' identical
in both files). Script: scratchpad `ref2.py` (read-only). Headline FAIL of 1,489 on VAL **stands and is stronger**
under this lens; several reported sub-claims are defective.

## Reproduced
- Threshold chosen on TRAIN only: top-tercile of TRAIN p_hgb = 0.40378 (matches 0.4038); kept flag 100 % consistent.
  Applied to VAL it keeps 41.3 % (VAL's own tercile would be 0.438) — distribution shift, see SPY-NaN below.
- VAL kept n 2,070, mean −0.0642 R, day-t −2.61 (iid −2.14), ex-top-5 % −0.173, ex-top-1 % −0.085, drop best 2 days
  (4/02 +67 R, 5/20 +20 R) −0.113. Negative under every tail cut; the best day alone carries +67 R of a −133 R book.
- VAL AUC 0.515 (LR 0.497). Deciles of p_hgb on VAL: net R' flat (−0.19 … +0.04), no monotone.

## Defects in the reported result
1. **TRAIN `passes_bar: true` is hard-coded**: `cell_1489.py` line ~174 `passes = holdout_name != "VAL" or (...)` —
   every non-VAL row passes by construction. On the actual clauses the TRAIN row FAILS the cache-share rule
   (kept 24.64 % vs pop 19.18 % = 5.46 pp > 5 pp).
2. **TRAIN-H2 sanity clauses scored on IN-SAMPLE predictions** (AUC 0.954 vs CV 0.643): "TRAIN same sign t ≥ 1" and
   "dropped < kept on both holdouts" are meaningless as scored; out-of-fold predictions were required. CV folds are
   `StratifiedKFold(shuffle=True)` not grouped by day, so the 0.643 CV AUC is itself inflated by same-day rows
   sharing context features.
3. **Paired ΔR +0.22 is NOT a classifier property**: VAL paired ΔR kept +0.225, dropped +0.221, population +0.223
   (real-SIP only: kept +0.222 vs dropped +0.217). It is the 1,481 retest-vs-break mechanism on every fill; the
   classifier adds +0.004 R. The result note "would clear +0.10" misattributes it. (ex-top-5 % of ΔR +0.155, so the
   mechanism lift itself is not tail-only.)
4. **Sparse-cache provenance lives in the POPULATION outcome**: cache-only rows (store_served_1438 = 1, zero rows in
   bars_sip.db) earn net R' +0.279 (TRAIN) / +0.147 (VAL) vs real-SIP −0.206 / −0.139 (Y rate 0.47/0.43 vs
   0.31/0.34). The only positive kept slice is cache-only (VAL +0.176, t 1.5, n 369); **on real-SIP rows the kept
   set is −0.116 R, day-t −3.82 (n 1,701)** — no positive kept mean survives on real SIP. This is the cell-1,427
   artifact footprint and it also inflates the 1,481/1,486 retest books (~+0.05 R at a 17 % cache-only share).
   VAL kept cache share 17.8 % vs pop 17.4 % (clause passes on VAL).
5. **SPY context NaN for 100 % of Apr–May 2026** (`ctx_spy_ret_fill_to_tr`, `arm_spy_ret_open_to_j`: 5 % NaN TRAIN,
   50 % VAL): SPY-NaN rows are kept at 55 % vs 27 % for SPY-present rows — the kept-set composition is driven by
   HGB NaN routing. Adequacy check: on SPY-present VAL rows AUC is 0.501 and kept −0.165 R (t −2.5), so the null
   is not a missing-feature artifact.
6. 1,490 SHORT is untested on VAL (0/5,016 matched outcomes) — a data-coverage void, not a measured fail; the
   PREREG's "FAIL → retest closed on both sides" cannot be claimed for the short side.

## Placebo / decoy
Placebo 0.495 and decoy 0.543 as reported (not refit here). Univariate store_served → Y AUC 0.47 on VAL (0.53
inverted): the decoy signal is real but under the 0.55 VOID line.

## Verdict
1,489 FAIL stands (stronger on real-SIP rows). Defects: hard-coded TRAIN pass flag, in-sample TRAIN clauses,
misattributed paired ΔR, population-level cache-only outcome inflation, 1,490 unscored.
