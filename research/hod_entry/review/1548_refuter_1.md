# Refuter 1 — cells 1,548–1,549 — LOOK-AHEAD AND OBTAINABILITY

Verdict under review: both cells FAIL on VAL (1548 SWEEP −0.025 R, 1549 WIDE −0.113 R).
Checks script: `research/hod_entry/review/1548_refuter_1_chk.py` (read-only; own trade walk).
Result: **NOT REFUTED.** The only material obtainability defect is FAVOURABLE to the cell; fixing it makes the FAIL deeper.

## Look-ahead
- d, s, W: `compute_dsw` filters `split == 'TRAIN'` & extender only; the RESULT's TRAIN extender quantiles are
  q50 0.9441 % = d and q75 1.9713 % = s; W = min(120, TRAIN p75 151.7 min) = 120. VAL anatomy tables are computed
  earlier in `main` but d/s/W do not read them. No VAL leak into the parameters.
- Kept flag: `PRED_CSV = model_1478_L3_v2_predictions.csv` (the amendment-3 arm-bar-closed build). v2 vs the leaky v1:
  kept flag agrees on 90.1 % of 9,911 rows, prob corr 0.92, so it is the different, corrected file. Kept counts 1,466/2,048 as the PREREG says.
- Level/consolidation low: over 9,911 fills no pre-fill-bar high exceeds the level by > 0.05 %, and no stop equals
  only the fill-bar low (0 rows). Both are known at the arm bar.
- The L3 label (cell_1478_L3.build_label_l3) uses 04:00–20:00 bars, so after-hours highs count as an extension. That explains the
  163 "no touch found" extenders. The label only feeds Part A and d/s/W, not the trades.

## Obtainability
- **Fill-bar sweep fills (favourable defect).** The resting bid may fill in the fill bar through its low, but at minute
  resolution that low can be a PRE-BREAK print. The population consists of fills, so membership depends on the break that follows.
  VAL kept: 276 of 1,641 sweep fills are in the fill bar, and 133 of them have the bar OPEN already below the limit, which means they were
  bought before the break. With fill-bar fills excluded (the window starts at fill_m+1):
  VAL 1548 = **−0.189 R, t −2.32** (n 1,592), down from −0.025; TRAIN = +0.366, down from +0.498. The FAIL is deeper.
- 1549 fill-bar low (pre-fill part) is conservative: only 18 VAL same-bar stops. The upper bound (walk from fill_m+1) is
  −0.108 R, t −2.03, which still fails.
- Target through-print: the builder uses h >= target and the PREREG says "exceeds". The strict `>` rule changes nothing (identical stats).
- Stop first on a bar touching both, gap-through at the open: correct in walk_path. The sweep's entry-bar stops (108 VAL) are
  assumed fill-then-stop, which is conservative.
- Anatomy drawdown includes the touch bar's full low, which can include post-touch prints. That is conservative on d/s and immaterial here.
- Halts: missing bars are walked through, and a reopen below the stop takes the open (gap-through). No imputation.

## Costs / units
Stop cost 0.88·2.9+0.12·94 = 13.83 bps (TRAIN) / 11.94 (VAL). EOD cost 11.5 / 9.7 bps, the RESULT_1443 eod means. Both are charged as
exit_px·bps/1e4/R. The 1549 entry cost is half_entry/R once, and the passive 1548 entry has no cost. The units are correct.

## Recompute
300 random filled trades were re-walked independently and agree to ≤ 4e-5 R. The only residual comes from the 4-decimal d/s used here.
Aggregate re-run: VAL 1548 n 1,641 −0.0249 t −0.29; VAL 1549 n 2,048 −0.1134 t −2.14, an exact match. TRAIN 1548 n 1,027
+0.498 vs builder 1,026 +0.487: one row differs (an R ≤ 0 or missing-bar exclusion), which is immaterial.
