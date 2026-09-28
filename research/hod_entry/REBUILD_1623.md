# REBUILD 1,623 -- the day's own feedback (causal intraday gate)

Independent rebuild from `research/hod_entry/PREREG_1623.md` prose only. cell_1623.py / cell_1623_fills.csv / RESULT_1623.md were NOT opened while writing this rebuild; cross-checking this output against those (kept-set Jaccard, mean agreement within 0.01 R) is a separate step this script does not perform.

## Data and source decisions
- Base fills: `causal_arming_causal.csv`, `status == 'fill'` (9911 rows, matches the PREREG's "9,911 fills"). `split` TRAIN == TRAIN-H2 per the PREREG; VAL is VAL; TEST is sealed (absent from this file).
- Outcome: `model_1478_L3_predictions.csv` `outcome_R` (standard-cost net R). Verified a clean 1:1 join on (day, symbol) against the base fills for all rows, with fill_min and split agreeing exactly on every row.
- Exit minute: `model_1478_L3_predictions.csv` has NO exit-minute column (header inspected: day,symbol,fill_min,split,why,outcome_R,L3,store_served_1438,hgb_prob_L3,hgb_kept_L3,lr_prob_L3,lr_kept_L3). Fallback per the task: `rebuild_1481_fills.csv` column `exit_m` is populated for only 8,973/9,911 rows -- exactly `filled == True`, cell 1481's OWN retest-fill condition, not the base rule -- so it cannot supply a complete exit-minute column. `causal_arming_causal.csv` already carries its own `exit_m` for all 9,911 fill rows, and its `net_R` matches `rebuild_1481_fills.csv`'s `base_net_R` exactly wherever both are present, confirming it IS the base rule's own exit. **Source used: `exit_m` from `causal_arming_causal.csv`, paired with `outcome_R` from `model_1478_L3_predictions.csv`** (recosting changes the R value, not which bar the trade closed on).
- Weeks spanned: {'TRAIN': 27, 'VAL': 22} (ISO-week count, for fills/week denominators).
- `rebuild_1623_fills.csv` columns: day, symbol, split, fill_min, exit_m, outcome_R, store_served_1438, F, n_res, gate_G_plus, gate_G_minus.

## Causality refuters checked
- F(f) sums only rows with `exit_m < fill_min` of f, computed inside a per-day group -- an EOD-forced exit (`why == 'eod'`, `exit_m == 955` = 15:55 ET) is that fill's own real exit minute, not a placeholder, so it resolves normally once past; a fill still open at f's entry (`exit_m >= fill_min`) is excluded by construction, never treated as resolved.
- No same-day aggregate is used anywhere except the explicitly-prior-exit subset -- F(f) never sees f itself or any fill that exits at/after f's own fill_min (asserted in code: every exit_m >= its own fill_min; 76 fills have exit_m == fill_min exactly, an instant `why == 'stop_bar'` stop on the entry bar -- real, not a data bug -- and the strict `<` comparison already excludes a row from its own resolved-set on that tie).
- Day-cohort look-ahead: the gate is fill-by-fill within a day (a running causal filter), not a day-level label computed from the FULL day and then applied backward to early fills.

## Holdout: TRAIN-H2 (n_total = 4398)

| subset | n | mean net R | day-clustered t | ex-top-5% | fills/wk (12/4 cap) |
|---|---|---|---|---|---|
| ungated (baseline) | 4398 | -0.1668 | -4.22 | -0.2795 | 50.63 |
| n_res < 2 (ungateable) | 1183 | -0.1416 | -1.61 | -- | -- |
| **G+ kept** (n_res>=2, F>=+0.5) | **331** | **-0.3899** | **-3.51** | -0.5174 | 5.44 |
| G+ dropped (remainder) | 4067 | -0.1486 | -- | -- | -- |
| G- kept (mirror, n_res>=2, F<=-0.5) | 1504 | -0.1918 | -3.59 | -0.3056 | 28.44 |
| G- dropped (remainder) | 2894 | -0.1538 | -- | -- | -- |

- G+ kept - dropped = -0.2413 R; G- kept - dropped = -0.0380 R.
- G+ kept cache-only share (store_served_1438): 22.1% (pass-bar reference 19.5% +/- 5pp).
- G+ kept top-single-day share of summed R: -7.5%.

Autocorrelation table (5 quantile bins of F, fills with n_res >= 1):

| F bin | n | mean F | mean outcome_R of f |
|---|---|---|---|
| (-1.5339999999999998, -1.126] | 725 | -1.2139 | -0.2068 |
| (-1.126, -0.643] | 709 | -0.8739 | -0.1837 |
| (-0.643, -0.278] | 718 | -0.4574 | -0.1156 |
| (-0.278, 0.264] | 717 | -0.0501 | -0.1976 |
| (0.264, 1.982] | 715 | 0.9965 | -0.1397 |

## Holdout: VAL (n_total = 5513)

| subset | n | mean net R | day-clustered t | ex-top-5% | fills/wk (12/4 cap) |
|---|---|---|---|---|---|
| ungated (baseline) | 5513 | -0.1705 | -4.40 | -0.2833 | 50.23 |
| n_res < 2 (ungateable) | 1137 | -0.0883 | -1.13 | -- | -- |
| **G+ kept** (n_res>=2, F>=+0.5) | **356** | **0.0490** | **0.34** | -0.0534 | 4.09 |
| G+ dropped (remainder) | 5157 | -0.1857 | -- | -- | -- |
| G- kept (mirror, n_res>=2, F<=-0.5) | 2046 | -0.2054 | -4.49 | -0.3194 | 31.14 |
| G- dropped (remainder) | 3467 | -0.1500 | -- | -- | -- |

- G+ kept - dropped = 0.2346 R; G- kept - dropped = -0.0554 R.
- G+ kept cache-only share (store_served_1438): 17.4% (pass-bar reference 19.5% +/- 5pp).
- G+ kept top-single-day share of summed R: 216.5%.

Autocorrelation table (5 quantile bins of F, fills with n_res >= 1):

| F bin | n | mean F | mean outcome_R of f |
|---|---|---|---|
| (-1.702, -1.106] | 963 | -1.2072 | -0.2779 |
| (-1.106, -0.668] | 958 | -0.8684 | -0.2090 |
| (-0.668, -0.296] | 959 | -0.4844 | -0.0959 |
| (-0.296, 0.133] | 962 | -0.1010 | -0.2464 |
| (0.133, 1.976] | 958 | 0.8023 | -0.1550 |

## Placebo (day-label shuffle, seed 1623)

Interpretation used (prose is terse; documented here for a re-checker): to placebo-test the gate on holdout H, the SAME F/n_res/G+ mechanism is recomputed on the OTHER holdout with that other holdout's `day` column permuted across its own rows (each row keeps its own fill_min/exit_m/outcome_R, but which day it nominally belongs to is randomized), which destroys genuine within-day resolved-before structure while preserving each split's marginal outcome distribution and day-count shape. The true holdout's G+ kept mean is compared to the shuffled OTHER holdout's G+ kept mean.

| true holdout | true G+ mean | shuffled source | shuffled n | shuffled G+ mean | margin | unpaired t |
|---|---|---|---|---|---|---|
| VAL | 0.0490 | TRAIN-H2 (shuffled) | 75 | -0.5725 | 0.6215 | 3.70 |
| TRAIN-H2 | -0.3899 | VAL (shuffled) | 92 | -0.1360 | -0.2539 | -1.53 |

## Pass bar for 1,623 (frozen, VAL) -- descriptive check only, no ship/kill verdict here

| criterion | pass? | value |
|---|---|---|
| kept mean net R >= +0.15 | FAIL | 0.0490 |
| day-clustered t >= 2.5 | FAIL | 0.34 |
| ex-top-5% > 0 | FAIL | -0.0534 |
| >= 3 fills/wk at 12/4 | PASS | 4.09 |
| dropped < kept, VAL | PASS | -0.1857 < 0.0490 |
| dropped < kept, TRAIN-H2 | FAIL | -0.1486 < -0.3899 |
| TRAIN-H2 same sign, t >= 1 | FAIL | mean=-0.3899 t=-3.51 |
| placebo margin >= +0.10 R, t >= 2 | PASS | margin=0.6215 t=3.70 |
| cache-only share within 5pp of 19.5% | PASS | 17.4 |

4/9 criteria met on this rebuild's numbers. This is a descriptive readout of the frozen bar, not a ship/kill call -- the PREREG requires an independent reimplementation to AGREE with the original cell_1623 before either is trusted, which is a separate comparison step outside this rebuild.

## Caveats (read as an adversary)
- Placebo mechanism is this rebuild's own interpretation of a terse prose line ("the OTHER holdout's days shuffled"); a different, equally defensible reading (e.g. within-day reordering on the SAME holdout) would give a different placebo number -- the true/shuffled margin should be treated as indicative, not as the frozen number, until reconciled against the original cell.
- No day-clustered SE on the placebo margin itself (an unpaired Welch t on the two kept samples is reported instead); the two samples are drawn from different holdouts of different size, so this is an approximation.
- G+ / G- kept counts are small relative to the 9,911-fill base population (see table) -- a few-fills-per-week gate is exactly the frequency risk this repo's pass bar is built to catch.
- TEST is sealed and not touched by this script (absent from every input file).
- VAL G+ kept top-single-day share is 216.5% of the summed kept R -- over 100% means the +0.049 R VAL mean is a single day plus a negative remainder, not a broad-based edge; this alone should block any read of VAL as a pass even before the t-stat (0.34) is considered.
- VAL placebo shuffled-kept n (75) is far from the true VAL G+ kept n (356) -- shuffling day labels changes how many fills end up with n_res>=2 and F>=+0.5 (days become arbitrary bags of unrelated trades), so the placebo compares differently-sized samples and its margin/t should be read as directional, not exact.
- TRAIN-H2 placebo shuffled-kept n (92) is far from the true TRAIN-H2 G+ kept n (331) -- shuffling day labels changes how many fills end up with n_res>=2 and F>=+0.5 (days become arbitrary bags of unrelated trades), so the placebo compares differently-sized samples and its margin/t should be read as directional, not exact.
