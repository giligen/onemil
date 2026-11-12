# RESULT 1,740 — M1 retest bid on TRUE high-of-day levels: FAIL (TEST sealed, not read) — 2026-11-12
Programme count 1,740. Files: `1740_gen.py`, `1740_score.py`, `1740_arms.csv` (literal level), `1740_arms_b1.csv` (corrected), `1740_orders[_b1].csv`, `1740_summary[_b1].csv`, `1740_gen_check[_b1].txt`.

**Level definition defect found.** `level_from_bars` (max h with m < entry_m) INCLUDES the break bar: only 55 % / 54 % (TRAIN/VAL) of break bars have h >= level. Diagnosis: B0 entry = open of the entry_m bar, the BREAK bar is entry_m-1 (it makes a new high vs max(h, m < entry_m-1) in 90 % of signals). Corrected level (H1740_OFF=1) = max h with m < entry_m-1: break bar h >= level 96.9 % / 96.5 %, B0 chase median +8 bp (p10 -46, p90 +45) = the live 8 bp. Both reads below; the corrected one is the answer. Stop/C conventions unchanged from 1,737 (stop = min low bars j-5..j-1, floored at level x 0.985).

**Corrected level (b1), net R, day-t, n filled (fill share of arms):**
| read | TRAIN | VAL | TRAIN H1 / H2 |
|---|---|---|---|
| strict | -0.088, t -0.83, 5,583 (76 %) | -0.183, t -4.30, 3,651 (77 %) | -0.005 / -0.176 |
| 10 bp | -0.148, t -1.41, 5,371 (73 %) | -0.238, t -5.80, 3,538 (75 %) | -0.070 / -0.232 |
| 1 % under (secondary) | -0.315, t -2.16, 3,684 (50 %) | -0.401, t -7.90, 2,460 (52 %) | -0.220 / -0.412 |

Ex-top-5 %: strict -0.20 / -0.30. Capped book (8/day, 4 conc.), strict: TRAIN n 1,793 -0.218 (t -5.8), VAL n 799 -0.126 (t -2.0). Placebo pctile strict 99 TRAIN / 69 VAL. Adverse selection: filled arms' B0 R -0.31 / -0.42 vs unfilled +0.68 / +0.73 (the bid fills the pullbacks, misses the runners).
Literal (uncorrected) level: strict -0.103 TRAIN / -0.175 VAL, also negative.

**Verdict: FAIL** (needs >= +0.10, t >= 2 on TRAIN and VAL). The 1,737 +0.18 R was the sparse-cache.db level artifact (fill share 45 % there vs 76 % here). TEST untouched.
Caveats: stop includes the break bar's low (j-5..j-1); the PREREG-literal "5 bars before the break bar" (j-6..j-2) not run; queue ignored (optimistic for fills).
