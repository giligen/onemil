# RESULT -- cells 1,491-1,492: the shallow stop (bars-only re-walk, PREREG_1491.md)

Base = the 9,911 `causal_arming_causal.csv` fills, entry unchanged; base comparison = cell_1478.build_outcome (amendment-slip-substituted). TEST does not exist in these files (never scored). Six books = 3 stops (level x (1-s), s in {0.25%,0.50%,0.75%}) x 2 targets (SCALP = 2*R_s; ASYM = the base target, unchanged reward).

| book | holdout | n | mean_net_R_own | mean_net_%price | base_%price | delta_%price | t_delta | ex_top5_% | fills/wk | R_s_%price_median | shippable | winner_capped_% | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SCALP_025 | TRAIN-H2 | 4295 | -1.306 | -0.406 | -0.265 | -0.140 | -1.87 | -0.460 | 159.07 | 0.302 | False | -0.406 |  |
| SCALP_025 | VAL | 5384 | -1.347 | -0.418 | -0.243 | -0.175 | -2.07 | -0.473 | 244.73 | 0.300 | False | -0.418 | False |
| ASYM_025 | TRAIN-H2 | 4295 | -1.227 | -0.382 | -0.265 | -0.117 | -1.72 | -0.581 | 159.07 | 0.302 | False | -0.382 |  |
| ASYM_025 | VAL | 5384 | -1.381 | -0.428 | -0.243 | -0.186 | -2.40 | -0.613 | 244.73 | 0.300 | False | -0.428 | False |
| SCALP_050 | TRAIN-H2 | 4302 | -0.626 | -0.351 | -0.267 | -0.085 | -1.18 | -0.430 | 159.33 | 0.551 | True | -0.351 |  |
| SCALP_050 | VAL | 5393 | -0.692 | -0.389 | -0.243 | -0.146 | -1.80 | -0.468 | 245.14 | 0.550 | True | -0.389 | False |
| ASYM_050 | TRAIN-H2 | 4302 | -0.590 | -0.331 | -0.267 | -0.065 | -1.08 | -0.576 | 159.33 | 0.551 | True | -0.331 |  |
| ASYM_050 | VAL | 5393 | -0.687 | -0.384 | -0.243 | -0.141 | -2.06 | -0.626 | 245.14 | 0.550 | True | -0.384 | False |
| SCALP_075 | TRAIN-H2 | 4305 | -0.377 | -0.305 | -0.267 | -0.038 | -0.60 | -0.407 | 159.44 | 0.801 | True | -0.305 |  |
| SCALP_075 | VAL | 5396 | -0.448 | -0.364 | -0.242 | -0.122 | -1.65 | -0.469 | 245.27 | 0.800 | True | -0.364 | False |
| ASYM_075 | TRAIN-H2 | 4305 | -0.381 | -0.307 | -0.267 | -0.040 | -0.82 | -0.573 | 159.44 | 0.801 | True | -0.307 |  |
| ASYM_075 | VAL | 5396 | -0.437 | -0.355 | -0.242 | -0.112 | -2.00 | -0.628 | 245.27 | 0.800 | True | -0.355 | False |

## Dip-depth table (report-only, retest tape)
Coverage: 9910/9911 base fills (100.0%).

| bucket | n | share % | base outcome_R |
|---|---|---|---|
| 0-25 bps | 1790 | 18.1 | -0.017 |
| 25-50 bps | 2245 | 22.7 | -0.194 |
| 50-75 bps | 2191 | 22.1 | -0.249 |
| 75-inf bps | 2948 | 29.7 | -0.455 |

## Judge (main session, 2026-09-26 17:35 UTC) — FAIL, all six books; the stop side is closed

Every shallow stop (0.25 / 0.50 / 0.75 % under the level, scalp or asymmetric target) is WORSE than the base on both
holdouts in % of price (VAL Δ −0.11 to −0.19 % of price, t −1.7 to −2.4; TRAIN-H2 same sign). The 0.25 % books also
fail the R-vs-spread rail (median R_s 0.30 % of price). The independent rebuild from the prose agrees on the sign and
the ranking of all six books and is MORE negative (its fill-bar stop was priced gap-through-at-open, a convention bug
against the PREREG's "conservative" wording; non-stop rows agree 98.4 %, all rows 82.7 % within 0.01 R). The ≥ 99 %
agreement bar was not met, and no re-run is ordered: no book is within 0.3 % of price of the pass bar in either
implementation, so the convention difference cannot flip the verdict. Dip-depth table (report-only, 9,910 fills):
≤ 25 bps 18 % of fills at −0.02 R, 25–50 bps −0.19 R, 50–75 bps −0.25 R, > 75 bps 30 % of fills at −0.46 R — the loss
of the base book is the deep-dip third; a stop under the level cuts it at nearly the same loss while clipping the
shallow dips that recover. Stop side now measured: level, consolidation low, 2.5 % floor, shallow grid. Count 1,492.
