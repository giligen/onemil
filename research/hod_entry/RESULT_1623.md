# RESULT -- cell 1,623: the day's own feedback gate

`PREREG_1623.md`. Base = the 9,911 fills of cell 1,438 (status=='fill', VAL union TRAIN-H2, TEST sealed/never read). outcome_R = cell 1,478's standard-cost net R (model_1478_L3_predictions.csv, matched 1:1, 0 unmatched). F(f) = mean outcome_R of the day's OTHER fills whose OWN exit_m is strictly before f's fill_min; n_res(f) = their count. G+: n_res>=2 and F>=+0.5 R (the candidate rule, "kept" below); G-: n_res>=2 and F<=-0.5 R (report-only, the mirror); remainder: n_res>=2 and -0.5<F<0.5; insufficient: n_res<2.

**Exit-minute source**: model_1478_L3_predictions.csv has no exit_m column (confirmed at runtime). Used causal_arming_causal.csv's own exit_m (complete, 0/9,911 missing, paired with the same row's fill_min). Cross-checked against fallback (2), rebuild_1481_fills.csv: present for 8973/9911 base rows, exact match 80.5% where present (the ~20% drift and the missing 9.5% are cell 1,481's own independent-rebuild disagree/no_retest population, not used here). store_served_1438 (features_1478_A.csv) agrees with model_1478_L3_predictions.csv's own copy of the same flag on 100.00% of rows.

## Main table -- G+ (kept) vs dropped, per holdout

| holdout | n kept | mean net R | t | ex-top5 | fills/wk (12/4) | n dropped | dropped mean | delta (kept-dropped) | kept cache-only % |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 331 | -0.3899 | -3.51 | -0.5174 | 5.44 | 4067 | -0.1486 | -0.2413 | 22.1 |
| VAL | 356 | 0.0490 | 0.34 | -0.0534 | 4.09 | 5157 | -0.1857 | 0.2346 | 17.4 |

## All four buckets, per holdout (G- and remainder/insufficient are report-only)

| holdout | bucket | n | mean net R | t | ex-top5 |
|---|---|---|---|---|---|
| TRAIN-H2 | G+ | 331 | -0.3899 | -3.51 | -0.5174 |
| TRAIN-H2 | G- | 1504 | -0.1918 | -3.59 | -0.3056 |
| TRAIN-H2 | remainder | 1380 | -0.1076 | -1.60 | -0.2170 |
| TRAIN-H2 | insufficient (n_res<2) | 1183 | -0.1416 | -1.61 | -0.2528 |
| VAL | G+ | 356 | 0.0490 | 0.34 | -0.0534 |
| VAL | G- | 2046 | -0.2054 | -4.49 | -0.3194 |
| VAL | remainder | 1974 | -0.2214 | -4.44 | -0.3369 |
| VAL | insufficient (n_res<2) | 1137 | -0.0883 | -1.13 | -0.1970 |

## 5-bin autocorrelation table: outcome_R of f vs F(f) (rows with F defined, n_res>=1)

| holdout | bin | n | F range | mean F | mean outcome_R | t |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 1/5 | 725 | [-1.53, -1.13] | -1.2139 | -0.2068 | -2.28 |
| TRAIN-H2 | 2/5 | 709 | [-1.13, -0.64] | -0.8739 | -0.1837 | -2.80 |
| TRAIN-H2 | 3/5 | 718 | [-0.64, -0.28] | -0.4574 | -0.1156 | -1.49 |
| TRAIN-H2 | 4/5 | 717 | [-0.28, 0.26] | -0.0501 | -0.1976 | -2.33 |
| TRAIN-H2 | 5/5 | 715 | [0.26, 1.98] | 0.9965 | -0.1397 | -1.32 |
| VAL | 1/5 | 963 | [-1.70, -1.11] | -1.2072 | -0.2779 | -5.36 |
| VAL | 2/5 | 958 | [-1.10, -0.67] | -0.8684 | -0.2090 | -3.32 |
| VAL | 3/5 | 959 | [-0.67, -0.30] | -0.4844 | -0.0959 | -1.14 |
| VAL | 4/5 | 962 | [-0.30, 0.13] | -0.1010 | -0.2464 | -3.40 |
| VAL | 5/5 | 958 | [0.13, 1.98] | 0.8023 | -0.1550 | -1.63 |

## Shuffle placebo (seed 1623, single permutation draw per holdout)

Gate applied to the OTHER holdout with each fill's day-context substituted by a randomly permuted day (own outcome_R kept real). margin = this holdout's real G+ kept mean minus the placebo's G+ kept mean; t = day-clustered t of the placebo kept sample's own mean.

| holdout | built on (shuffled) | days permuted | unshuffled by chance | placebo n | placebo mean | margin | t |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | VAL | 102 | 1 | 186 | -0.3616 | -0.0283 | -2.71 |
| VAL | TRAIN-H2 | 128 | 0 | 208 | -0.1058 | 0.1548 | -0.61 |

## Pass-bar checklist (frozen; PREREG_1623.md lines 36-39, scored on VAL)

- [ ] VAL kept mean net R >= 0.15 (got 0.0490)
- [ ] VAL day-clustered t >= 2.5 (got 0.34)
- [ ] VAL ex-top-5% > 0 (got -0.0534)
- [x] VAL fills/wk at 12/4 >= 3.0 (got 4.09)
- [ ] dropped < kept on BOTH holdouts (VAL -0.1857<0.0490; TRAIN-H2 -0.1486<-0.3899)
- [ ] TRAIN-H2 same sign as VAL and t >= 1.0 (got mean -0.3899, t -3.51)
- [ ] placebo margin >= 0.1 and t >= 2.0 (got margin 0.1548, t -0.61)
- [x] kept cache-only share within 5.0pp of 19.5% (got 17.4%)

**Verdict: FAIL**

## Caveats (read as an adversary)

- The shuffle placebo is a SINGLE permutation draw (seed 1623, as the PREREG specifies one seed, not a distribution) -- noisier than the 1,000-draw count-matched null used elsewhere in this line (cell 1,445's `null_percentile_of`); a different draw could move the margin/t meaningfully for small kept-n holdouts. Re-run with several seeds before treating a narrow pass/fail as final.
- "Margin and t" for the placebo is read here as (this holdout's real kept mean minus the placebo's own kept mean) and (day-clustered t of the placebo kept sample alone) -- matching how every other t in this line attaches to one sample's mean -- rather than a two-sample difference test; a stricter reader could ask for the latter too.
- The autocorrelation table and the G-/remainder/insufficient buckets are report-only, not part of the frozen pass bar; do not read a bin's or bucket's sign as a verdict on its own.
- exit_m is causal_arming_causal.csv's own column, not independently re-walked in this script; the cross-reference above measures agreement with cell 1,481's independent rebuild but this script does not re-derive exit_m from bars_fills_1478.db itself.
- TEST is untouched by this script (VAL/TRAIN-H2 only), per the frozen spec.


## Judge (main session, 2026-09-28 18:10 UTC) — 1,623 / 1,624 / 1,625 all FAIL; the day, the crowd and the index carry nothing
* 1,623 day-feedback gate G+: VAL +0.05 R (t 0.3, tail-carried: median −0.89 R, one day = 216 % of the kept sum),
  TRAIN-H2 −0.39 R (t −3.5) — opposite signs; the 5-bin table of outcome vs the day's resolved mean is flat. The day's
  early outcomes do not predict its later ones. Rebuild identical (Jaccard 1.0); the shuffle placebo reading differed
  between builds (ambiguous prose) — moot. Refuter: a same-minute look-ahead in the "resolved before" rule (integer
  exit bar vs fractional fill) — correcting it makes the gate no better.
* 1,624 break breadth: every bucket ≈ −0.17 to −0.18 R on both holdouts (top tercile VAL −0.17, t −3.1; bottom −0.18);
  refuter: the window included the fill's own minute and the hour adjustment still left a time-of-day tilt — neither
  helps. 65,974 arm events replayed to a 100 % match with the base file.
* 1,625 index instrument (SPY/IWM 60 min after a break-count burst): VAL SPY +1.3 bps (t 0.4), TRAIN-H2 −4.2 (t −1.6),
  sign-flipped; IWM likewise. The builder's "passing" placebo margin (+5.9 bps) was a time-of-day artefact (random
  minutes drawn at midday vs signals at 09:51); against a same-minute baseline the margin is +0.7 bps (t 0.2).
Programme count 1,625. Frames beyond the fill — the day's own feedback, the cross-section of breaks, the index as the
instrument — are closed on this population.
