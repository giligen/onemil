# PREREG — cells 1,489–1,490: the RETEST-INSTANT classifier — which pullbacks recover (buy) and which continue (short)

FROZEN 2026-09-26 before any 1,481/1,482 number is read (the rebuild's partial −0.11 / −0.16 R and the 87 %/13 %
withdrawal split are the only retest numbers seen). Programme count on the HOD line: 1,488 → 1,490. Owner 9/26: "data
data data to tell apart clear losers that will continue and head down (maybe even buy short later) vs those that
statistically are more likely to go back up."

## Population and label
Population: every base fill with a 1,481 retest fill (the strictly-below rule, entry = level − $0.01, 15-minute window),
taken from the INDEPENDENT rebuild's fills (`rebuild_1481_fills.csv`, the correct limit-price convention). Label
Y = the retest long's net R′ > 0 under the 1,481 standard (report-only: target hit). Base rate per holdout.

## Features at the retest instant t_r (nothing after the retest fill print enters; each with a timestamp proof)
1. Arm-bar inheritance: the cell 1,478-L3 probability (HGB) and every 1,478 feature (they are all ≤ bar j).
2. The break itself: minutes from the base fill to t_r; the base fill's distance above the level (bps); the break bar's
   volume relative to the mean bar volume of the day through j; the high reached between the base fill and t_r as %
   of the level (how far it ran before pulling back).
3. The dip: from the tape window of the retest minute (`sip_cache_1481/`, and the bars between): number of prints ≤ the
   level before the fill print; the lowest print ≤ t_r relative to the level (bps); the dip's speed (minutes from the
   post-break high to t_r); volume of the dip bars vs the break bar; odd-lot share and mean size of the prints ≤ level
   before t_r; the NBBO at t_r (spread bps; bid depth not available); whether the bid stepped down through the level
   (bid at t_r − 5 s vs at t_r) — the tape features are NaN where the window lacks the prints (coverage reported).
4. Context at t_r: breadth at the retest minute (features_1478_B's breadth matrix at that minute), SPY return from the
   base fill to t_r (cache.db SPY bars), the number of the day's prior retests of the same level (0 for the first).
5. Decoys in a metadata-only model: the store served in 1,438, the tick-window coverage flags — VAL AUC ≤ 0.55 or VOID.

## Models and selection
As PREREG_1478: HistGradientBoostingClassifier, seed 1489, 5-fold CV inside TRAIN-H2 over the fixed grid, fit once,
applied once to VAL; logistic regression beside; threshold = the TRAIN-H2 top-tercile probability; label-shuffled placebo.
| cell | book |
|---|---|
| 1,489 BUY | the 1,481 retest long on fills with probability ≥ the threshold |
| 1,490 SHORT | the 1,480 short (entry at the bid at t_r, stop = max(break-bar high + $0.01, entry × 1.01), target 2 R, cover 15:55, shortable flag, SSR excluded, costs as 1,480) on fills with probability ≤ the TRAIN-H2 BOTTOM-tercile threshold |

## Pass bar (frozen; VAL, per cell)
Kept mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at 12/4, dropped mean < kept mean on
both holdouts, TRAIN-H2 same sign t ≥ 1, VAL AUC ≥ 0.60, placebo AUC ≤ 0.53, kept cache-only share within 5 pp of the
population's, and (1,489 only) the kept set's paired ΔR vs the base book on the same fills ≥ +0.10. TEST once for the
single best passing cell across 1,481 / 1,482 / 1,486 / 1,489 / 1,490.

## Independent check and consequences
Feature matrix rebuilt from this prose (Jaccard of the kept VAL set ≥ 0.95, means within 0.03 R); refuters' first lens
= the timestamp proof of every feature with non-zero importance (t_r is the boundary, not bar j). PASS on 1,489 → the
live engine's resting-bid entry mode gains a "retest score" gate computed from the print watch at the fill instant —
dry run 5 sessions, then $50 real orders; PASS on 1,490 → a short book proposal for the owner (borrow, SSR, his manual
shorts on the same account). FAIL → the retest is closed on both sides with the AUC on record.

## Not allowed
Any feature after the retest fill print; choosing thresholds on VAL; more than one TEST read; the builder's
fill-at-print convention.
