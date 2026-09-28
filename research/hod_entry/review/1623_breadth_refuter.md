# Refuter — cell 1,624, break breadth (the crowd), `PREREG_1623.md`

2026-09-28 · adversarial refuter · inputs: the builder's `cell_1624_fills.csv` and `cell_1624_arm_events_cache.csv`,
`causal_arming_causal.csv`, and the rebuild's numbers (`REBUILD_1624.md`). Every number below comes from
`research/hod_entry/review/1623_breadth_refuter_chk.py`, which is read-only, prints to stdout and runs in about 2 minutes.

## Verdict: the FAIL stands (refuted = false)

I found four defects, and none of them changes the verdict. I built the gate seven ways: the builder's version, a
window that follows the spec, B60, a count of distinct names, 15-minute cutpoints, ranks set inside each holdout, and
the rebuild's fill-only count. On VAL, the kept mean falls between −0.19 and −0.14 R in all seven, and kept minus
dropped falls between −0.04 and +0.05 R (day-bootstrap SE ≈ 0.06 R). The frozen bar needs a kept mean of at least
+0.15 R. The VAL population averages −0.17 R, so the gate would have to lift the kept fills by at least 0.32 R, which
is 5 SE or more beyond anything measured. Breadth carries no information about outcome inside an hour: the
within-hour Spearman correlation between B30 and outcome_R is −0.011 on TRAIN-H2 and −0.008 on VAL.

## Reproduction and causality of the inputs
- **Builder's numbers reproduce exactly.** B30 matches in 100.00 % of the 9,911 fills when rebuilt from the event
  cache with the builder's window. tercile30 matches 100.00 %. The VAL kept set comes out at n 2,940, −0.1726 R, t −3.10
  (the builder reports −3.08, a small-sample factor).
- **The event cache matches n_cross.** The per-symbol-day event count matches `causal_arming_causal.csv` n_cross on
  100.0 % of 33,852 symbol-days:

  | status | rows | events |
  |---|---|---|
  | fill | 9,911 | 61,505 |
  | nofill | 2,010 | 4,469 |
  | not_armed | 21,931 | 0 |

  No event comes from outside the base. Events and fills cover the same days, 2025-07-01 to 2026-05-29, TRAIN-H2 and
  VAL only, so no TEST day is used.
- **The population filter adds no day-level label.** The superset keeps a name-day when the day's high ≥ open ×
  (1 + min_dist), the high ≥ floor + 1 c, and the prior-day adv20 clears its minimum. Every armed crossing already
  satisfies all three, so counting events only from this superset leaks nothing about the day.

## Lens 1: the arm-count window (defect, verdict unchanged)
`fill_min` is fractional in 100 % of rows (for example 605.31). The builder counts m_hi ∈ [fill_min − 30, fill_min).
In whole minutes that is floor − 29 through floor: it includes the fill's own minute and leaves out minute floor − 30.
The PREREG says "in the 30 minutes before f's fill minute", and its refuter list says "B counts arms before the fill
minute".
- **Look-ahead inside the minute.** 78.5 % of fills have at least one other name's event inside their own fill minute
  (mean 4.85, max 92). That crossing can happen later in the minute than the fill.
- **The own-event caveat is wrong in minutes.** f's own event falls inside its own fill minute for 97.8 % of fills.
  RESULT's caveat that it lands "strictly before fill_min" holds in seconds but not in the spec's minutes.
- **How much B30 moves.** With the spec window [floor − 30, floor − 1], mean B30 is 64.0 against the builder's 68.3
  (Spearman 0.993). 5.4 % of tercile labels change, and the VAL kept-set Jaccard is 0.945.
- **The spec-window gate is slightly worse:**

  | holdout | kept mean | t | ex-top-5 % | kept − dropped |
  |---|---|---|---|---|
  | TRAIN-H2 | −0.1746 | −2.77 | not computed | −0.012 |
  | VAL | −0.1850 | −3.49 | −0.298 | −0.031 |

## Lens 2: "any status" and "no fills counted from after"
- **Coverage is correct.** Every armed crossing on fill and nofill name-days counts, not only the fills. not_armed
  name-days have no crossings by definition. Nothing from after the fill minute enters the spec window.
- **Chop is counted as breadth.** The builder counts every crossing: about 6.2 per filled name-day. A single choppy
  name re-crossing its high of day therefore adds many events.
- **Distinct names change nothing.** Counting distinct names in the spec window gives kept − dropped of +0.041 on
  TRAIN-H2 and −0.042 on VAL (VAL kept −0.189 R, t −3.65). The sign flips between holdouts, so there is no information.

## Lens 3: the hour adjustment (defect, verdict unchanged)
- **Hour buckets are too coarse for the first hour.**

  | fill time | TRAIN-H2 kept | VAL kept | mean R, TRAIN-H2 / VAL |
  |---|---|---|---|
  | 9:30–9:44 | 0 % (n 151) | 2 % (n 335) | −0.05 / −0.06 |
  | 9:45–9:59 | 40 % | 62 % | not listed here |

  For early fills the 30-minute window reaches back before the first arm (9:36), so B30 rises mechanically with the
  clock. The within-hour Spearman correlation between B30 and minute is +0.11 on TRAIN-H2 and +0.08 on VAL, so the
  gate is still partly a time-of-day gate. It also hurts itself: the 9:30–9:44 fills it never keeps do better than the
  population average of −0.17 R.
- **15-minute cutpoints still fail.** VAL kept is −0.1614 R (t −3.02), with kept − dropped +0.019.
- **The count drifts upward, so the VAL "tercile" is half the book.** TRAIN-H2's raw cutpoints keep 53.3 % of VAL fills.
  The monthly kept share climbs from 11 % in 2025-07 to 71 % in 2026-05.
- **A true VAL tercile still fails.** With ranks set inside each holdout, VAL keeps 1,809 fills at −0.1393 R (t −2.04),
  kept − dropped +0.046.

## Lens 4: the shuffle placebo (reporting defect, verdict unchanged)
- **The builder ran a different placebo from the PREREG's.** The builder permuted B30 within each hour inside the same
  holdout (seeds 1624/1625). The PREREG asks for "the same gate on the OTHER holdout's days shuffled (seed 1623)".
- **Its "placebo t" is not the margin's t.** The reported −3.86 / −4.33 are the t of the placebo kept set's own mean.
  The bar asks for the t of the margin.
- **The PREREG's day swap fails.** Using the other holdout's days, seed 1623 plus 199 more seeds:

  | holdout | margin | z | seed-1623 margin alone |
  |---|---|---|---|
  | TRAIN-H2 | +0.025 | +0.55 | −0.056 |
  | VAL | +0.002 | +0.02 | −0.031 |

- **500 within-hour permutations:** the margin is −0.008 on TRAIN-H2 (p 0.60) and −0.011 on VAL (p 0.71).
- **Day-bootstrap kept − dropped:**

  | holdout | builder window | spec window |
  |---|---|---|
  | VAL | −0.004 (SE 0.061, t −0.07) | −0.031 (SE 0.058, t −0.53) |
  | TRAIN-H2 | +0.004 | −0.012 |

The bar (margin ≥ +0.10 R, t ≥ 2) fails under every placebo design.

## Lens 5: tails and day concentration
- **Tails hide nothing.** VAL kept − dropped under different trims:

  | trim | builder window | spec window |
  |---|---|---|
  | none | −0.004 | −0.031 |
  | ex-top-1 % | −0.004 | −0.031 |
  | ex-top-5 % | −0.004 | −0.032 |
  | winners capped at +3 R | −0.004 | −0.031 |
  | ex-bottom-5 % | −0.005 | −0.032 |

- **The gate mostly selects days.**

  | | TRAIN-H2 | VAL |
  |---|---|---|
  | days with a kept fill | 73 of 128 | 91 of 102 |
  | kept fills on the top 10 days | 52 % | 32 % |
  | mean of per-day kept means | −0.41 R | −0.35 R |
  | kept days with a positive R-sum | 34 % | 25 % |

- **Thrust days: inside them the gate keeps the worse fills.** Thrust days here are the 10 days with the highest mean
  B30. That is a whole-day label and is not causal, so this is a description only.

  | on the 10 thrust days | TRAIN-H2 | VAL |
  |---|---|---|
  | kept fills | n 759, −0.09 R | n 874, +0.11 R |
  | dropped fills | n 184, +0.46 R | n 145, +0.49 R |

  Outside the thrust days, kept fills average −0.245 R (t −3.11) on TRAIN-H2 and −0.293 R (t −5.99) on VAL.
- **A few days carry the loss.** The five worst days hold 68 % (TRAIN-H2) and 36 % (VAL) of the kept R-sum. Without
  them, kept is still −0.07 and −0.12 R.
- None of this brings the kept mean near +0.15 R. The day label is look-ahead (the ignition lesson), not a finding.

## Lens 6: cache-only share
- **Inside the rail.** The population is 19.45 % cache-only (TRAIN-H2 20.44 %, VAL 18.66 %). The VAL kept set is 18.2 %
  (builder window) or 18.4 % (spec window), within 5 pp. The gate does not select on source: mean B30 is 75.2 for
  non-cache rows and 73.9 for cache-only rows.
- **Side note, outside this cell's scope:** on VAL, cache-only rows average +0.14 / +0.17 R and the rest −0.24 /
  −0.25 R. That is a +0.4 R gap by data source in the base population, and every cell in this series inherits it. The
  gate does not separate fills within either source:

  | source | kept | dropped |
  |---|---|---|
  | non-cache | −0.248 | −0.242 |
  | cache-only | +0.166 | +0.143 |

## Builder vs rebuild (Jaccard 0.76)
The disagreement comes from the definition, not a coding error. The builder counts every armed crossing on every name
(verified 100 % against n_cross). The rebuild counts only the fills' own arm_m, and says so. The verdict does not depend
on the choice: the rebuild's VAL kept is −0.1689 (t −3.10), and all seven constructions sit between −0.19 and −0.14 R.
This review re-verified the builder's event cache and B30 independently: n_cross 100 %, B30 100 %, terciles 100 %.

## Adequacy: what this null covers
- **Detection limit.** On VAL the day-bootstrap SE of kept − dropped is about 0.06 R. The smallest lift this test
  detects (80 % power, 5 % two-sided) is about 0.17 R. The upper 95 % bound of the observed lift is +0.11 R (builder
  window) or +0.08 R (spec window).
- **What it rules out.** Raising the traded book by the owner's +0.2 R through a tercile gate needs kept − dropped of
  about +0.30 R, which is excluded. Passing the frozen bar needs about +0.48 R, also excluded.
- **What it does not rule out:** lifts smaller than about 0.17 R, other crowd definitions (sector or peer breadth,
  index breadth), and other populations.

## Corrections for RESULT_1624.md before it is relayed (this refuter did not edit builder files)
1. **Window:** report the spec window [floor(fill_min) − 30, floor(fill_min) − 1] (numbers in Lens 1), and correct
   the caveat that the own event is "strictly before fill_min".
2. **Placebo:** relabel the "placebo t" as the t of the placebo kept set, and add the PREREG's day swap on the other
   holdout's days.
3. **Disclosures:** the VAL kept share is 53 % because the count drifts upward, and fills at 9:30–9:44 are almost
   never kept, a leftover time-of-day gate.
4. **Day selector:** disclose that the top 10 days hold 52 % / 32 % of kept fills.
