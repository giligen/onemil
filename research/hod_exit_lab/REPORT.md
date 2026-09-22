# HOD exit lab — PASS 1 verdict (cells 1,359–1,379) and the two-pass synthesis. 2026-09-22

Workflow `hod-exit-lab` (Haiku inventory, Sonnet harness + scorer, Sonnet critic; 393K tokens) on the complete
causal HOD-break population: 12,135 signals with cached per-minute paths (`paths.parquet`, one DB pass after the
20:05 UTC blackout), B0 = the live rule (stop, +2 R target, 15:55 open exit, measured half-spread + 2 bp both
legs). **B0: net −0.221 R TRAIN (n 7,315), −0.304 R VAL (n 4,695).** Every cell is scored PAIRED against B0 on the
same trades; the paired SE on VAL is ≈ 0.02 R, so an exit effect of +0.05 R would have shown.

## Exit cells — 0 of 14 pass; every variant is within ±0.04 R of B0
| cell | TRAIN ΔR | VAL ΔR | VAL t | halves |
|---|---|---|---|---|
| X1 no target | −0.000 | −0.005 | −1.0 | −0.00/+0.00 |
| X2 target +1 R | −0.042 | −0.016 | 0.2 | −0.05/−0.03 |
| X3 target +3 R | +0.014 | +0.003 | −0.0 | +0.01/+0.02 |
| X4 target +5 R | −0.001 | +0.006 | −0.3 | −0.01/+0.01 |
| X5 breakeven lock @ +1 R | −0.017 | −0.018 | −1.3 | −0.02/−0.01 |
| X5b ORB lock 1.5 → 0.5 R | −0.006 | −0.014 | −1.5 | −0.01/−0.00 |
| X6 trail MFE − 1 R | −0.028 | −0.020 | −0.8 | −0.04/−0.02 |
| X7 time stop 60 min | −0.027 | −0.004 | −0.3 | −0.05/−0.00 |
| X7b time stop 30 min | −0.036 | −0.011 | −0.6 | −0.04/−0.03 |
| X8 50 % @ +1 R + runner | −0.033 | −0.009 | −0.4 | −0.05/−0.02 |
| X9 VWAP exit | −0.002 | −0.016 | −1.1 | +0.01/−0.01 |
| X10 close at 12:00 | −0.097 | −0.001 | −0.6 | −0.18/−0.00 |
| X11 stop 1.5× (baseline-R) | −0.008 | +0.000 | −0.8 | −0.01/−0.01 |
| X12 stop 0.75× (baseline-R) | −0.008 | −0.011 | −0.5 | −0.02/−0.00 |

## Condition cells — 0 of 7 pass
D1 SPY above open +0.024/+0.014 (dropped −0.28/−0.34); D2 uncrowded −0.061/+0.026; D3 no-lunch −0.109/+0.011;
D4 HMM calm −0.031/−0.021; D5 first-signal VOID (n_prior capped at 20 in 81 % of rows); W1 week gate −0.024/+0.002
(halves −0.25/+0.10); W2 daily kill −0.120/+0.035. Nothing same-signed and ≥ +0.10 on both splits.

## The drift exhibit (DRIFT.md) read together with the cells — this is the finding
Unmanaged paths: mean best excursion +1.85 R, mean worst −1.65 R, medians +1.25 / −1.2 R, median time to the
peak 80–90 min; 22–24 % of signals reach +1 R and close ≤ 0, giving back 3.3–3.6 R from the peak; the mean
unmanaged path is only +0.08 R at 2 h and +0.12 R at 4 h with the median below zero most of the day.
**Every exit rule lands on the same number because the path after entry carries no usable information**: a
±1.7 R excursion around a +0.1 R drift is a random walk in R units, and any stop, target, lock, trail, time or
VWAP exit on a random walk has the same expectation, minus cost. This is the same message as AUC 0.51 on the entry
features and the placebo shorts (only generic drift): the signal "new high of day on a gapper", as this population
defines it, does not predict the next hours. Exit design cannot rescue a signal with no information.

## Caveats (read as an adversary)
* Reproduction gate: B0 matches the study's PUBLISHED slotted baseline (−0.212) within 0.009 R but is 0.063 R below
  the study's raw-population figure (fill convention: target buffer 1.002 and 0.1 % stop slip baked into
  features.csv). The absolute level carries ±0.06 R; the paired deltas do not.
* 125 of 12,135 signals (1 %) are absent from the B0 book (paths incomplete near session boundaries); logged, not
  material at this effect size.
* Cadence C1/C2/C7 came out N/A on the unslotted population (the scorer's label wiring); irrelevant for a negative
  book, must be fixed before any positive cell is reported.
* The live dry run's +0.06 R over 44 trades (SE ≈ 0.23 R) is consistent with this population.

## What is different enough to be worth running next (the population, not the exit)
1. **A new information channel** — the owner's order-flow-imbalance idea: bars carry nothing, the quote/trade
   tape at the break minute is the only untested source. Needs Databento MBP-1 windows for the signal minutes
   (spend to be approved).
2. **A new signal definition** — pre-market-high break (a level the crowd watches) or a break after ≥ 20 min of
   consolidation at the high (the flag lineage), each a NEW population under its own PREREG, with the R-vs-spread
   gate and the drift exhibit as the first deliverable.
3. Not worth running: any further exit/filter/entry/slot variant on this population (37 cells say the same thing).
Programme count 1,388.
