# HOD exit lab — PASS 2 verdict (cells 1,380–1,388). 2026-09-22

Workflow `hod-exit-lab-pass2` (Sonnet scorer, 3 Sonnet refuters, Sonnet critic; 408K tokens). Inputs: the pass-1
cache (`paths.parquet`, `signals.parquet`, `b0_trades.csv`). Baseline **B0** = the live rule on the complete causal
population (12,135 signals): **net −0.221 R TRAIN, −0.304 R VAL** (the causal study's −0.04 R gross after ≈ 0.15 R
cost and the stop/target physics). Tables: `CELLS_PASS2.md`, `cells_pass2.json`, `trades2/`.

| cell | rule | TRAIN ΔR | VAL ΔR | VAL t | halves | dropped cohort R (TR/VAL) | verdict |
|---|---|---|---|---|---|---|---|
| 1,380 T1 | drop r_pct < 1.5 % | +0.040 | +0.055 | −0.27 | +0.04/+0.03 | −0.30 / −0.40 | fail (t) |
| 1,381 T2 | drop r_pct < 2.5 % | +0.057 | +0.129 | 0.57 | +0.08/+0.00 | −0.24 / −0.34 | fail (t); monotone dose |
| 1,382 M1 | signal in prior 5 sessions | +0.055 | +0.107 | 1.11 | +0.12/−0.00 | −0.25 / −0.37 | fail; decays H1→H2 |
| 1,383 M2 | prior-5-session signals net + | +0.039 | +0.064 | 1.12 | +0.08/+0.01 | −0.23 / −0.32 | fail |
| 1,384 E1 | retest entry (≤ 15 min) | +0.130 | +0.118 | 18.8 (paired) | +0.15/+0.11 | — | **refuted at verification** (see below) |
| 1,385 O1 | overnight runner | −0.023 | +0.007 | 1.41 | −0.04/−0.00 | — | fail; worst night −10.5 R, P1 −4.9 R |
| 1,386 O2 | O1, exit 10:00 | — | — | — | — | — | VOID: no next-day minute bars in bars_sip (signal days only) |
| 1,387 S1 | 8 concurrent | −0.081 | −0.000 | 0.02 | −0.09/−0.07 | −0.17 / −0.30 | fail; more slots = more of a negative edge |
| 1,388 S2 | daily cap 20 | −0.048 | −0.015 | 0.31 | −0.03/−0.05 | −0.20 / −0.30 | fail |

## E1 — the one paired pass, and why it does not stand
Two refuters cleared it (look-ahead: causal, entry at the next open after the retest low; tail: ex-top-5 % ΔR still
+0.08 with t 13.8). The fills/cost refuter killed the READING, per the PREREG's own clause: the 21–24 % of signals
that never retest had B0 net **+0.119 R on TRAIN** (−0.039 on VAL) — the best trades do not pull back (the
passive-entry adverse-selection memory, confirmed). Corrected full-book effect (retest book vs the full B0 book,
unpaired): **+0.046 R**, and the retest book is still **−0.175 / −0.258 R** net. A smaller loss is not an edge.

## What pass 2 establishes
* On the complete causal population the HOD-break long loses ≈ 0.2–0.3 R net per trade at the live exit, and no
  trade-level filter (cost floor, symbol memory), entry tweak (retest), overnight extension or slot rule moves it
  above zero; the largest honest shift is +0.13 R (T2, n 1,093 VAL, t 0.6).
* The cost floor works in the direction the law predicts (dropped cohorts −0.3/−0.4 R, monotone dose) but the
  kept cohort is still −0.18 R: cost is not the only leak.
* The dry run's +0.06 R over 44 trades (SE ≈ 0.23 R) is statistically indistinguishable from this population.
* Cadence C1/C2/C7 were N/A in the scorer's block on the unslotted population (critic gap) — irrelevant to a
  negative book; must be wired before any positive cell is reported.
Programme count 1,388. The remaining lever is pass 1 (the exit) and its drift profile.
