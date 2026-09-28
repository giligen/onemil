# PREREG — cells 1,610–1,616: COST-AWARE BARRIERS on the HOD break (what got eaten in each direction, and whether re-scaling R buys it back)

FROZEN 2026-09-28 16:35 UTC before any number. Programme count: 1,609 → 1,616. Owner 9/28: "you might want to change the
Rs on the spread, do an R×0.75 or whatever got eaten in each direction — do a deep job."

## What was seen (disclosed) and the question
* Gross R is flat across spread quintiles on the 9,911 base fills (cells 1,607–1,609); net falls with the spread by
  about the spread. The 35-exit lab (base entry) and the 55-cell surface (retest entry) were all within ±0.04 R / all
  negative. First-passage on the retest surface: P(+1 % before −2 %) = 0.62 vs the driftless 0.67, P(+2 % before −1 %)
  = 0.27 vs 0.33 — both slightly BELOW driftless (a small downward tilt), never above.
* The arithmetic: on a driftless path E[gross] = 0 for every stop/target pair, so re-scaling R cannot offset a cost.
  This cell tests the premise, not the arithmetic: is the path driftless at the scale of the cost, per spread bucket?

## Data
The 9,911 base fills (`causal_arming_causal.csv` status == fill), minute bars `bars_fills_1478.db`, the per-fill costs
of the standard: entry half-spread `half_entry` (features_1478_A), the arm-time quoted spread `spread_bps_at_arm`
(features_1478_C, the causal field), the stop-limit exit standard (2.9 / 3.2 bps + 12 % tail at 94 / 76 bps) and the
EOD bid cost (11.5 / 9.7 bps). R = fill − consolidation low. TRAIN-H2 / VAL as always; TEST absent.

## Part A — the drift map (report-only, the deliverable even if every cell fails)
For each spread quintile (edges on TRAIN-H2, the causal arm-time spread) and for the whole population: the empirical
first-passage probability P(+k before −m) from the fill for k, m ∈ {0.25, 0.5, 1, 1.5, 2, 3} % of price (36 pairs; a
path that reaches neither by 15:55 counts as "neither", reported), beside the driftless value k/(k+m), with the
binomial 95 % interval; the mean signed excursion at 5, 15, 30, 60, 120 minutes after the fill (bps); the realised
entry cost (fill − level, and fill − bid at the fill instant where the tape exists) and the realised stop-exit cost
per quintile. This map says, per bucket and per scale, whether there is any drift a barrier could harvest.

## Part B — the barrier cells (paired on the same fills, standard cost, base entry)
Let c_in = the fill's entry cost (half_entry + (fill − level)) and c_out = the expected exit cost of a stop (the
stop-limit standard in price) — "what got eaten in each direction", per fill.
| cell | stop | target |
|---|---|---|
| 1,610 R×0.75 | fill − 0.75 R | fill + 1.5 R (2 × the new R) |
| 1,611 R×0.75, base target | fill − 0.75 R | fill + 2 R |
| 1,612 target + eaten | fill − R | fill + 2 R + c_in + c_out |
| 1,613 stop + eaten | fill − R − c_out | fill + 2 R |
| 1,614 both + eaten | fill − R − c_out | fill + 2 R + c_in + c_out |
| 1,615 spread-scaled | fill − R × (1 + s) | fill + 2 R × (1 + s), s = spread_bps_at_arm / (R in bps), capped at 1 |
| 1,616 quintile-optimal (report-only) | per spread quintile, the (k, m) pair of Part A with the highest TRAIN-H2 net mean | the same pair on VAL |
Walk: `sip_rebuild.walk_path` semantics on minute bars (stop first on a bar touching both; gap-through at the open;
the fill bar's low ≤ stop ⇒ stopped), 15:55 at the bid. Report per cell and holdout: n, mean net R (in the BASE R
unit and in % of price), day-clustered t, ex-top-5 %, paired Δ vs the base rule on the same fills with its
ex-top-5 %, exit mix, and the same statistics per spread quintile.

## Pass bar (frozen; VAL, per cell)
Mean net ≥ +0.15 % of price AND paired Δ vs the base ≥ +0.10 % of price with day-clustered t ≥ 2.5, ex-top-5 % of the
Δ > 0, TRAIN-H2 same sign t ≥ 1, ≥ 3 fills/week, median R ≥ 0.5 % of price. Cell 1,616 is report-only (its pair is
chosen on TRAIN-H2 per quintile — read on VAL, never selected).

## Independent check and consequences
Rebuild from this prose (Part A probabilities within 0.02, Part B ≥ 99 % of rows within 0.01 R); refuters: the
first-passage estimator (censoring at 15:55; the fill bar), the cost fields (causal arm-time spread; c_in from the
fill instant; no double-charge against the standard net R), tails and day concentration, the quintile edges on TRAIN.
PASS → the HOD engine's stop/target rule changes for the $50 run (a config `barrier_mode`) after a dry parity day.
FAIL → the barrier side is closed with the drift map on record: no re-scaling of R offsets the cost on this population.

## Not allowed
Adding barrier cells after seeing TRAIN-H2; choosing 1,616's pair on VAL; reading a HOD TEST.
