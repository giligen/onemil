# PREREG — cell 1,659: MINIMUM STOP DISTANCE 1.5 % (a cost veto), paper first

FROZEN 2026-09-29 16:05 UTC after the 1,658 table and before any live number. Programme count: 1,658 → 1,659.

## Evidence (cell 1,658, the 9,911-fill population of 1,438 at the measured 13-bps round trip)
| stop distance | share of fills | win % | gross R | net R TRAIN-H2 (t) | net R VAL (t) |
|---|---|---|---|---|---|
| 0.75–1.5 % | 44 % | 36 % | −0.005 | −0.109 (−3.96) | −0.115 (−4.43) |
| 1.5–3 % | 45 % | 38 % | +0.023 | −0.009 (−0.83) | −0.068 (−2.75) |
| ≥ 3 % | 10 % | 41 % | +0.086 | −0.076 (−1.56) | +0.138 (+0.50) |
The tight-stop bucket is negative on BOTH halves at t ≤ −4: its gross is ≈ 0 and a 13-bps round trip is ≈ 0.10 R of
a 1.2 % stop — cost, not signal. No bucket is positive on both halves (the ≥ 3 % promise in VAL is contradicted by
TRAIN-H2), so nothing is SELECTED; the veto only removes a bucket that loses consistently for a known mechanism.

## Rule
`hod_break.min_r_pct: 1.5` (was 1.0), checked at arm time as today (the realized stop distance at a resting fill is
within the 0.15 % limit of the arm-time estimate — the agent's reading of `min_r_pct`). Nothing else changes.
Expected effect on the population: net −0.08 R → ≈ −0.025 R per fill (gross ≈ +0.035 R on the remaining 56 %),
fills/week ≈ 0.56 × today's. The book stays ≈ breakeven; this is a cost hygiene rule, not an edge claim.

## Consequences
Paper session from 2026-09-30 (the HOD paper account; loaded at the 12:30 UTC start). Live only when the paper
mechanics are clean AND the owner says so; the 30-fill cost gate continues with today's 10 live fills.
Not allowed: raising the floor further after reading paper results; any selection of the ≥ 3 % bucket without a
new PREREG on a fresh sample.
