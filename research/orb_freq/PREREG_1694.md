# PREREG — cell 1,694: where ORB's money actually is — runners, who gets the risk, and letting them run (FROZEN 2026-10-01 15:30 UTC)

Owner 10/1 15:25 UTC: "unacceptable. act as a true researcher, find the money." The sweeps asked one question in
many forms (can a filter or exit raise the mean) and every honest answer was "the body is zero, the money is in the
tail". This cell asks the tail's questions directly, on the production book with real entry minutes (the 1,679/1,693
reconstruction, validated to the minute) and the completed bar store. Both directions (select 2025 → test 2026;
select 2026 → test 2025), 2024H2 as the out-of-regime read; MDE beside every t; $ at $375 risk per fill.

## Part A — let the runners run (the whole production book; NOT in 1,693, which had no whole-book exit rows)
Exits, fixed: A1 target 3 R (lock as live); A2 target 4 R; A3 NO target — hold to the 15:45 close with the live lock;
A4 half at +2 R, rest no target; A5 target 3 R with the lock moved to breakeven at +2 R; A6 target 2 R (production,
reference). Reads: mean R, t, ex-top-5 %, the SHARE of fills ≥ 2 R / ≥ 3 R / ≥ 5 R captured, give-back saved vs
continuation gained decomposition, weekly P10 per fill, strong-week gap (scripts/cadence_bar.py), $/yr.

## Part B — who gets the risk: a risk tilt, not a filter (sizing is linear; EV per unit of risk is the object)
Slice means under the PRODUCTION exit (1,693's slice rows with exit E1) define, on the selection half only, a tilt:
for each admission feature (price band, prior-day volume tercile, 5-min range tercile, RVOL tercile, gap band,
pre-market $ vol tercile) the expected R by slice → risk multiplier m ∈ {0.5, 1.0, 1.5} by tercile of expected R
(bottom/mid/top). Variants: B1 single feature (6 reads), B2 additive score across the six features (tercile edges on
the selection half), B3 B2 capped at total weekly risk = production's. Read on the test half: EV per unit of risk
(ΣR·m / Σm) vs production (ΣR / n), $/yr at the SAME total weekly risk budget as production, ex-top-5 % of the tilted
book, weekly P10 per unit of risk, strong-week gap, max drawdown in $. Both directions; a tilt is ROBUST only if the
slice ORDERING agrees across directions (Spearman of slice ranks ≥ 0.6) and the test-half EV/risk gain ≥ +10 %.

## Part C — runner anatomy and the missed runners
C1 the fills that reached ≥ 2 R and ≥ 3 R: their arm-time profile (every admission feature + time of first ≥ 1 R,
minutes to peak) vs the rest; the pre-entry classifier for "≥ 2 R runner" (HistGradientBoosting, defaults, both
scorings, placebo) — AUC and the lift of the top decile of P(runner) in mean R and in runner share.
C2 the MISSED runners: every candidate the pipeline admitted but did not enter (vetoed by PDR / G1 / Q1 / spread /
slot / score) in 2025–26, with its counterfactual outcome under the production entry and exit: the share of ≥ 3 R
outcomes among the dropped vs the taken, by veto; the $ left on the table per veto per year; the tail share of the
dropped set if it were traded at half size.

## Pass bar
Part A: an exit replaces production's only if paired ΔR ≥ +0.05 R with day-clustered t ≥ 2.0 in BOTH directions and
the ≥ 3 R capture share rises (that is the mechanism) — then rebuild → paper as one mechanics change. Part B: a
ROBUST tilt (ordering agreement + ≥ 10 % EV/risk gain both directions) → rebuild → paper via per-pool/per-slice risk
multipliers. Part C: descriptive; a veto whose dropped set's runner share ≥ the taken set's and whose counterfactual
$/yr is positive in both halves becomes a pre-declared gate-relaxation cell (not a change by itself).

## Multiplicity
A: 6 exits × 3 windows; B: (6 + 2) tilts × 2 directions; C: ≈ 20 reads. Stated.

## Output
`research/orb_freq/RESULT_1694.md` (≤ 160 lines: A table, B table, C tables), `1694_reads.csv`, `1694_runners.csv`,
`1694_missed.csv`, `1694_money.py`, `1694_money.log`. The agent returns ≤ 200 words.
