# PREREG — cell 1,438: the fill book as a LIVE resting order would produce it (causal arming)

Frozen 2026-09-25 ~19:40 UTC, after cell 1,427's TEST PASS (+0.330 R, t 2.12) and before any number of this cell.
Programme count 1,437 → 1,438.

## The gap found in the judge's causality trace
`trading/hod_break.py::detect` accepts bar i as the signal only if `rv_profile(cumv[i], adv20, m[i])` is inside
[rv_lo, rv_hi) — `cumv[i]` is the cumulative volume THROUGH bar i, i.e. it includes the break bar's whole volume, which
is not known when a resting order fills mid-bar at the first cross. Every other condition (consolidation of bars ≤ i−1,
level = HOD through i−1, distance from the open, last entry minute) is known at the close of bar i−1. Cell 1,427's book
therefore conditions each fill on a quantity observed after the fill. A live resting order cannot: it fills on every
cross while armed, including crosses on bars whose full-bar rv ends outside the band, and it never fills on bars the
spec took only because their full-bar volume pulled rv into the band.

## The rule (what the live engine will do; nothing else changes vs 1,427)
At the close of each bar j (≥ K+1 bars into the session, m[j+1] ≤ last_entry_minute), the order for bar j+1 is ARMED
iff: a valid consolidation ends at j (`consolidation_low`), level = HOD through j, level ≥ open × (1 + min_dist), and
rv_profile(cumv[j], adv20, m[j]) ∈ [rv_lo, rv_hi) — all through bar j. While armed, a buy-stop-limit rests at
level + $0.01 with limit level × 1.0015; it fills at the NBBO ask at the first consolidated print ≥ trigger inside bar
j+1 if ask ≤ limit; a print ≤ stop after the fill inside that bar = stopped; then the B0 path rules (stop =
consolidation low, 2 R target from the fill, 15:55 exit), measured half-spread cost. One entry per symbol-day (the
first fill); an armed bar with no cross re-arms at the next close if the conditions still hold. Report-only variant:
rv evaluated with cumv[j] + the tick volume of bar j+1 up to the fill instant.

## Sample and bar
TRAIN-H2 and VAL first (the population is now EVERY symbol-day the spec's causal superset covers, i.e. the same
symbol-days as `research/bf_zero/spec_trades.csv` / `b0_trades.csv`, not only the spec's signal bars; the minute bars
come from `data/cache.db`, the ticks from the SIP cache plus new fetches for bars that were not signals). Pass bar on
VAL: fill mean net R ≥ +0.10, day-clustered t ≥ 2, ex-top-5 % > 0, coverage ≥ 80 %, gap ≤ 5 pp, ≥ 3 fills/week. If VAL
passes, TEST is read a SECOND time (disclosed as such; the rule is a mechanical correction toward the live order, not
a fitted parameter) with the same bar. Also report the overlap with cell 1,427's fills and the outcome of the fills
that only this rule takes.

## Consequences (pre-committed)
PASS → this rule, not 1,427's, is what the HOD dry run implements for 10 sessions, then the exploration-tier proposal
at $100 risk with guardrails. FAIL → the fill-condition edge was an artefact of the rv look-ahead; closed.

## Not allowed
Changing any constant of `HodBreakParams` or the 15 bps limit; excluding symbol-days for any reason other than the
tape rail; reading TEST before VAL passes.
