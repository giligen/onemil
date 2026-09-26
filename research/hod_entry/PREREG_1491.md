# PREREG — cells 1,491–1,492: the SHALLOW STOP — exit a few ticks under the break instead of at the consolidation low

FROZEN 2026-09-26 before any number. Programme count on the HOD line: 1,490 → 1,492. Owner 9/26: "like several ticks
below the break as the exit… just based on what you said."

## What was seen (disclosed)
87 % of fills dip through the level within 15 minutes; those earn −0.35 R at the consolidation-low stop, the 13 % that
do not earn +0.67 R; 23 % of the withdrawers still hit +2 R. Exiting at the first print through the level was measured
worse than holding (cell 1,480 long leg, −0.04 / −0.10 R) because that exit costs ≈ 0.6 % of price with the measured
35 bps slip on a 1.6 %-of-price R. The question left open: a stop a FEW TICKS under the level — deep enough that the
shallow shakeouts do not trigger it, shallow enough that the failures cost 0.5 % instead of 1.6 %.

## Rule (paired re-walk on the 9,911 base fills; entry unchanged = the base fill)
Grid: stop = level × (1 − s) for s ∈ {0.25 %, 0.50 %, 0.75 %} (three stops), each with two targets:
* 1,491 SCALP: R_s = fill − stop_s, target = fill + 2 R_s (a small target close to the entry);
* 1,492 ASYMMETRIC: target = the BASE target (fill + 2 × the consolidation R — unchanged reward, smaller risk).
Six books (3 × 2). Walk on `bars_fills_1478.db` with `sip_rebuild.walk_path` semantics; the fill bar's low ≤ stop_s ⇒
stopped (conservative); stop exits carry the verified stop-limit slip (2.9 / 3.2 bps filled + 12 % tail at 94 / 76 bps,
in units of the book's own R); target = limit; 15:55 exit at the bid (1,443 EOD means). Report every book in TWO units:
its own R and % of price (the owner's question is about dollars, not R), paired against the base on the same fills (ΔR
in the BASE's R units and Δ% of price), with the exit-type mix. Report-only: the dip-depth table from the retest tape
(`sip_cache_1481/`, the 1,481 outputs): the distribution of the lowest print below the level within 15 minutes (bps),
and the base outcome by dip-depth bucket (≤ 25 bps, 25–50, 50–75, > 75) — the shape of the shakeout vs the failure.
The R-must-exceed-the-spread rail: any book whose median R_s < 0.5 % of price is reported but flagged NOT SHIPPABLE.

## Pass bar (frozen; VAL, per book)
Paired Δ(% of price) vs the base ≥ +0.15 % with day-clustered t ≥ 2.5 AND the book's own mean net % of price ≥ +0.20 %
(≈ a positive book, not only a lift), ex-top-5 % > 0, ≥ 3 fills/week, TRAIN-H2 same sign t ≥ 1, winner-capped at +3 R
of the base still positive. TEST once for the single best book if it passes.

## Independent check and consequences
Rebuild from the prose on the bar store (≥ 99 % within 0.01 R); refuters: fill-bar convention, slip units, the target
definition per book, tail dependence. PASS → the live stop rule gets a `stop_mode: shallow` (stop = level × (1 − s)) —
dry 5 sessions with the parity ledger, then $50 real orders under the 9/25 fixes. FAIL → the stop side is closed: level,
consolidation low, floor at 2.5 %, and the shallow grid all measured.

## Not allowed
Adding stops or targets to the grid; choosing s on VAL for TEST; reading TEST more than once.
