# PREREG — cell 1,723: HOD live book re-decomposition at 500 fills (frozen 2026-10-30, run when `hod_break` paper fills ≥ 500, ≈ late Dec 2026)

Written NOW so the questions are not post hoc when the data arrives. No number is read before the 500th fill.

1. Decomposition as in 1,721 (gross at intended levels, entry slip, stop slip, target slip, net) on all fills and on
   the fills AFTER 1,721's window (out of sample for the 1,721 numbers). Pass for "the gross is real": gross > 0 in both
   windows and pooled t ≥ 2 (trade-clustered); report the CI.
2. Target geometry (mechanism from 1,721: the EOD-exit cohort, n 53, closed at +0.36 R with p90 +1.23 R — winners that
   never reached 2 R): replay every fill with the SAME stop and a 1.5 R target, and with a 2 R target + move-stop-to-
   break-even at +1 R, using the bars in `research/bf_zero/bars_sip.db` if they cover the dates (else from the ledger's
   exit path only — then report "not computable"). Pass: ≥ +0.10 R over the live rule in both halves (alternating dates).
3. Cuts: ONLY the three best-mechanism cuts from 1,721 (fill time of day, fills-per-day, R as % of price) at the same bar.
4. Cost floor: with entry quotes recorded (hardening item), per-fill spread cost vs slip; state how much of the 0.10 R is
   the quoted spread (irreducible for a chase entry) and how much is beyond it.
No other variants. A/B bar as in 1,721. Everything else is a new PREREG.

5. Runner-vs-fader at 14:00: the two pre-stated hold rules of PREREG_1725.md (near-the-high; momentum-intact), at its pass bar.

6. Cell 1,726 items at the larger n: passive squeeze A1/A2 and passive EOD A3 (cost lever, bar +0.02 R pooled with the live flat charged 12 bps), the uncensored 14:00->15:55 path of the R1 hold side, the B1 double-down leg (bar +0.10 R add-leg, both halves), and gate (C) for a dedicated 14:00 runner scan (>= +30 bps net in both halves).
