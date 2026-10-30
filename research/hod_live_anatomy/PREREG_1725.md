# PREREG — cell 1,725: can an afternoon runner be told from a fader at 14:00? (owner 2026-10-30)

EXPLORATORY at today's n (67 positions open at 14:00 ET in the 231-fill live book); the same rules are frozen into the
500-fill re-decomposition (PREREG_1723 item 5) where the bar can actually be met. No fitted model, no feature search.

## Definitions
Runner: a position open at 14:00 that exits (target or 15:55 flat) at ≥ +1.0 R. Fader: everything else open at 14:00.
Features at the 14:00 bar close, computed from the cell-1,724 bars only (no look-ahead): open P&L in R; MFE so far in R;
distance of the 14:00 close to the day's high so far, in R; afternoon volume ratio = volume 13:30–14:00 / volume in the
30 minutes after the fill.

## Two pre-stated hold rules (hold = live rule; squeeze = take the profit at 14:00 if open P&L ≥ 0, else live stop/EOD)
R1 **near the high**: hold iff open P&L ≥ +0.5 R AND the 14:00 close is within 0.5 R of the day's high so far.
R2 **momentum intact**: hold iff MFE so far ≥ +1.0 R AND the 14:00 close ≥ 50 % of MFE (has given back less than half).

## Report
Runner rate overall and under each rule's hold / squeeze sides (n, runners, rate, Fisher exact p); paired Δ R vs the
live rule for "hold R-side, squeeze the rest" with t and ex-top-5 %; the A/B halves. Pass bar (for the 500-fill run,
not today): hold-side runner rate ≥ 2× the squeeze side with Fisher p ≤ 0.01 in both halves AND paired Δ ≥ +0.10 R
with ex-top-5 % > 0. Today's output is labelled EXPLORATORY and ships nothing.
