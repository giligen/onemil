# PREREG — cell 1,700p: are the two mid-week leads real? Neighbours of 1,700l M5 and M4b (FROZEN 2026-10-02 18:30 UTC)

1,700l left two cells that improved both CAGR and max DD on the headline but whose paired weekly lift is negative
ex-top-5 %: M5 gap-down exit with replacement (28.5 % / −40.1 % / $562K) and M4b Wednesday-open rebalance (29.3 % /
−41.1 % / $596K). 1,700o showed how to settle such a lead: the conditional hedge's 32 neighbours split 16/16 and its
median neighbour was the reference → a favourable draw. Same method here, fixed now:

## Family G — gap-down exit (a held name opening ≥ g % below the prior close is sold at that open)
g ∈ {4, 6, 8, 10, 12} × slot handling {replaced by the next-ranked name the same morning, left in cash until Monday}
× re-entry {allowed at the next Monday reset, barred for 4 weeks} = 20 cells (M5 = 8 %, replaced, allowed).

## Family D — rebalance day and time
Day ∈ {Mon, Tue, Wed, Thu, Fri} × time {open, close} = 10 cells, signal through the prior close in every case
(reference = Monday open; M4b = Wednesday open). Read as a placebo family: a real day-of-week effect needs its
neighbours (Tue and Thu) to lean the same way.

## Reads per cell
CAGR, max DD, ratio, end $ from $50K, Sharpe, the five 1,700j episodes' depths, turnover and cost, halves 2017–2021 /
2022–2026 (ratio vs REF in each), paired weekly difference vs REF (mean, t, ex-top-5 %), and for family G the count of
exits, the share of exits where the sold name was lower one and four weeks later (the mechanism check).

## "Real" rule (per family, as 1,700o)
≥ 75 % of the family's cells improve BOTH CAGR and max DD vs REF, both halves improve the ratio in ≥ 75 % of cells,
and (family G) the sold names are lower four weeks later in ≥ 55 % of exits. If real, the recommendation is the
family's MEDIAN cell, never the best. Otherwise the lead is recorded as a favourable draw and the sleeve stays plain.
Then, only for families judged real: the joint cell (median G + median D) is read once.

## Output
`1700p_neighbours.py` (reuse 1700l_midweek.py's lean loader and engine), `1700p_cells.csv`, `RESULT_1700p.md`
(≤ 70 lines). ONE process through scripts/research_run.sh -m 2500M. The agent returns ≤ 130 words.
