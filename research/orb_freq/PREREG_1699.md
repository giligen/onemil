# PREREG — cell 1,699: the second layer batch on the ORB stack (FROZEN 2026-10-01 18:30 UTC)

Owner 10/1: "think how hard I had to push… push harder." Same book, method, directions and statistics as PREREG_1698;
each layer read on top of the stack (base → RVOL tilt → add at +1 R), in dollars, both directions, Q3 weekly table.

## Layers (fixed)
 L9  add level sweep: the add at +0.5 R / +1.0 R / +1.5 R (one add, original stop) — the level that is same-signed in
     both directions with the best EV joins; also 2 units at +1 R.
 L10 lock re-read for the two-unit position: lock arm at {+1.5, +1.75, +2.0, +2.5} R → stop to {+0.5, +1.0} R on
     the COMBINED position (the live 1.75 → 0.5 was tuned for one unit); the live setting is the reference.
 L11 P1 on the stack: the P1 pool (gap 3–5 %, run ≥ 5 % by 09:35) with its regime-specific half-out at +1 R added
     as a frequency layer; the union's $, weekly P10 per fill, worst week, max drawdown vs the stack without it; and
     P1 with the tilt and the add applied to it as well (does the stack transfer to the pool?).
 L12 gapper-count tilt: the day's number of production candidates at 09:35 (known at the decision) in terciles
     (selection-half edges), multipliers 0.5 / 1.0 / 1.5 by cell mean, clamped; robustness rule as the tilts.
 L13 adds on both base and P1 at the chosen level.
 L14 the stack with compounding: risk per fill = 0.5 % of a $65K account growing with realised P&L (the ramp's
     above-water rule applied weekly), reported as the equity curve and max drawdown in $ for 2025-01..2026-09 and
     for Q3 — informational, the scale question.

## Pass rule
As 1,698: a layer joins only if paired ΔR on top of the stack ≥ +0.03 R per fill, same sign both directions, ex-top-5 %
ΔR ≥ −0.02 (tilts: ordering rule + ≥ 10 % EV/risk both ways). Drawdown layers reported for the owner.

## Output
`research/orb_freq/RESULT_1699.md` (≤ 120 lines), `1699_reads.csv`, `1699_weekly_q3.csv`, `1699_equity.csv`,
`1699_layers2.py`, `1699_layers2.log`. The agent returns ≤ 180 words.
