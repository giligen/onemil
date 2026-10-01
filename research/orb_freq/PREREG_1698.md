# PREREG — cell 1,698: more layers on the ORB stack (base + RVOL tilt + add at +1 R), each read on top of the stack (FROZEN 2026-10-01 18:10 UTC)

Owner 10/1: "Love the new P&L. More. Layers?" The stack so far on the production selection (never changed): L1 the RVOL
risk tilt (1,694 Part B; both directions +20 % / +13 % EV per unit of risk), L2 the add of one unit at +1 R with the
original stop (1,687 idea 43; +0.14 / +0.28 R same-signed). This cell reads six further layers, EACH ON TOP of the
stack that precedes it (interaction, not isolation), in both directions, with Q3 2026 shown week by week in dollars.

## Book and method
The production book with real entry minutes (research/orb_freq/1693_pool_exits.py reconstruction; walker of
1687_cells.py idea43_walk for the add; tilt terciles from PARITY_1697_tilt.md) on the bar store (read-only, with a
30 s backoff when the store is busy); windows: select 2025 → test 2026 and the reverse; 2024H2 reported; $ at $375
base risk per fill; every read: paired ΔR vs the stack below it, iid and day-clustered t, ex-top-5 % ΔR, MDE, weekly
P10 PER FILL, worst week $, max drawdown $, the realised risk multiple (Σ multipliers ÷ n).

## Layers (fixed definitions)
 L3 second add at +2 R: one more unit at +2 R, original stop for all units (variants: the second add with its stop at
    +1 R = breakeven of the first add).
 L4 the add unit's stop: (a) original stop (= L2), (b) breakeven of the add (entry + 1 R), (c) the live lock applied
    to the add unit as if it were its own position.
 L5 day-risk budget: cap the day's total open base risk at B × $375, B ∈ {2, 3, 4} (fills beyond the cap are sized to
    the remaining budget, then skipped); read the worst week, max drawdown and the EV lost.
 L6 two-dimensional tilt: RVOL tercile × price band ($3–10 / 10–20 / 20–30), multipliers from the selection half's
    cell means (0.5 / 1.0 / 1.5 by tercile of expected R across the 9 cells), clamped inside the 1.5× cap; robust only
    if the 9-cell ordering agrees across directions (Spearman ≥ 0.6 on 9 ranks is attainable) and EV/risk gains ≥ 10 %
    both ways.
 L7 market-context tilt: SPY's return from the prior close to 09:35 (from the store's SPY bars) in terciles
    (selection-half edges), multipliers 0.5 / 1.0 / 1.5 by cell mean; same robustness rule.
 L8 base partial at +2 R once the add is on: sell 50 % of the BASE unit at +2 R, the add unit and the rest run with
    the live lock.

## Pass rule per layer
A layer joins the stack only if its paired ΔR on top of the stack ≥ +0.03 R per fill with the same sign in both
directions and ex-top-5 % ΔR ≥ −0.02, and (for tilts) the ordering rule above. Layers that reduce the worst week by
≥ 25 % at an EV cost ≤ 0.02 R are reported as drawdown layers (the owner decides). Q3 2026 week table for the final
stack beside base.

## Multiplicity
8 variants × 2 directions ≈ 16 paired reads + the Q3 table. Programme count: ORB line + 1.

## Output
`research/orb_freq/RESULT_1698.md` (≤ 120 lines: the layer table first, then the Q3 weekly table of the final stack),
`1698_reads.csv`, `1698_weekly_q3.csv`, `1698_layers.py`, `1698_layers.log`. The agent returns ≤ 180 words.
