# PREREG — cell 1,700v: continuous-path momentum ("frog in the pan") on the guarded sleeve (FROZEN, before any number)

Why: 1,700t showed the reference's deepest drawdown was partly names whose 12-month return was one jump (AMC ×15,
ABVX, AMRN, OCGN) — removing them by a crude data guard lifted 27.2 % / −44.5 % to 29.3 % / −38.3 %. The published
mechanism (Da, Gurun, Warachka 2014): momentum built from many small same-direction days persists (investors
under-react to information that arrives continuously); momentum built from a few large days does not and reverses.
The guard is the extreme case of that idea found by accident; this cell tests the idea itself, as a family.

## Signal
Information discreteness over the formation window: ID = sign(window return) × (share of down days − share of up
days); low (negative) ID = a continuous path. Everything else = the guarded reference GREF (1700s_lowvix.py engine,
guard ON; must reproduce 29.34 % / −38.3 % / $596,394): universe, risk-adjusted score, Monday open, 1/N reset, costs.

## Family (16 cells, declared now; judged as a family, median cell recommended, never the best)
selection {A50: drop names above the cross-sectional median ID, rank the rest by the sleeve's score ·
A67: drop the top third by ID · B40: the 40 best by score, keep the N lowest ID · B60: the 60 best by score, keep the
N lowest ID} × N {20, 30} × ID window {t−252..t−21, t−126..t−21}.

## Reads per cell (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026)
CAGR, max DD, ratio, end $, Sharpe, worst year, years beating SPY /10, GREF's three deepest episodes and the cell's
depth in each, beta to SPY and alpha t, weekly correlation and holdings overlap with GREF, turnover and cost drag,
paired weekly difference vs GREF (mean, t, ex-top-5 %), by-year table for the median cell.

## Pass rule
A cell "improves" if its CAGR/DD ratio ≥ GREF's (0.77) + 0.10 AND CAGR ≥ 25 %. The family is REAL only if ≥ 12 of 16
improve AND the ratio beats GREF's in BOTH halves in ≥ 12 of 16 AND the paired weekly difference is ≥ 0 ex-top-5 % in
≥ 8 of 16 AND the median cell is shallower by ≥ 3 points in ≥ 2 of GREF's 3 deepest episodes. If real: the MEDIAN
cell goes to an independent rebuild, then a default-off flag on the paper sleeve for the owner's go. 6–11 improve:
"partial" — report the separating axis, no recommendation. Otherwise FAIL. Cells on this line: +16.

## Output
`1700v_fip.py` (reuse 1700s_lowvix.py's engine and loader; ONE process, panel loaded once), `1700v_cells.csv`,
`RESULT_1700v.md` (≤ 60 lines, adversary caveats). Through `bash scripts/research_run.sh -m 2500M`. Agent returns ≤ 150 words.
