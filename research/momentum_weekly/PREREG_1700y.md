# PREREG — cells 1,700y: volatility-scaled momentum (Barroso & Santa-Clara 2015) on the guarded sleeve (FROZEN 2026-10-03)

Published form: scale the momentum book's exposure by target vol ÷ its own trailing realized vol; the paper reports the
crashes largely removed and Sharpe roughly doubled on US momentum 1927–2011. Not yet tested on this sleeve (1,700s/u scaled by
the VIX term structure, an external gauge — this cell uses the book's OWN realized vol, the documented rule). Owner's
standing ask: documented strategies first.

## Fixed specification (operates on the guarded daily curve `1700u_curves_daily.csv` column `guard`; no engine change)
Exposure for the coming week, set every Monday open = min(cap, target ÷ σ126), σ126 = annualised std of the sleeve's daily
returns over the trailing 126 trading days (6 months, the paper's window), known at Friday's close. target ∈ {20 %, 30 %,
40 %}; cap ∈ {1.0 (no leverage), 1.5 (leverage, REPORTED not recommended — margin cost 6 %/yr on the excess)}. The unexposed
share earns nothing. Changing exposure trades (new − old) × equity of the book at the band cost 17.5 bp per traded dollar, on
top of the sleeve's own costs already inside the curve. Cells: 3 × 2 = 6 (+6, 1,700y-1…6). Window 2017-01 → 2026-09 (the
first 126 days at exposure 1.0), halves 2017–21 / 2022–26.

## Reads
CAGR, max DD, CAGR/DD, worst week, weekly P10, green-week share, GREF's three deepest episodes under each cell, mean and
range of exposure, exposure in GREF's 10 worst weeks (did it de-risk BEFORE them?), extra turnover per year, both halves.

## Pre-committed rule (same bar as 1,700x)
RECOMMENDED for the paper sleeve only if max DD improves ≥ 8 pt vs GREF (−38.3 %) AND CAGR/DD ≥ 0.77 + 0.10 AND both halves'
ratio ≥ GREF's half (0.67 / 0.99) AND worst week not worse by more than 2 pt — among cap 1.0 cells only; a cap 1.5 cell that
passes is reported as "needs the owner's leverage decision". Otherwise closed: the sleeve's DD is not repaired by its own vol.

## Output
`research/momentum_weekly/1700y_volscale.py`, `1700y_cells.csv`, `1700y_exposure.csv` (weekly exposure per cell),
`RESULT_1700y.md` ≤ 45 lines. `bash scripts/research_run.sh -m 1500M`. Agent (Haiku) returns ≤ 150 words.
