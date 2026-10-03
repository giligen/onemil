# PREREG — cell 1,703i: the guarded sleeve + a small BTC-trend weight (FROZEN 2026-10-03, before any number)

1,703e/h judged the BTC trend as a 50/50 stack component and it failed (50/50 ratio 0.68 vs GREF alone). Nobody would run
it at 50 %. Owner's standing rule (9/29): sleeves that trigger on different days stack — judge the portfolio SUM at realistic
weights and the shared tail. Weekly correlation BTC-trend vs sleeve ≈ 0.17.

## Fixed specification
Sleeve = GREF daily equity (`research/momentum_weekly/1700u_curves_daily.csv` column `guard`, 2017-01 → 2026-09).
Component = BTC trend rule, TWO forms from 1,703h (no new parameter): mom_20 and sma_100, daily, 10 bp per switch, calendar
days (the sleeve's equity is carried flat on non-trading days). Portfolio weights w ∈ {0, 10, 20, 30} % to the component,
rebalanced to target every Monday (the sleeve's own rebalance day), 1 bp cost on the rebalance flow. Cells: 2 forms × 3
non-zero weights = 6 (+6, 1,703i-1…6). Window 2017-01 → 2026-09; halves 2017–21 / 2022–26.

## Reads
CAGR, max DD, CAGR/DD, worst week, worst month, weekly P10, green-week share; the sleeve's 10 worst weeks and the component's
return in those same weeks (shared tail); for each weight the change vs w = 0 in CAGR, max DD and ratio.

## Pre-committed rule
A weight is RECOMMENDED for the paper stack only if CAGR/DD ≥ GREF's + 0.05 AND max DD not worse than GREF's by more than
1 pt AND both halves show the ratio ≥ GREF's half ratio AND the component's mean return in the sleeve's 10 worst weeks ≥ 0.
If no weight passes: the BTC trend is a personal-holding matter for the owner, not a sleeve; close the stack question.

## Output
`research/known_strategies/1703i_stack.py`, `1703i_cells.csv`, `1703i_worst_weeks.csv`, `RESULT_1703i.md` ≤ 40 lines.
Through `bash scripts/research_run.sh -m 2000M`. Agent returns ≤ 150 words.
