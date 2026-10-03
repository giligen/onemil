# PREREG — cells 1,700x: hedging the guarded sleeve's drawdown with a short overlay (FROZEN 2026-10-03, before any number)

The guarded sleeve (GREF 28.7 % / −38.1 % band cost; 1,700t/u) fails every drawdown repair tried so far: VIX gates (1,700s/u),
residual and path momentum (1,700r/v), stack components (1,703a–i). The untried lever is a SHORT overlay — a research
question about the book's shape, not a decision to trade short (that stays the owner's).

## Fixed specification (daily engine of 1700s_lowvix.py / 1700u, guarded universe, same costs and Monday open timing)
Long leg = the guarded top-20 book exactly as 1,700u `guard` (unchanged). Overlay families, each at hedge ratios
h ∈ {25 %, 50 %} of the long book's equity, reset every Monday open with the long leg:
- X-beta: short SPY (one name, borrow 0.5 %/yr, 2 bp per traded dollar).
- X-loser: short the 20 LOWEST-scored names of the same guarded eligible universe (12-1 return ÷ vol, equal weight), borrow
  3 %/yr, cost as the long leg (band 15–20 bp); a name that fails the hygiene guard is ineligible on both sides.
Cells: 2 families × 2 ratios = 4 (+4, 1,700x-1…4). Window 2017-01 → 2026-09, halves 2017–21 / 2022–26 (as 1,700u).
Short proceeds earn nothing (conservative). No stop, no gate, nothing else varies.

## Reads
CAGR, max DD, CAGR/DD, worst week, weekly P10, green-week share, the depth of GREF's three deepest episodes under each cell,
the overlay's own return in GREF's 10 worst weeks (shared tail), both halves, and the overlay's annual cost + borrow.
Known hazard, read explicitly: momentum crashes (2020-04..06, 2021-02, 2026-04) — the loser leg rallies hardest; print the
loser overlay's worst week and the month-by-month 2020-03..2020-07 and 2026-03..2026-05.

## Pre-committed rule
A cell is RECOMMENDED for the paper sleeve (as a design to show the owner, who decides on shorting) only if max DD improves
by ≥ 8 pt vs GREF AND CAGR/DD ≥ GREF's + 0.10 AND both halves' CAGR/DD ≥ GREF's half AND worst week not worse than GREF's by
more than 2 pt. Otherwise: the short overlay is closed as a DD repair at these ratios, and the sleeve's drawdown is accepted
as the price of its return (sizing is the remaining lever).

## Output
`research/momentum_weekly/1700x_hedge.py` (reuse the 1700u engine head verbatim), `1700x_cells.csv`, `1700x_curves_daily.csv`,
`RESULT_1700x.md` ≤ 50 lines. Through `bash scripts/research_run.sh -m 2500M` (service down today — Saturday; one process).
Agent (Sonnet, real code) returns ≤ 150 words. Any cell meeting the rule is rebuilt independently before the owner sees it.
