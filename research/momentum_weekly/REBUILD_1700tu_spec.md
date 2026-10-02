# Independent rebuild — the guarded sleeve and the half-size gate (prose spec, 2026-10-02)

Purpose: the guarded reference (cell 1,700t) and the half-size gate (cell 1,700u, `VIXratio|w252|p20|half`) each come
from ONE build. Before those numbers are final for the owner they are rebuilt from this prose by a builder that has
NOT read the first implementation. The builder MAY read and extend `REBUILD_1700_sleeve.py` (the earlier independent
build of the plain sleeve, reconciled in `RECON_1700_sleeve.md`) and MUST NOT open `1700j_frontier.py`,
`1700l_midweek.py`, `1700p_neighbours.py`, `1700r_residual.py`, `1700s_lowvix.py`, `1700u_gate_guarded.py`,
`1700u_yby.py`, `1700v_fip.py`, `G_dump.py`, `H_dump.py`, or `trading/momentum_sleeve.py`.

## The plain sleeve (already rebuilt — keep as is, with the two reconciliation fixes in RECON_1700_sleeve.md)
Universe each signal date: prior close ≥ $10, 20-day average dollar volume ≥ $200M, ≥ 273 bars of history, name
exclusions by WORD-boundary match (ETF, ETN, FUND, TRUST, WARRANT, UNIT …), delisted names included, SPY excluded.
Signal: return from 252 bars back to 21 bars back ÷ the standard deviation of the last 252 daily returns. Top 20.
Every Monday (first session of the week) at the OPEN: all 20 names reset to 1/20 of equity; signal from the prior
session's close. Costs 5 bps + a spread proxy per traded dollar, as the reconciled build. $50K, 2017-01 → 2026-09.

## Guard (new)
On a signal date a name is ineligible, BEFORE ranking, if inside its last 273 bars up to and including the signal
date it has (a) a one-day close-to-close move above +200 % or below −75 %, or (b) more than 10 calendar days between
two consecutive bars. The first bar of a name carries no move.

## Half-size gate (new)
Daily closes of VIX and VIX3M from the CBOE history files
(https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv and VIX3M_History.csv).
ratio = VIX ÷ VIX3M. Percentile on a date = among the previous 251 ratios, the share strictly below that date's
ratio plus half the share equal to it (needs ≥ 126 observations). Each rebalance Monday: if the percentile at the
signal date (the prior session's close) is below 20 %, every name is held at 1/40 of equity for that week and the
rest is cash earning 0; otherwise 1/20. Trades to move between half and full size pay the same costs.

## Compare against (first build; state every difference, do not tune to match)
| | CAGR | max DD (daily) | end $ | by year 2017 → 2026 |
|---|---|---|---|---|
| guarded sleeve | 29.34 % | −38.3 % | $596,394 | 19.6, −5.0, 14.9, 85.3, 25.4, 0.0, 16.9, 40.8, 38.2, 73.0 |
| + half-size gate | 30.67 % | −37.1 % | $658,510 | not published — report yours |
Gate: 143 of ~504 rebalance weeks at half size (28.4 %), 54 spells. Guarded top-20 on 2021-02-08, 2025-12-29 and
2026-06-29: `recon/G_holdings.csv`. Gate state + weights on 2021-08-30 (half), 2024-01-08 (full), 2024-12-02 (half):
`recon/H_gate.csv`. Names the guard removed from holdings in the first build: AMC ×15, AMRN ×6, NBIS ×6, ABVX ×5,
OCGN ×3, WOLF ×1 (36 name-weeks).

## Agreement bar
Guarded: CAGR within 1.0 pt, max DD within 3 pts, every year within 4 pts, holdings ≥ 19/20 on each of the three
dates, guard-removed name-weeks within ±5 with the same six names. Gate: half-size weeks within ±3 of 143, the three
dated states identical, CAGR within 1.0 pt and max DD within 3 pts. Anything outside = DISAGREE: report the first
week where the two equity paths separate by > 1 % and the holdings on that week; do not adjust the spec.

## Output
`REBUILD_1700tu.py` (may import REBUILD_1700_sleeve.py or copy from it), `REBUILD_1700tu_by_year.csv`,
`REBUILD_1700tu_weekly.csv` (date, guarded equity, gated equity, half flag), `REBUILD_1700tu.md` (≤ 40 lines: the
comparison table, AGREE / DISAGREE per line of the bar, differences explained).
