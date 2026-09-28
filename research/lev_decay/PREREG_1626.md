# PREREG — cells 1,626–1,629: LEVERAGED SINGLE-STOCK ETF DECAY — short both sides of a 2x pair

FROZEN 2026-09-28 18:30 UTC before any number. Programme count: 1,625 → 1,629. Idea 1 of `research/IDEAS_20260928.md`.

## Mechanism (documented: Cheng & Madhavan 2009; Avellaneda & Zhang 2010)
A daily-rebalanced L× ETF on an underlying with daily variance σ² loses ≈ ½·(L² − L)·σ² per day relative to L× the
underlying's cumulative return (the rebalancing/volatility drag), plus its expense ratio and financing. A position
short the 2x LONG and short the 2x SHORT on the same underlying, dollar-balanced, is delta-neutral at each rebalance
and collects the drag of both, minus borrow. On an 80 %-vol single stock (σ_daily ≈ 5 %) the drag is ≈ 0.25 %/day per
leg. The trade loses when the underlying trends strongly in one direction without pullbacks (the path where leverage
compounds in the long's favour) and when borrow is expensive or recalled. Nothing here is a directional bet.

## Population and data
Every pair of 2x (or 1.5x/1.75x/3x, reported separately) long and short daily ETFs on the SAME single-stock
underlying in `data/research/databento/alpaca_assets_all_20260905.csv` (664 leveraged names; ≈ 100 single-stock on
the large underlyings), matched by the underlying ticker in the name and the leverage factor (state the parser; hand-
check 20). Daily bars for every pair member and its underlying from Alpaca (2022-01 → 2026-09; most listed 2022–2025 —
report each pair's first date); the shortable / easy-to-borrow flags from the Alpaca asset endpoint (today's snapshot
— disclosed as hindsight; the borrow RATE is not available: rails at 5 / 15 / 30 %/yr per leg).
Splits: TRAIN = 2022-01..2025-03, VAL = 2025-04..2026-09 (by calendar; pairs enter when listed). TEST: none (the
forward paper/live book is the test).

## Cells
* 1,626 PAIR-SHORT, static: short $1 of the long-2x and $1 of the short-2x at the close, rebalance to dollar-neutral
  every 5 sessions (the drift between rebalances is the delta exposure — reported), hold indefinitely; P&L per pair per
  day in bps of gross short notional ($2). Costs: 5 bps per rebalance leg, borrow at the three rails.
* 1,627 PAIR-SHORT, vol-gated: enter only while the underlying's 20-day realised vol ≥ 60 % annualised; exit below 40 %.
* 1,628 SINGLE-LEG hedged (report-only): short the 2x long, long 2 × the underlying (the classic version; needs margin
  for the long leg) — the same drag on one side.
* 1,629 the 3x / 1.5x pairs (report-only).
Report per cell and split: n pair-days, mean bps/day of gross notional, day-clustered t (the day across pairs), the
share of pairs positive, the worst pair-month and the worst day, drawdown, the realised-vol tercile table (the drag
must rise with σ² — the mechanism check), the theoretical drag ½·(L² − L)·σ² per pair-day vs the realised P&L
(the calibration line), and the P&L at each borrow rail.

## Pass bar (frozen; VAL, per cell)
Mean ≥ +4 bps/day of gross notional net at the 15 %/yr borrow rail (≈ +10 %/yr on the capital at risk after margin),
day-clustered t ≥ 2.5, ≥ 60 % of pairs positive, TRAIN same sign t ≥ 1, the realised drag within 30 % of the theory
in the top vol tercile (the mechanism, not a fluke), worst month ≥ −3 % of gross notional, and the trade must be
executable: ≥ 10 pairs with easy-to-borrow flags on both legs at the snapshot.

## Independent check and consequences
Rebuild from the prose (pair set Jaccard ≥ 0.95, bps within 0.5); refuters: the pair matching (wrong underlying or
factor), the rebalance convention and its delta drift, the borrow assumption (flags are today's), survivorship
(delisted/closed ETFs — the asset list is current; count the ones that closed), dividends/expense in the bars, the
trend-path tail (the worst months). PASS → a paper book on the ORB paper account (shorts allowed) for 4 weeks, then
live at 5 % of equity gross notional per side on the owner's word. FAIL → closed with the calibration on record.

## Not allowed
Choosing the vol gate or the rebalance interval after a number; adding pairs on non-single-stock underlyings without
a separate cell.
