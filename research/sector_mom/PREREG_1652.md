# PREREG — cells 1,652–1,654: SECTOR-ETF MOMENTUM (12-1, monthly), a stackable allocation sleeve

FROZEN 2026-09-29 07:50 UTC before any number. Programme count: 1,651 → 1,654. Re-queued under the owner's stacking rule
(9/29). No paid data. Honest prior: LOW — industry momentum (Moskowitz & Grinblatt 1999) is documented, but sector-ETF
momentum has been weak since 2009; the cell exists because it is free and the sleeve would hold capital that the day
books do not use overnight. Honest scale: ≈ +0.3 %/month excess on $60K ≈ $180/month if the literature's number holds.

## Data (free)
Alpaca daily bars (adjustment all) 2015-01 → 2026-09-04 for the SPDR sectors XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU,
XLB, XLRE (from 2015-10), XLC (from 2018-06) and SPY; a sector enters the ranking once it has 12 months of history.
Sample 2016-01 → 2023-12 for the read; TEST 2024-01..2026-09 sealed (one read for the single best passing cell).
MDE printed: monthly excess SD ≈ 3 % over 96 months → SE ≈ 0.31 % → MDE ≈ 0.77 %/month at t 2.5, far above the
documented effect, so the t item is informational and the alternative rule decides (below).

## Signals and trades (decided at the close of the month's last session, executed MOC that session — the same auction
as the TOM sleeve, reported for collision)
* 1,652 TOP-3 LONG: rank sectors by the 12-1 return (the 11 months ending one month before the decision), hold the top
  three equal-weight for the next month; rebalance monthly; 1 bp per leg on turnover.
* 1,653 TOP-3 minus BOTTOM-3 (long/short, dollar-neutral, borrow 1 %/yr on the short leg; report-only unless 1,652
  passes).
* 1,654 ABSOLUTE-MOMENTUM GATE on 1,652: hold cash (0 %) in months where SPY's own 12-1 return ≤ 0.
Report per cell: monthly mean net return, the excess over equal-weight all-sectors held the same way (the benchmark),
Newey-West t (3 lags) of the monthly excess, annualised excess, max drawdown vs the benchmark's, per-year excess sign
table (2016–2023), turnover, the worst month, the share of months the top-3 set changes, and the stacking line
(expected $/month at $60K, capital window = all month, collision = the day books' overnight margin and the TOM sleeve's
four nights).

## Pass bar (frozen; 2016–2023 pooled)
Monthly excess over the equal-weight benchmark ≥ +0.25 % net, NW t ≥ 2.5 OR (the MDE rule) ≥ 6 of 8 years with positive
excess AND max drawdown no worse than the benchmark's by more than 5 pp AND the worst month no worse than the benchmark's
worst by more than 3 pp; ex-top-5 % of months still ≥ 0 excess.

## Independent check and consequences
PASS → rebuild from the prose (monthly holdings Jaccard ≥ 0.98) and refuters (the 12-1 window alignment at month ends,
XLRE/XLC entry dates, dividends in adjusted bars, the 2020 months) before the number reaches the owner; then the
sleeve is added to `scripts/tom_sleeve.py`'s family as a monthly MOC rebalance on the paper account for two months.
FAIL → closed; no rebuild needed for a fail unless the first build's own caveats raise a coding doubt.

## Not allowed
Changing the lookback, the top-N, the rebalance day or the benchmark after a number.
