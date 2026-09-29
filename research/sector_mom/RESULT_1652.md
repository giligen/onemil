# RESULT — cells 1,652–1,654: sector-ETF 12-1 momentum — FAIL (closed)

Judged 2026-09-29 08:20 UTC by the main session from `RESULT_1652_build.md` (Sonnet, `cell_1652.py`). Programme count:
1,654. The PREREG allows no rebuild on a fail unless the build's own caveats raise a coding doubt; the review found none
(turnover, borrow, benchmark, TEST seal verified; holdings CSV ends 2023-12).

Data limit disclosed: the Alpaca plan returns bars from 2016-01-04 for every symbol, so the 12-1 signal starts in
2017-01 (83 of the planned 96 months; 7 years in the sign table). VAL-type detectability was already stated as poor
(MDE ≈ 0.77 %/month); the alternative rule decides.

| cell | monthly excess vs equal-weight sectors | NW t | years positive (of 7) | max DD vs bench | worst month vs bench | ex-top-5 % excess | verdict |
|---|---|---|---|---|---|---|---|
| 1,652 top-3 long | +0.13 % | 0.68 | 2 | −15.8 % vs −22.7 % | −9.7 % vs −14.4 % | −0.14 % | FAIL |
| 1,653 top-3 minus bottom-3 | −0.81 % | −1.16 | 2 | — | — | −1.82 % | FAIL |
| 1,654 SPY-gated top-3 | +0.03 % | 0.13 | 2 | — | — | −0.40 % | FAIL |

Reading: the long-only tilt lowered drawdown (fewer sectors, the momentum sectors happened to be the defensive ones in
the bad months) but earned nothing beyond the benchmark once its top 5 % of months are removed; 2 of 7 years positive
is a coin. The long/short version is negative. Consistent with the post-2009 literature on sector-ETF momentum.
Stacking line: nothing to add. Closed.
