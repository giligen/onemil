# RESULT — cells 1,643–1,645: leveraged-ETF rebalancing flow into the close — FAIL (closed)

Judged 2026-09-29 06:40 UTC by the main session from `RESULT_1643_build.md` (Sonnet, `cell_1643.py`) and
`REBUILD_1643.md` (independent rebuild from the prose, `rebuild_1643.py`). Programme count: 1,645. Free data
(Alpaca minute bars 2016-01-04..2026-09-04, nine underlyings, 13.8 M bars; 21 early closes excluded — the PREREG's
"last bar before 15:59" test never fires on SIP bars with extended hours, so the build used a 13:00–16:00 volume-share
rule verified against the NYSE calendar; the rebuild used the calendar itself: same sessions).

## Agreement
Event sets: Jaccard 1.000 TRAIN, 0.9997 VAL (one event at exactly |r| = 1.00 %). VAL means within 0.15 bps; the only
row-level divergence is the short-return denominator convention on four March-2020 XLE/GDX/XLF events (≤ 53 bps on
those rows, < 0.4 bps on any split mean) — a spec ambiguity, not a bug; the long cell is unaffected.

## Numbers (net bps per event; MDE printed beside each)
| cell | split | n | ev/wk | mean | clustered t | MDE at t 2.5 | ex-top-5 % | |r| terciles (low/mid/high) |
|---|---|---|---|---|---|---|---|---|
| 1,643 long, up days | TRAIN 2016–20 | 2,187 | 8.4 | +0.8 | 0.14 | 4.1 | −6.2 | +0.8 / +0.2 / +1.4 |
| 1,643 | VAL 2021–23 | 1,630 | 10.5 | −3.2 | −2.56 (rebuild −2.29) | 2.0 | −7.6 | −3.5 / −2.0 / −4.0 |
| 1,645 short, down days | TRAIN | 1,805 | 6.9 | +2.8 | 0.57 | 3.7 | −5.0 | +3.9 / +4.0 / +0.4 |
| 1,645 | VAL | 1,563 | 10.1 | −1.6 (rebuild −1.5) | −1.19 | 2.5 | −6.0 | −2.4 / −0.2 / −2.3 |
Underlyings positive in VAL: 2 of 9 (both cells). 1,644 (hold to the next open; 15:45 entry) net negative in VAL.

## Adequacy
The test had the power the PREREG asked for (MDE 2–4 bps against a +5 bps bar). The result is not "no power": VAL
is negative at t −2.3..−2.6 for the long side, and the mechanism check fails on both halves — the flow, if it exists,
does not show in the last half hour of the underlying since 2016, consistent with execution moving into the closing
auction and the effect being arbitraged. The mirror (fade the day's move in the last 30 minutes) is +3 bps in VAL
but ≈ −1..−3 bps in TRAIN: no consistent lead, so no mirror PREREG.

## Verdict
FAIL on the frozen bar, both builds. Closed with the tercile tables on record; no variant on this population.
