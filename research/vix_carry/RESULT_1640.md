# RESULT — cells 1,640–1,642: VIX basis carry with a defined loss — FAIL (closed on this mechanism)

Judged 2026-09-28 21:00 UTC by the main session from `RESULT_1640_build.md` (first build, `cell_1640.py`) and
`REBUILD_1640.md` (independent rebuild from the prose, `rebuild_1640.py`). Programme count: 1,642.

## Agreement
In-market day sets: Jaccard 1.000 on every pass-bar split; mean bps/day equal to 0.002 bps; NW t equal. The only
difference is a warm-up day-count convention on the 1,642 TRAIN −1x era (542 vs 540 calendar rows, identical
in-market days). Both builds are the same book.

## Numbers (VAL 2020-01..2023-12, cell 1,640 SVXY −0.5x era, the frozen two-close rule)
| item | value | bar | pass |
|---|---|---|---|
| net bps/day in market | +5.15 | ≥ +4 | yes |
| Newey-West t (5 lags) | 0.76 | ≥ 2.5 | **no** |
| days in market | 74.9 % | ≥ 40 % | yes |
| basis-decile table monotone | corr −0.30 (decile means), −0.06 per day; ex-worst-5-days the same sign | monotone on both halves | **no** (TRAIN +0.54) |
| worst day | −16.9 % of notional | ≥ −25 % | yes |
| max drawdown | −32.9 % | ≥ −35 % | yes |
| gate beats always-in on worst day and drawdown | yes | yes | yes |
| TRAIN same sign, t ≥ 1 (−1x era 2016-01..2018-02 only) | yes | | yes |
Price-scale check: seven |daily return| > 40 % days, every one a known vol-spike day (2018-02-05/06, Brexit, COVID, 2024-08-05);
no split artefact. Data limit disclosed: the Alpaca plan starts 2016-01-04, so TRAIN is 2016–2019, not 2011–2019 (VAL unaffected).

## Adequacy review (a null is a claim about my test first)
* The t ≥ 2.5 bar was unreachable by construction: with a daily SD near 2.7 % and ≈ 750 in-market days the minimum
  detectable mean at t 2.5 is ≈ 25 bps/day (≈ 60 %/yr). No carry strategy clears that. **Spec error, recorded** —
  a PREREG must print the MDE beside every t-bar. The t item therefore carries no information here.
* The decisive item is the mechanism: the basis predicted next-day short-vol returns in TRAIN (+0.54) and did not in
  VAL (negative under both correlation conventions, with and without the worst five days). Not a tail artefact.
* The economics at the owner's tail budget: a −60 % day capped at $450 allows ≈ $750 notional; +5 bps/day × 75 % in
  market ≈ $7/month. At $5K notional ≈ $47/month against a −$850 worst day and a −$1,650 drawdown. Return/drawdown ≈
  0.3 even if the point estimate were real.

## Verdict and consequences (per the frozen PREREG)
FAIL → closed. The option expression (UVXY put spreads / SVIX call spreads) is NOT pursued on this mechanism: the
timing signal it would rely on did not hold out of sample, and the untimed carry is a −33 % drawdown for +13 %/yr.
No re-run without a different signal (e.g. the VIX-of-VIX or the futures curvature) under its own PREREG.
