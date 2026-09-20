# orb_inplay — Cell E: liquid sub-universe (price >= $20, ADV20 >= 5M)

SAME book/exit/cost model as the base cell (`score.py`'s stop=0.10xATR14, target=10R, time
exit 15:55; measured per-leg half-spread + $0.0035/share). Universe restricted BEFORE ranking
to `prev_close >= $20 AND adv20 >= 5,000,000`, top-20 RVOL re-ranked within it. TEST sealed,
not queried. PREREG committed `c443c44` before scoring.

## Sub-universe / picks
- Sub-universe size/day: median **310** names (vs 1,861 for the whole ADV>=1M base universe).
- Picks: 18.4/day, 6,677 total (2024-12-02..2026-05-29); TRAIN+VAL window 6,624: ok 6,409 (96.8%),
  Reg SHO excluded 116 (1.8%), doji 86, **no_bars 13 (0.2%) — availability rail PASSES easily**
  (99.8% scoreable, far above the 80% floor). Median pick price $51.21 (p10 $23.07) — richer than
  the base cell's $22-27.
- Cost coverage: no new NBBO fetch was needed. Cell-E entries fall in price bands c ($20-50) /
  d (>=$50); both already have >=70 measured legs (>= the 20/cell pre-reg minimum) in the existing
  `hs_table.json` stratified sample, which is price-band x clock, not symbol-based, so it applies
  to any name in-band. Nearly all cell-E entries land in band d: measured entry-leg half-spread
  **median 0.140% of price (p90 0.140% — almost the whole distribution is one band), median 0.44R
  / p90 1.46R** (R = 0.10 x ATR14, so tiny relative to the spread at these low-vol names).

## Results (1x = $66,000 unlevered; 2x reported alongside)

| split | side | n | fills/wk | gross R (t) | net R (t_iid / t_cl) | MDE | WR | $ P&L | $/wk | MDD |
|---|---|---|---|---|---|---|---|---|---|---|
|TRAIN|combined|387|7.3|+0.431 (+2.44)|**-1.038** (-3.90/-3.89)|0.746|15.8%|-30,768|-580|-32,231|
|TRAIN|long|198|3.7|+0.466 (+1.91)|-1.046 (-2.44/-2.39)|1.202|16.7%|-14,637|-276|-19,844|
|TRAIN|short|189|3.6|+0.394 (+1.54)|-1.029 (-3.32/-3.36)|0.869|14.8%|-16,131|-304|-16,577|
|VAL|combined|160|7.3|-0.011 (-0.04)|**-1.369** (-4.61/-4.41)|0.832|11.2%|-11,481|-522|-11,598|
|VAL|long|95|4.3|+0.154 (+0.46)|-1.044 (-2.75/-2.75)|1.064|12.6%|-1,514|-69|-6,651|
|VAL|short|65|3.1|-0.251 (-0.79)|-1.844 (-3.90/-3.92)|1.324|9.2%|-9,967|-453|-9,497|
|2x|VAL combined|185|8.4|+0.095 (+0.41)|-1.187 (-4.32/-4.05)|0.770|12.4%|-28,556|-1,298|-27,965|

avg round-trip cost: TRAIN 1.47R/trade, VAL 1.36R/trade (1x); 1.28R/trade (2x VAL) — from
`score_e.out`.

## Verdict vs pass bar (net >=+0.10R, clustered t>=2.0, TRAIN halves same-signed, >=3 fills/wk)

**FAIL, decisively, on every side and combined.** Fills/week clears the >=3 floor easily (3.1-8.4).
Gross is positive on TRAIN (long/short/combined, t 1.5-2.4) but already ~flat-to-negative on VAL
gross (combined -0.011, t -0.04) — restricting to a liquid, expensive sub-universe did not recover
a gross edge on VAL, let alone survive costs. Net R/trade is -1.0 to -1.8 on both splits, clustered
t -2.4 to -4.4 (all far past -2.0 in the wrong direction). The mechanism is unchanged from the base
cell: R = 0.10 x ATR14 is small relative to price, so a ~0.14%-of-price half-spread alone eats
~0.4-1.5R per entry leg, and the round trip + commission runs 1.3-1.6R/trade against a book that
only wins 10-17% of the time (need the rare 10R touch). Liquidity concentration (median pick price
$51, band d) did not shrink the spread-to-R ratio enough to flip the sign.

## Cadence

`scripts/cadence_bar.py` was not found anywhere in the repo (searched 3 levels from root) — could
not be run. Substituting the report's own green-week diagnostic (`score_e.out`): VAL combined green
weeks 18.2% vs a count-matched null of 50.3% (p=1.000) — i.e. VAL is GREEN in fewer weeks than a
coin-flip null of the same trade counts would produce by chance, consistent with the negative net.

## The one caveat that alone could explain the headline

The 10R-target / 0.10R-stop shape itself: at these R sizes ANY plausible measured spread — even a
liquid, $50-median-price universe — is a multiple of R, so this book design cannot pass a real-cost
bar regardless of universe. Restricting to liquidity was the wrong lever; the exit geometry is the
binding constraint (consistent with Cell D's range-stop/lock redesign being the more promising path).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
