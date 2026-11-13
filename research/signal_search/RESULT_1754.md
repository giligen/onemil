# RESULT 1,754 — auction-to-auction gapper (MOO long, MOC out): KILL (sign is negative)

Daily caches 1748/1746 (7,292 symbols), PIT rank ≤ 3,000 at t−1, px ≥ $1, attention = prior vol ≥ 1.5× or range ≥ 5 %. Scale check dropped 0.03 % of name-days. TEST never loaded. Cost 2 bp/leg, 0 spread; sensitivity adds measured entry+exit half-spread (400 extreme-loser name-days, not gappers: mean 94 / median 62 bp).

Names/wk (TRAIN / VAL, $1–20): g 5 % 71 / 65; g 10 % 17.5 / 17.0; g 20 % 4.5 / 4.7.

Primary (g 10 %, $1–20, excess open-to-close):
| split | n | excess bp | day-t | ex-top-5 % bp |
|---|---|---|---|---|
| TRAIN 16–18 | 1,419 | −330 | −9.4 | −510 |
| TRAIN 19–20 | 2,930 | −116 | −2.5 | −344 |
| VAL 21–23 | 2,557 | −185 | −4.5 | −399 |

Halves same sign. Years: 16 −348, 17 −339, 18 −301, 19 −216, 20 −96, 21 −299, 22 −90, 23 −86. Capped +20 %: −280 bp. Green months 17 %. MDE 115 bp. Every price and rank tier negative; fade deepens with gap (g 20 % VAL −287 bp). Net 0-spread −187 bp; with measured half-spread −250 (median) / −281 (mean). Kill rules hit: VAL excess ≤ +30, day-t < 2, ex-top-5 % ≤ 0, 2022 negative, half-spread net ≤ 0. No paper test.

Caveats: survivor-leaning cache; events/yr rise 4× in 2020 (thin early coverage); cache open assumed = opening cross.

Winners vs losers (excess bp, TRAIN → VAL, terciles): gap low/mid/high −41/−155/−362 → −133/−116/−294; prior range low/high −66/−274 → −29/−416; price < $2.9 / > $7.2 −279/−117 → −236/−203. Winners: higher price, higher prior volume ratio, narrower range, more liquid. The fade grows with gap and prior-day range (exhaustion).

CHILDREN:
1. Family mean-reversion harvest (short side), parent 1,754: short the same gappers at the open, cover at close, only gap ≥ 15 % AND prior range ≥ 15 %, price ≥ $2, rank ≤ 2,500 (daily bars on disk). Kill: VAL gross < +100 bp after 4 bp + 30 bp borrow allowance, OR ex-top-5 % (squeeze tail) ≤ 0, OR 2022/2023 negative, OR TRAIN halves opposite.
2. Family new levels/timing, parent 1,754: split the fade into open→10:00 vs 10:00→close with minute bars (bars_sip.db, causal). Short at 10:00 only if the later leg carries it. Kill: VAL 10:00→close short gross < entry half-spread + 40 bp, OR ≥ 80 % of the fade falls in open→10:00, OR ex-top-5 % ≤ 0.
