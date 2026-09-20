# REPORT C — the chase-tolerant short (cell 1,273)

**Verdict: NO SHIP.** The 62.9 % gap-through no-fill rate was the leading caveat of the Stage-A/B report; it is
now tested and was **not** the explanation — letting the fast breakdowns in nearly doubles the book and leaves the
raw edge at zero, the selected book negative on VAL. Pre-registered in `PREREG.md` §"Cell C" (`5c564a9`), scored
after. TEST (>= 2026-06-01) never queried.

## 1. What changed — the entry only

Stop **order** at the break: fill = the **next bar's open**, capped at `range_low × (1 − 100 bps)`; a next open
below the cap is a **skip, not a chase**. SSR names still need `open > that bar's low`. Stop stays `range_high`, so
**R grows when the fill is lower** — every R below is on the actual fill. `buildC.py` re-walks only the
`gap_through` rows (all other no-fill reasons are unchanged by construction: same fill bar, same test).

## 2. Fill rate — before / after
| | signals | fills | rate |
|---|---|---|---|
| Stage B (no-chase stop-limit, `L = range_low × (1−30bps)`) | 2,615 | 903 | **34.5 %** |
| **Cell C** (stop, next open, cap −100 bps) | 2,615 | **1,871** | **71.5 %** |

Of the 1,645 gap-through triggers **968 now fill**, 608 skip over the cap (> 1 % through), 69 fail the SSR
uptick. Cost: **3,732 / 3,742 legs measured** (Alpaca SIP NBBO), global median **11.9 bps**; imputed 0.5 % of
legs, 1.4 % of the VAL Stage-B book.

## 3. Stage A — raw, no selection (vs the matched non-triggering control)
| split | n | net R | cl-se | t | ex-1 % | ex-5 % | cap 3R | $10K | $50K |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN raw | 1,250 | **+0.002** | 0.170 | +0.01 | −0.070 | −0.228 | −0.069 | −2,042 | −10,150 |
| TRAIN control | 932 | −1.481 | 0.156 | −9.50 | −1.560 | −1.667 | −1.520 | −24,335 | −122,160 |
| VAL raw | 621 | **+0.058** | 0.161 | +0.36 | −0.013 | −0.175 | −0.016 | +2,438 | +12,244 |
| VAL control | 475 | −1.602 | 0.235 | −6.83 | −1.671 | −1.782 | −1.623 | −12,172 | −61,124 |
| TRAIN H1 / H2 | 758 / 492 | +0.095 / −0.142 | 0.273 / 0.120 | +0.35 / −1.18 | +0.027 / −0.227 | −0.127 / −0.385 | +0.029 / −0.220 | +3,277 / −5,319 | +16.5K / −26.7K |

VAL cuts: no-SSR (n=282) −0.030 R · no-wrapper (n=577) +0.037 R. Stage A was −0.07 / +0.04 R under the stop-limit and is +0.00 / +0.06 R with the fast breakdowns added — inside
one standard error, and **every ex-tail and capped column is negative in both splits**.

## 4. Stage B — selection (composite / signs / quintiles / vetoes / 8 slots, refit on cell-C TRAIN)
Picks 1,260 · vetoed 757 (pdr 631, range-size 147, G1 664) · no-fill picks 373.
| split | n | net R | cl-se | t | ex-1 % | ex-5 % | cap 3R | $10K | $50K |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | 174 | **−0.045** | 0.133 | −0.33 | −0.159 | −0.318 | −0.147 | −25 | −96 |
| **VAL** | 138 | **−0.139** | 0.146 | **−0.96** | −0.178 | −0.352 | −0.187 | **−885** | −4,422 |
| TRAIN H1 / H2 | 70 / 104 | +0.114 / −0.151 | 0.250 / 0.142 | +0.46 / −1.07 | −0.081 / −0.212 | −0.268 / −0.352 | −0.081 / −0.191 | +629 / −654 | +3,176 / −3,273 |

VAL cuts: no-SSR (n=67) −0.219 R · no-wrapper (n=134) −0.134 R.
VAL: **138 fills / 22 weeks = 6.27 per week** (Stage B was 3.8), green weeks **27.3 %** vs count-matched null
**44.8 %**, **$10K −$40/wk · $50K −$201/wk**, SSR 51.4 % of fills, wrapper 2.9 %. Selection still **subtracts**
from raw on VAL (−0.139 vs +0.058): the composite refit on cell-C TRAIN is no better than on the stalled fills.

## 5. Pass bar, line by line (VAL)
| # | bar | value | |
|---|---|---|---|
| 1 | net >= +0.15 R | **−0.139 R** | **FAIL** |
| 2 | day-clustered t >= 2.0 | **−0.96** | **FAIL** |
| 3 | ex-top-5 % >= 0 | **−0.352 R** | **FAIL** |
| 4 | TRAIN H1/H2 same-signed | +0.114 / −0.151 | **FAIL** |
| 5 | >= 3 fills / week | 6.27 | PASS |
| 6 | green weeks > null | 27.3 % vs 44.8 % | **FAIL** |
| 7 | Stage B − Stage A control >= +0.10 R | +1.463 R | PASS (a control that is −1.6 R by construction — a stop at `range_high` from a 09:36 short on a name that never broke; this line is nearly free and should be retired as a discriminator) |

**2 of 7. NO SHIP.**

## 6. Power
VAL Stage B day-clustered se **0.146 R** → **MDE 0.41 R** at 80 % (Stage A VAL se 0.161 → 0.45 R). The bar
(+0.15 R) sits **below** the MDE, so a +0.15 R truth is not resolvable at n=138 — but the point estimate is
negative in both splits and in every tail/SSR/wrapper cut: a measured miss, not only a power failure. Frequency
was the binding constraint before and no longer is (6.3 fills/wk); the sign did not improve.

## 7. The ONE caveat that alone could explain the headline
**The cap is itself a selection rule: it still rejects 608 of the FASTEST breakdowns (37 % of gap-throughs).**
Cell C tolerates a chase only to −100 bps, so the prints that gapped further through the level — exactly the
capitulation a short wants — remain absent; and R on the fills admitted is inflated by the lower entry against
an unchanged `range_high` stop, which mechanically shrinks every R-denominated number. A wider cap, or a stop
re-anchored to the fill, is a different cell and is **not** scored here.

## 8. Mid-run changes
One, declared: the first score ran at 52 % imputed cost while the NBBO fetch for the 1,936 new legs was in
flight; the tables above are the post-fetch run (0.5 % imputed) — deltas <= 0.03 R, no pass-bar line flipped.
Artifacts: `sigC.csv`, `legsC.csv`, `book_stage_{a,b}_cellC.csv`, `scoreC.log`. `score.py --cell c` selects the
variant; the default path is unchanged.
