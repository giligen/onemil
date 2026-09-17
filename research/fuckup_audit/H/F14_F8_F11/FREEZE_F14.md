# FREEZE — book F14 (second break), written 2026-09-17 BEFORE `--val` was ever run

Written after step 3 (the TRAIN stack) and before any VAL number for this book existed. The only VAL numbers
that existed at this moment are Stage C's own unfiltered cells (the parity anchor, `h0_parity.csv`).

## The book
`C/pop_c.csv` rows with `fam == 'F14'` and `cfg == '{"N": 15}'` (second break: the first F8-15 opening-range break
filled at the next open under the 0.6% cap and STOPPED OUT under the 2R walk; the next break of the same level that
day is the signal; stop = the lowest low from the stop bar through the signal bar), next-open fill, `next_entry`
present, fill >= $5, `next_entry_m <= 841`, `next_r_pct >= 1`, `range_so_far_pct >= 5`, exit = hold to 15:55 with
the touch stop, contract (c), book `run_book(12, 4)`.

Parity anchor reproduced exactly (`h0_parity.csv`): TRAIN n 768, +0.0564 R, t 1.32 · VAL n 363, +0.0503, t 0.91 —
`dn = 0`, `|dR| = 0.0` against `C/score5_results.csv`.

## The frozen stack — three vetoes, in this order, no re-tuning after this line

1. **V4 `dist_open_pct > 0.8`** — the break level must sit more than 0.8% above the day's 09:30 open.
   *Mechanism*: F14 is a continuation trade. A level within 0.8% of the open is a level the day never actually
   left; there is nothing for a "second break" to continue, and the trade is a coin flip on a name that has gone
   nowhere. *Live source*: the day's 09:30 open and the break level, both known at the signal bar.
2. **V2 `vwap_dist_pct >= 0`** — the break level must be at or above the running session VWAP.
   *Mechanism*: a break level BELOW VWAP means the average share traded today is held above the breakout price;
   every buyer since the open is underwater and sells into the move. This is the same rule the live bull-flag book
   ships as `trading.bull_flag.vwap_gate`. *Live source*: running session VWAP at the signal bar.
3. **V1 `rv_adv` is not NaN** (equivalently: `adv20` exists — the symbol has at least 20 prior daily bars).
   *Mechanism*: without 20 prior daily bars there is no liquidity baseline, relative volume cannot be computed,
   and the name is a fresh listing whose float and holder base are unknown. Live ORB ships exactly this veto
   (`g1_veto.short_history_veto`, added after every short-history pick lost). *Live source*: the count of daily
   bars for the symbol at 09:30.

## TRAIN, frozen (both halves, ex-top-5%, +3R cap)

| cell | n | tr/wk | net R | gross | t | WR% | stop% | wkR | green | worst wk | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 768 | 14.5 | +0.0564 | +0.0836 | 1.32 | 45.2 | 21.2 | +0.82 | 0.45 | −8.1 | −31.3 | −0.1189 | +0.0125 |
| + V4 | 651 | 12.3 | +0.0952 | +0.1209 | 1.97 | 45.5 | 20.0 | +1.17 | 0.47 | −8.1 | −22.8 | −0.0927 | +0.0433 |
| + V2 | 620 | 11.7 | +0.1100 | +0.1341 | 2.19 | 45.5 | 19.2 | +1.29 | 0.49 | −8.1 | −22.5 | −0.0836 | +0.0555 |
| + V1 = **STACK** | 598 | 11.3 | **+0.1260** | +0.1502 | **2.44** | 46.3 | 18.9 | +1.42 | 0.51 | −7.4 | −20.4 | −0.0671 | +0.0695 |
| STACK H1 (Jan–Jun 25) | 264 | 5.0 | +0.1277 | +0.1522 | 1.48 | 44.7 | 20.8 | | | | | −0.0875 | +0.0534 |
| STACK H2 (Jul–Dec 25) | 334 | 6.3 | +0.1246 | +0.1486 | 1.99 | 47.6 | 17.4 | | | | | −0.0507 | +0.0823 |

## The pass rule for VAL (METHOD step 4), fixed here
PASS = the stack's VAL mean net R improves vs the unfiltered VAL book **AND** each vetoed bucket is negative on VAL
**AND** VAL mean net R > 0 **AND** >= 55% of VAL weeks green. TEST is read only if VAL passes, once, with the tail
tests, a permutation p over every cell this sub-stage looked at for F14, and the money line.

## Declared sensitivities (reported, never gate cells)
(a) the `2R stop-1%` exit on the same stack; (b) the no-`range_so_far_pct` floor twin. Both are re-scores of the
frozen stack, not new searches.
