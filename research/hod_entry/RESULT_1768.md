# RESULT 1,768 (J7): early short of the gap >= 10 % gapper

`1768_score.py`, `1768_grid.csv`. Price >= $5, TRAIN = 2025H2, VAL = 2026 (read once), TEST sealed. Cost: 4 bp fees, 2x symbol-median half-spread (not NBBO), borrow 50 bp/day. Stops fill at max(stop, bar open) + 50 bp; minute bars cannot time intrabar.

## VAL, gap >= 10 %, 10:30 cover
| Entry | net bp | t | n/wk | stop-hit | worst-5 % day |
|---|---|---|---|---|---|
| E1 09:35 | +142 | 4.7 | 16.5 | 34 % | -501 |
| E2 09:45 | +79 | 2.7 | 16.5 | 23 % | -547 |
| E3 09:45 faded | +96 | 2.6 | 10.9 | 11 % | -772 |
| E4 range-low break | +91 | 2.7 | 10.4 | 9 % | -527 |

## Primary E3 x 10:30
TRAIN halves: H1 +33 bp, H2 -67 bp (opposite signs). TRAIN -20 bp (t -0.45, n 266). VAL ex-top-5 % +21 bp, net at borrow 100 bp +46 bp, max loss -2,132 bp, weekly P10 -216 bp, MDE80 about 100 bp.

## Verdict: KILL
Rule fired: TRAIN halves opposite. All other rules pass.

## Adversarial read
TRAIN is all 2025 and VAL all 2026, so the split confounds year with rule. TRAIN 10:30 cells sit within one MDE of zero; the VAL stop-hit share is far lower than TRAIN (E3 11 % vs 17 %, E1 34 % vs 44 %), so the VAL gain looks like squeeze frequency, not the entry rule. Gap 5-10 % loses in both samples (-41 to -87 bp at 10:30).

CHILDREN:
1. 1,769 regime: separate TRAIN squeezes from VAL calm days with a causal 09:30 regime read.
2. 1,770 E1 with a volatility-scaled stop (range high + k x ATR), the only cell with VAL t 4.7.
3. 1,771 borrow and real-NBBO audit of VAL's E1 names before further work.
