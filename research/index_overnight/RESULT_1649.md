# RESULT — cells 1,649–1,651: index overnight sleeves — 1,649 FAIL, 1,650 FAIL, 1,651 PASS on the letter of the frozen bar, tail-carried on its spirit → PAPER only

Judged 2026-09-29 07:20 UTC by the main session from `RESULT_1649_build.md` (Sonnet, `cell_1649.py`) and `REBUILD_1649.md`
(independent rebuild from the prose, `rebuild_1649.py`). Programme count: 1,651. Free data (cached minute bars, 2016–2023;
TEST 2024+ sealed and untouched).

## Agreement
Night/event sets: Jaccard 0.9925 (1,649) and 0.979 (1,651) — the whole gap is the rebuild excluding 15 early-close
dates the first build kept; matched rows agree to 0.01 bps (corr 0.998 / 1.000). Same verdicts on every item.

## Numbers (net bps; build / rebuild)
| cell | read | mean | t | MDE | rule that decides | result |
|---|---|---|---|---|---|---|
| 1,649 unconditional overnight, 2016–23 pooled | per night | +3.3 / +3.2 | 1.85 / 1.79 | 4.7 | alt: 6/8 years ✓, 3/3 ETFs ✓, overnight Sharpe > buy-and-hold: IWM only (0.86 vs 0.50; SPY 0.43 vs 0.79; QQQ 0.51 vs 0.89) | **FAIL** |
| 1,650 heavy-close overnight, VAL 2020–23 | per night (464 nights) | +8.8 / +8.7 | 2.57 / 1.95 | 11.1 | alt: excess over complement +8.8 / +6.9 bps at t 1.57 / 1.34 (< 2); tercile table not monotone in VAL | **FAIL** |
| 1,651 turn-of-month, 2016–23 pooled | per 4-night event (285) | +32.1 / +32.0 | 1.41 / 1.39 | 60 | alt: 6/8 years ✓, 3/3 ETFs ✓, ex-top-5 % +4.4 / +5.3 > 0 ✓ | **PASS on the letter** |

## Adequacy and the tail
* 1,649: since 2016 SPY and QQQ earned intraday as well, so overnight-only trails holding; the documented pattern shows
  only in IWM. Not a sleeve for this account.
* 1,650: +8.8 bps on heavy-close nights is real-looking but the excess over ordinary nights is inside noise and the
  mechanism table breaks; no capital.
* 1,651: the letter of the frozen bar is met, but 83 % of the mean is carried by the top 5 % of events and without 2020
  the mean is +14 bps (t 0.64). Ex-top-5 % (+5 bps per four nights) is BELOW four ordinary nights (≈ 13 bps): strip the
  tail and the window is worse than random nights. The literature's effect (McConnell & Xu 2008, 1926–2005) may be
  real; this sample cannot show it apart from 2020. Worst event Feb 2018 (−6.1 %).

## Stacking line (owner's rule 9/29)
1,651 at $20K per ETP, three ETPs, four nights a month: expected ≈ +$190/month (≈ +$85 ex-2020), worst event ≈ −$3.7K,
capital window = four nights a month while the day books are flat, collision = a gap-down morning with any other
overnight sleeve. 1,649/1,650 add nothing on this evidence.

## Consequences (per the frozen PREREG)
1,651 → the PAPER MOC/MOO sleeve (paper account, $20K per ETP) from the next event (buy MOC 2026-09-30, sell MOC on the
third October session) as an execution rehearsal; capital only on the owner's word with the tail line above in front of
him. 1,649 and 1,650 closed. Refuter left open: the dividend refuter for 1,651 was not data-verified (no unadjusted
series cached; ex-dividend dates of SPY/QQQ/IWM fall mid-quarter, outside the windows — stated, not verified).
