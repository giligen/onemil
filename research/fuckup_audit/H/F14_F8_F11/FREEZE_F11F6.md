# FREEZE — book F11 close-confirmed F6, written 2026-09-17 BEFORE `--val` was run for this book

Written after step 3. The only VAL number for F11(F6) that existed at this moment is Stage C's own unfiltered cell
(the parity anchor: all-day VAL n 543, +0.0775 R, t 1.28; the >=10:00 twin VAL n 442, +0.0853, t 1.59).

## The book
`C/pop_c.csv` rows with `fam == 'F11'` and `cfg == '{"base": "F6"}'` — the F6 red-to-green base (the day opens BELOW
the previous close; the level IS the previous close; the stop is the running low before the signal) with CLOSE
confirmation: the signal is the first bar whose close is at or above the level (its high necessarily reached it).
Next-open fill under the 0.6% cap, `next_entry` present, fill >= $5, `next_entry_m <= 841`, `next_r_pct >= 1`,
`range_so_far_pct >= 5`, exit = hold to 15:55 with the touch stop, contract (c), book `run_book(12, 4)`,
all-day window (09:30-14:01).

Parity anchor reproduced exactly on BOTH windows: all-day TRAIN n 1239, −0.0115, t −0.35 / VAL n 543, +0.0775,
t 1.28; `>= 10:00` TRAIN n 1018, −0.0045, t −0.18 / VAL n 442, +0.0853, t 1.59 — `dn = 0`, `|dR| = 0.0`.

## Booking convention
**PRIMARY = no-refill** (as declared in `FREEZE_F8N30.md`): the veto runs POST-ranking, the vetoed pick's slot stays
empty. The refill column is reported beside every cell.

## The frozen stack — TWO vetoes (the third candidate made TRAIN worse and was dropped ON TRAIN)
1. **P1 `prev_day_range_pct >= 5`** — the previous day's high-low range must be at least 5% of its close.
   *Mechanism*: a red-to-green reclaim is a CONTINUATION trade — it pays when the name was already in motion
   yesterday and today's open-below-close is a dip inside a live move. A quiet previous day means the reclaim is a
   fresh one-day pop with no second-day flow behind it. Live ORB ships exactly this veto (`prev_day_range_veto`,
   "day-2 of the fireworks, not day-1"), at 8%; 5% is the coarse cut chosen here and 8% is reported beside it.
   *Live source*: the previous daily bar from `daily_bars`.
2. **B1 `sig_body_pct < 1.0`** — the confirming bar's own body (close/open − 1) must be under 1%.
   *Mechanism*: F11 confirms on a bar CLOSE above the level and the fill is the NEXT bar's open. The bigger that
   confirming bar's body, the further above the level the fill lands and the wider the stop it inherits — it is a
   chase, and every live engine here caps the chase. *Live source*: the signal bar's open and close.

Dropped on TRAIN: **PR1 `price < 50`** (the BF `max_entry_price` analogue). It passes both era legs on its own
(vetoed bucket −0.096 H1 / −0.053 H2 population, −0.176 / −0.109 booked) but LOWERS the stack on TRAIN
(+0.0804 -> +0.0711), so it is not in the frozen stack.

## TRAIN, frozen (no-refill; both halves, ex-top-5%, +3R cap)

| cell | n | tr/wk | net R | gross | t | WR% | stop% | wkR | green | worst wk | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1239 | 23.4 | −0.0115 | +0.0216 | −0.35 | 43.5 | 26.2 | −0.27 | 0.43 | −12.8 | −50.4 | −0.1926 | −0.0557 |
| + P1 | 955 | 18.0 | +0.0414 | +0.0731 | 0.99 | 45.7 | 29.5 | +0.75 | 0.51 | −9.0 | −22.3 | −0.1589 | −0.0158 |
| + B1 = **STACK** | 659 | 12.4 | **+0.0804** | +0.1127 | **1.62** | 48.0 | 25.5 | +1.00 | 0.57 | −8.0 | **−19.4** | −0.1179 | **+0.0207** |
| STACK H1 | 317 | 6.0 | +0.1489 | +0.1820 | 1.84 | 47.6 | 24.3 | | | | | −0.0840 | +0.0570 |
| STACK H2 | 342 | 6.5 | +0.0169 | +0.0485 | 0.29 | 48.2 | 26.6 | | | | | −0.1472 | −0.0130 |

The stack does not clear G1 (t 1.62 < 2.0) and its H2 half is only +0.017. It is carried to VAL per METHOD step 4.

## The pass rule for VAL (METHOD step 4), fixed here
PASS = the stack's VAL mean net R improves vs the unfiltered VAL book (which is **+0.0775** — a high bar, stated
here so it cannot be moved afterwards) AND each vetoed bucket is negative on VAL AND VAL mean net R > 0 AND >= 55%
of VAL weeks green. TEST only if VAL passes.

## Declared sensitivities (reported, never gate cells)
(a) the `2R stop-1%` exit; (b) the no-floor twin; (c) the refill booking; (d) the `>= 10:00` window.
