# FREEZE — book F8 N=30 (30-minute opening-range break), written 2026-09-17 BEFORE `--val` was run for this book

Written after step 3. No VAL number for F8 N=30 existed at this moment other than Stage C's own unfiltered cell
(the parity anchor, `h0_parity.csv`: VAL n 456, −0.0107 R, t −0.32).

## The book
`C/pop_c.csv` rows with `fam == 'F8'` and `cfg == '{"N": 30}'` (level = the highest high of the first 30 one-minute
bars, stop = the lowest low of those 30 bars, signal = the first bar at or after bar 30 whose high reaches the
level), next-open fill under the 0.6% cap, `next_entry` present, fill >= $5, `next_entry_m <= 841`,
`next_r_pct >= 1`, `range_so_far_pct >= 5`, exit = hold to 15:55 with the touch stop, contract (c),
book `run_book(12, 4)`.

Parity anchor reproduced exactly: TRAIN n 1124, +0.0079 R, t 0.34 · VAL n 456, −0.0107, t −0.32 — `dn = 0`,
`|dR| = 0.0` against `C/score5_results.csv`.

## Booking convention, declared here (applies to F8 N=30 and F11 from this line on)
**PRIMARY = no-refill**: the veto runs POST-ranking and the vetoed pick's slot stays EMPTY (book `run_book(12,4)`
on the unfiltered population, then drop the vetoed trades). This is the convention every shipped ORB veto uses;
the refill form was measured toxic there (PDR veto: 2025H2 -> ~$0, MDD -$29K -> -$50K) and it is toxic here too
(M1 alone: +0.026 no-refill vs **−0.031** refill — the freed slot is filled by a worse candidate). The refill
column is reported beside every cell.

## The frozen stack — three vetoes, in this order, no re-tuning after this line
1. **W1 `asset_class != 'wrapper'`** — no 2x / inverse single-stock ETFs.
   *Mechanism*: a wrapper carries no company-level information. Its "break" is a geared echo of an underlying that
   has already broken, and the product bleeds on the path. ORB's own rulebook records that wrappers have no company
   events and that same-morning underlying news on them is crowding, negative in all three eras.
   *Live source*: `trading/orb_asset_class.classify_asset`.
2. **M1 `sig_close_pos >= 0.5`** (NaN keeps) — the breakout bar must CLOSE in the top half of its own high-low range.
   *Mechanism*: a bar that pokes the level and closes back at its own low is a failed break, not a break. This is
   ORB's shipped touchgo Rule M, at the same 0.5 threshold.
   *Live source*: the signal bar's OHLC, known at its close.
3. **G1 `gap_pct < 3`** (NaN keeps) — the name must not already be up 3% or more at the 09:30 open.
   *Mechanism*: gappers fade. The program has already settled that 0 of 27 gap x dollar-volume cells deliver a
   positive open-to-close and that prior-day-attention names lose 20-60 bps in the first hour (PLAN §2).
   A 30-minute range break on a name that spent its move overnight is buying the fade.
   *Live source*: the previous close and the 09:30 open.

## TRAIN, frozen (no-refill; both halves, ex-top-5%, +3R cap)

| cell | n | tr/wk | net R | gross | t | WR% | stop% | wkR | green | worst wk | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1124 | 21.2 | +0.0079 | +0.0318 | 0.34 | 48.4 | 15.1 | +0.17 | 0.57 | −9.7 | −40.5 | −0.1045 | +0.0006 |
| + W1 | 878 | 16.6 | +0.0272 | +0.0504 | 1.10 | 50.1 | 11.7 | +0.45 | 0.58 | −7.9 | −37.5 | −0.0744 | +0.0230 |
| + M1 | 562 | 10.6 | +0.0323 | +0.0551 | 1.02 | 50.7 | 12.1 | +0.34 | 0.53 | −5.3 | −18.4 | −0.0752 | +0.0287 |
| + G1 = **STACK** | 428 | 8.1 | **+0.0581** | +0.0814 | **1.62** | 53.0 | 11.4 | +0.47 | 0.55 | −5.7 | **−9.0** | −0.0472 | +0.0549 |
| STACK H1 | 210 | 4.0 | +0.0681 | +0.0917 | 1.27 | 52.4 | 11.0 | | | | | −0.0487 | +0.0622 |
| STACK H2 | 218 | 4.1 | +0.0485 | +0.0714 | 1.01 | 53.7 | 11.9 | | | | | −0.0348 | +0.0479 |

The stack does NOT clear G1 on TRAIN (t 1.62 < 2.0). It is carried to VAL anyway, per METHOD step 4, because the
stack's whole claim is shape: MDD −40.5 -> −9.0 R, ex-top-5% −0.105 -> −0.047, both halves positive.

## Declared COMPANION stack (reported, counted, never primary — no mechanism sentence exists for its buckets)
`S1 spy_gap not in [0, 0.3)` + `X1 spy_vs_sma20 not in [0, 2)` + `D1 not Friday`: TRAIN +0.0598 (t 1.26), H1 +0.0827
/ H2 +0.0027, 30% of weeks green. Rejected as primary on the H2 collapse and the missing mechanism.

## The pass rule for VAL (METHOD step 4), fixed here
PASS = the stack's VAL mean net R improves vs the unfiltered VAL book AND each vetoed bucket is negative on VAL AND
VAL mean net R > 0 AND >= 55% of VAL weeks green. TEST only if VAL passes.

## Declared sensitivities (reported, never gate cells)
(a) the `2R stop-1%` exit on the same stack; (b) the no-`range_so_far_pct` floor twin; (c) the refill booking.
