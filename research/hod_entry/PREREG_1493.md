# PREREG — cells 1,493–1,547: the RETEST BOUNCE — the whole exit surface from the retest instant

FROZEN 2026-09-26 18:00 UTC before any number. Programme count on the HOD line: 1,492 → 1,547 (55 exit cells, selected
on TRAIN-H2, one VAL read). Owner 9/26: "go over all these recent experiments, find and deduct the edge they are all
taking us one step closer — remember we can shift entry, exit, everything."

## What was seen (disclosed) and the deduction
All on the 9,911 base fills of cell 1,438 (TRAIN-H2 / VAL, TEST sealed), standard cost (half-spread once, stop-limit
exit standard, EOD at the bid):
* Base book (ask at the break, stop = consolidation low, target 2 R): −0.16 to −0.21 R net, gross ≈ 0. Every filter,
  exit, slot and entry variant on the break entry sits within ±0.04 R of it (exit lab 35 cells; 1,445–1,467; 1,491).
* The break is a sweep: 55 % of break fills trade ≥ 0.25 % back under the level INSIDE the fill minute, 27 % ≥ 0.5 %,
  13 % ≥ 0.75 % (1,491 stop-bar rates). 87–91 % dip through the level within 15 minutes.
* Retest entry (bid at level − $0.01 for 15 min, 1,481): fills 91 %; paired +0.17 / +0.22 R better than the break on
  the same fills (t 22–35, ex-top-5 % of the lift +0.15) — the immediacy cost of the break, recovered — yet the book is
  −0.09 R (VAL, t −2.2) at the consolidation-low stop with a 2 R′ (≈ 3.2 %) target. Deeper retest (1,482): −0.07 R.
* The failed-break SHORT at the same instant (1,480: bid at the first print ≤ level − $0.01, stop max(break-bar high +
  $0.01, entry × 1.01), target 2 R): −0.53 / −0.68 R, t −8.6 / −12.6 — the strongest statistic of the programme.
  A short with a ~1 % stop losing that much means the price bounces ≥ 1 % before it drops 2 % in roughly three
  retests out of four (driftless: two out of three).
* Confirmation at minute 16 (1,487): −0.05 R (real-SIP rows −0.22 R); pyramid −0.57 R vs holding; shallow stops
  (1,491) all worse than the consolidation low; L3 (extension ≥ 5 %) AUC 0.72 but its top tercile −0.09 R under both
  the break and the retest entry; the dip-depth table: > 75 bps pullbacks (30 % of fills) −0.46 R, ≤ 25 bps −0.02 R.
Deduction: the path after the break is a wide oscillation around the level with no 3 % drift — a 2 R target is rarely
reached, a 1.6 % stop is often hit, and every long so far was built for a trend. The one directional regularity measured
is the BOUNCE after the dip (the short's loss). No cell has yet been long FOR the bounce: retest entry (no spread), a
bounce-sized target, a stop wider than the bounce. Order of magnitude if the short's numbers mirror: +0.2–0.3 % of price
per fill at ~50 fills/week — the size that clears the owner's bar. Caveat on record: part of the short's loss is its
own cost (spread at the bid, borrow, SSR-excluded subset), so the bounce may be smaller than the mirror suggests.
This cell measures the whole surface once instead of guessing one more exit.

## Population and entry (unchanged from 1,481)
The 1,481 retest fills of the independent rebuild (`rebuild_1481_fills.csv`, status == fill, 8,973 fills; builder
agreement 99.96 %): entry = level − $0.01 at t_r = the first print strictly below it within 15 RTH minutes after the
base fill bar. Report-only replication of the selected cell on the 1,482 fills (`cell_1482_fills.csv`, builder only).

## Exits — the grid (55 cells)
Stop s below the entry ∈ {0.5 % (report-only: fails the R-vs-spread rail), 1.0, 1.5, 2.0, 3.0 %, CL = the consolidation
low of the base fill}; exit e ∈ {limit target +0.5, +0.75, +1.0, +1.5, +2.0, +3.0 % above the entry; NONE (15:55 only);
T30 / T60 = flat at the open of the 30th / 60th RTH minute after t_r unless stopped or ended}. 6 × 9 = 54, plus M = the
mirror of the 1,480 short (stop = min(dip-bar low − $0.01, entry × 0.99), target entry × 1.02) = 55. Every cell is also
computed on the stratum "L3 top tercile" (`model_1478_L3_predictions.csv`, hgb_prob_L3 ≥ 0.3070) — report-only.
Path: inside the retest minute after the fill print the tape decides (the first print ≤ stop → stopped; the first print
> target → target; whichever is earlier); from minute m_r + 1 the minute bars of `bars_fills_1478.db` with `walk_path`
semantics (stop first on a bar touching both; gap-through at the open; a target needs high > target and fills at the
target). 15:55 exit at the bid. Costs: entry passive (none); target limit (none); stop = the stop-limit standard
(2.9 / 3.2 bps on filled stops + 12 % no-fill tail at 94 / 76 bps, TRAIN-H2 / VAL, `RESULT_1463_unbiased.md`); EOD and
time exits at the bid (`RESULT_1443.md` EOD means). Units: primary = net % of price per fill; also net R with
R = entry − stop price; fills/week under the live cap (12/day, 4 concurrent); the first-passage matrix P(target first),
P(stop first), P(time or EOD); median holding minutes. The whole surface is written for both holdouts.

## Placebo — same name, same day, another hour
For the selected cell and every NONE cell: the same exits applied to the same symbol-days at one random RTH minute in
[09:45, 15:00] (seed 1493, one per fill; never inside the fill's own retest window), entry at that minute's open (the
same passive standard, no spread). Paired by symbol-day: the cell must beat its placebo by ≥ +0.10 % of price with
day-clustered t ≥ 2. This separates the retest timing from the day's drift in in-play names.

## Selection and pass bar (frozen)
On TRAIN-H2 only: among cells with stop ≥ 1.0 %, ≥ 3 fills/week and day-clustered t ≥ 2, pick the single best by
mean net % of price (ties → the wider stop). VAL is read for that ONE cell: mean net ≥ +0.15 % of price, day-clustered
t ≥ 2.5, ex-top-5 % > 0, winner-capped at +3 % of price still positive, count-matched null (random base fills on the
same days, their base outcome in % of price, 1,000 draws, seed 1493) ≥ 99, the placebo margin above, the cache-only
share of the cell's fills within 5 pp of the population's 19.5 %, and neighbour stability: the grid neighbours of the
selected cell (adjacent stop, adjacent exit) all same-signed on VAL. The full VAL surface is written for the record,
labelled unselected. TEST: one read of the selected cell if VAL passes.

## Independent check and consequences
Rebuild from this prose (never reading the builder's code) on the same fills, tape and bars: ≥ 99 % of (fill, cell)
rows within 0.01 % of price; the same selected cell. Refuters: obtainability (a passive limit sell at the target under
the through-print rule; the stop-limit tail; halts), look-ahead (nothing after t_r decides the entry; the L3 stratum is
the arm-bar probability), statistics (tails, day concentration, drop the best 2 days, the 55-cell multiplicity spent on
TRAIN-H2, the neighbour check, the placebo). PASS → the live rule: after the break, rest the bid at level − $0.01 for
15 minutes; target and stop as selected — dry 5 sessions on the parity ledger, then $50 real orders under the 9/25
fixes. FAIL (a flat surface) → the retest is closed for every exit; with 1,487 and 1,491 this closes the long side of
the population at every entry and every exit measured, and the report says so with all the numbers.

## Not allowed
Adding grid points after seeing TRAIN-H2; selecting on VAL; reading TEST more than once; using the tape to improve the
entry; treating the 0.5 % stop rows or the L3 stratum as selectable.
