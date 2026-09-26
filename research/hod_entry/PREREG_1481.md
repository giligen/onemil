# PREREG — cells 1,481–1,482: BUY THE RETEST, NOT THE BREAK (the resting bid one tick under the level after the break)

FROZEN 2026-09-26 before any number of these cells. Programme count on the HOD line: 1,480 → 1,482.

## What was seen (disclosed) and why this cell exists
Cell 1,480 (failed-break flip short, tape-priced): the short leg loses −0.53 R (TRAIN-H2, t −8.6) and −0.68 R (VAL,
t −12.6), and bailing out of the long at the failed-break print instead of holding to the consolidation-low stop made the
long worse by 0.04 / 0.10 R. Its detector found a dip to ≤ level − $0.01 within 15 minutes after the fill bar on 8,627 of
the 9,911 fills (87 %). Read together: the break is followed by a retest almost every time, and price recovers strongly
after it. The base book's entry — the ask at the first print above the level — is therefore the worst point of the whole
sequence, and its −0.2 R is, in part, the price of buying the top of a two-way move. The retest-long numbers themselves
have NOT been seen.

## Rule
Population: the 9,911 fills of cell 1,438 (the same arming, the same break). At the instant of the base fill, instead of
buying at the ask, a BUY LIMIT rests at level − $0.01 for 15 minutes (through the end of the 15th RTH minute after the
fill bar). Obtainability (conservative, the passive-limit standard): the order is filled at the limit price at the first
print STRICTLY BELOW the limit within the window (traded through); a print exactly at the limit does not fill (report-only
variant: fills at-or-below). No print below → no trade (the runners that never retest are reported separately with their
base net R, so the selection cost is visible). Stop = the consolidation low (the base's stop); R′ = entry − stop; target
= entry + 2 R′; 15:55 exit. Path: inside the retest minute the tape decides (a print ≤ stop after the fill = stopped at
the stop-limit standard; a print ≥ target = target), then the minute bars of `bars_fills_1478.db` with `walk_path`
semantics. Costs: entry passive (no spread charged); target = a limit (no slip); stop = the verified stop-limit standard
(2.9 / 3.2 bps on filled stops, 12 % tail at 94 / 76 bps); EOD exit at the bid (cell 1,443's EOD measure).
Tape: the 3,002 windows already cached under `sip_cache_1480/` plus fresh `fetch_window` calls for the rest (resumable
cache `sip_cache_1481/`; no cap — the population is the point). Report R′ as % of price, the fill share, the median dip
below the level, the time from break to retest.

| cell | variant |
|---|---|
| 1,481 | as above (limit at level − $0.01, 15-minute window) |
| 1,482 | 1,481 with the window extended to 30 minutes and the limit at level − 0.2 % (the deeper retest) |
| report-only | at-or-below fills; the never-retest cohort's base net R; the base book's net R on the SAME retest fills (the paired comparison: same trades, entry at the ask vs at the bid) |

## Pass bar (frozen)
VAL: mean net R′ ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at first-12/day 4-concurrent, count-
matched null percentile ≥ 99, TRAIN-H2 same sign with t ≥ 1; the paired ΔR vs the base on the same fills ≥ +0.10 on both
holdouts (the mechanism must show in the pairing, not only in the level). TEST read once for the better cell if it passes.

## Independent check and consequences
Rebuild from this prose (never reading the builder's code) on the same tape windows: fill-set Jaccard ≥ 0.99, net R′
within 0.01 R on ≥ 99 % of fills. Refuters: obtainability (queue priority on a passive limit; halts; the traded-through
convention), look-ahead (the 15-minute window must start at the base fill, not at the arm), statistics (tail dependence,
day concentration, drop the best 2 days). PASS → the live engine's entry changes from a stop-limit at the ask to "after
the break, rest a bid at level − 1 tick for 15 minutes" — a smaller change than the current code — dry run 5 sessions
with the parity ledger (the ledger already records the tape cross), then $50 real orders under the 9/25 fixes.
FAIL → the population's long side is closed at the retest as well; the tape windows stay for any future entry study.

## Not allowed
Moving the tick, the window or the stop; choosing between 1,481 and 1,482 on VAL for TEST (the better VAL cell is read on
TEST once, disclosed); counting an at-the-limit print as a fill in the primary book; reading TEST otherwise.

## Amendment 1 (2026-09-26 16:40 UTC, before any 1,481/1,482 number is read) — the joint with the extension predictor
Cell 1,486 = the 1,481 retest entry (limit level − $0.01, 15-minute window, strictly-below fill rule, base stop, 2 R′
target, standard costs) restricted to fills whose cell 1,478-L3 probability (HGB, `model_1478_L3_predictions.csv`,
column for L3/HGB) is ≥ the TRAIN-H2 top-tercile threshold already frozen there (0.3070). Same pass bar as 1,481 (VAL
mean net R′ ≥ +0.15, t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week, null ≥ 99, TRAIN-H2 same sign, paired ΔR vs the base on
the same fills ≥ +0.10). Motivation on record: L3's VAL AUC 0.72 with a negative kept set under the break entry. TEST:
still one read, for the single best of 1,481 / 1,482 / 1,486 on VAL. Programme count 1,482 → 1,486 (1,483–1,485 are
the catalyst/runway cells).
