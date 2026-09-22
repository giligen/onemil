# PREREG — HOD-break EXIT LAB + day/week conditions. Cells 1,359–1,379 (+ descriptive drift profile)

Owner 2026-09-22: full conviction in HOD-break with proper filters (day / week / trade); creative exits welcome.
Facts this lab builds on: (1) no causal ENTRY feature predicts the outcome (cells 1,343–1,355, AUC 0.51); (2) the
population drifts UP intraday (placebo shorts lose 0.6–0.9 R); (3) the live rule stops 41–45 %, hits +2 R on
33–39 %, closes the rest at 15:55; (4) the dry run gives back half its intraday gain into the close (9/22: +2.5 R at
12:26 ET → +1.1 R at 15:04). Therefore the exit, never studied, is the lever. **Cost gate (R-must-exceed-spread):**
the HOD long stop sits ≈ 2 % of price below entry → round-trip cost ≈ 0.15 R; cells that change the stop distance
report cost in their own R AND in baseline R.

## Population, baseline, cost, splits (unchanged from the causal-filter study)
`research/bf_zero/causal_filter/features.csv` (12,135 signals; day, symbol, entry_m, entry, stop, r_pct, …),
1-minute bars `research/bf_zero/bars_sip.db` (coverage 100 % after the 9/22 backfill). Baseline **B0** = the live
rule: long at `entry`, stop at `stop`, target entry + 2 R, else exit at the 15:55 open. Fill physics: stop fills at
the stop price, gap-through at the bar open; target fills at the target only if the bar's high ≥ target; a bar that
touches BOTH stop and target counts as a stop (conservative); exits at the 15:55 bar open. Cost: the study's
per-signal NBBO half-spread on both legs + 2 bp per side slippage, as `cells.py`. TRAIN 2025 (halves H1/H2),
VAL 2026-01..05, TEST sealed. **Reproduction gate first:** B0 must reproduce the causal study's TRAIN mean R within
0.01 R before any variant is scored.

## Method — one DB pass, then everything on cached paths
The harness walks each signal ONCE from entry to 15:55 and stores its 1-minute path (o/h/l/c, VWAP since 09:30,
minutes since entry) in `paths.parquet`. Every cell below is then a function of the cached path; the DB is not
touched again. Variants are scored PAIRED against B0 on the same trades (per-trade ΔR), which is the powerful test.
Heavy DB work must not run between 13:25 and 20:05 UTC (live engine; the 9/22 stall).

## Descriptive exhibit (no cell, reported first): the drift profile
Mean and median excursion from entry by minute-since-entry (0–390), the MFE and MAE distributions in R, minutes to
MFE, and the **give-back share**: trades with MFE ≥ +1 R that end ≤ 0 under B0, and the R they gave back. Same
by time-of-day bucket of the entry. This exhibit informs pass 2; it does not change any cell in this pass.

## Exit cells (each ONE change from B0; everything else identical)
| cell | rule |
|---|---|
| 1,359 X1 | no target: stop or 15:55 only |
| 1,360 X2 | target +1 R |
| 1,361 X3 | target +3 R |
| 1,362 X4 | target +5 R |
| 1,363 X5 | breakeven lock: when a bar's high ≥ entry + 1 R, stop → entry (from the next bar) |
| 1,364 X5b | ORB-style lock: high ≥ entry + 1.5 R → stop → entry + 0.5 R |
| 1,365 X6 | trailing stop: once MFE ≥ 1 R, stop = MFE − 1 R (ratchets on bar highs, applied from the next bar) |
| 1,366 X7 | time stop: at entry + 60 min, exit at the next open if the close is below entry |
| 1,367 X7b | time stop: at entry + 30 min, exit if the close is below entry + 0.25 R |
| 1,368 X8 | partial: 50 % at +1 R, runner with X5 lock and no target, 15:55 close |
| 1,369 X9 | VWAP exit added: exit at the next open after the first 1-min close below session VWAP (stop still live) |
| 1,370 X10 | clock exit: everything closed at the 12:00 ET open (no afternoon) |
| 1,371 X11 | stop widened to 1.5× the distance (R changes — report in own-R and baseline-R; cost gate) |
| 1,372 X12 | stop tightened to 0.75× the distance (same reporting) |

Look-ahead rule for every path rule: a level computed from bar t's high/low can only act from bar t+1; within a
bar, stop before target (conservative); a lock/trail stop that is gapped through fills at the open.

## Day / week / trade-context cells (scored on B0 only; each ONE cut; never stacked with an exit cell)
| cell | rule (kept cohort) | data |
|---|---|---|
| 1,373 D1 | SPY above its 09:30 open at the signal minute | `research/index_orb/cache/SPY_1min.parquet` (to 2026-05) |
| 1,374 D2 | breadth: signals fired that day before this one ≤ the TRAIN-H1 median (uncrowded) | features.csv |
| 1,375 D3 | exclude entries 12:00–14:00 ET | entry_m |
| 1,376 D4 | HMM calm state on the day | `research/regime/hmm_labels.csv` |
| 1,377 W1 | week gate: trade week t only if week t−1 net R > 0 under B0 | weekly series |
| 1,378 W2 | daily kill: no new entries after the day's realized R ≤ −3 | daily series |
| 1,379 D5 | first signal of the day per symbol only (drop later re-breaks of the same name) | features.csv |
Report-only tables (no cell): day-of-week; time-of-day buckets; entry minute deciles.

## Pass bar (per cell; all)
1. **Paired**: mean ΔR vs B0 ≥ +0.10 on TRAIN and on VAL, day-clustered t of ΔR ≥ 2 on VAL;  2. ΔR > 0 in both
TRAIN halves;  3. ex-top-5 % net R of the variant ≥ B0's;  4. weekly MDD (R) not worse than 1.25× B0's on either
split;  5. for D/W cells: kept cohort ≥ 3 fills/week on VAL and the dropped cohort mean R < 0 on both splits;
6. for X11/X12: the bar is met in BASELINE-R units and the cost-in-own-R is stated.
Cadence block (`scripts/cadence_bar.py`) on VAL for B0 and for every passing cell.

## Verification of any pass (before it is called a result)
Three independent refuters with distinct lenses on the passing cell's code and trades: (a) intra-bar sequencing and
look-ahead; (b) fill obtainability and cost; (c) tail dependence and split stability. Then an independent rebuild
of the passing cell from THIS prose by an agent that has not seen the harness; trade-by-trade agreement on
(day, symbol) ≥ 99 %. A cell passes only if all four agree.

## Decision rule
Any exit cell passes → PREREG for the dry run to carry BOTH exits side by side (`[HOD DRY]` lines log the variant
exit's would-be P&L), 10 sessions, then the owner's word. Any D/W cell passes → the same, as a would-skip tag.
Nothing passes → pass 2 is designed from the drift profile (the give-back anatomy), not from more entry filters.

## Multiplicity and not-allowed
21 cells; programme count 1,379. Not allowed: stacking cells; re-tuning a threshold after VAL; any TEST row;
scoring an exit cell on a D/W-filtered cohort in this pass.
