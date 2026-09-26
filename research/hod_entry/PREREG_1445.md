# PREREG — cells 1,445–1,456: reviving the resting-order fill book by a CAUSAL restatement of its positive cohort

FROZEN 2026-09-26 02:20 UTC, before any cell number is computed (the two marginal splits disclosed below are the only
numbers seen). Owner's instruction 9/26: "go back to the resting order cell that you thought was good and find the way to
revive it." Programme count on the HOD line: 1,444 → 1,456.

## What is being revived, exactly
Cell 1,427 (E1: buy stop-limit at HOD + $0.01, limit 15 bps, fill at the NBBO ask at the first consolidated print
≥ trigger) showed its FILLS at +0.29 / +0.24 / +0.33 R (TRAIN-H2 / VAL / sealed TEST). Cell 1,438 (same order, correct
levels, causal arming) showed the whole armed-and-crossed population at −0.21 R (−0.11 after the double-charged entry
half-spread is removed; raw ≈ 0). The adversarial check decomposed 1,438's fills: the cohort that 1,427 also filled is
+0.24 / +0.28 R (n 1,114 / 1,388; +0.34 / +0.38 after the cost correction), and what separates it mechanically is that
`data/cache.db` held complete bars for the symbol-day. Those bars were written by backtest cache builders whose selection
includes the FULL-DAY range ≥ 10 % (a look-ahead) — but ALSO price-band, float and dollar-volume terms that are knowable
at the arm bar. The check tested seven single features in bins (distance from open, rv at j, ask distance, time of day,
R %, range to j, arm index) and found no positive bin; it never tested float, the price band, change vs the PRIOR CLOSE,
or the live scanner's own qualification predicate, which is a deterministic function of the bars and is therefore
replayable point-in-time. That replay is cell 1,444's hypothesis made testable.

## Base book (fixed)
The 9,911 fills of cell 1,438 (`research/hod_entry/causal_arming_causal.csv`, status = fill), TRAIN-H2 and VAL; TEST
sealed. Outcomes are NOT recomputed: this is a FILTER study — every cell partitions the same fills and scores the kept
subset. No constant of the order changes.

## Cost convention (new standard for every HOD number from here on)
net R = raw R − (exit half-spread + measured stop slip)/R. The entry is an ask fill (`causal_arming.resolve_window`
sets fill = ask), so no entry half-spread is charged on top of it: `sip_rebuild.trade_result` charges
(half_entry + exit_half + 2 bps × exit)/R, which double-counts the entry side. Recomputation from the CSV without a
refetch: rows with exit_half_src = 'fill_instant' (3,388 of 9,911) have exit_half = half_entry, so half_entry =
(cost_R × R − 0.0002 × exit_price)/2; rows with exit_half_src = 'nbbo' (6,523) take exit_half from
`research/bf_zero/causal_filter/nbbo.csv` (spread_mean/2 on (day, symbol)) and half_entry = cost_R × R − exit_half −
0.0002 × exit_price. corrected net R = net_R + half_entry/R. Stop slip: cell 1,443's per-trade measured slip where its
per-day cache (`sip_cache_stopslip/`, keyed (symbol, exit_m, why, fill_min)) has it, else the holdout mean (35 bps);
EOD exits the same with 1,443's EOD measure. The flat 30 bps variant is reported beside for continuity.

## Disclosure — numbers seen before the freeze
While checking whether the float term is usable at all (float exists only as a CURRENT snapshot in cache.db `universe`,
known for 5,045 of 9,911 fills, updated 2026-04..2026-09), two marginal splits of the base book were read: float ≤ 50M
mean net R −0.28 (n 3,225, uncorrected cost) vs −0.22 unknown; level ≤ $30 −0.23 vs > $30 −0.21. Neither separates on
its own; no cell below was chosen because of them. The float term stays in the composite as the scanner uses it and is
disclosed as snapshot-dated (a stock's float changes slowly; offerings/splits are the exception).

## Cells (all evaluated at the close of arm bar j; nothing after bar j enters any feature; ONE joint cell is allowed at the end: the best liquidity cell ∧ the best mover cell, pre-declared as 1,456 with its own VAL read — that is the only composition permitted)
| cell | condition kept (evaluated with data through arm bar j only) | source of each term |
|---|---|---|
| 1,445 PRIMARY | Q = the live scanner's intraday qualification replayed at bar j (`scanner/criteria.py::evaluate_intraday` + universe membership): prev_close ∈ [$1, $30] ∧ float known ∧ 0 < float ≤ 50M ∧ close_j ∈ [$1, $30] ∧ max(gap_j, range_j) ≥ 15 %, where gap_j = (close_j − prev_close)/prev_close × 100 and range_j = (running RTH high_j − running RTH low_j)/running RTH low_j × 100 | prev_close: Databento PIT daily close of the prior session; float: cache.db `universe.float_shares` (current snapshot — disclosed); bars: bars_sip.db RTH bars ≤ j |
| 1,446 | Q without the two price terms (prev_close band and close_j band removed) | as above |
| 1,447 | 1,446 with the threshold at 10 % (the cache builder's value) | as above |
| 1,448 | 1,446 without the float term = the pure mover term max(gap_j, range_j) ≥ 15 % | bars, prior close |
| 1,449 | float known ∧ ≤ 50M alone | float snapshot |
| 1,450 | HOD level ≥ prior session's high (multi-day breakout, no overhead supply) | Databento PIT daily high of the prior session |
| 1,451 | HOD level ≥ max high of the prior 20 sessions | Databento PIT daily highs |
| 1,452 | LIQUIDITY: share of RTH minutes from 09:30 through m[j] that have a bar in bars_sip.db ≥ 0.90 (the causal analogue of the check's "tracked" = minute completeness; thin names print far fewer bars) | bars_sip.db |
| 1,453 | dollar volume through bar j ≥ $1M (Σ v × c over RTH bars ≤ j) | bars_sip.db |
| 1,454 | quoted spread at the fill instant ≤ 10 bps, spread = 2 × half_entry / fill with half_entry recovered as in the cost section (the last valid NBBO at the trigger print) | the CSV's cost columns (+ nbbo.csv) |
| 1,455 PLACEBO (look-ahead, report-only) | `trading/bf_selection.mover_day_qualifies` on the FULL day's PIT daily bar: (range ≥ 10 % ∨ (high − prev_close)/prev_close ≥ 10 %) ∧ close ∈ [$1, $30] (current config; historical builds may have used $2–20 — disclosed) ∧ (float unknown ∨ ≤ 50M) | Databento PIT daily bar incl. the close |
| 1,456 JOINT | the best liquidity cell (1,452–1,454) ∧ the best mover cell (1,445–1,448), each chosen by TRAIN-H2 kept mean ONLY; VAL read once for the joint | as above |

Definitions: "arm bar j" = the last RTH bar with minute < fill_min (the fill is in bar j+1); "running" = over RTH bars
from 09:30 through j; bars_sip.db timestamps are UTC (convert to ET with zoneinfo; RTH = 09:30–15:59 ET); bars are
trade-driven, so a missing minute is a minute without a bar. Databento rows join on instrument_id with
`equs_instrument_symbol_map.csv` (d0 ≤ day ≤ d1); "prior session" = the previous bar_date present for that
instrument. Test tickers (^Z[A-Z]ZZT$) are already excluded from the base book.

The placebo must reproduce the +0.24 / +0.28 cohort (≈ +0.34 / +0.38 corrected) — it is the calibration of the target,
never a rule. A causal cell is "isolating the cohort" when its kept set's mean net R is ≥ half the placebo's.
Expected sizes: 1,445 is tiny on this $20+ universe (only $20–30 names) — reported regardless; 1,446–1,448 carry the
hypothesis. Power: a cell keeping 20 % of VAL (≈ 1,100 fills, σ ≈ 1.2 R) has SE ≈ 0.036 R, so +0.15 R is detectable.

## Pass bar (frozen; applies to each causal cell separately)
On VAL: kept-fill mean net R ≥ +0.15 (corrected cost, measured slip), day-clustered t ≥ 2.5 (eleven causal cells incl. the joint → the
bar is raised from 2.0), ex-top-5 % > 0, ≥ 3 kept fills/week at first-12/day 4-concurrent (`run_consol.simulate_slots`),
same sign on TRAIN-H2 with t ≥ 1, and the DROPPED set's mean net R < the kept set's on both holdouts (a filter must
separate, not just shrink). Report for every cell: n kept / dropped, both means, ΔR, t, ex-top-5 %, fills/wk, winner-
capped (+3 R) mean, and the count-matched null (the mean of 1,000 random subsets of the same size). TEST is read ONCE,
for the single best causal cell on VAL, only if it passes the bar.

## Consequences (pre-committed)
PASS on 1,445–1,448 → the live engine arms only names in the scanner's qualified set (`_qualified_stock_data` is in-process);
dry 5 sessions with the parity ledger + `scanner_qualified_at_arm` column, then $50 real orders under the 9/25 fixes and
caps. PASS on 1,450/1,451 → the same with the multi-day-high check from `daily_bars` at arm time; PASS on 1,452–1,454 →
the same with a bar-density / dollar-volume / spread gate at arm (all three are one-line checks in the engine). FAIL on
all → the positive cohort is not causally isolable with these terms; the residual is reported with its MDE and the study
is closed on this population.

## Independent check (before any number reaches the owner)
A second agent that has not read the first implementation rebuilds every flag from this prose and compares per
(day, symbol): the flag agreement must be ≥ 99 %, the kept-set means within 0.02 R. Three refuters then attack the best
cell on look-ahead (every term's timestamp), data (float snapshot date; symbol mapping; delisted names), and statistics.

## Not allowed
Adding a cell after a number exists; choosing the joint cell's components on VAL; moving a threshold; recomputing outcomes; reading TEST for more than one cell; using
cache.db for any feature; using a float or prior-close value dated after the symbol-day.

## Execution plan (cascade; the main session only judges)
1. Builder (Sonnet): `research/hod_entry/cell_1445.py` + unit tests; writes `cell_1445_features.csv` (one row per base
   fill: day, symbol, fill_min, split, every flag, corrected net R, slip variant) and `RESULT_1445.md` (one table: cell ×
   holdout). 2. Independent rebuild (Sonnet, has not read the builder's code): every flag from this prose →
   `cell_1445_rebuild.csv`; agreement per flag and kept-set means. 3. Three refuters (look-ahead / data / statistics)
   on every cell that clears the bar on VAL. 4. Judge. TEST read once only after 1–4, for the single best cell.
