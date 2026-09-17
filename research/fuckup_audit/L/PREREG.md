# Stage L — TRANSFER FILTERS. Pre-registration.

Written 2026-09-17 21:xx UTC, **before any scoring run of this stage**. Nothing below is changed after the
first number exists; amendments are appended at the bottom with their timestamp and the reason.

## 0. Why this stage exists and what makes it different

Every filter tested in Stages H and D5 was derived from the book's OWN losers and then measured on the same
book's VAL. All of them inverted (`H/F6/REPORT.md` §5.2, `H/F14_F8_F11/REPORT.md`: 13 of 22 vetoed buckets flip
sign on VAL; `LOG.md` 2026-09-17). Stage L tests ONLY filters whose evidence comes from **another book or
another period**, applied at the threshold that other book already ships, **with no tuning of any threshold
inside this stage**. A filter that fails here fails as a transfer, which is a different and more useful fact
than a filter that fails as a fit.

## 1. Books (6). Each is taken AS ALREADY SCORED — no book is re-derived.

| id | book | per-trade source | R definition | costs | slot rule |
|---|---|---|---|---|---|
| **B1** | bull flag, raw detector (regen-7 honest cache) | `data/bull_flag_cache_causal_full_20260905.csv`, rows with `entry_price > stop_loss` and `shares > 0` | `R = pnl / (shares x (entry_price - stop_loss))` | already inside `pnl` (regen-7 exit simulation); no extra spread charge | `trading.hod_break.run_book(12, 4)` |
| **B2** | F6 red-to-green, FIRST-BREAK hold (implementation B) | `H/F6_rebuild/scan_ai.csv` (7,606 pre-book candidates) minus `^Z[A-Z]ZZT$` | net R from `H/F6_rebuild/book.py::net_r` (`hold` legs) | `book.py` cost contract (cost_curve median spread by price x hour band; entry 0.25 x half; stop 0.875, eod 0.412, target 0.875) | `book.py::run_book_idx(12, 4)` |
| **B3** | F14 second break, hold | `H/F14_F8_F11/pop_F14.csv` (= `C/pop_c.csv` fam F14 cfg `{"N": 15}`) | `hcore.scoreable(outcome='hold', floor=True)` net | Stage-C contract (c) | `trading.hod_break.run_book(12, 4)` |
| **B4** | F8 N=30 opening-range break, hold | `H/F14_F8_F11/pop_F8N30.csv` | same | same | same |
| **B5** | ORB B+ honest fills | `analysis_results/orb_features_20260916_2053.csv` + the resim dump `D1_orb/candidates_dump.csv` | `R = pnl_pct / range_size_pct` (the `orb_timestop_validation.py` definition) | inside the pipeline's exit physics (30/10 bps) | ORB's own: composite rank, Q4-preference, family/super-group dedup, top N=3, then the shipped post-selection vetoes |
| **B6** | S1 gap-fade SHORT, hold | `G/candidates_short.csv`, `fam == 'S1'`, universe **UB** (`in_u12 == 1` AND `range_so_far_pct >= 5`), `ssr != 1`, entry >= $10, `entry_m <= 841`, `r_pct >= 1` | `G/score_short.py` net (mirrored contract, borrow 0 = primary) | Stage-G contract | `trading.hod_break.run_book(12, 4)` |

**Standing rules applied to every book** (PLAN §1): test tickers `^Z[A-Z]ZZT$` excluded everywhere;
every feature with < 100% coverage gets a missingness table per split BEFORE it is used, and a filter whose
feature is missing on a trade **keeps** that trade (fail-open) — a missing value never removes a trade.
Splits TRAIN 2025-01-01..2025-12-31 / VAL 2026-01-01..2026-05-31 / TEST 2026-06-01 onward. TEST is read ONCE,
only for cells that pass on BOTH TRAIN and VAL, after this file is frozen.

## 2. Filters (8). Threshold and form fixed here; NOTHING is tuned in this stage.

| id | rule | evidence it transfers FROM | live-computable? |
|---|---|---|---|
| **T1** | keep only `prev_day_range_pct >= 8` | ORB's shipped PDR veto (`trading/orb_pdr_veto.py`; ships at 11.0 in B+, the study threshold is 8.0 and 8.0 is what `H/F6` and `I` transferred) | yes, prior daily bar |
| **T2** | keep only `range_so_far_pct >= 5` at the SIGNAL bar (strictly prior bars) | Stage E (8 of 10 long family x split pairs), Stage G (12 of 12 short pairs) | yes, streamed bars |
| **T3** | keep only trades where SPY's return from the 09:30 open to the **close of the minute before entry** is > 0 (for **B6, the short book, the mirror**: SPY return < 0) | `H/F6` §A3 near-miss, `probe_days.md` (day direction separates the book, t 4.6; the only causal carrier is intraday) | yes, one ETF quote |
| **T4** | keep only the FIRST trade the book takes on that day (entry ordinal 1 of the booked sequence) | `D5_r2g/REPORT.md`: entry ordinal holds its sign in all three splits (+0.164/+0.681/+0.148 at ordinal 1, decaying to +0.038/+0.033/-0.067 at 4) | yes |
| **T5** | keep only entries with `entry_m < 600` (before 10:00 ET) | `research/bf_consistency/README.md` §6e + BF live (10:00-10:15 -0.15R 2025 / -0.01R 2026; live BF 10:xx -$10,945 on 26 trades) | yes |
| **T6** | **exit rule, not a veto**: at entry_m + 10 minutes, if `(open of that bar - entry) / R < +0.25` (short: `(entry - open)/R < +0.25`), exit at that open; else keep the book's own exit | `orb_timestop_validation.md` (lifts the raw ORB population in all three splits: -0.133 -> -0.056 TRAIN, +0.026 -> +0.044 VAL, +0.022 -> +0.056 TEST) | yes |
| **T7** | **exit rule, not a veto**: if the FILL minute's volume is `< 1.5x` the mean of the 5 bars before it, exit at the open of the next bar; else keep the book's own exit | `live_followthrough.md` (2 of 3 periods on the BF cache; the live BF separator) | yes, one minute after the fill |
| **T8** | keep only trades whose stop is `>= 3%` of the entry price (`r_pct >= 3.0`) | the cost mechanism, `RESULTS.md` stage 2b / `probe_costs.md` (spread cost 0.19R at a 1-2% stop vs 0.05R at 5-8%; stop rate 51% vs 17%) | yes |

**Cost of a forced exit (T6, T7).** A forced exit is a market order at a bar open. It is charged the STOP
coefficient (0.875 x half), the most expensive leg in the contract — deliberately conservative, so a T6/T7
pass cannot be bought with a cheap-exit assumption. For **B1** and **B5**, whose costs live inside the book's
own P&L rather than in an explicit spread contract, the forced exit is charged **0.3% of R** (the convention
`orb_timestop_validation.py` already used) on top of the recomputed gross.

**Book rule after filtering.** The book's own slot rule is RE-RUN on the filtered population so freed slots
refill (12/day + 4 concurrent for B1-B4 and B6; for B5 the filter is applied to the candidate universe
BEFORE ranking so the pipeline's top-N selection refills by construction, and every post-selection shipped
veto stays exactly as it is). No cell is ever "the base book minus some trades".

## 3. Applicability. A filter is skipped where it is structurally degenerate, and the reason is declared HERE.

| book | skipped | reason |
|---|---|---|
| B2 | T1 | the book's universe is already `prev_day_range_pct >= 8` (`H/F6_rebuild/prefilter.py`) — 100% pass |
| B2 | T2 | the book's own scan already applies the `range_so_far_pct >= 5` floor — 100% pass |
| B3, B4 | T2 | the Stage-C population contract applies the same floor — 100% pass |
| B5 | T1 | ORB ships the PDR veto (`prev_day_range_pct >= 11` in B+) — already in the book |
| B5 | T5 | ORB enters at 09:35 only — 100% pass |
| B6 | T2 | universe UB is defined as `range_so_far_pct >= 5` — 100% pass |

Skipped cells are still COUNTED (they were looked at to decide they are degenerate).

## 4. Cells

6 books x 8 filters = **48** single-filter cells, of which **7 are declared degenerate above** and **41 are run**,
plus **6 best-two stacks** (one per book) = **54 declared cells**. The stack for a book is, mechanically:
*the two filters with the largest TRAIN improvement among those whose VAL improvement sign is >= 0*; if fewer
than two qualify, the stack is not run and that is reported as the cell's outcome. Every additional number
printed (missingness tables, per-split base statistics, tail tests, permutation draws) is a diagnostic of a
declared cell, not a new cell, and the total is restated in the report.

## 5. Decision rule (frozen)

A cell **passes** iff all three hold:
1. TRAIN improvement in mean net R per trade `>= +0.03`;
2. VAL improvement `>= 0`;
3. VAL mean net R of the filtered book `> 0`.

Only passing cells have TEST read, and TEST is read once, in a single scripted pass, after this file is frozen.
For every passing cell the report additionally prints, without gating on them: the tail test (mean with the top
5% of trades removed; mean with winners capped at +3R), the per-split trade count and trades/week, and the
$/month.

**Multiplicity.** A 200-draw day-level sign-flip null is run over ALL declared cells: within a draw each
trading day's net R is multiplied by a random +-1 (the same sign for the base book and the filtered book on
that day, so the pairing is preserved), every cell's improvement is recomputed, and the maximum |improvement|
across the grid is recorded. The observed maximum is reported against that null's 95th percentile and the
implied p. Booking is independent of net R (`trading.hod_break.run_book` orders by entry minute and symbol), so
the draw does not need to re-book.

**$/month.** B1-B4 and B6 at **$300 of risk per trade** (the capacity number `H/F6_sizing/REPORT.md` and
`I/REPORT.md` settled for the small-cap intraday books): `$/month = total net R over the split / months x $300`.
B5 at the shipped **$10K stage sizing**, read straight off the pipeline's own book P&L. No number in this stage
is a forward expectation; all of them are relative arithmetic on a fixed window.

**Smallest visible effect.** For every book the report prints `MDE = 2.8 x SE` of the BASE book's per-trade net
R on TRAIN and on VAL. A filter whose true effect is under that MDE cannot be seen here, and the report says so
rather than calling it absent.

**Phrasing.** No cell result is ever written as "no edge exists". The supported form is
"no improvement was detectable for THIS filter on THIS book over THIS window at THIS book size, and the
smallest per-trade improvement the test could have seen is X R".

## 6. Verification required before any cell number is reported

1. Each book's BASE statistics must reproduce the already-published anchor for that book (`H/F6_rebuild/REPORT.md`
   §2 for B2, `H/F14_F8_F11/REPORT.md` for B3/B4, `G/REPORT.md` for B6, `D1_orb/REPORT.md` for B5,
   `live_followthrough.md` for B1's un-booked mean R), or the deviation is explained in writing before anything
   else is read. Where a book is booked here for the first time (B1) the un-booked anchor is the check.
2. Availability audit for every filter feature, per book and per split, printed before the grid.
3. Every forced-exit fill (T6, T7) must be inside the bar that supplies it (`low <= fill <= high`); the share of
   fills failing that test is printed and those trades keep their original exit.

## 7. Resources

One `nice -n 10` python process at a time, `ulimit -v 1500000`, bars read per (symbol, day) from
`research/bf_zero/bars_sip.db` with `data/cache.db::intraday_bars_1min` as the fallback, both `mode=ro`;
SPY/IWM from the cached `H/F14_F8_F11/etf_minute_ret.csv` (built from `research/lit_review_2026/etf_1min.db`).
Everything this stage writes lives under `research/fuckup_audit/L/`. No config, service, cache, cron or order is
touched.

---

### Amendments

**A1 — 2026-09-17, disclosed immediately after the step-1 run, not repaired.** `l1_pop.py`'s parity table
prints the BASE (unfiltered) book's statistics for all three splits, TEST included, because the anchor
comparison is written as one loop over `C.SPLITS`. So the base book's TEST mean net R was seen before the grid
was scored: B1 −0.0863, B2 −0.0420, B3 −0.0455, B4 −0.0449, B6 −0.0098. Three of those five were already
published (`H/F6_rebuild/REPORT.md` for B2, `live_followthrough.md` for B1, and B6's population is Stage G's);
B3's and B4's TEST base numbers are new here. This is an unfiltered aggregate, it selects nothing, and no
filter cell's TEST number is read until the freeze — but it is a deviation from "TEST is read once" and is
recorded as one, exactly as Stage H recorded the same slip.

**A2 — 2026-09-17, disclosed after the B5 runs, not repaired.** `study_orb_pipeline_static_lock.py` prints and
writes the WHOLE-WINDOW book, so every B5 cell's 21-month dollar total was visible the moment its run finished,
TEST months included: base $7,186.64 / T2 $3,132.81 / T3 $2,010.32 / T4 $2,745.66 / T6 $4,709.48 / T7 $3,179.60
/ T8 $11,544.76 / STACK T4+T8 $5,732.37. Per-SPLIT TEST statistics were still computed only for the two B5 cells
that passed. The alternative — re-writing the pipeline to hide its own output — would have been a bigger risk
than the disclosure, so the totals are reported here in full rather than quietly used.

**A3 — 2026-09-17, found at run time.** T5 (entries before 10:00) returns an EMPTY population on B3 and B4:
F14's second break needs a first break, and F8 N=30's opening range is 30 minutes long, so neither family can
produce a signal before 10:00 (min `entry_m` = 601 on both). Those two cells are reclassified as structurally
degenerate alongside the seven declared in §3, taking the degenerate count from 7 to 9 and the run count from
41 to 39 single-filter cells. The declared total of 54 cells is unchanged.

