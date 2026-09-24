# PREREG — the breakout thermometer: does "breakouts have been working lately" predict the next day? Cells 1,420–1,422

Frozen 2026-09-24 ~19:20 UTC, before any number below exists. Programme count 1,419 → 1,422.

## Why
ORB made +0.27 R/fill in 2025 and ≈ 0 in 2023–24 (`research/orb_2023/REPORT.md`); the bull flag's 2024H2 book is
~breakeven. The books look like one regime bet. If the regime is PERSISTENT and visible in real time, a book can size up
while it is on and stand down while it is off — the only honest path from a regime-bound book to the $10K/month north
star. Earlier regime work used SPY trend/volatility states (`research/regime/`, cells 1,310–1,311, NO-SHIP) and an
intraday risk-on tape (reversed out of regime). This is different: the thermometer is the recent outcome of the WHOLE
breakout population (every candidate that triggered, not just the few the book took) — strategy-return momentum
(factor momentum: Gupta-Kelly 2019, Ehsani-Linnainmaa 2022), measured on hundreds of trades per window, not on the
book's own ~40.

## Data (all exist; nothing fetched)
* ORB population, ONE code version (`2026-09-05.entered_inclusive`), production seed (gap ≥ 5 %, open $3–30):
  2023-01..2024-06 `research/orb_2023/out_1418/orb_features_20260924_0439.csv`; 2024-07..12
  `research/orb_2024/out_1415/orb_features_20260923_2052.csv`; 2025-01..2026-09-23
  `analysis_results/orb_features_20260923_2049.csv`. Read with `trading/orb_csv.read_orb_csv` (ticker NA).
* ORB book (live config, catalyst veto OFF, 8 slots, $375): `research/orb_2023/book_1418.csv`,
  `research/orb_2024/book_1415.csv`, and 2025-01..2026-09 REBUILT with the identical invocation
  (`ORB_BT_FEATURES_CSV=<the 2025-26 CSV> ORB_BT_BOOK_OUT=research/thermo/book_2025_26.csv ORB_CATALYST_VETO=0
  python3 study_orb_pipeline_static_lock.py`). Consistency check before use: its 2025 fills must match
  `research/orb_seed_wide/out/runB_true.csv` 2025 within 5 % on n and total $ (else STOP and report).
* HOD population and book: `research/hod_exit_lab/b0_trades.csv`, splits TRAIN and VAL ONLY — TEST rows are dropped on
  read and never touched.

## Thermometers (causal: day d uses trading days strictly before d)
* **T_ORB(d)** = win rate (share with `pnl > 0`) of ALL `entered == 1` rows of the ORB population over the 20 trading
  days before d (every triggered breakout, selected or not). Requires ≥ 40 breakouts in the window, else undefined.
* **T_HOD(d)** = mean B0 `net_R` of ALL HOD signals over the 20 trading days before d (≥ 40 signals, else undefined).
* **Hot(d)** iff T(d) > the EXPANDING median of all earlier defined T values (≥ 60 of them, else undefined). No
  parameter is fit anywhere: the 20-day window, the 40 / 60 minimums and the median cut are fixed here.

## Cells
| cell | book | thermometer | sample |
|---|---|---|---|
| 1,420 | ORB live-config fills (R = `_sized_pnl` / 375) | T_ORB | every defined day 2023-01 .. 2026-09 |
| 1,421 | HOD B0 signals (`net_R`) | T_HOD | TRAIN-H2 and VAL (TRAIN-H1 is the burn-in) |
| 1,422 | bull flag P1 Stage-2 trades 2024H2 + 2025-26 (the files `research/bf_2024/REPORT.md` scored) | T_ORB (cross-book) | report-only |

Statistic per cell: hot − cold mean R, t of the hot dummy in OLS `R ~ 1 + hot` with day-clustered SE (iid t beside);
quintile table of next-day R by T (report-only); per-period breakdown (ORB: 2023-24 vs 2025-26; HOD: TRAIN-H2 vs VAL);
hot-cohort ex-top-5 % mean; fills/week in hot weeks; share of days hot per period.

## Pass bars (frozen)
* **1,420 PASS** iff hot − cold ≥ +0.15 R/fill pooled with day-clustered t ≥ 2, AND hot − cold > 0 inside 2023-24 and
  inside 2025-26 separately (each cohort ≥ 15 fills there), AND hot-cohort ex-top-5 % mean > 0.
* **1,421 PASS** iff hot − cold ≥ +0.10 R on TRAIN-H2 AND on VAL, VAL day-clustered t ≥ 2, AND the hot cohort's VAL
  mean net R ≥ 0 (it must make money, not just lose less), AND ≥ 3 hot fills/week at first-12/day 4-concurrent
  (`research/hod_consol/run_consol.simulate_slots`).
* Adversary lenses before relaying either: (i) oracle bound — the same split on the SAME-day T (look-ahead) must be
  larger than the causal one, else the causal number is noise; (ii) a stale thermometer (window ending 60 trading days
  before d) is reported beside; (iii) the ORB pooled result must not come only from the 2024→2025 level shift — that
  is what the inside-period legs are for.

## Consequences (pre-committed)
* 1,420 PASS → PREREG a live ORB flag: cold days trade at 0.5× stage risk, hot days at stage risk; the ramp counts
  hot-day fills only; T_ORB computed each morning from the nightly features CSV (the same file the study uses).
  FAIL → no regime sizing on ORB; record the MDE.
* 1,421 PASS → the HOD dry run tags hot/cold days for 10 sessions side by side, then an exploration-tier live
  proposal for hot days only. FAIL → record; the HOD population is also time-uninformative at this horizon.
* 1,422 is report-only (26 + ~79 trades cannot carry a claim); it informs whether a PASS on 1,420 transfers to BF.

## Not allowed
Changing the window, minimums, cut, the outcome definition or the pass bars after any number exists; reading HOD
TEST; refitting anything in the books.

## Amendment 2026-09-24 ~22:05 UTC (before any thermometer number: the first run STOPPED at the consistency gate)
The gate's reference `runB_true.csv` was built with the catalyst veto ON (`research/orb_seed_wide/REPORT.md` line 12,
"catalyst veto at its code default = ON"); the 2023-24 books, the rebuild and LIVE (since 9/21) are veto OFF — so the
rebuild (2025: 211 fills / $8,377) could never match it (85 / $6,561). The gate's purpose is to prove the rebuild
MACHINERY, so it is re-pointed: a veto-ON rebuild of the same features CSV (`book_2025_26_vetoON.csv`) must match
`runB_true` 2025 within 5 % on n and $; if it does, cell 1,420 is scored on the veto-OFF book exactly as frozen.
Nothing else changes.
