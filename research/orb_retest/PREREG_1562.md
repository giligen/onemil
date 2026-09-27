# PREREG — cells 1,562–1,563: ORB with the HOD learnings — buy the retest one tick under the opening-range high

FROZEN 2026-09-27 05:40 UTC before any number. Programme count: 1,561 → 1,563. Owner 9/27: "can ORB benefit from our
learnings on spread and pricing and potentially buying a tick below the price on its way down (if the same pattern
exists)".

## What was seen (disclosed) and why this cell is different from the HOD retest
* HOD (cells 1,481–1,547): the retest bid at level − $0.01 fills 91 % of breaks and is +0.17 / +0.22 R better than the
  break entry on the same fills (t 22–35, ex-tails +0.15) — the immediacy cost recovered — but the HOD population's gross
  is zero, so the retest book stays at −0.09 R and every exit from the retest instant is negative (0 of 55 cells).
* ORB at the live config: +0.105 R/fill in the backtest (t 3.3, n 473, 2025-01..2026-09; 2023–24H1 +0.02, pooled
  out-of-regime ≈ 0) — the ONE book with a positive gross. Cell 1,426 (`research/orb_latency_bt/REPORT.md`) replayed
  every backtest fill on XNAS ticks with the engine's chase rule: +0.028 R/fill at zero latency, only 21 % of fills
  match the backtest's 30 bps entry model, ~72 bps adverse ask-to-fill drift on the live book. Read together: ORB's
  edge is real and its immediacy cost eats most of it. If the opening-range break withdraws to the level as often as
  the HOD break does, the retest bid recovers that cost on a population whose gross is positive. The retest numbers
  of ORB have NOT been seen.

## Population and data
The cell 1,426 population: every backtest fill at the live config with XNAS tape (`research/orb_latency_bt/
population.csv`, entered == True; the tape windows `research/hod_ofi/raw/*.parquet`, kind == signal, keyed by
symbol_win and entry_m; the replay engine `research/orb_latency_bt/replay.py` defines the trigger level = the
opening-range high, the chase rule, the stop and the live exit walk — the builder reuses its exit walk, the rebuild
implements the exit rule from the prose of replay.py's docstring and REPORT.md, never from the builder's code).
Splits: TRAIN = entries in 2025, VAL = entries in 2026-01..2026-09 (disclosed: no sealed TEST exists for this book;
the forward dry run is the test). Real-fill parity: the live parity ledger is not used here.

## Rule
Base = the zero-latency replay fill of cell 1,426 (results.csv, delay_s == 0, status filled; fill_price and
pnl_replay), re-costed with the stop-limit exit standard so both legs share one cost model. Retest (1,562): at the
trigger print (first XNAS print ≥ level + $0.01), instead of chasing, a BUY LIMIT rests at level − $0.01 for 15 minutes
(through the end of the 15th RTH minute after the break minute); it fills at the limit at the first print STRICTLY
BELOW it (report-only: at-or-below). No print below → no trade (the never-retest cohort is reported with its base
outcome — the runners lost). Stop = the live ORB stop (as replay.py); R′ = entry − stop; target and time exit = the
live ORB exit rule recomputed from R′ (2 R′ if the live rule is a multiple of R; otherwise the rule's own price);
15:55 exit at the bid if the rule holds that long. Path: inside the retest minute the tape decides (a print ≤ stop
after the fill → stopped; a print ≥ target → target), then the minute bars the replay engine uses. Cell 1,563: the
same with the window extended to 30 minutes and the limit at level × (1 − 0.002) (the deeper retest). Costs: entry
passive (none); target limit (none); stop = the stop-limit standard (2.9 / 3.2 bps filled + 12 % no-fill tail at
94 / 76 bps, in units of the book's own R); EOD at the bid (RESULT_1443 means). Report per cell and split: n signals,
fill share, the withdrawal share within 15 min (the HOD pattern check: 87 % there), median dip below the level (bps),
minutes to the retest, mean net R′, day-clustered t, ex-top-5 %, winner-capped +3 R, fills/week, R′ as % of price
(median; the R-vs-spread rail at 0.5 %), the paired ΔR vs the base on the SAME signals with its ex-top-5 %, the
never-retest cohort's base outcome, and the exit mix.

## Pass bar (frozen)
Paired ΔR vs the base ≥ +0.10 R on BOTH splits with day-clustered t ≥ 2.5 on the pooled book and same sign in each
split; the retest book's own VAL mean net R′ ≥ +0.15 with t ≥ 2 (n is small — ORB fires ~4/week — so the paired test
carries the decision and the own-mean is the size check), ex-top-5 % of both > 0, winner-capped positive, ≥ 3
fills/week, never-retest loss disclosed (the book's mean INCLUDING the never-retest signals as zero-trades must stay
≥ the base's mean), median R′ ≥ 0.5 % of price. The better of 1,562 / 1,563 on VAL is named; there is no TEST.

## Independent check and consequences
Rebuild from this prose on the same tape (fill-set Jaccard ≥ 0.99, net R′ within 0.01 R on ≥ 99 % of fills). Refuters:
obtainability (queue priority on a passive bid; the through-print rule; halts; the fill inside the break minute may
only use prints AFTER the trigger print), look-ahead (the level is the opening-range high known at the range close;
the window starts at the trigger print), statistics (tails, day concentration, drop the best 2 days, the two splits).
PASS → the ORB engine gains `entry_mode: retest_bid` beside the chase entry: dry run 5 sessions on the ORB parity
ledger after the execution-repair rehearsal, then live at the current ORB size on the owner's word. FAIL → ORB keeps
the chase entry; the withdrawal share and the paired number stay on record.

## Not allowed
Moving the tick, the windows or the stop after seeing a number; reusing HOD's numbers as ORB evidence; any read of the
live parity ledger as a research input.

## Amendment 1 (2026-09-27 05:50 UTC, before any number) — population and splits as the tape actually is
`population.csv` holds 638 entered backtest fills 2023-01-12..2026-09-23 (not only 2025–26). At zero latency the replay
finds a trigger print inside [09:35, 09:40) ET for 410 of them (301 filled + 109 skipped by the chase guard); 207 have
no trigger print in that window and 21 lack tick data. Population of this cell = the 410 signals with a trigger print
(the chase-guard skips are exactly the signals a resting bid could still take); the 228 others are excluded and
counted. Splits by entry date: TRAIN = 2023-01-12..2025-06-30, VAL = 2025-07-01..2026-09-23 (≈ halves). The base for
the paired test on the 109 guard-skipped signals is a zero trade (the chase rule did not enter) — reported separately
from the 301 paired against the replay fill. The exit rule: replay.py does not walk exits (it re-prices the BT's own
P&L); the builder walks the live ORB exit rule (`trading/orb_engine.py` exit logic as documented in
`research/orb_2023/REPORT_liveexit.md` and the BT walker `study_orb_pipeline_static_lock.py`) on the 1-minute bars the
BT uses (`data/cache.db` intraday bars, READ-ONLY), from R′; the rebuild implements the same documented rule from the
prose of those files, never from the builder's code.
