# PREREG — HOD-break ENTRY: a resting buy-stop at the level, filled on the tape inside the break bar. Cells 1,423–1,425

Frozen 2026-09-25 ~07:45 UTC before any number exists. Programme count 1,422 → 1,425.

## Why
Every HOD-break study so far (causal filters 0/12, exit lab 35 cells, rank, breadth, order flow) kept the SAME entry:
the OPEN of the bar after the break bar closes (`research/hod_exit_lab/PREREG.md` B0). That is the latest possible entry:
it pays the whole move from the level to the next open. The break itself has ≈ 0 gross expectancy at that entry and
−0.22 R net. The one lever never tested is the entry price. A resting buy-stop at the level is what a human breakout
trader (and the live ORB engine) actually does; we now own XNAS tick data for the break bar of every signal
(`research/hod_ofi/raw/`, window [S−300 s, S+120 s] where S = entry_m × 60 and the break bar is [S−60, S)), so the
fill can be read off the tape, not assumed. Mechanism: the distance level → next open is paid by the late entry and
kept by the resting order; the stop is unchanged, so R shrinks and every exit lands in fewer R of adverse move.

## Data
* Book: `research/hod_exit_lab/b0_trades.csv`, TRAIN and VAL only (TEST sealed, dropped on read); columns entry (next
  open), stop, target, R, cost_R, net_R, split, half, wk. Paths after entry: `research/hod_exit_lab/paths.parquet`.
* Level: the HOD level the signal broke — locate its definition in `research/bf_zero/spec_sim.py` /
  `build_candidates.py` (the running max of closed 1-min highs before the break bar) and reproduce it from the minute
  bars the walker used (`data/cache.db` intraday bars — reads before 13:25 UTC only). Tick: 0.01.
* Tape: XNAS trades and mbp-1 quotes per signal (`research/hod_ofi/pipeline.py` loaders; `sec` = ET seconds).
  Availability rail: a signal is usable iff the break bar has ≥ 1 XNAS trade AND a two-sided quote prevails at the
  fill instant; usable ≥ 80 % of signals and winner/loser missingness gap ≤ 5 pp, else VOID.

## Entries (all with B0's stop, B0's 2 R target recomputed from the new R, B0's 15:55 exit, B0's path rules)
* **E1 (cell 1,423) — resting stop-limit at the level.** Trigger = level + 0.01, limit = level × 1.0015 (15 bps chase
  cap, fixed here). The order rests before the break bar. Fill instant = the first XNAS print ≥ trigger inside the
  break bar; fill price = the prevailing XNAS ask at that instant (from mbp-1, strictly prior record) if ask ≤ limit,
  else NO FILL (the signal is skipped, never chased). Inside the remainder of the break bar, a print ≤ stop after the
  fill = stopped at the stop price (conservative). From the next bar on, the B0 walker on `paths.parquet`.
* **E2 (cell 1,424) — idealised level fill** at trigger exactly (no ask, no cap): report-only upper bound.
* **E3 (cell 1,425) — B0 entry re-costed** with the same measured half-spread at the next-bar open (the pairing control;
  B0's published cost_R is the study cost model). Every comparison is E1 vs E3 on the SAME signals.
* Cost for E1/E2/E3: half the quoted XNAS spread at the fill instant on entry, and B0's exit cost rule unchanged.

## Statistics
Paired ΔR (E1 − E3) per signal with day-clustered t (OLS on day clusters), on TRAIN-H2 and VAL; E1's own mean net R;
fill rate (share of usable signals E1 fills); ex-top-5 % of E1; E1 fills/week at first-12/day 4-concurrent
(`research/hod_consol/run_consol.simulate_slots`); the level→next-open distance in R (the chase) as a distribution.
Lenses: (i) XNAS is one venue — report how often the first XNAS print ≥ trigger comes ≥ 5 s after the break bar's open
(a consolidated tape would trigger earlier or later; state the direction of the bias); (ii) the fill-rate–edge trade-off:
E1 at limits 5 / 15 / 30 bps (report-only); (iii) E1 with the fill instant's ask + 1 tick (slippage) as a sensitivity.

## Pass bar (frozen)
E1 PASS iff E1 mean net R ≥ +0.05 on TRAIN-H2 AND on VAL (a BOOK, not merely better than B0), paired ΔR vs E3 ≥ +0.10 on
both with VAL day-clustered t ≥ 2, fill rate ≥ 70 %, E1 ex-top-5 % > 0 on VAL, ≥ 3 E1 fills/week, and the ask+1-tick
sensitivity still ≥ 0 on VAL.

## Consequences (pre-committed)
PASS → the HOD dry run switches its entry to the resting stop-limit (dry, zero orders) for 10 sessions side by side with
the current entry; if the dry book tracks the backtest, an exploration-tier live proposal at $100 risk. FAIL → record
with the MDE; the entry lever on this population is closed too.

## Not allowed
Changing the trigger, limit, cost rule, stop, target or the pass bar after any number exists; reading TEST; using bar
highs instead of prints to assume a fill.
