# PREREG — round 3 (cells 1,466–1,467 + verification of 1,463/1,464): the execution fixes, verified and composed once

FROZEN 2026-09-26 06:50 UTC before any new number. Programme count on the HOD line: 1,465 → 1,467. Disclosed as
post-hoc: the components below were chosen after their own cells' numbers were seen (1,454 as the strongest separator
of round 1, 1,464 as the only passing cell of round 2, 1,463 pending an unbiased measurement). The bar is unchanged and
TEST is read once for 1,466 only.

## Verification of 1,464 (R floor at 2.5 % of price) — must pass before it counts
Independent rebuild from this prose by an agent that has NOT read `cell_1440.py` / `cell_1457.py`: for every base fill
(`causal_arming_causal.csv`, status = fill), stop' = min(stop, fill × 0.975), R' = fill − stop', target' = fill + 2 R';
the fill bar (minute = fill_min) is stopped at stop' if its LOW ≤ stop' (conservative); otherwise walk the minute bars
after the fill bar with `sip_rebuild.walk_path` semantics (stop-first on a bar touching both, gap-through at the open,
the 15:55 bar exits at its open); bars from `causal_arming.load_day_bars` (the source with more RTH bars, SIP on a
tie); cost = exit half-spread (unchanged in $) + the measured or fallback stop slip in $ (unchanged bps), both re-scaled
by R'; net R' = raw' − cost'. Fills whose stop is unchanged keep the base outcome exactly. Agreement bar: ≥ 99 % of fills
within 0.01 R of the builder's `net_R_1464` column in `cell_1457_features.csv`; paired ΔR on VAL within 0.02 R.
Refuters (look-ahead in the re-walk; data: bar source, fill-bar convention; statistics: paired t by day, week P10,
ex-top-5 %, drop the best 2 days). PASS stands only if the rebuild agrees and no refuter refutes with high confidence.

## Unbiased 1,463 (stop-limit exit)
Random sample, seed 1463, of 400 stop exits per holdout from the base book (why ∈ {stop, stop_bar}), FRESH tape for the
exit minute via `causal_arming.fetch_window(symbol, day, exit_m, exit_m + 1)` (no reuse of the 1,443 cache). Per stop:
t0 = first print ≤ stop; stop-market = the NBBO bid at t0 + 250 ms; stop-limit L = stop × (1 − L bps) for L ∈ {20, 50}:
filled at the bid at t0 + 250 ms if bid ≥ L, else at the first print ≥ L after t0 within the minute, else (no-fill tail)
at the minute's last print. Report per variant × holdout: n, mean / median / p90 slip bps vs the stop, no-fill count and
its mean slip, and the stop-market mean (must agree with 1,443's 35 bps within 5 bps or the sample is VOID). Ship bar
unchanged: mean slip lower by ≥ 10 bps AND no-fill mean slip ≤ 100 bps on both holdouts.

## Cells
| cell | condition | role |
|---|---|---|
| 1,466 JOINT | spread at the fill ≤ 10 bps (1,454's flag) ∧ R floor 2.5 % (1,464's re-walk) ∧ stop-limit exit slip (the verified 1,463 variant's holdout mean applied to the re-walked stop exits in place of the measured stop-market slip) | the only composition; VAL read once; TEST once if it passes |
| 1,467 (report-only) | the WHOLE base book under R floor 2.5 % ∧ stop-limit slip — the "execution-fixed" population book | the honest expectation of the live rule if both mechanisms ship |

Pass bar for 1,466: VAL kept mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at 12/4,
TRAIN-H2 same sign t ≥ 1, dropped mean < kept mean on both holdouts.

## Consequences (pre-committed)
* 1,464 verified → the live engine gets `stop_floor_pct` (stop = min(consolidation low, fill × (1 − 2.5 %)), default
  off, ON for the HOD dry run) with a parity test against the research walk; 1,463 verified → the StopMonitor HOD exit
  becomes a stop-limit at the verified offset (default off, ON for the dry run). Both are engineering items with their
  own rehearsal; neither turns real orders on.
* 1,466 PASS → dry run 5 sessions with the spread gate at arm, then $50 real orders under the 9/25 fixes and caps.
* 1,466 FAIL → the resting-order HOD-break book is closed as a money book at every filter and every exit on this
  population; what survives are the two execution mechanisms (for any future book) and the ceiling number (+0.17 R with
  perfect foresight at the old execution) as the reason. No further cell on this population.

## Not allowed
Any other composition; moving 2.5 %, 10 bps or the limit offsets; reading TEST for 1,467; treating the biased 1,463
subset as evidence.
