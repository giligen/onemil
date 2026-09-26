# RESULT — cells 1,466 (JOINT, pass bar applied) / 1,467 (report-only)

1,464 R-floor rebuild verdict: **NOT_REPRODUCED** (49.9 % of fills within 0.01 R of the builder's `net_R_1464`,
bar is ≥99 %; VAL paired Δ -0.0088 R, within the ±0.02 bar) — per PREREG_1466.md's own fallback rule this composes
on `cell_1464_rebuild.csv` (the independent re-walk), not the builder's `net_R_1464`/`slip_bps_1464` columns (which
also lack the per-fill `R_new`/`exit_price_new` the stop-limit substitution needs). The disagreement traces to a
disclosed interpretation choice in the rebuild's own docstring: it decomposes cost using the *pooled* holdout-mean
stop-slip bps on both sides, while the builder charges *per-fill measured* slip where available — a real,
documented divergence, not a coding bug. 1,463's 20 bps stop-limit variant is taken as given-verified (holdout
mean slip 2.9 bps TRAIN-H2 / 3.2 bps VAL) and substituted for the pooled 35.9/34.8 bps fallback on stop/stop_bar
exits only; non-stop exits are untouched. Scored with cell_1445/1457's own functions (day-clustered t, ex-top-5 %,
fills/wk via `run_consol.simulate_slots`, +3R winner cap, count-matched null, seed 1466).

| cell | holdout | n_kept | n_dropped | kept_mean | dropped_mean | t_kept | ex_top5 | fills/wk | week_p10 | green_wk % | null_pctile | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1466 | TRAIN-H2 | 1061 | 3337 | +0.122 | -0.107 | 2.21 | +0.023 | 25.0 | -7.94 | 63.0 % | 100.0 | FALSE |
| 1466 | VAL | 1223 | 4290 | +0.054 | -0.114 | 1.08 | -0.047 | 27.0 | -10.28 | 54.5 % | 100.0 | FALSE |
| 1467 | TRAIN-H2 | 4398 | 0 | -0.051 | n/a | -1.28 | -0.159 | 39.7 | -36.96 | 37.0 % | 95.9 | n/a (report-only) |
| 1467 | VAL | 5513 | 0 | -0.076 | n/a | -1.90 | -0.185 | 41.0 | -51.23 | 22.7 % | 70.9 | n/a (report-only) |

1,466 FAILS: VAL kept mean +0.054 < +0.15, VAL t 1.08 < 2.5, VAL ex-top-5 % negative (TRAIN-H2 t 2.21 and both
holdouts' dropped<kept clauses do pass). 1,467 (whole book, both execution fixes applied) stays clearly negative
on both holdouts (-0.05/-0.08 R) — the two execution mechanisms alone do not turn the population positive. Per
PREREG_1466.md's pre-committed consequence: **1,466 FAIL closes the resting-order HOD-break book as a money book
at every filter and every exit tried on this population; no further cell.** TEST was not sought (absent from
these files, and moot given the VAL fail).

## Judge's verdict (2026-09-26 07:40 UTC) — round 3 closes the revival
* 1,463 stop-limit exit 20 bps below the stop: VERIFIED on fresh tape (random 400 stops per holdout, seed 1463; the
  stop-market control reproduces cell 1,443's 35 bps): filled stops slip 2.9 / 3.2 bps, no-fill tail 14 % / 11 % at
  94 / 76 bps → ≈ 13 bps blended vs 35. Ships to the dry run as an instrument (`RESULT_1463_unbiased.md`). Caveat for
  the live version: the fill-at-print convention assumes queue priority; the live StopMonitor measure decides.
* 1,464 R floor 2.5 %: NOT verified. Two independent implementations agree on the aggregate (+0.134 / +0.118 R vs
  +0.140 / +0.127, t 8.6 vs 9.7) but only 49.9 % of fills within 0.01 R (builder defect: a fill whose exact minute is
  missing from the bar table silently keeps the base outcome; spec gap: how an existing stop row's cost splits into
  spread and slip). Refuted on tail dependence with high confidence: ex-top-5 % the lift is +0.008 / −0.005; the top 5 %
  of VAL fills (276) carry 104 % of the ΔR sum — noise stop-outs rescued into targets/EOD, not a broad gain. It stays a
  logged counterfactual in the dry ledger, not a rule.
* 1,466 joint (spread ≤ 10 bps ∧ R floor ∧ stop-limit): VAL +0.054 R, t 1.1, ex-top-5 % −0.05 — FAIL. 1,467 whole book
  with both fixes: −0.05 / −0.08 R — FAIL. Per the pre-commitment the resting-order HOD-break book is CLOSED as a money
  book at every filter and every exit tried on this population (23 cells, 1,445–1,467, each independently rebuilt).
