# REBUILD_1619 -- independent rebuild of PREREG_1617.md Frame B (burst fade)

Built from PREREG_1617.md prose only; cell_1619.py / cell_1619_fills.csv / RESULT_1619.md were not opened. See rebuild_1619.py's module docstring for the full mechanism as read and every disclosed interpretation choice.

## VAL (primary)
- n_pop: 5513
- n_elig: 3576
- n_have_tape: 3531
- n_filled: 2202
- fill_share: 0.6236
- retest_share: 0.8474
- stop_share: 0.1526
- eod_share: 0.0000
- cover_within_15min_share: 0.8451
- runner_n: 0
- runner_mean_net_R_f: nan
- mean_net_R_f: -0.6003
- mean_net_pct_price: -0.3596
- median_R_f_pct_price: 0.5991
- t_stat: -26.9118
- ex_top5_mean: -0.6430
- fills_per_week_raw: 100.0909

## TRAIN-H2 (sign check only)
- n_pop: 4398
- n_elig: 2655
- n_have_tape: 2610
- n_filled: 1662
- fill_share: 0.6368
- retest_share: 0.8261
- stop_share: 0.1739
- eod_share: 0.0000
- cover_within_15min_share: 0.8237
- runner_n: 0
- runner_mean_net_R_f: nan
- mean_net_R_f: -0.5997
- mean_net_pct_price: -0.3593
- median_R_f_pct_price: 0.5991
- t_stat: -22.1777
- ex_top5_mean: -0.6424
- fills_per_week_raw: 61.5556

## Pass bar (frozen, PREREG_1617.md)
mean net R_f >= +0.15 AND >= +0.10% of price, t >= 2.5, ex-top-5% > 0, >= 3 fills/week, TRAIN-H2 same sign, median R_f >= 0.5% of price.
**Rebuild verdict: FAIL** (this rebuild's own numbers only -- the independent-check agreement bar in the PREREG, fill-set Jaccard >= 0.98 and >= 99% within 0.01 R_f against cell_1619, is for the comparator to run, not this script).

## Obtainability / refuters (PREREG "Independent check" section, B)
- Population 9911 base fills; excluded not-shortable 3680; no cached tape at all 90.
- Exit phase mix among fills: 
exit_phase
tick    3681
bar      183
- Tick-tape coverage is short per signal (tens to ~100s of seconds of real prints in the sampled files); most retest/stop resolution for fills more than a few minutes from the close falls through to the 1-minute-bar walk (bars_fills_1478.db), mirroring sip_rebuild.walk_path's semantics for a short. This is a material, disclosed departure from a fully tick-priced retest for those rows -- see the module docstring.
- **eod_share is 0.0000 and runner_n is 0: this rebuild's bar-phase walk continues on real, full-day 1-minute bars all the way to 15:55, and at R_f as tiny as ~0.6% of price essentially every position touches either the stop or the retest level somewhere over the rest of the day, so almost nothing "never retests." The PREREG's own report list asks for a "runner losses (the never-retest cohort's cost)" line, which implies cell_1619 expected a non-trivial such cohort -- most likely because its walk is bounded by the ~15-minute tape window itself (no bar fallback), so a position unresolved at that data horizon is booked as the "eod"/runner case there. THIS IS THE SINGLE MOST LIKELY POINT OF DIVERGENCE between this rebuild and cell_1619 -- a comparator should check it first, ahead of any per-row price arithmetic.

## Caveats for the comparator
- half_entry, base_outcome_R joins are on (day,symbol,fill_min,split); any float round-trip mismatch would show as an unmatched row (see [load] WARNING lines in the run log) -- none were logged in this run unless noted above.
- Stop and EOD leg costs are modeled as the COMPLETE cost for that leg (SLIP_STOP_BPS or EOD_BPS alone, no separately-added half-spread+2bp on top). The alternative reading -- additive on top of the standard half-spread+2bp -- was considered and rejected because the PREREG phrases each as "= ..." a single complete recipe; a comparator disagreement concentrated in stop/eod rows' cost_R most likely traces to this choice.
- No SSR column exists in borrow_flags.csv; only `shortable` gates the population.
- Borrow cost (3%/yr pro rata) is on notional at the short's entry price, actual holding seconds; it is negligible (sub-bp) for every intraday hold here and does not drive any conclusion.
