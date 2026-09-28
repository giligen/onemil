# PREREG — cells 1,623–1,625: the DAY, the CROWD and the INSTRUMENT (frames beyond the fill)

FROZEN 2026-09-28 17:15 UTC before any number. Programme count: 1,622 → 1,625. Owner 9/28: "we just need to improve
0.2R… out of the box, different thinking." Every property of the fill itself is measured (1,622 cells); these frames
condition on things outside the fill: the day's own resolved outcomes, the cross-section of breaks, the instrument.

## Data (all on disk)
Base fills: `causal_arming_causal.csv` (status == fill; day, symbol, split, fill_min; also every ARM row of the day,
any status, with its arm minute — the crowd); outcomes and exit minutes: `model_1478_L3_predictions.csv` (outcome_R,
exit_m); minute bars `bars_fills_1478.db`; SPY / IWM minute bars from `data/cache.db` (READ-ONLY) or Alpaca if absent
(state the source). TRAIN-H2 / VAL as always; TEST sealed.

## Cell 1,623 — the day's own feedback (causal intraday gate)
For each fill f on day d, F(f) = the mean outcome_R of the day's fills whose EXIT minute is strictly before f's fill
minute (resolved before the decision), and n_res(f) = their count. Gate G+: trade f only if n_res ≥ 2 and F ≥ +0.5 R;
gate G−: trade f only if n_res ≥ 2 and F ≤ −0.5 R (the mirror, report-only); the ungated remainder and the fills with
n_res < 2 reported. Report per holdout: n kept, mean net R, day-clustered t, ex-top-5 %, fills/week at 12/4, the
kept-vs-dropped difference, and the autocorrelation table: outcome_R of f vs F(f) in five bins of F. Placebo: the same
gate computed on the OTHER holdout's days shuffled (seed 1623) — the gate must beat the shuffled version.

## Cell 1,624 — break breadth (the crowd)
B30(f) = the number of ARM events (any status, all names) in the 30 minutes before f's fill minute; B60 likewise.
Terciles set on TRAIN-H2 by minute-of-day-adjusted rank (the count rises through the morning — rank within the same
hour bucket, so the gate is not a time-of-day gate). Gate: trade f only in the TOP tercile of B30 (pre-declared);
report the bottom tercile and B60 beside. Same statistics as 1,623.

## Cell 1,625 — the index as the instrument
Signal: a burst — B30 (all names) in its TRAIN-H2 top decile for the first time that day, at minute m*. Trade: buy SPY
at the open of minute m* + 1, exit at the open of minute m* + 61 (60-minute hold) or 15:55, whichever first; a second
leg: IWM the same; cost 2 bps round trip (penny spread on a $500 ETF) + no stop (the exposure is 60 minutes of index).
One signal per day at most. Report per holdout: n days, mean return in bps, day-clustered t, ex-top-5 %, winner-capped
+1 %, the same trade at a random minute of the same day (placebo, seed 1625), and the 30-/120-minute holds beside
(report-only). Pass bar in bps: mean net ≥ +8 bps per signal, t ≥ 2.5, ex-top-5 % > 0, placebo margin ≥ +5 bps t ≥ 2,
≥ 2 signals/week, TRAIN-H2 same sign.

## Pass bar for 1,623 / 1,624 (frozen; VAL)
Kept mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at 12/4, dropped < kept on both
holdouts, TRAIN-H2 same sign t ≥ 1, placebo/shuffle margin ≥ +0.10 R with t ≥ 2, kept cache-only share within 5 pp of
19.5 %. TEST once for the single best passing cell.

## Independent check and consequences
Rebuild from the prose (kept-set Jaccard ≥ 0.99, means within 0.01 R; 1,625 signal days Jaccard ≥ 0.99). Refuters:
causality (F uses exits strictly before the fill minute — the exit minute of a resolved fill must be its own exit, not
the day's 15:55 for open ones; B counts arms before the fill minute; the SPY entry at the NEXT minute's open),
the time-of-day confound in 1,624, day-cohort look-ahead (the ignition lesson: no day-level label leaks into the
gate), tails and day concentration, the shuffle placebo. PASS → the gate goes into the HOD engine as a runtime
counter (1,623 / 1,624) for the $50 run, or an index leg (1,625) as a dry instrument. FAIL → these three frames close;
the report says what the population's days, crowd and instrument do not carry.

## Not allowed
Tuning the thresholds (+0.5 R, n ≥ 2, top tercile, top decile, 60 min) after a number; selecting on VAL.
