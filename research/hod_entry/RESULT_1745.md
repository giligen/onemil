# RESULT 1,745 — HOD-break spec on the liquid top-300 (monthly), 2026-01-02..11-12: FAIL

Spec: `trading.hod_break.arm_state` + `arm_gate_reject`, defaults + min_r_pct 1.5, price >= $5, 8/day, 4 concurrent
(`run_book`), 2R target, flat 15:55. Bar-level fills (bars only; open above limit = no fill; fill bar with low <= stop =
stopped, conservative). Cost: NBBO half-spread measured at EVERY fill minute (1,414 requests, 0 errors; 400-fill
calibration sample median 5.5 bps, p90 14.9; all fills 5.7 / 14.6) + 8 bps chase; stop leg 13 bps; target/eod leg =
per-name x hour median half-spread + 2 bp. Files: 1745_score.py, 1745_fills_scored.csv, 1745_summary.csv.

Population: 1,414 fills (847 TRAIN / 567 VAL), 34/30 uncapped fills/wk, capped 22/17.6; r_pct median 1.9;
obtainability 99.4 % TRAIN / 99.3 % VAL (open > limit only 0.4 %).

| capped book | n | mean net R | day-t | ex-top-5 % | stop share | wk P10 | green wk |
|---|---|---|---|---|---|---|---|
| TRAIN | 541 | -0.047 | -0.69 | -0.156 | 0.53 | -6.0 | 30 % |
| VAL | 334 | -0.125 | -1.50 | -0.235 | 0.52 | -6.9 | 35 % |
| TRAIN H1 / H2 | 238 / 303 | -0.151 / +0.034 | -1.55 / 0.37 | | | | |

Uncapped net: TRAIN -0.069 (t -0.87), VAL -0.165 (t -1.65). ZERO-cost capped: TRAIN +0.081 (t 1.2), VAL +0.012 (t 0.14);
uncapped zero TRAIN +0.059, VAL -0.025. Doubled half-spread: capped -0.097 / -0.183. Optimistic fill-bar: unchanged.
Price >= $20 filter: -0.021 / -0.081. Placebo (300 draws, same geometry from a random later minute, capped): real
percentile 92.7 TRAIN / 95.0 VAL, placebo mean -0.096 / -0.183 (the placebo is worse than the signal, but both < 0).
Cadence scorer (its window is hard-coded 2025-01..2026-05, so it diluted fills/wk to 5.7 and P10 with empty weeks; own
weekly P10 above): C1-C4, C7 fail. Gross edge at the level ≈ 0 (+0.08 / +0.01 R), cost 0.13-0.14 R even on liquid names.

Verdict: FAIL the pre-committed bar (mean net >= +0.10 and t >= 2 on TRAIN and VAL: both negative; TRAIN halves flip sign;
ex-top-5 % < 0; placebo < 95 on TRAIN). MDE of this test ~ ±0.17 R (t 2, n 541) so +0.10 R is only barely resolvable; the
point estimate is negative and the zero-cost book is ≈ 0, so the liquid population is closed for this spec. Not
independently rebuilt (no pass). Caveats: bar-level fills (no tape), universe from cache.db daily_bars, TEST not used.
