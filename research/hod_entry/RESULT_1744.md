# RESULT 1,744 — no-stop hold to the close on the cheapest half-spread tercile: FAIL

Programme count 1,743 -> 1,744 (5 reads). TRAIN/VAL only; TEST never read (1735_per_signal.csv has none). Join loss 0 of 12,135;
flow matched 100 %. Scorer `1744_score.py`, rows `1744_rows.csv`, table `1744_summary.csv`. TRAIN cuts (half-spread bps of price): 10.30 / 24.33.
V1F = v1_net_R x 2. Capped book = first 4 by entry minute per day (all positions held to the close, so concurrency binds at 4).

| Primary V1F cheap | n | mean R | day-t | iid t | ex-top-5 % | ex-top-1 % | green wk (null) | wk P10 | capped | B0 |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN H1 | 1280 | +0.296 | 1.02 | 3.41 | -0.17 | +0.15 | 0.50 (0.35) | -23.4 | +0.12 | -0.07 |
| TRAIN H2 | 1158 | -0.029 | -0.22 | -0.39 | -0.34 | -0.12 | 0.48 (0.36) | -34.3 | -0.29 | -0.06 |
| TRAIN | 2438 | +0.141 | 0.82 | 2.45 | -0.26 | +0.01 | 0.50 (0.36) | -30.1 | -0.09 | -0.07 |
| VAL | 1335 | -0.012 | -0.11 | -0.20 | -0.27 | -0.09 | 0.45 (0.36) | -42.3 | +0.08 | -0.08 |

Downside uncapped: 31 % of rows < -1 R, 15-17 % < -2 R. Max DD 162-206 R.
Pass bar: 5 of 7 conditions FAIL (TRAIN and VAL mean/day-t, halves sign flip, VAL ex-top-5 %, VAL weekly P10); fills/wk and VAL capped pass. **VERDICT FAIL.**

Dose check: mid tercile TRAIN +0.004 / VAL -0.049; dear -0.41 / -0.43. Cheap beats mid by +0.14 on TRAIN, but only by +0.04 on VAL: cost
helps in dose, yet the gross on cheap names is not there. Exploratory: X2 (cheap, r_pct >= 2 %) VAL +0.305 day-t 2.28 but TRAIN +0.07, halves
+0.30 / -0.27 (not a pass, one-split). X1 (print) = primary. X3 V2F cheap ~ 0 (TRAIN -0.02, VAL +0.006).
Reading: day-clustered t (<= 1) is far below iid t (2.5-3.4): the TRAIN H1 mean is day/regime clustering, not signals. The close-hold on
cheap names is tail-carried (ex-top-5 % -0.26 / -0.27) with a -30..-42 R weekly P10. MDE at day-t 2: roughly 0.25 R on n 1.3-2.4k.
Per synthesis the HOD line is closed at the engine level.
