# RESULT — cell 1,430: exits on the winning fills (PREREG_WEEKEND.md, frozen 2026-09-25 18:35 UTC)

Base fills: E1 status=='fill' rows of cell 1,427, `research/hod_entry/sip_rebuild_val.csv`
(TRAIN-H2 = split TRAIN/half H2, n=1165; VAL = split VAL, n=1443). Paths: `research/hod_exit_lab/
paths.parquet`. Cost: E1's own cost_R reused unchanged per fill (measured entry half-spread + B0's
own exit half-spread) + a new 30 bps stop-slip charge on variant stop exits (reported figures
below; without-slip ΔR TRAIN/VAL in parentheses). Slots not required by this cell's own pass bar.

| variant | n (TRAIN/VAL) | ΔR TRAIN-H2 | ΔR VAL | VAL t | ex-top-5 % (VAL) | worst week (var / B0, R) | verdict |
|---|---|---|---|---|---|---|---|
| (a) 90-min time stop | 1165/1443 | -0.119 (-0.051) | -0.132 (-0.065) | -6.56 | -0.220 | -12.10 / -17.17 | FAIL |
| (b) breakeven lock (+1R→BE) | 1163/1442 | -0.153 (-0.054) | -0.170 (-0.071) | -8.64 | -0.229 | -26.14 / -17.17 | FAIL |
| (c) ORB lock (+1.5R→+0.5R) | 1164/1440 | -0.108 (-0.015) | -0.135 (-0.039) | -8.14 | -0.206 | -17.68 / -17.17 | FAIL |
| (d) 50% scale-out at +2R | 1157/1433 | -0.008 (+0.083) | -0.040 (+0.050) | -1.49 | -0.168 | -20.35 / -20.49 | FAIL |
| (e) VWAP-close after +0.5R | 1163/1442 | -0.091 (-0.021) | -0.096 (-0.029) | -5.92 | -0.152 | -20.93 / -17.17 | FAIL |
| (f) close at 14:30 | 1165/1443 | -0.076 (-0.002) | -0.076 (-0.001) | -6.98 | -0.130 | -24.51 / -17.17 | FAIL |

All six variants FAIL the frozen 1,430 pass bar (ΔR ≥ +0.05 both holdouts, VAL t ≥ 2, worst week
not worse than B0's); every point estimate is negative or near-zero and every VAL t is negative.
Best variant is (d) 50% scale-out at +2R: with the new 30 bps stop-slip charge it is flat-to-slightly-negative
(TRAIN -0.008, VAL -0.040); without the charge it clears +0.05 on both holdouts (+0.083 / +0.050)
but VAL t stays -1.49 (computed on the slip-charged series per PREREG) — still a clear FAIL. B0's
own 15:55/2R exit remains the best exit on this population; no variant ships.
