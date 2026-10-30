# RESULT cell 1,727 - double down on the 14:00 runner (PREREG_1727.md)

Population: cell 1,681 live-rule fills, n 5506 (1663_features x 1676_features; per-fill file used = `research/hod_entry/1663_features.csv` + `1681_paths.parquet` bars cached from bars_sip.db; the DB itself was not re-queried). Dates 2025-07-01..2026-05-29; TEST is absent from this population (nothing excluded, nothing sealed was read). Full population, no subsample. Replay exit_type differs from stored on 13 / 5506 fills; 0 eligible fills lack a 15:55 bar (last close used).

## 1. Eligible counts (fill <= 13:30 ET, live-rule position still open at the 14:00 bar)

| split | fills | fill<=13:30 | open at 14:00 | R1 | R2 | R1 share |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 2349 | 2296 | 512 | 127 | 157 | 24.8% |
| VAL | 3157 | 3108 | 787 | 181 | 240 | 23.0% |

## 2. Uncensored path 14:00 open -> 15:55 open (live exits ignored), per eligible position

| split | group | n | mean bps | median bps | mean R | median R | MFE after (R) | MAE after (R) | share up | day-clustered t (bps) |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | R1 | 127 | +17.8 | -6.6 | +0.050 | -0.026 | +0.80 | -0.80 | 48.0% | -0.33 |
| TRAIN-H2 | non-R1 | 385 | +3.9 | +2.0 | +0.018 | +0.007 | +0.67 | -0.65 | 50.4% | +0.31 |
| TRAIN-H2 | R2 | 157 | +22.2 | +24.4 | +0.065 | +0.114 | +0.81 | -0.82 | 52.9% | -0.11 |
| VAL | R1 | 181 | -15.1 | -8.6 | -0.077 | -0.040 | +0.69 | -0.69 | 48.1% | -1.06 |
| VAL | non-R1 | 606 | -3.8 | +9.4 | -0.000 | +0.038 | +0.67 | -0.67 | 52.0% | -0.51 |
| VAL | R2 | 240 | +3.5 | +7.3 | -0.000 | +0.033 | +0.74 | -0.68 | 52.1% | -0.11 |

## 3. B1 add leg (R1 names, original share count, risk unit = original R per share)

| set | n | adds/wk (BT) | mean R | day-clust t | ex-top-5 % | winner-capped +2R | add stopped | add at target |
|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 127 | 4.86 | -0.094 | -2.21 | -0.161 | -0.094 | 0.20 | 0.40 |
| VAL | 181 | 8.62 | -0.058 | -0.38 | -0.122 | -0.058 | 0.17 | 0.35 |
| TRAIN-H2 wk-A | 59 | 4.51 | -0.082 | -0.92 | -0.146 | -0.082 | 0.20 | 0.42 |
| TRAIN-H2 wk-B | 68 | 5.20 | -0.105 | -2.46 | -0.167 | -0.105 | 0.19 | 0.38 |
| ALL | 308 | 6.53 | -0.073 | -1.89 | -0.135 | -0.073 | 0.18 | 0.37 |

### Combined position vs the live rule (R1 positions only; original leg unchanged, add = 1x size)

| split | n R1 | live mean R | live + add mean R | delta per R1 position | delta per ALL fill in split (sum add_R / fills) |
|---|---|---|---|---|---|
| TRAIN-H2 | 127 | +1.055 | +0.961 | -0.094 | -0.0051 |
| VAL | 181 | +1.060 | +1.002 | -0.058 | -0.0033 |

## 4. Dollar translation (VAL rate and VAL mean; live/BT fill ratio = 50 / 116.8 BT fills per week = 0.428)

BT adds/week VAL 8.62 -> live-scaled 3.69 adds/week.

| risk per original position | add-leg $ per month (VAL mean R x live adds/wk x 52/12) | with VAL mean - 1 SE |
|---|---|---|
| $50 | -46 | -99 |
| $150 | -139 | -296 |

## 5. Frozen pass bar

* FAIL: mean >= +0.10 R in TRAIN-H2 (-0.094)
* FAIL: mean >= +0.10 R in VAL (-0.058)
* FAIL: day-clustered t >= 2 on VAL (-0.38)
* FAIL: ex-top-5 % > 0 on VAL (-0.122)
* FAIL: winner-capped (+2R) > 0 (TRAIN-H2 and VAL) (-0.094 / -0.058)
* PASS: both TRAIN halves same sign (-0.082 / -0.105)
* PASS: >= 3 adds/week at the live fill rate (3.69)

**VERDICT: FAIL** (2 / 7 items).

## 6. MDE

* TRAIN-H2: add-leg day-clustered SE 0.079 R, MDE (alpha .05 two-sided, 80 % power) = 0.222 R on n 127; the +0.10 R bar is below the MDE.
* VAL: add-leg day-clustered SE 0.066 R, MDE (alpha .05 two-sided, 80 % power) = 0.184 R on n 181; the +0.10 R bar is below the MDE.

## 7. Caveats (own implementation)

* 1-minute bars: stop and target touches inside a bar are ordered stop-first (conservative for the add, and for the live replay); the add's stop fill is `min(stop, bar open) - 12 bps`, a target sell carries NO cost (limit sell at the printed target, as in the live base).
* Add buy fills at the 14:00 bar open + 12 bps: obtainable (the 14:00 open is known only at 14:00:00, so a real order would hit the first prints; 12 bps is the PREREG's flat charge, not measured NBBO). EOD sells use the 15:55 bar open - 12 bps; stored live EOD uses the same open with 6 bps.
* Original-leg stop: the add stop (13:59 close - 1R) is always above the original stop (open P&L >= +0.5R), so the add stops first; the original leg is never altered, hence the combined delta is exactly the add leg.
* R1 is evaluated on one state per position (the PREREG definition); the 14:00 cohort is selected on the live book's own survival, so TRAIN halves by alternating weeks are not independent of the VAL window in the calendar sense but are disjoint in weeks.
* Adds/week live scaling assumes R1 frequency per fill is the same on live as in the BT (the live fills use the same population rule but BT fill rate is overstated vs live per cells 1,438/1,443).
* The population is the 1,681 full-day-range-free causal population; 1,681's own fills at 82 % fill rate; no change here. Single cell with one definition set: multiplicity = 1 new cell (1,727), no parameter varied.
