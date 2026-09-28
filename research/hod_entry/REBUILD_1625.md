# REBUILD_1625 -- Cell 1,625 independent rebuild from the prose only

Rebuilt 2026-09-28T17:33:11.088585+00:00. Source: PREREG_1623.md '## Cell 1,625' section only -- cell_1625.py / cell_1625_signals.csv / RESULT_1625.md were never opened.

TRAIN-H2 top-decile (p90) B30 threshold: **23.000** (pooled over 4398 TRAIN-H2 fill events).

Burst (signal) days found: **57** of 230 causal_arming days (TRAIN-H2=15, VAL=42).


## Primary metric: SPY, 60-minute hold, net bps (2 bps round-trip cost)

| split | n days | mean net bps | day-clustered t | ex-top-5% bps | winner-capped(+100bps) bps | signals/week |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 15 | 0.25 | 0.03 | -4.79 | 0.25 | 1.50 |
| VAL | 42 | -0.45 | -0.10 | -3.83 | -0.45 | 2.33 |

## IWM leg, 60-minute hold, net bps

| split | n days | mean net bps | day-clustered t | ex-top-5% bps | winner-capped(+100bps) bps | signals/week |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 15 | 0.50 | 0.05 | -6.77 | 0.35 | 1.50 |
| VAL | 42 | 1.16 | 0.15 | -4.77 | 0.22 | 2.33 |

## Placebo (random minute, same day, seed 1625) -- SPY 60-minute hold

| split | n days | placebo mean net bps | real - placebo margin bps | margin t |
|---|---|---|---|---|
| TRAIN-H2 | 15 | -5.24 | 5.48 | 0.79 |
| VAL | 42 | -0.36 | -0.09 | -0.01 |

## Report-only: 30- and 120-minute holds (SPY)

| split | hold | n days | mean net bps | day-clustered t |
|---|---|---|---|---|
| TRAIN-H2 | 30min | 15 | -1.85 | -0.44 |
| TRAIN-H2 | 120min | 15 | 11.53 | 1.19 |
| VAL | 30min | 42 | -2.98 | -0.78 |
| VAL | 120min | 42 | -0.60 | -0.10 |

## Pass bar (VAL, SPY leg, 60-minute hold, frozen in PREREG_1623.md)

- [FAIL] mean net >= +8 bps (value: -0.4499311138332004)
- [FAIL] t >= 2.5 (value: -0.09878519933684861)
- [FAIL] ex-top-5% > 0 (value: -3.8266919970728877)
- [FAIL] placebo margin >= +5 bps (value: -0.08511288847751988)
- [FAIL] placebo margin t >= 2 (value: -0.012894327155905227)
- [PASS] >= 2 signals/week (value: 2.3333333333333335)
- [FAIL] TRAIN-H2 same sign (value: TRAIN-H2=0.25 VAL=-0.45)

**Overall: FAIL**


## Caveats (read as an adversary)

1. **B30 undercounts breadth.** The prose defines B30 over arm events of 'any status'; causal_arming_causal.csv only carries a minute for status=='fill' rows (verified: 0/2,010 'nofill' and 0/21,931 'not_armed' rows have a fill_min, and bars_fills_1478.db's fetch_log has exactly the 9,911 fill symbol-days, no more). This rebuild's B30 is fills-only and is a lower bound on the prose's breadth measure by construction -- it will systematically undercount by roughly the 2,010/11,921 (~17%) share of arm events that never filled. If the original cell_1625.py recovered nofill arm minutes some other way, the burst-day sets and the Jaccard check in the Independent-check section will disagree on that gap and the discrepancy should be attributed here, not to a coding error.

2. **Minute rounding.** fill_min is a fractional ET minute; m* is taken as floor(fill_min) of the triggering event to land on the 1-minute SPY/IWM bar grid. This is a unit-conversion choice, not a tuned threshold.

3. **Percentile method.** The TRAIN-H2 top-decile threshold uses numpy's default linear-interpolation percentile over the pooled per-fill B30 values (not hour-of-day adjusted -- the prose only states the hour adjustment for cell 1,624's terciles, not for 1,625's decile, so none was added here per 'Not allowed: tuning the thresholds ... after a number').

4. **Bar-lookup fallbacks.** forward_filled=0, missing=0, no_day_data=0 (see log above); a missing bar drops that leg/day from that leg's stats only, logged at WARNING, never silently zero-filled.
