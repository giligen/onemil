# RESULT 1,743 — variance ratio / first passage on stored HOD paths (TRAIN 7,390 / VAL 4,745; TEST never read)

Join day+symbol: 12,135/12,135 signals matched paths and b0 (0 lost, 0 duplicates); 125 b0 rows have NaN net_R (excluded from the B0 means only). Paths run minutes_since_entry 0..378, so there is NO pre-entry segment: the pre-break reference VR is not available (PREREG "if present"). Programme count 1,743.

## VR over minutes 1..120 (close-to-close, signal-block bootstrap 95 % CI; Lo-MacKinlay robust z)
| split | VR(10) [CI] | VR(30) [CI] | z(10) / z(30) |
|---|---|---|---|
| TRAIN | 0.867 [0.848, 0.885] | 0.737 [0.714, 0.759] | -15.7 / -19.3 |
| VAL | 0.834 [0.816, 0.853] | 0.772 [0.741, 0.807] | -19.8 / -16.1 |
| TRAIN H1 / H2 | 0.906 / 0.816 | 0.714 / 0.765 | same sign, all CIs < 1 |

Windows: minutes 16..120 VR(10) 0.837 TRAIN / 0.835 VAL (below 1); minutes 1..15 VR(10) 1.028 [0.986, 1.074] TRAIN, 0.874 VAL (undecided/below). VR(2) 0.94/0.93 includes bid-ask bounce bias. VR from `vwap` is 3.5-6.7 but meaningless: it is the cumulative day VWAP, smooth by construction (not a tradable price).

## First passage (closes, minutes >= 1, gross, no cost)
| split | geometry | P(stop first) | P(target first) [CI] | neither |
|---|---|---|---|---|
| TRAIN | -1R/+2R | 0.527 | 0.254 [0.244, 0.264] | 0.219 |
| VAL | -1R/+2R | 0.549 | 0.264 [0.252, 0.276] | 0.187 |
| TRAIN / VAL | -1R/+1R | 0.446 / 0.460 | 0.463 / 0.473 | |
| TRAIN / VAL | -0.5R/+1R | 0.622 / 0.638 | 0.345 / 0.348 | |
Driftless bounds: stop first 2/3, target 1/3 (-1/+2). Target / (target+stop) = 0.325 / 0.325, i.e. about the driftless 1/3 once the 19-22 % unresolved paths are set aside.

## Drift (bps/min, signal SE)
| split | 1..15 | 16..60 | 61..120 | entry to 15:55 |
|---|---|---|---|---|
| TRAIN | -0.92 (0.16) | +0.14 (0.06) | +0.03 (0.04) | +21 bps, +0.173 R (day-clustered SE 0.21) |
| VAL | -0.18 (0.17) | +0.29 (0.07) | -0.06 (0.05) | +17 bps, +0.139 R (day-clustered SE 0.10) |

## Subgroups (9 reads x 2 splits, exploratory)
Every tercile has VR(10) CI below 1 on both splits (0.73-0.96). Highest-VR cells (half_spread T1 0.96/0.90, entry_m T1 0.95/0.85) are still < 1. No candidate for a PREREG. B0 net R tracks cost (half_spread T3 -0.45/-0.61), not VR.

## Pre-committed reading that holds
VR(10) and VR(30) CIs entirely below 1.0 on TRAIN and VAL AND P(target first, -1/+2) = 0.25/0.26 <= 1/3: **mean-reverting after the break; no stop-based long design can be net positive here; stop testing geometry on this population, move to population/cost.**

## Caveats
Close-to-close 1-min returns carry bounce (biases VR(2), less VR(30)); first passage uses closes only (misses intrabar touches, so stops are under-counted); the positive end-of-day drift (+0.14..0.17 R gross; iid t about 5 but day-clustered t 0.8 TRAIN / 1.4 VAL, i.e. day-driven) coexists with VR < 1 and is a buy-and-hold-to-close effect the stop geometries do not capture; z uses within-signal lags only. MDE in VR units about 0.02 (CI half-width).
