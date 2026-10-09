# REPORT3 — cell 3, vol line: 5-day tracking gate (2026-10-09)

**VERDICT: VOID at the PREREG_3 gate. Strategy NOT run (no trades3.csv / weekly3.csv / scorecard). Cell 3 produced no
strategy evidence; the SPVXSP rebuild from free CBOE VX settlements cannot stand in for the ETPs.**

## Gate (synthetic leg = rebuilt index, era-correct leverage, fees pro rata, vs real Alpaca adjusted closes)
Overlapping log returns, `run3.py` -> `tracking3.csv`, `gate3.json`. Annualised difference = (synthetic - real) log
return over the window / years.
| leg | window | 5-day R^2 | 5-day beta | 20-day R^2 | 20-day beta | ann. diff syn-real | gate |
|---|---|---|---|---|---|---|---|
| VXX (+1x) | 2018-01-18 -> 2026-10-08 (8.72 y, n 2188) | **0.945** | 0.924 | 0.960 | 0.951 | **-4.52 %/yr** | R^2 < 0.95 and |diff| > 3: FAIL |
| SVXY (era -1x then -0.5x) | 2016-01-04 -> 2026-10-08 (10.76 y, n 2702) | **0.751** | 0.605 | 0.924 | 0.688 | **-11.70 %/yr** | R^2 < 0.95 and |diff| > 3: FAIL |
Gate required 5-day R^2 >= 0.95 AND |ann. diff| <= 3 %/yr on both legs. VXX misses R^2 by 0.005 and the return
difference by 1.5 pt; SVXY fails both by a wide margin. Total log return, real vs synthetic: VXX -4.617 vs -5.012;
SVXY +0.309 vs -0.949.

## Reading (own caveats)
* The 16:15-vs-16:00 timing offset predicted by PREREG_3 does wash out at 5 days for VXX (daily 0.855 -> 0.945 -> 20d
  0.960) but not enough, and the offset cannot explain a persistent -4.5 %/yr (VXX) / -11.7 %/yr (SVXY) drift: that is
  a level bias in the rebuild (candidates, NOT tested here: the 820 Settle=0 -> Close substitutions mostly 2011-14 do not
  touch these windows; the ETPs' actual roll/rebalance schedule vs my business-day weights; SVXY -1x-era daily reset vs
  16:15 settles; the 2018-02-05/06 pair, where the synthetic -1x leg lost -96.4 % vs real -88.4 %).
* SVXY 20-day R^2 0.924 with beta 0.69 means the synthetic -1x-era leg is too volatile relative to the product
  (beta < 1 on regress real-on-synthetic), consistent with 16:15-settle noise plus the 2018-02 pair.
* Not computed: any strategy number, any diagnostic excluding the 2018 pair at 5 days (the gate is pre-declared as
  all-window; an ex-2018 variant would be a new, post-hoc gate and would need its own PREREG).
* This gate was pre-declared before any strategy number was read (PREREG_3); no threshold was moved.

## Consequence
Vol line on free data: cells 1 (completeness), 2 (daily tracking), 3 (5-day tracking + return bias) are all VOID. The
free-data route to a 2011-2018 test of the term-structure sleeve is closed unless a different construction is
validated first (e.g. Databento VX/ETP history, a paid purchase needing the owner's GO and evidence first).
Only the 2018-2026 real-ETP window (VXX from 2018-01, SVXY from 2016-01) is testable without the rebuild, which
cannot cover TRAIN-A (2011-2017) and so cannot satisfy the PREREG_1 two-half pass bar.
