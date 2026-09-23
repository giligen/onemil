# REPORT_2024.md — cells 1,413–1,414 on the untouched 2024H2 holdout (frozen rule, frozen code)

Availability: 100.0% of candidate symbol-days have Alpaca RTH bars (PASS). Signals walked: 1834; kept (BR ≥ 0.6115): 388 on 48 days; weeks: 27.

## Cell 1413 — all kept signals

- n 388 on 48 days · net -0.131 R · t -1.97 · day-weighted -0.255
- placebo (random risk-on long, same name-day) -0.182 R → setup − placebo +0.051 R
- rest (BR below the edge) -0.027 R (n 1446) → kept − rest -0.105 R
- weekly: Sharpe -0.37, green 29% of decided weeks, worst week -14.7 R, best week +9.3 R, fills/week 14.4
- days: worst -11.4 R, best +9.3 R, top 10 % of days = -55% of R
- slotted (first 4 at once, 12/day): n 195 net -0.181 R
- **PASS = False**

## Cell 1414 — kept, skipping each day’s first four

- n 241 on 27 days · net -0.035 R · t -0.42 · day-weighted -0.039
- placebo (random risk-on long, same name-day) -0.076 R → setup − placebo +0.041 R
- (rest comparison: cell 1,413 only)
- weekly: Sharpe -0.08, green 36% of decided weeks, worst week -10.0 R, best week +10.9 R, fills/week 8.9
- days: worst -10.0 R, best +10.9 R, top 10 % of days = -230% of R
- slotted (first 4 at once, 12/day): n 128 net +0.055 R
- **PASS = False**

