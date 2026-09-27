# RESULT_1567 — defined-risk put-credit-spread ladder on SPY

Equity fixed at $65,000 for the whole backtest; B = 10% = $6,500.00. 133 entry Mondays, 7 with no usable listed expiry (VOID before any cell).

## TRAIN selection (highest monthly Sharpe, n_cycles>=12, green_months>=55%)

Selected cell **1574** (Delta=0.15, W=10.0, M=B, G=1): TRAIN monthly Sharpe 7.60, n_cycles 19, green months 100%.


## Full TRAIN table (all 24 cells)

|   cell |   delta |   width | mgmt   |   gate |   n_cycles |   mean_monthly_ret_on_B |   monthly_sharpe |   green_month_share |   worst_month_usd |   max_dd_usd |   win_rate |
|-------:|--------:|--------:|:-------|-------:|-----------:|------------------------:|-----------------:|--------------------:|------------------:|-------------:|-----------:|
|   1574 |    0.15 |      10 | B      |      1 |         19 |             0.0248055   |        7.60305   |            1        |            71.94  |       -0     |   1        |
|   1572 |    0.15 |      10 | A      |      1 |         19 |             0.0107046   |        4.09485   |            0.888889 |           -26.24  |       26.24  |   0.842105 |
|   1588 |    0.3  |      10 | A      |      1 |         20 |             0.013511    |        1.71386   |            0.8      |          -233.49  |      400.73  |   0.8      |
|   1573 |    0.15 |      10 | B      |      0 |         42 |             0.0196134   |        1.70191   |            0.941176 |          -848.203 |      848.203 |   0.97619  |
|   1584 |    0.3  |       5 | A      |      1 |         19 |             0.0153085   |        1.45103   |            0.8      |          -446.48  |      446.48  |   0.736842 |
|   1586 |    0.3  |       5 | B      |      1 |         19 |             0.0282368   |        1.14665   |            0.8      |          -858.12  |     1528.36  |   0.894737 |
|   1568 |    0.15 |       5 | A      |      1 |         18 |             0.00274667  |        0.836412  |            0.666667 |          -106.48  |      173.92  |   0.722222 |
|   1569 |    0.15 |       5 | B      |      0 |         48 |             0.0124741   |        0.812787  |            0.888889 |          -858.407 |     1696.65  |   0.958333 |
|   1580 |    0.2  |      10 | A      |      1 |         21 |             0.007452    |        0.801004  |            0.7      |          -426.36  |      500.1   |   0.761905 |
|   1578 |    0.2  |       5 | B      |      1 |         26 |             0.0127895   |        0.50681   |            0.916667 |         -1659.63  |     1659.63  |   0.923077 |
|   1585 |    0.3  |       5 | B      |      0 |         49 |             0.0199008   |        0.479323  |            0.764706 |         -2485.86  |     3954.22  |   0.836735 |
|   1570 |    0.15 |       5 | B      |      1 |         18 |             0.00647446  |        0.460157  |            0.9      |          -838.24  |      838.24  |   0.944444 |
|   1577 |    0.2  |       5 | B      |      0 |         50 |             0.0139803   |        0.447367  |            0.888889 |         -2525.75  |     3141.27  |   0.92     |
|   1576 |    0.2  |       5 | A      |      1 |         26 |             0.00459385  |        0.428222  |            0.818182 |          -660.227 |      660.227 |   0.769231 |
|   1583 |    0.3  |       5 | A      |      0 |         49 |             0.00575994  |        0.347373  |            0.6875   |          -848.96  |     1803.8   |   0.673469 |
|   1589 |    0.3  |      10 | B      |      0 |         44 |             0.0122411   |        0.326487  |            0.823529 |         -2431.68  |     3932.11  |   0.818182 |
|   1590 |    0.3  |      10 | B      |      1 |         20 |             0.00983718  |        0.317832  |            0.8      |         -1500.43  |     2345.49  |   0.85     |
|   1582 |    0.2  |      10 | B      |      1 |         21 |             0.00519776  |        0.201339  |            0.909091 |         -1668.68  |     1668.68  |   0.904762 |
|   1581 |    0.2  |      10 | B      |      0 |         42 |             0.0036702   |        0.119535  |            0.888889 |         -2565.41  |     3261.5   |   0.904762 |
|   1579 |    0.2  |      10 | A      |      0 |         42 |             0.000704917 |        0.0896778 |            0.529412 |          -426.36  |      616.003 |   0.666667 |
|   1571 |    0.15 |      10 | A      |      0 |         42 |             0.000215023 |        0.0381286 |            0.705882 |          -256.68  |      613.95  |   0.666667 |
|   1587 |    0.3  |      10 | A      |      0 |         44 |            -0.00132124  |       -0.121122  |            0.470588 |          -406.413 |     1247.55  |   0.659091 |
|   1575 |    0.2  |       5 | A      |      0 |         50 |            -0.00774932  |       -0.751842  |            0.470588 |          -660.227 |     1474.25  |   0.6      |
|   1567 |    0.15 |       5 | A      |      0 |         48 |            -0.00814832  |       -1.28567   |            0.5      |          -333.72  |     1268.71  |   0.625    |


## VAL read of the selected cell

n_cycles=16, mean monthly return on B=1.32%, monthly Sharpe=1.05, green months=88%, ex-top5% cycle PnL=$577, worst month=$-565 (>= -B: True), max DD=$565 (<=1.5B=9,750: True).

SPY buy-and-hold over the same VAL window: return 24.49%, $1,592 on B.


Pass-bar checklist: mean monthly return >= 4%=FAIL, monthly Sharpe >= 1.0=PASS, green months >= 60%=PASS, ex-top5% cycles positive=PASS, worst month >= -B=PASS, max DD <= 1.5B=PASS


**Overall: FAIL** (neighbour check and TRAIN-same-sign must also be read from the full VAL table below before shipping).



## Full VAL table (all 24 cells, unselected ones labelled for the record)

|   cell |   delta |   width | mgmt   |   gate |   n_cycles |   mean_monthly_ret_on_B |   monthly_sharpe |   green_month_share |   worst_month_usd |   max_dd_usd |   win_rate |
|-------:|--------:|--------:|:-------|-------:|-----------:|------------------------:|-----------------:|--------------------:|------------------:|-------------:|-----------:|
|   1590 |    0.3  |      10 | B      |      1 |         16 |             0.0458024   |         5.04837  |            1        |           160.94  |       -0     |   1        |
|   1589 |    0.3  |      10 | B      |      0 |         34 |             0.0555393   |         3.28525  |            0.928571 |          -828.06  |      828.06  |   0.970588 |
|   1573 |    0.15 |      10 | B      |      0 |         33 |             0.022561    |         2.6384   |            0.923077 |          -426.847 |      426.847 |   0.969697 |
|   1581 |    0.2  |      10 | B      |      0 |         38 |             0.0371466   |         2.29513  |            0.923077 |          -883.06  |      883.06  |   0.973684 |
|   1584 |    0.3  |       5 | A      |      1 |         16 |             0.0181037   |         2.25623  |            0.8      |          -129.22  |      129.22  |   0.8125   |
|   1585 |    0.3  |       5 | B      |      0 |         37 |             0.0548344   |         2.01976  |            0.928571 |         -1650.24  |     1650.24  |   0.945946 |
|   1569 |    0.15 |       5 | B      |      0 |         41 |             0.0232553   |         1.75283  |            0.923077 |          -813.8   |      813.8   |   0.97561  |
|   1586 |    0.3  |       5 | B      |      1 |         16 |             0.033488    |         1.67128  |            0.9      |          -842.12  |      842.12  |   0.9375   |
|   1575 |    0.2  |       5 | A      |      0 |         39 |             0.0109748   |         1.44018  |            0.846154 |          -259.96  |      259.96  |   0.717949 |
|   1582 |    0.2  |      10 | B      |      1 |         21 |             0.0238135   |         1.25713  |            0.888889 |          -883.06  |      883.06  |   0.952381 |
|   1583 |    0.3  |       5 | A      |      0 |         37 |             0.0168217   |         1.23358  |            0.769231 |          -565.72  |      565.72  |   0.72973  |
|   1578 |    0.2  |       5 | B      |      1 |         21 |             0.0206982   |         1.20209  |            0.9      |          -844.12  |      844.12  |   0.952381 |
|   1577 |    0.2  |       5 | B      |      0 |         39 |             0.0256489   |         1.09462  |            0.928571 |         -1592.36  |     1592.36  |   0.948718 |
|   1574 |    0.15 |      10 | B      |      1 |         16 |             0.0131953   |         1.04827  |            0.875    |          -564.56  |      564.56  |   0.9375   |
|   1587 |    0.3  |      10 | A      |      0 |         35 |             0.00794786  |         0.766869 |            0.666667 |          -404.01  |      454.1   |   0.657143 |
|   1576 |    0.2  |       5 | A      |      1 |         21 |             0.00482423  |         0.67712  |            0.75     |          -222.307 |      222.307 |   0.666667 |
|   1579 |    0.2  |      10 | A      |      0 |         39 |             0.00450535  |         0.470188 |            0.642857 |          -567.487 |      655.63  |   0.692308 |
|   1570 |    0.15 |       5 | B      |      1 |         19 |             0.00811128  |         0.449415 |            0.875    |          -919.12  |      919.12  |   0.947368 |
|   1571 |    0.15 |      10 | A      |      0 |         34 |             0.000895419 |         0.180766 |            0.666667 |          -239.36  |      347.137 |   0.617647 |
|   1588 |    0.3  |      10 | A      |      1 |         16 |             0.00180303  |         0.176061 |            0.7      |          -404.01  |      404.01  |   0.625    |
|   1567 |    0.15 |       5 | A      |      0 |         41 |            -0.00101546  |        -0.115052 |            0.642857 |          -503.533 |      700.247 |   0.560976 |
|   1580 |    0.2  |      10 | A      |      1 |         21 |            -0.00371481  |        -0.320855 |            0.75     |          -567.487 |      757.467 |   0.571429 |
|   1568 |    0.15 |       5 | A      |      1 |         19 |            -0.00352733  |        -0.550292 |            0.571429 |          -288.053 |      288.053 |   0.578947 |
|   1572 |    0.15 |      10 | A      |      1 |         16 |            -0.00407314  |        -0.882418 |            0.625    |          -171.073 |      287.063 |   0.5625   |


## Data caveats (carried from FETCH_1567.md)

* Option bars are trade OHLCV, not quotes: "mid" is the mean trade close in the 10:00-10:05 ET sub-window (entry) and the as-of daily close (ongoing management), both documented approximations.

* Management marks use daily closes, not intraday quotes (minute bars were fetched for the entry Monday only, per FETCH_1567.md's scope reduction); the STOP exit uses the next session's daily OPEN as a stand-in for "next session's 10:00".

* WARNING/fallback counters: {'entry_mid_fallback': 12383, 'void_no_expiry_spot': 3}

## Judge (main session, 2026-09-27 15:10 UTC) — FAIL on the frozen bar, and the test itself is inadequate

* Builder's TRAIN pick (1574: Δ 0.15, $10, hold, IV gate) earned +1.3 %/month on B in VAL ($86/month) — far under the
  4 % bar — and both refuters showed the TRAIN pick was an artifact: (a) cycles whose long leg had no 10:00 trade print
  were VOIDed, and those VOIDs removed exactly the two April-2025 maximum losses (SPY lists every strike; a live engine
  would trade the pair) — pair-aware selection drops 1574's TRAIN Sharpe from 7.6 to 0.2 with a −$1,673 month; (b)
  months without an exit were dropped from the monthly series instead of counted as $0. With both fixed the TRAIN rule
  selects cells whose VAL is 0.0–2.3 %/month. TRAIN-to-VAL rank correlation across the 24 cells is −0.07: the selection
  carries no information at 19–41 cycles per cell.
* Independent rebuild disagrees on the cycle set (Jaccard 0.38, 41 % of cycles within $5) and selects a different
  cell — the disagreement is the DATA: option "mids" were taken from sparse TRADE prints in a 5-minute window (12,383
  fallbacks), deep-OTM legs often have no print, daily marks are forward-filled closes. Trade bars cannot price OTM
  spreads; quotes are needed. This is a claim about the test, not about the premium.
* What the VAL table shows anyway (unselected, multiplicity 24): 3 of 24 cells clear 4 %/month — the Δ 0.30 / $10-wide
  cells (1585, 1589: +5.5 % / +5.6 % of B per month ≈ $360, monthly Sharpe 2.0–3.3, worst months −$830 / −$1,650 =
  13–25 % of B, win rate 95–97 %). Order of magnitude consistent with the disclosed prior; not evidence until a
  quote-based test selects it on TRAIN. Comparison lines on VAL: naked short put at the same Δ +$6,893 (undefined risk),
  SPY buy-and-hold on B +$1,592; the cap gives up most of the premium, as expected.
* Budget assertion held in every cell (worst month ≥ −B by construction; the largest realised was −$1,673 = 26 % of B).
Consequence: FAIL as frozen. Adequacy review → Amendment 2 (below): quote-based v2 on the 8 cells the premium can
support (Δ 0.20 / 0.30, W $10, management A / B, gate on / off), cells 1,591–1,598. Programme count 1,590 → 1,598.
