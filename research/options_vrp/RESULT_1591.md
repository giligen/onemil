# RESULT_1591 -- v2 quote-era put-credit-spread ladder (Amendment 2/2a, tick-priced)

Equity fixed at $65,000; B = 10% = $6,500.00. 133 entry Mondays, 7 with no usable listed expiry.

## TRAIN selection (highest monthly Sharpe, n_cycles>=12, green_months>=55%)

Selected cell **1597** (Delta=0.3, M=B, G=0): TRAIN monthly Sharpe -0.04, n_cycles 23, green months 71%, VOID share 0.0%.


## Full TRAIN table (8 cells)

|   cell |   delta | mgmt   |   gate |   n_cycles |   void_cycles |   mean_monthly_ret_on_B |   monthly_sharpe |   green_month_share |   worst_month_usd |   max_dd_usd |   win_rate |
|-------:|--------:|:-------|-------:|-----------:|--------------:|------------------------:|-----------------:|--------------------:|------------------:|-------------:|-----------:|
|   1592 |     0.2 | A      |      1 |         12 |             0 |             0.0028467   |        0.626925  |            0.411765 |           -283.24 |       283.24 |   0.833333 |
|   1598 |     0.3 | B      |      1 |         10 |             0 |             0.00357086  |        0.313886  |            0.235294 |           -806.06 |       806.06 |   0.9      |
|   1594 |     0.2 | B      |      1 |         12 |             0 |             0.00232941  |        0.23755   |            0.352941 |           -722.12 |       722.12 |   0.916667 |
|   1596 |     0.3 | A      |      1 |         10 |             0 |             0.000135023 |        0.0175283 |            0.235294 |           -545.12 |       676.36 |   0.8      |
|   1597 |     0.3 | B      |      0 |         23 |             0 |            -0.000888688 |       -0.0403339 |            0.705882 |          -1586.12 |      2392.18 |   0.782609 |
|   1593 |     0.2 | B      |      0 |         25 |             0 |            -0.00445593  |       -0.211495  |            0.647059 |          -1619.18 |      2354.36 |   0.88     |
|   1591 |     0.2 | A      |      0 |         25 |             0 |            -0.0040543   |       -0.572809  |            0.470588 |           -423.48 |       888.28 |   0.6      |
|   1595 |     0.3 | A      |      0 |         23 |             0 |            -0.00686552  |       -0.686949  |            0.470588 |           -545.12 |      1602.32 |   0.608696 |


## VAL read of the selected cell

n_cycles=18, VOID share=0.0%, mean monthly return on B=3.17%, monthly Sharpe=4.25, green months=79%, ex-top5% cycle PnL=$2,862, worst month=$0 (>=-B: True), max DD=$-0 (<=1.5B: True), slip $0.05/leg monthly ret=3.10%, slip $0.10/leg monthly ret=2.91%.

SPY buy-and-hold over the same VAL window: return 24.49%, $1,592 on B, max DD -10.23%.


Pass-bar checklist: mean monthly return >= 4%=FAIL, monthly Sharpe >= 1.0=PASS, green months >= 60%=PASS, ex-top5% cycles positive=PASS, worst month >= -B=PASS, max DD <= 1.5B=PASS, VOID share <= 10%=PASS


**Overall: FAIL**



## Full VAL table (8 cells)

|   cell |   delta | mgmt   |   gate |   n_cycles |   void_cycles |   mean_monthly_ret_on_B |   monthly_sharpe |   green_month_share |   worst_month_usd |   max_dd_usd |   win_rate |
|-------:|--------:|:-------|-------:|-----------:|--------------:|------------------------:|-----------------:|--------------------:|------------------:|-------------:|-----------:|
|   1597 |     0.3 | B      |      0 |         18 |             0 |             0.0317031   |        4.2533    |            0.785714 |              0    |        -0    |   1        |
|   1598 |     0.3 | B      |      1 |          8 |             0 |             0.0156651   |        2.32797   |            0.428571 |              0    |        -0    |   1        |
|   1593 |     0.2 | B      |      0 |         24 |             0 |             0.0177534   |        1.25709   |            0.714286 |           -883.06 |       883.06 |   0.958333 |
|   1595 |     0.3 | A      |      0 |         19 |             0 |             0.00867956  |        1.07298   |            0.571429 |           -441.24 |       441.24 |   0.736842 |
|   1594 |     0.2 | B      |      1 |         13 |             0 |             0.0050244   |        0.378731  |            0.428571 |           -883.06 |       883.06 |   0.923077 |
|   1596 |     0.3 | A      |      1 |          8 |             0 |             0.00205538  |        0.312126  |            0.357143 |           -441.24 |       441.24 |   0.75     |
|   1591 |     0.2 | A      |      0 |         24 |             0 |             0.000385934 |        0.0547284 |            0.5      |           -404.48 |       596.96 |   0.666667 |
|   1592 |     0.2 | A      |      1 |         13 |             0 |            -0.000786374 |       -0.111709  |            0.285714 |           -404.48 |       614.84 |   0.615385 |


## Amendment 2a caveats carried into this result

* Strike SELECTION still uses the cached 10:00 entry-minute trade price (a delta estimate), never a tick -- only the fill and every P&L dollar are tick-priced.

* Entry: a leg with no trade in 10:00:00-10:05:00 ET -> the CYCLE is VOID (counted). Exit (Management A only): a leg with no trade in that window on the exit session falls back to the session's daily OPEN (counted, NOT void) -- Amendment 2a states this asymmetry explicitly.

* WARNING/fallback/VOID counters: {'entry_mid_fallback': 4155, 'strike_shift_due_to_missing_print': 83, 'tick_void_no_trade': 147, 'tick_fallback_5m_window': 224, 'void_entry_tick_missing': 69, 'exit_fallback_daily_open': 240, 'void_no_expiry_spot': 1}
