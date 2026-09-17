# Stage G step 0 — the short universe, SSR and the attention list

- panel rows 5,043,226 -> 5,043,226 after (symbol, bar_date) dedup (0 duplicate keys dropped); symbols 15,448; days 420 2025-01-02..2026-09-04
- E/members.csv 1,570,771 rows, calendar 410 days 2025-01-17..2026-09-04
- U1uU2 198,318 -> dvol20_med >= $10M 115,501 -> open >= $10 98,074 -> asset_class==stock **70,713** (35.7% of U1uU2, 2,672 symbols, 410 days)
- M18 selection: 4 placeholder tickers and 708 non-matching tickers dropped; ranking pool 1,984,237 symbol-days over 410 days; selected 16,360 (8,180 attention / 8,180 control)
- attention/control on the 410-day calendar: 16,360 of 16,360 (100.0%) — the rest fall on early-close days the calendar excludes
- attention/control rows: 16,360 -> panel row on the trade day 16,342 -> dvol20_med >= $10M 10,026 -> open >= $10 8,653 -> stock **6,504** (3,013 attention / 3,491 control)
- union: 70,713 U1uU2 + 6,504 attention/control -> 72,793 distinct keys (4,424 keys in both, attention flag carried onto 4,424 U1uU2 rows)
- **attention names that are ALSO in U1uU2: 2,500 of 3,013 = 83.0%** (the S5-strict variant of PREREG §1)

## SSR (Rule 201 fired on t-1: low_{t-1} <= 0.90 x close_{t-2})

           n   ssr  unknown  ccdown  ssr_pct  ccdown_pct
split                                                   
TEST   14977  1941        0     927    12.96        6.19
TRAIN  36034  4736        0    2408    13.14        6.68
VAL    21782  2676        0    1254    12.29        5.76

- close-to-close marker (prev_ret_cc <= -10%) agrees with the low-based trigger on 68,029 of 72,793 rows (93.5%); the low-based trigger is the rule.

- provenance check, E/members.csv vs research/lit_review_2026/daily_panel.parquet on 70,713 shared keys (both built from the SAME Databento parquet): open median |diff| 0.000002%, p99 0.000005%, max 0.0005%; prev_close median 0.000002%, p99 0.000005%
- E/bars_causal index covers 40,037 of 72,793 G keys (55.0%); the rest must come from bars_sip.db or attention.db and the builder records every miss in G/coverage_short_missing.csv

- **members_g.csv: 72,793 symbol-days over 410 days, 2,690 symbols** (TRAIN 36,034 / VAL 21,782 / TEST 14,977)
