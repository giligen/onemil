# RESULT 1,646-1,648 -- pre-announcement run-up (BUILD, independent check pending)

Total resolved/traded firm-quarters (TRAIN+VAL): 33048. Causality-check violations (E within 300d of its own R1): 0 (expected 0 -- E is always R1 + ~365d by construction).

## Pass bar (frozen; VAL, per cell) -- mean net >=40bps, t>=2.5, ex-top-5%>0, >=20 ev/wk, TRAIN same-sign t>=1, SPY-adj net >=25bps, hit rate >=70%

| cell | n | mean net bps | t | ex-top-5% | events/wk | TRAIN t | SPY-adj net | hit rate | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1646 | n=11478 | 10.8 (no) | t=0.62 (no) | ex5=-71.4 (no) | 147.15/wk (OK) | TRAIN t=1.09 (OK) | spy_adj=-1.0 (no) | hit=0.47 (no) | **FAIL** |
| 1647 | n=11478 | 25.3 (no) | t=1.1 (no) | ex5=-97.6 (no) | 147.15/wk (OK) | TRAIN t=1.57 (OK) | spy_adj=-15.9 (no) | hit=0.47 (no) | **FAIL** |

1,648 (report-only, mechanism split of 1,647): volume-tercile table monotone on VAL = True, on TRAIN = False.

## Full stats (TRAIN + VAL, both cells)

|   cell | split   |     n |   events_wk |   mean_net_bps |    t |   ex_top5_bps |   ex_top1_bps |   winner_capped_bps |   spy_adj_net_bps |   hit_rate |   early_arrival_share |   mde_bps | years_positive              |
|-------:|:--------|------:|------------:|---------------:|-----:|--------------:|--------------:|--------------------:|------------------:|-----------:|----------------------:|----------:|:----------------------------|
|   1646 | TRAIN   | 21570 |      136.52 |           23.6 | 1.09 |         -70.3 |          -7.8 |                19.1 |               4.8 |      0.437 |                 0.382 |      12.7 | 2019:-/2020:+/2021:-/2022:+ |
|   1646 | VAL     | 11478 |      147.15 |           10.8 | 0.62 |         -71.4 |         -18   |                 7   |              -1   |      0.474 |                 0.304 |      14.7 | 2023:-/2024:+               |
|   1647 | TRAIN   | 21570 |      136.52 |           42.2 | 1.57 |         -90.5 |          -0.4 |                29.6 |               7.3 |      0.437 |                 0.382 |      17.6 | 2019:-/2020:+/2021:+/2022:+ |
|   1647 | VAL     | 11478 |      147.15 |           25.3 | 1.1  |         -97.6 |         -16.4 |                13.5 |             -15.9 |      0.474 |                 0.304 |      22   | 2023:-/2024:+               |

## 1,648 -- volume-ratio tercile table (mean net bps, 1,647 exit rule)

| split   |   tercile |    n |   mean_net1647_bps |    t |
|:--------|----------:|-----:|-------------------:|-----:|
| TRAIN   |         1 | 7082 |               29.8 | 1.19 |
| TRAIN   |         2 | 7080 |               55.1 | 1.95 |
| TRAIN   |         3 | 7082 |               44.6 | 1.17 |
| VAL     |         1 | 3341 |               12.3 | 0.42 |
| VAL     |         2 | 4319 |               26.2 | 1.1  |
| VAL     |         3 | 3802 |               35.7 | 1.25 |

## Per-quarter table (net bps, both cells)

| split   | year_quarter   |    n |   mean_net1646_bps |   mean_net1647_bps |
|:--------|:---------------|-----:|-------------------:|-------------------:|
| TRAIN   | 2019Q4         |   49 |              -91.1 |              -76.9 |
| TRAIN   | 2020Q1         | 1728 |             -177.6 |             -391.9 |
| TRAIN   | 2020Q2         | 1758 |              268.2 |              398.8 |
| TRAIN   | 2020Q3         | 1745 |              176.8 |              235.3 |
| TRAIN   | 2020Q4         | 1710 |               31.2 |              152.8 |
| TRAIN   | 2021Q1         | 1824 |                6.5 |               -3.6 |
| TRAIN   | 2021Q2         | 1633 |             -103   |              -96.4 |
| TRAIN   | 2021Q3         | 1784 |               20   |               49.2 |
| TRAIN   | 2021Q4         | 1799 |               35.7 |               77   |
| TRAIN   | 2022Q1         | 1879 |              -82.4 |             -104   |
| TRAIN   | 2022Q2         | 1776 |             -206.1 |             -357   |
| TRAIN   | 2022Q3         | 1896 |              242   |              358.9 |
| TRAIN   | 2022Q4         | 1989 |               52.6 |              153.4 |
| VAL     | 2023Q1         | 1928 |               38   |               39.9 |
| VAL     | 2023Q2         | 1884 |              -13.1 |              -32.6 |
| VAL     | 2023Q3         | 1916 |              -45.5 |              -99.6 |
| VAL     | 2023Q4         | 1903 |              -62.9 |               50.6 |
| VAL     | 2024Q1         | 1927 |               88.6 |              100.9 |
| VAL     | 2024Q2         | 1920 |               58.2 |               91.3 |

## Size split (<=$1B vs larger)

UNAVAILABLE -- no shares-outstanding/market-cap source on disk (same documented gap as cell_1633/RESULT_1633.md). Not computed; not silently defaulted.

## Caveats for the independent-check agent (read as an adversary)

1. **"Same fiscal quarter" is inferred, not observed.** events_raw.csv has no reportDate/period-of-report column, so R1/R2 are found by nearest-date matching (target +/-45d) rather than an explicit fiscal-quarter key. An item-2.02 8-K that is NOT a regular quarterly release (e.g. a preliminary/ad hoc results filing) can pollute a firm's release calendar and get selected as an anchor; not filtered here.
2. **Duplicate/amendment merge is a 14-day heuristic** (consecutive-gap cumsum, not a true greedy-interval merge from the last KEPT release) -- see build_release_calendar docstring for the exact edge case this can miss.
3. **Early-arrival clip**: when the real 8-K's reaction session lands ON OR BEFORE the planned entry session (E-5), cell 1,646 closes on the entry session itself (same-day round trip) rather than dropping the event -- see resolve_trades' n_early_arrival_clip count in the run log. This keeps n intact but can print a near-zero-duration trade for a badly-estimated E; hit_window/hit rate still penalizes these appropriately.
4. **Volume-ratio window (1,648) is a documented choice, not in the PREREG's numeric detail**: mean dollar volume over R1-1..R1+1 divided by dvol20 at R1-6. A different window could change the tercile table materially -- rebuild agent should treat this as a named parameter to vary, not assume it is the only choice.
5. **SPY-adjustment**: SPY IS present in both parquet panels (confirmed before writing this script) -- no Alpaca fetch fallback was implemented or needed.
6. **Corpse gate (session_gap_too_wide) added after the first run found 150x-300x "returns"** on FLG/HAPN/VSXY: their own per-symbol bar array has a multi-month gap (real halt, or a recycled ticker used by a different company later -- both seen on inspection) spanning the same calendar window that load_prices()'s zero-OHLCV drop silently removes non-trading days, so `idx_E+1` landed on a bar many months after `idx_E-6`. Fixed by requiring bar_date[idx_E+1] - bar_date[idx_E-6] <= 20 calendar days (a nominal 7-session window is ~9-12 days with holidays); see resolve_trades' session_gap_too_wide lost count. A cell_1633-convention SPLIT_GUARD (any consecutive close/close ratio outside [0.40, 2.50] along the E-6..E+1 path) is also applied (split_guard lost count) -- it caught a real WMT 3-for-1 split (2024-02-26) that had produced a -6,510bps 'trade' in the first guarded run. Not screened beyond this: intra-day halts/circuit breakers and genuine >50% single-name crashes (biotech trial failures, meme-stock reversals) are real returns, kept as-is, and still drive the tail-dependence gap between mean and ex-top-5% below -- read that gap as real single-name risk, not further data corruption, unless the independent check finds otherwise.
7. **TEST is sealed**: 2024-07-01 onward was dropped immediately after split assignment (assign_split), before any statistic in this file. Nothing above used it.
