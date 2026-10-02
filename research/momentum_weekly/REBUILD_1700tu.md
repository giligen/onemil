# REBUILD_1700tu - independent rebuild of guarded sleeve + half-size gate (2026-10-02)
Code `REBUILD_1700tu.py` (memory-safe pyarrow load; RECON fixes applied: word-boundary exclusions, all 20 names reset
every Monday, costs on traded delta dollars, min(5bp+0.5*proxy,20bp)). Dotted tickers kept. VIX/VIX3M from CBOE CSVs.

| book | CAGR | max DD (daily close marks) | end $ |
|---|---|---|---|
| plain (repro, expect 26.6-27.2 / -42..-44.5 / ~508K) | 26.24 | -42.9 | 480,739 |
| guarded (first build 29.34 / -38.3 / 596,394) | 28.34 | -38.1 | 564,611 |
| guarded + half gate (first 30.67 / -37.1 / 658,510) | 29.66 | -36.7 | 623,605 |

Plain is ~0.4 pt under the expected band; end $ 5% lower. Every book sits ~1.0 pt CAGR under the first build.
Gate: 142 half-size weeks of 503 (first 143), 54 spells. 2021-08-30 half (pct 0.080), 2024-01-08 full (0.378), 2024-12-02 half (0.060).
Guard removed 36 name-weeks: AMC 15, AMRN 6, NBIS 6, ABVX 5, OCGN 3, WOLF 1 (identical to the first build).
Holdings vs recon/G_holdings.csv: 20/20 on 2021-02-08, 2025-12-29, 2026-06-29.

By year guarded (2017..2026), my daily-close calendar-year marks: 18.8 -4.0 14.9 76.3 28.5 -0.2 14.6 39.8 38.2 75.4
Spec row:                                                  19.6 -5.0 14.9 85.3 25.4  0.0 16.9 40.8 38.2 73.0
Largest difference 2020 (-9.0). The first build's daily curve is not a close-mark curve (it jumps 7% day to day where
mine moves 1%), so year-end marks are not comparable. On the SAME Monday-open marks the first build's guard curve
(1700u_curves_daily.csv, read only after the fact) vs mine differ by -0.1..-2.8 pts per year (REBUILD_1700tu_by_year_mondays.csv);
the cumulative gap is a steady ~0.5 %/yr, no holdings difference.
Gated by year (daily marks): 15.5 -7.4 18.8 84.9 58.9 6.3 11.0 46.5 20.7 61.5

## Agreement bar
Guarded CAGR within 1.0: AT THE LIMIT (-1.00)  | max DD within 3: AGREE (0.2) | every year within 4: DISAGREE on spec marks
(2020 -9.0, 2023 -2.3 ok), AGREE on same-Monday marks (max 2.8) | holdings >=19/20 x3: AGREE | removed name-weeks +-5, same
six names: AGREE (0 diff).
Gate half weeks +-3 of 143: AGREE (142) | three states: AGREE | CAGR within 1.0: DISAGREE marginally (-1.01) | DD within 3: AGREE (0.4).
First separation > 1% of the equity paths (Monday marks): week of 2020-08-31 (cumulative -1.1%; that week alone -0.47%);
no holdings gap is visible, the drift is cost/weight-convention level (my cost charged on traded deltas; first build unknown).
Not tuned. Files: _by_year.csv (daily basis), _by_year_mondays.csv, _weekly.csv, _removed.csv, _daily_*.csv, .log.
