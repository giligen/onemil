# REBUILD_1564 -- independent rebuild of PREREG_1564/1565/1566 (prose-only)

Builder files (cell_1564.py, test_cell_1564.py, cell_1564_events.csv, RESULT_1564.md) were NOT read. Quote cache research/orb_failure/quotes_1564/ was absent -- **100% of events use the minute-of-day half-spread FALLBACK (25.0 bps, this rebuild's own placeholder, no canonical table found in the repo)**, not measured per-trade NBBO. Net-R numbers below are provisional per CLAUDE.md's own warning that a spread band can flip a book's sign.

## Population / event counts
{'none': 5695, 'success': 3319, 'failure': 1939, 'break_no_resolution': 1903}

## Per-cell, per-split summary
### 1564_failed_break_short
{'TRAIN': {'n': 391, 'events_per_week': 7.667, 'mean_net_R': np.float64(-0.1879), 'mean_raw_R': np.float64(-0.0188), 't_day': -2.723, 'ex_top5': -0.2944, 'ex_top1': np.float64(-0.2096), 'winner_capped': np.float64(-0.1879), 'median_r_pct_price': np.float64(4.955), 'exit_mix': {'eod': 0.598, 'stop': 0.238, 'eod_fallback': 0.092, 'target': 0.072}}, 'VAL': {'n': 247, 'events_per_week': 6.861, 'mean_net_R': np.float64(-0.2752), 'mean_raw_R': np.float64(-0.1002), 't_day': -3.855, 'ex_top5': -0.3703, 'ex_top1': np.float64(-0.2922), 'winner_capped': np.float64(-0.2752), 'median_r_pct_price': np.float64(4.416), 'exit_mix': {'eod': 0.66, 'stop': 0.251, 'eod_fallback': 0.049, 'target': 0.04}}}

### 1565_failed_break_short_vwap
{'TRAIN': {'n': 54, 'events_per_week': 1.688, 'mean_net_R': np.float64(-0.5128), 'mean_raw_R': np.float64(0.0721), 't_day': -1.908, 'ex_top5': -0.7869, 'ex_top1': np.float64(-0.6212), 'winner_capped': np.float64(-0.5822), 'median_r_pct_price': np.float64(1.677), 'exit_mix': {'stop': 0.519, 'target': 0.463, 'eod': 0.019}}, 'VAL': {'n': 32, 'events_per_week': 1.684, 'mean_net_R': np.float64(-0.8814), 'mean_raw_R': np.float64(-0.2847), 't_day': -2.973, 'ex_top5': -1.0324, 'ex_top1': np.float64(-0.9547), 'winner_capped': np.float64(-0.8814), 'median_r_pct_price': np.float64(1.374), 'exit_mix': {'stop': 0.531, 'target': 0.469}}}

### 1566_held_break_long
{'TRAIN': {'n': 1292, 'events_per_week': 24.377, 'mean_net_R': np.float64(-0.0673), 'mean_raw_R': np.float64(0.0123), 't_day': -1.129, 'ex_top5': -0.1724, 'ex_top1': np.float64(-0.0881), 'winner_capped': np.float64(-0.0673), 'median_r_pct_price': np.float64(6.744), 'exit_mix': {'eod': 0.601, 'stop': 0.22, 'eod_fallback': 0.106, 'target': 0.073}}, 'VAL': {'n': 1559, 'events_per_week': 39.974, 'mean_net_R': np.float64(-0.0135), 'mean_raw_R': np.float64(0.0655), 't_day': -0.242, 'ex_top5': -0.1127, 'ex_top1': np.float64(-0.034), 'winner_capped': np.float64(-0.0135), 'median_r_pct_price': np.float64(6.602), 'exit_mix': {'eod': 0.695, 'stop': 0.183, 'target': 0.063, 'eod_fallback': 0.06}}}

## Universe placebo (all candidates, same 1564 trade)
{'TRAIN': {'n': 678, 'events_per_week': 12.792, 'mean_net_R': np.float64(-0.1431), 'mean_raw_R': np.float64(0.076), 't_day': -1.385, 'ex_top5': -0.248, 'ex_top1': np.float64(-0.1645), 'winner_capped': np.float64(-0.1431), 'median_r_pct_price': np.float64(4.082), 'exit_mix': {'eod': 0.518, 'stop': 0.273, 'target': 0.121, 'eod_fallback': 0.088}}, 'VAL': {'n': 509, 'events_per_week': 13.051, 'mean_net_R': np.float64(-1.3308), 'mean_raw_R': np.float64(-0.0693), 't_day': -1.522, 'ex_top5': -1.4877, 'ex_top1': np.float64(-1.3622), 'winner_capped': np.float64(-1.3308), 'median_r_pct_price': np.float64(3.294), 'exit_mix': {'eod': 0.491, 'stop': 0.342, 'target': 0.092, 'eod_fallback': 0.075}}}

## Count-matched null (1000 draws, seed 1564)
null mean=-0.7409891349795984, VAL obs mean_net_R=-0.2752, percentile=70.0

## HOD calibration (report-only)
{'after_FAILURE_TRAIN': {'n': 4, 'mean_R': np.float64(-0.952), 't': -7.264}, 'after_FAILURE_VAL': {'n': 10, 'mean_R': np.float64(0.3791), 't': 0.722}, 'after_SUCCESS_TRAIN': {'n': 36, 'mean_R': np.float64(0.0681), 't': 0.289}, 'after_SUCCESS_VAL': {'n': 50, 'mean_R': np.float64(-0.3608), 't': -2.223}}
