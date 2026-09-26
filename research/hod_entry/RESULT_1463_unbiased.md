# RESULT 1,463 (unbiased) -- stop-limit exit slip, fresh random sample

| variant | holdout | n | mean bps | median bps | p90 bps | no-fill n | no-fill mean bps |
|---|---|---|---|---|---|---|---|
| stop-market | TRAIN-H2 | 316 | 35.1 | 23.6 | 71.0 | 0 | n/a |
| stop-limit-20bps | TRAIN-H2 | 272 | 2.9 | 7.5 | 17.6 | 44 | 93.9 |
| stop-limit-50bps | TRAIN-H2 | 302 | 19.4 | 18.5 | 43.6 | 14 | 181.2 |
| stop-market | VAL | 320 | 35.8 | 24.3 | 86.2 | 0 | n/a |
| stop-limit-20bps | VAL | 286 | 3.2 | 7.0 | 17.3 | 34 | 75.8 |
| stop-limit-50bps | VAL | 308 | 18.4 | 17.1 | 44.1 | 12 | 112.1 |

Stop-market mean vs cell_1443 ref (35.9/34.8, tol 5.0): TRAIN-H2 35.1, VAL 35.8 -> sample_valid=True.
Ship bar (mean slip <= stop-market by >= 10.0 bps AND no-fill mean <= 100.0 bps, both holdouts): 20bps=True, 50bps=False.
Unmeasured/flagged: {'no_print_le_stop': 152, 'no_tape': 12} (0 stop_bar fill-instant flags).
