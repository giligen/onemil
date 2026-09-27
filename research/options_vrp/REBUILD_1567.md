# REBUILD_1567 -- independent rebuild of PREREG_1567
Entry Mondays used: 133. build_cycles meta: {'skipped_no_spot': 0, 'skipped_no_expiry': 1, 'skipped_no_strikes': 0}

## Selection
18/24 cells cleared the TRAIN screen (n>=12, green>=55%).
Selected cell: delta=0.3 width=5.0 mgmt=B gate=gate15

### TRAIN (2024-02-05..2025-06-30)
{'n_cycles': 23, 'n_void': 0, 'n_months': 9, 'mean_monthly_ret': np.float64(0.08616683760683762), 'monthly_sharpe': np.float64(2.4918068634522954), 'green_share': np.float64(0.8888888888888888), 'worst_month': np.float64(-566.5999999999995), 'max_dd': np.float64(566.5999999999995), 'ex_top5_positive': np.True_, 'win_rate': np.float64(0.8260869565217391), 'total_pnl': np.float64(5040.760000000001)}

### VAL (2025-07-07..2026-08-17)
{'n_cycles': 30, 'n_void': 0, 'n_months': 11, 'mean_monthly_ret': np.float64(0.09984167832167833), 'monthly_sharpe': np.float64(1.573668991567167), 'green_share': np.float64(0.8181818181818182), 'worst_month': np.float64(-1672.3), 'max_dd': np.float64(1944.9), 'ex_top5_positive': np.True_, 'win_rate': np.float64(0.9), 'total_pnl': np.float64(7138.680000000002)}

Buy-and-hold SPY on B=6500.0: TRAIN ret=0.2544 maxdd$=1536.75; VAL ret=0.2449 maxdd$=665.21

## Full VAL table (unselected, for the record)
| delta | width | mgmt | gate | n | green% | mean_mo_ret | sharpe | worst_$ | maxdd_$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 5.0 | A | gate15 | 30 | 64% | -0.69% | -0.67 | -528 | 757 |
| 0.15 | 5.0 | A | none | 57 | 60% | -0.75% | -0.71 | -528 | 1223 |
| 0.15 | 5.0 | B | gate15 | 30 | 82% | 2.38% | 1.46 | -836 | 836 |
| 0.15 | 5.0 | B | none | 57 | 93% | 4.23% | 2.88 | -770 | 770 |
| 0.15 | 10.0 | A | gate15 | 28 | 73% | -0.32% | -0.42 | -376 | 447 |
| 0.15 | 10.0 | A | none | 53 | 64% | -0.05% | -0.07 | -376 | 407 |
| 0.15 | 10.0 | B | gate15 | 28 | 90% | 2.52% | 2.25 | -487 | 487 |
| 0.15 | 10.0 | B | none | 53 | 93% | 3.95% | 3.94 | -422 | 422 |
| 0.2 | 5.0 | A | gate15 | 30 | 40% | -1.59% | -1.77 | -499 | 1029 |
| 0.2 | 5.0 | A | none | 57 | 43% | -1.64% | -1.72 | -499 | 1595 |
| 0.2 | 5.0 | B | gate15 | 30 | 91% | 1.21% | 0.40 | -1884 | 1884 |
| 0.2 | 5.0 | B | none | 57 | 93% | 3.82% | 1.43 | -1778 | 1778 |
| 0.2 | 10.0 | A | gate15 | 28 | 55% | -0.30% | -0.35 | -386 | 676 |
| 0.2 | 10.0 | A | none | 49 | 67% | 0.14% | 0.16 | -472 | 659 |
| 0.2 | 10.0 | B | gate15 | 28 | 91% | 2.62% | 1.45 | -957 | 957 |
| 0.2 | 10.0 | B | none | 49 | 93% | 4.34% | 2.53 | -957 | 957 |
| 0.3 | 5.0 | A | gate15 | 30 | 73% | 4.46% | 1.07 | -519 | 702 |
| 0.3 | 5.0 | A | none | 57 | 67% | 3.47% | 1.04 | -519 | 702 |
| 0.3 | 5.0 | B | gate15 | 30 | 82% | 9.98% | 1.57 | -1672 | 1945 |
| 0.3 | 5.0 | B | none | 57 | 86% | 12.35% | 1.95 | -2496 | 2769 |
| 0.3 | 10.0 | A | gate15 | 30 | 70% | 1.20% | 0.76 | -604 | 618 |
| 0.3 | 10.0 | A | none | 57 | 80% | 1.29% | 0.90 | -618 | 709 |
| 0.3 | 10.0 | B | gate15 | 30 | 82% | 4.08% | 1.35 | -1534 | 1731 |
| 0.3 | 10.0 | B | none | 57 | 86% | 7.60% | 2.23 | -1954 | 2151 |
