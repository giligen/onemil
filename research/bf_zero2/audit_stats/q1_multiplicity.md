pool rows 862,626 | keys 27 | days {'TRAIN': 250, 'VAL': 102, 'TEST': 68}
cells rebuilt: 108

## observed — the 12 cells with the highest min-across-splits t
                   key exit  slots  TRAIN_n  VAL_n  TEST_n  TRAIN_meanR  VAL_meanR  TEST_meanR  TRAIN_t  VAL_t  TEST_t  min_t  all_pos
                 F6 {}   e4      4    996.0  408.0   272.0        0.283      0.675       0.575    2.860  3.946   3.231  2.860     True
                 F6 {}  e1c      4    996.0  408.0   272.0        0.124      0.320       0.353    2.776  4.290   3.869  2.776     True
                 F6 {}   e4     20   4756.0 2016.0  1356.0        0.079      0.246       0.268    2.646  4.924   3.648  2.646     True
F5 {"K": 3, "X": 0.04}   e4     20   5000.0 2040.0  1360.0        0.121      0.135       0.172    3.081  2.585   2.721  2.585     True
                 F6 {}  e1c     20   4756.0 2016.0  1356.0        0.033      0.151       0.182    1.962  5.248   4.814  1.962     True
F5 {"K": 5, "X": 0.04}   e4     20   5000.0 2040.0  1360.0        0.081      0.179       0.087    2.343  3.592   1.744  1.744     True
F5 {"K": 5, "X": 0.06}   e4     20   5000.0 2040.0  1360.0        0.096      0.182       0.056    3.100  3.988   1.234  1.234     True
F5 {"K": 8, "X": 0.06}   e4     20   5000.0 2040.0  1360.0        0.072      0.041       0.054    2.829  1.223   1.505  1.223     True
F5 {"K": 8, "X": 0.04}  e1c      4   1000.0  408.0   272.0        0.045      0.067       0.110    1.124  1.090   1.461  1.090     True
F5 {"K": 5, "X": 0.06}  e1c     20   5000.0 2040.0  1360.0        0.019      0.083       0.087    1.064  2.954   2.494  1.064     True
F5 {"K": 3, "X": 0.02}   e4     20   4999.0 2040.0  1360.0        0.138      0.166       0.069    2.980  2.625   0.994  0.994     True
F5 {"K": 3, "X": 0.06}   e4     20   5000.0 2040.0  1360.0        0.120      0.101       0.052    3.290  2.363   0.924  0.924     True

cells positive on all three splits: 25 of 108
cells with t > 2.8 on all three: 1   | t > 2.5: 4 | t > 2.0: 4
the winner: F6 {} e4 slots 4 min_t 2.86

## effective number of independent tests (eigenvalues of the daily-R correlation matrix)
  TRAIN  cells 108  mean |corr| 0.195  M_eff(Cheverud)  13.9  M_eff(Li-Ji)  57.0  eigenvalues >1: 27, top-5 share 43.2%
  VAL    cells 108  mean |corr| 0.229  M_eff(Cheverud)  11.3  M_eff(Li-Ji)  53.0  eigenvalues >1: 25, top-5 share 46.5%
  TEST   cells 108  mean |corr| 0.250  M_eff(Cheverud)   9.9  M_eff(Li-Ji)  46.0  eigenvalues >1: 23, top-5 share 52.4%
  ALL    cells 108  mean |corr| 0.202  M_eff(Cheverud)  13.6  M_eff(Li-Ji)  60.0  eigenvalues >1: 26, top-5 share 42.7%
  bootstrapped TRAIN: t matrix (108, 10000), median n per cell 1195
  bootstrapped VAL: t matrix (108, 10000), median n per cell 499
  bootstrapped TEST: t matrix (108, 10000), median n per cell 360

## Q1 — null distribution of the SEARCH (10,000 day-block resamples, every cell centered to zero mean)
  P(at least one of the 108 cells has t > 2.00 on ALL THREE splits) = 0.0225
  P(at least one of the 108 cells has t > 2.50 on ALL THREE splits) = 0.0043
  P(at least one of the 108 cells has t > 2.78 on ALL THREE splits) = 0.0010
  P(at least one of the 108 cells has t > 2.80 on ALL THREE splits) = 0.0010
  P(at least one of the 108 cells has t > 3.00 on ALL THREE splits) = 0.0003
  P(at least one of the 108 cells has t > 3.50 on ALL THREE splits) = 0.0001
  P(a SINGLE pre-specified cell clears t > 2.8 on all three)          = 0.000011
  null quantiles of max-over-cells min-over-splits t: 50% 0.82 | 90% 1.51 | 95% 1.75 | 99% 2.26
  observed max-over-cells min-over-splits t = 2.86  ->  search-adjusted p = 0.0009
  null #cells positive on all three: mean 13.0, 95th pct 35 (observed 25) — the cells are strongly correlated, so this is NOT ~K/8

## the wider census (score3 is only the last stage)
  score3 grid (this bootstrap)           cells   108  M_eff ~   13.6  P(some cell t>2.8 on all 3) ~ 0.0002
  score2, 275 cells                      cells   275  M_eff ~   34.7  P(some cell t>2.8 on all 3) ~ 0.0004
  bf_zero 8fam x 21cfg x 4 exits         cells   672  M_eff ~   84.9  P(some cell t>2.8 on all 3) ~ 0.0009
  lit-review cells (RESULTS.md census)   cells   153  M_eff ~   19.3  P(some cell t>2.8 on all 3) ~ 0.0002
  all of the above                       cells  1208  M_eff ~  152.6  P(some cell t>2.8 on all 3) ~ 0.0017