# Stage G — the pure M18 spec on the right population

- S5 filled rows, borrowable, SSR excluded: **4,229** (attention 1,656 / control 2,573)
- of which `r_pct >= 1.0` keeps 2,166 (51.2%) — the filter that contaminated the first version of this table: for S5 it means "the 09:30-09:34 high is at least 1% above the 09:35 open", i.e. the name had ALREADY dropped in the first five minutes. M18 has no such filter.

## A. The M18 spec as published: every attention/control name, no stop, no R filter

                  pop     group      horizon split    n  gross_bps  net_bps     t  net_borrow_bps  MDE_bps
ALL (correct for M18) attention 09:35->10:30 TRAIN  998       -4.6    -22.5 -1.65           -27.8     38.1
ALL (correct for M18) attention 09:35->10:30   VAL  429       32.9     15.1  0.81             9.9     52.6
ALL (correct for M18) attention 09:35->close TRAIN  990       -4.6    -22.5 -1.29           -27.7     48.8
ALL (correct for M18) attention 09:35->close   VAL  429        2.0    -15.8 -0.56           -21.1     78.2
ALL (correct for M18)   control 09:35->10:30 TRAIN 1532       -5.8    -23.9 -3.94           -29.2     17.0
ALL (correct for M18)   control 09:35->10:30   VAL  646       -3.7    -21.8 -2.03           -27.1     30.0
ALL (correct for M18)   control 09:35->close TRAIN 1524       -0.9    -19.0 -1.96           -24.2     27.1
ALL (correct for M18)   control 09:35->close   VAL  646      -18.2    -36.3 -2.34           -41.6     43.4

## B. The same on the r_pct >= 1.0 subset (what score_short.py reported — kept for the record)

            pop     group      horizon split   n  gross_bps  net_bps     t  net_borrow_bps  MDE_bps
r_pct>=1 subset attention 09:35->10:30 TRAIN 565      -19.3    -36.9 -1.70           -42.2     60.6
r_pct>=1 subset attention 09:35->10:30   VAL 272       56.2     38.4  1.62            33.2     66.3
r_pct>=1 subset attention 09:35->close TRAIN 562      -18.6    -36.2 -1.38           -41.5     73.3
r_pct>=1 subset attention 09:35->close   VAL 272       17.3     -0.5 -0.01            -5.8    107.4
r_pct>=1 subset   control 09:35->10:30 TRAIN 653      -13.0    -30.8 -2.72           -36.1     31.7
r_pct>=1 subset   control 09:35->10:30   VAL 327        7.3    -10.5 -0.65           -15.8     45.5
r_pct>=1 subset   control 09:35->close TRAIN 650       -0.6    -18.3 -1.04           -23.6     49.1
r_pct>=1 subset   control 09:35->close   VAL 327      -13.9    -31.7 -1.33           -37.0     66.6

## C. By M18 dollar-volume band, population A, open(09:35)->10:30, in bps of the SHORT

   band     group split   n  short_gross_bps  stock_move_bps
$10-50M attention TRAIN 487            -15.9            15.9
$10-50M attention   VAL 221             21.0           -21.0
$10-50M   control TRAIN 658             -4.3             4.3
$10-50M   control   VAL 249             -1.2             1.2
  >$50M attention TRAIN 511              6.2            -6.2
  >$50M attention   VAL 208             45.6           -45.6
  >$50M   control TRAIN 874             -7.0             7.0
  >$50M   control   VAL 397             -5.2             5.2

`short_gross_bps` > 0 = the stock FELL (the fade M18 measured); `stock_move_bps` is the same number with M18's sign (the stock's own return), so it can be laid directly against `research/lit_review_2026/open_fade.md` — remembering that M18 measures from the 09:30 OPEN and this measures from 09:35, so the first five minutes of the fade (M18 TRAIN r0935: -37 bps at $10-50M, -14 at >$50M) is NOT in these numbers.
