SPY days 420  dispersion days 410

## Q2a — regime level by split (trading days in the book period)
split   days  SPY rv20 ann  SPY |ret| bps  xs disp bps  xs MAD bps  F6 cand/day
TRAIN    240         16.17           74.8        305.2       175.4         54.2
VAL      102         12.84           64.7        344.4       197.0         70.7
TEST      68         13.33           63.9        366.6       205.6         78.5

weeks in regression: 87  ({'TEST': 14, 'TRAIN': 51, 'VAL': 22})

## Q2b — weekly book R on split dummies ONLY (TRAIN is the base)
        coef     se      t
const  2.399  0.919  2.610
VAL    3.569  1.894  1.885
TEST   4.399  2.206  1.994
R2 0.073

## Q2c — weekly book R on regime variables ONLY
           coef     se      t
const     4.009  0.777  5.158
spy_rv20  0.013  0.484  0.027
xs_disp   1.007  0.900  1.120
n_cand   -1.253  0.657 -1.906
R2 0.028

## Q2d — weekly book R on regime variables AND split dummies (do the split effects survive?)
           coef     se      t
const     2.164  1.011  2.140
spy_rv20  0.260  0.469  0.554
xs_disp  -0.137  0.947 -0.145
n_cand   -1.152  0.492 -2.342
VAL       4.065  2.100  1.936
TEST      5.077  2.418  2.099
R2 0.098

## Q2e — same, per-TRADE mean R as the dependent (weeks weighted by trades)
           coef     se      t
const     0.122  0.055  2.227
spy_rv20  0.019  0.027  0.699
xs_disp   0.005  0.060  0.086
n_cand   -0.060  0.029 -2.102
VAL       0.175  0.129  1.362
TEST      0.235  0.129  1.817
R2 0.074

## Q2f — the control: mean R of the FULL F6 candidate population (no selection) by split
  TRAIN  n  13323  pop meanR -0.0062  t -0.80   book meanR +0.1297
  VAL    n   7214  pop meanR +0.0800  t +7.19   book meanR +0.3218
  TEST   n   5339  pop meanR +0.0165  t +1.22   book meanR +0.3499

## Q2g — book R and pool R per month, side by side
         pool_meanR  pool_n  book_meanR  book_R  book_n
mo                                                     
2025-01      -0.058     593       0.233  17.698      76
2025-02       0.036     848       0.144  10.948      76
2025-03      -0.095     981      -0.016  -1.383      84
2025-04      -0.031    4042       0.131  11.005      84
2025-05      -0.010     760       0.144  12.088      84
2025-06      -0.085     565      -0.094  -7.492      80
2025-07       0.022     664       0.452  39.777      88
2025-08       0.052     770       0.099   8.318      84
2025-09       0.128     679       0.220  18.478      84
2025-10       0.000    1097       0.131  12.050      92
2025-11      -0.017    1513       0.022   1.643      76
2025-12       0.100     811       0.069   6.029      88
2026-01       0.009     957       0.152  12.194      80
2026-02       0.036    1364       0.384  29.220      76
2026-03       0.050    1812      -0.014  -1.222      88
2026-04       0.193    1406       0.549  46.082      84
2026-05       0.093    1675       0.563  45.009      80
2026-06       0.006    1871       0.499  41.891      84
2026-07      -0.056    1798       0.042   3.695      88
2026-08       0.129    1490       0.557  46.829      84
2026-09      -0.081     180       0.172   2.755      16

## Q2h — weekly regression of the book-minus-pool EXCESS on regime (is the selection edge regime-driven?)
           coef     se      t
const     0.111  0.051  2.197
spy_rv20  0.020  0.025  0.790
xs_disp  -0.005  0.048 -0.097
n_cand   -0.060  0.022 -2.754
VAL       0.132  0.107  1.236
TEST      0.264  0.121  2.184
R2 0.087
  mean excess by split: TRAIN +0.123 VAL +0.229 TEST +0.354