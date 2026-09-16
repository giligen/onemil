# F6 red-to-green, hold to close with a -1R stop — verification | rows 25,876 | weeks {'TRAIN': 53, 'VAL': 22, 'TEST': 14}

## cost sensitivity and book size (net R per trade / R per week)
  slots  4 spread  40 bps | TRAIN +0.283R  +5.3/wk t +2.9 green 31/53 mdd -13 | VAL +0.675R +12.5/wk t +4.0 green 15/22 mdd -14 | TEST +0.575R +11.2/wk t +3.2 green 11/14 mdd -12
  slots  4 spread  60 bps | TRAIN +0.251R  +4.7/wk t +2.5 green 30/53 mdd -14 | VAL +0.639R +11.9/wk t +3.7 green 15/22 mdd -17 | TEST +0.541R +10.5/wk t +3.0 green 11/14 mdd -13
  slots  4 spread  80 bps | TRAIN +0.218R  +4.1/wk t +2.2 green 30/53 mdd -15 | VAL +0.603R +11.2/wk t +3.5 green 14/22 mdd -20 | TEST +0.507R  +9.8/wk t +2.9 green 11/14 mdd -14
  slots  4 spread 120 bps | TRAIN +0.153R  +2.9/wk t +1.6 green 28/53 mdd -31 | VAL +0.532R  +9.9/wk t +3.1 green 14/22 mdd -25 | TEST +0.439R  +8.5/wk t +2.5 green 10/14 mdd -16
  slots 10 spread  40 bps | TRAIN +0.124R  +5.8/wk t +2.5 green 28/53 mdd -38 | VAL +0.437R +20.2/wk t +5.1 green 18/22 mdd -24 | TEST +0.307R +14.9/wk t +2.9 green 11/14 mdd -25
  slots 10 spread  60 bps | TRAIN +0.097R  +4.5/wk t +1.9 green 26/53 mdd -41 | VAL +0.406R +18.8/wk t +4.7 green 17/22 mdd -27 | TEST +0.276R +13.4/wk t +2.6 green 10/14 mdd -28
  slots 10 spread  80 bps | TRAIN +0.069R  +3.2/wk t +1.4 green 24/53 mdd -47 | VAL +0.375R +17.4/wk t +4.4 green 17/22 mdd -30 | TEST +0.245R +11.9/wk t +2.3 green 9/14 mdd -31
  slots 10 spread 120 bps | TRAIN +0.014R  +0.7/wk t +0.3 green 24/53 mdd -92 | VAL +0.314R +14.6/wk t +3.7 green 13/22 mdd -38 | TEST +0.183R  +8.9/wk t +1.7 green 7/14 mdd -39
  slots 20 spread  40 bps | TRAIN +0.079R  +7.0/wk t +2.6 green 31/53 mdd -59 | VAL +0.246R +22.5/wk t +4.9 green 15/22 mdd -34 | TEST +0.268R +25.9/wk t +3.6 green 11/14 mdd -30
  slots 20 spread  60 bps | TRAIN +0.055R  +4.9/wk t +1.9 green 28/53 mdd -71 | VAL +0.220R +20.2/wk t +4.4 green 14/22 mdd -39 | TEST +0.240R +23.2/wk t +3.3 green 11/14 mdd -38
  slots 20 spread  80 bps | TRAIN +0.031R  +2.8/wk t +1.1 green 28/53 mdd -100 | VAL +0.194R +17.8/wk t +3.9 green 14/22 mdd -44 | TEST +0.212R +20.5/wk t +2.9 green 9/14 mdd -46
  slots 20 spread 120 bps | TRAIN -0.016R  -1.5/wk t -0.6 green 25/53 mdd -190 | VAL +0.142R +13.0/wk t +2.9 green 11/22 mdd -62 | TEST +0.156R +15.1/wk t +2.1 green 6/14 mdd -62

## the 4-slot book at 40 bps, in detail
  TRAIN: n 996 (4.0/day) meanR +0.283 t 2.86 WR 33.7% weekly +5.3R green 31/53 worst -12.2R maxDD -12.7R
  VAL: n 408 (4.0/day) meanR +0.675 t 3.95 WR 40.7% weekly +12.5R green 15/22 worst -9.6R maxDD -14.0R
  TEST: n 272 (4.0/day) meanR +0.575 t 3.23 WR 44.1% weekly +11.2R green 11/14 worst -12.4R maxDD -12.4R

## monthly R (4 slots, 40 bps)
           sum  count
mo                   
2025-01    3.5     76
2025-02   23.7     76
2025-03    9.9     84
2025-04   17.1     84
2025-05   -6.0     84
2025-06   45.3     80
2025-07   67.2     88
2025-08   -5.0     84
2025-09   54.3     84
2025-10   45.0     92
2025-11    5.7     76
2025-12   21.5     88
2026-01   19.9     80
2026-02   78.6     76
2026-03  -12.8     88
2026-04  113.3     84
2026-05   76.2     80
2026-06   90.9     84
2026-07   -4.6     88
2026-08   69.3     84
2026-09    0.7     16

## R distribution
count    1676.00
mean        0.43
std         3.18
min        -7.94
5%         -1.42
25%        -1.11
50%        -1.03
75%         0.86
95%         5.27
max        35.08

## sub-samples (mean net R, n) — the result must not be one bucket
  price: (5.0, 10.0] +0.42 (n 663) | (10.0, 20.0] +0.48 (n 464) | (20.0, 50.0] +0.44 (n 357) | (50.0, 1000000000.0] +0.28 (n 192)
  entry_m: (570, 600] +0.43 (n 1675) | (600, 660] +0.05 (n 1)
  gap_pct: (-1000000000.0, -10.0] +0.76 (n 10) | (-10.0, -5.0] +0.16 (n 169) | (-5.0, -2.0] +0.31 (n 713) | (-2.0, 0.0] +0.59 (n 784)
  r_pct: (1.0, 2.0] +1.15 (n 310) | (2.0, 4.0] +0.48 (n 584) | (4.0, 8.0] +0.10 (n 697) | (8.0, 1000000000.0] +0.05 (n 85)
  rv_profile: (0.0, 1.0] +0.42 (n 941) | (1.0, 2.0] +0.47 (n 382) | (2.0, 5.0] +0.43 (n 252) | (5.0, 1000000000.0] +0.89 (n 56)
  adv20: (0.0, 500000.0] +0.25 (n 642) | (500000.0, 2000000.0] +0.46 (n 528) | (2000000.0, 10000000000000.0] +0.70 (n 461)
  wrapper: 0 +0.37 (n 1378) | 1 +0.70 (n 298)