# Q3 — block bootstrap of the weekly and daily series

weeks per split {'TRAIN': 53, 'VAL': 22, 'TEST': 14} | trading days {'TRAIN': 249, 'VAL': 102, 'TEST': 68}

mean R per week (daily series rescaled to a week), 10,000 circular block resamples
  (block = 2 weeks for the weekly series, 5 days for the daily series)
unit  split  n_units   obs    lo     hi  p_below_0  p_below_2  p_below_5    sd
week  TRAIN       53 2.437 0.862  4.008      0.001      0.290      1.000 0.799
week    VAL       22 5.967 3.000  8.895      0.000      0.004      0.263 1.515
week   TEST       14 6.798 2.917 10.495      0.000      0.008      0.180 1.937
week POOLED       89 3.996 2.531  5.418      0.000      0.004      0.915 0.739
 day  TRAIN      249 2.437 0.791  4.088      0.002      0.300      0.999 0.846
 day    VAL      102 5.967 3.107  8.809      0.000      0.004      0.251 1.461
 day   TEST       68 6.798 2.990 10.518      0.000      0.006      0.179 1.927
 day POOLED      419 3.996 2.555  5.410      0.000      0.003      0.918 0.728


# Q4 — stability


## by month
  TRAIN  2025-01 +0.233 (n76, +18R) | 2025-02 +0.144 (n76, +11R) | 2025-03 -0.016 (n84, -1R) | 2025-04 +0.131 (n84, +11R) | 2025-05 +0.144 (n84, +12R) | 2025-06 -0.094 (n80, -7R) | 2025-07 +0.452 (n88, +40R) | 2025-08 +0.099 (n84, +8R) | 2025-09 +0.220 (n84, +18R) | 2025-10 +0.131 (n92, +12R) | 2025-11 +0.022 (n76, +2R) | 2025-12 +0.069 (n88, +6R)
  VAL    2026-01 +0.152 (n80, +12R) | 2026-02 +0.384 (n76, +29R) | 2026-03 -0.014 (n88, -1R) | 2026-04 +0.549 (n84, +46R) | 2026-05 +0.563 (n80, +45R)
  TEST   2026-06 +0.499 (n84, +42R) | 2026-07 +0.042 (n88, +4R) | 2026-08 +0.557 (n84, +47R) | 2026-09 +0.172 (n16, +3R)

## by price band
  TRAIN  (5.0, 10.0] +0.091 (n424, +39R) | (10.0, 20.0] +0.202 (n285, +58R) | (20.0, 50.0] +0.126 (n188, +24R) | (50.0, 1000000000.0] +0.091 (n99, +9R)
  VAL    (5.0, 10.0] +0.244 (n137, +33R) | (10.0, 20.0] +0.464 (n109, +51R) | (20.0, 50.0] +0.246 (n93, +23R) | (50.0, 1000000000.0] +0.353 (n69, +24R)
  TEST   (5.0, 10.0] +0.249 (n81, +20R) | (10.0, 20.0] +0.454 (n81, +37R) | (20.0, 50.0] +0.346 (n82, +28R) | (50.0, 1000000000.0] +0.351 (n28, +10R)

## by entry-minute band
  TRAIN  (569, 575] +0.142 (n900, +128R) | (575, 580] -0.076 (n66, -5R) | (580, 585] +0.598 (n16, +10R) | (585, 590] -0.385 (n8, -3R) | (590, 600] +0.006 (n5, +0R) | (600, 842] +0.055 (n1, +0R)
  VAL    (569, 575] +0.327 (n402, +132R) | (575, 580] +0.154 (n4, +1R) | (580, 585] -0.442 (n2, -1R)
  TEST   (569, 575] +0.350 (n272, +95R)

## by leveraged-ETF wrapper flag
  TRAIN  0.0 +0.125 (n900, +112R) | 1.0 +0.219 (n93, +20R)
  VAL    0.0 +0.229 (n307, +70R) | 1.0 +0.604 (n101, +61R)
  TEST   0.0 +0.303 (n167, +51R) | 1.0 +0.424 (n105, +45R)

## by R size (% of price)
  TRAIN  (1.0, 2.0] +0.402 (n172, +69R) | (2.0, 4.0] +0.201 (n324, +65R) | (4.0, 8.0] +0.013 (n437, +6R) | (8.0, 1000000000.0] -0.173 (n63, -11R)
  VAL    (1.0, 2.0] +0.462 (n88, +41R) | (2.0, 4.0] +0.397 (n156, +62R) | (4.0, 8.0] +0.134 (n151, +20R) | (8.0, 1000000000.0] +0.655 (n13, +9R)
  TEST   (1.0, 2.0] +0.618 (n51, +32R) | (2.0, 4.0] +0.511 (n100, +51R) | (4.0, 8.0] +0.068 (n111, +8R) | (8.0, 1000000000.0] +0.499 (n10, +5R)

## concentration — largest single symbol / single day as a share of the split profit
split    total R    top sym    sym R   share      top day    day R   share  top trade share
TRAIN      129.2       RYET      7.7    6.0%   2025-07-01      8.0    6.2%             1.5%
       top-5 symbols 25.9% of profit (['RYET', 'COEP', 'RGTIW', 'CORZW', 'SION']); top-5 days 29.1%; symbols with >10% of profit: []; days with >10%: []
VAL        131.3       ASTN     12.9    9.8%   2026-02-09      8.0    6.1%             1.5%
       top-5 symbols 30.0% of profit (['ASTN', 'IONZ', 'SMX', 'BEX', 'AXTI']); top-5 days 30.5%; symbols with >10% of profit: []; days with >10%: []
TEST        95.2       CBRZ      4.9    5.2%   2026-08-04      8.0    8.4%             2.1%
       top-5 symbols 22.0% of profit (['CBRZ', 'ARQQ', 'ASTN', 'ARCT', 'RUM']); top-5 days 40.2%; symbols with >10% of profit: []; days with >10%: []

## trimming the best trades (and, for contrast, the worst)
split      n   total R |  drop best 1%       5%      10% |  drop worst 1%       5%      10%
TRAIN    996     129.2 |         109.2     29.2    -70.8 |          163.9    225.3    286.9
VAL      408     131.3 |         121.3     89.3     49.3 |          149.0    177.3    203.6
TEST     272      95.2 |          89.2     67.2     39.2 |          106.9    126.6    143.7
  (share of profit surviving a 10% best-trim: TRAIN -55%, VAL 38%, TEST 41%)

  exit-reason mix and the left tail
  TRAIN  eod 283 (+0.16R) | stop 419 (-1.20R) | target 294 (+2.00R) | worst trade -7.94R | trades < -2R: 13 (1.3%)
  VAL    eod 92 (+0.24R) | stop 160 (-1.27R) | target 156 (+2.00R) | worst trade -4.16R | trades < -2R: 8 (2.0%)
  TEST   eod 63 (+0.34R) | stop 106 (-1.25R) | target 103 (+2.00R) | worst trade -5.71R | trades < -2R: 5 (1.8%)


# Q5 — does the result depend on WHICH four of the day's candidates are taken?

A  first-come, alphabetical tie-break (THE CLAIM): TRAIN +0.124 (n996) VAL +0.320 (n408) TEST +0.353 (n272)
D  LAST-come control (latest four of the day):   TRAIN +0.023 (n996) VAL -0.053 (n408) TEST -0.039 (n272)
E  the whole qualifying population, no book:     TRAIN -0.006 (n13323) VAL +0.080 (n7214) TEST +0.017 (n5339)

B  first-come, RANDOM tie-break  (200 draws)
  split      mean      sd     p2.5    p97.5    claim  pctile of claim   n/draw
  TRAIN    +0.131   0.014   +0.104   +0.159   +0.124           28.0%      996
  VAL      +0.298   0.037   +0.229   +0.367   +0.320           71.5%      408
  TEST     +0.382   0.050   +0.294   +0.490   +0.353           28.5%      272

C  four RANDOM candidates per day  (200 draws)
  split      mean      sd     p2.5    p97.5    claim  pctile of claim   n/draw
  TRAIN    +0.015   0.028   -0.038   +0.064   +0.124          100.0%      996
  VAL      +0.049   0.046   -0.040   +0.142   +0.320          100.0%      408
  TEST     -0.020   0.055   -0.130   +0.084   +0.353          100.0%      272

## the same thing as R per week (multiply mean R by trades/week)
  TRAIN  random-4: +0.28 R/wk   vs claim +2.32 R/wk
  VAL    random-4: +0.91 R/wk   vs claim +5.93 R/wk
  TEST   random-4: -0.40 R/wk   vs claim +6.85 R/wk

## what "first-come" selects (population mean R by within-day entry rank)
  TRAIN  (0, 4] +0.124 (n996) | (4, 8] +0.009 (n996) | (8, 16] +0.004 (n1924) | (16, 32] +0.012 (n2589) | (32, 64] +0.013 (n2082) | (64, 1000000] -0.059 (n4736)
  VAL    (0, 4] +0.320 (n408) | (4, 8] +0.248 (n408) | (8, 16] +0.053 (n809) | (16, 32] +0.067 (n1432) | (32, 64] +0.049 (n1831) | (64, 1000000] +0.051 (n2326)
  TEST   (0, 4] +0.353 (n272) | (4, 8] +0.174 (n272) | (8, 16] +0.088 (n544) | (16, 32] +0.091 (n1047) | (32, 64] -0.071 (n1472) | (64, 1000000] -0.055 (n1732)

## and the mechanical correlate: R size (% of price) by within-day entry rank
  TRAIN  (0, 4] r_pct 3.98% | (4, 8] r_pct 4.86% | (8, 16] r_pct 5.42% | (16, 32] r_pct 5.78% | (32, 64] r_pct 5.83% | (64, 1000000] r_pct 6.27%
  VAL    (0, 4] r_pct 3.59% | (4, 8] r_pct 4.46% | (8, 16] r_pct 5.11% | (16, 32] r_pct 5.59% | (32, 64] r_pct 5.87% | (64, 1000000] r_pct 5.82%
  TEST   (0, 4] r_pct 3.80% | (4, 8] r_pct 4.16% | (8, 16] r_pct 4.81% | (16, 32] r_pct 5.45% | (32, 64] r_pct 5.97% | (64, 1000000] r_pct 6.42%


# Q6 — power

split      n   sd(R)      se  observed mean  MDE 80% power  MDE in R/wk  obs/MDE
TRAIN    996   1.404  0.0445        +0.1297         0.1246         2.34     1.04
VAL      408   1.503  0.0744        +0.3218         0.2085         3.87     1.54
TEST     272   1.506  0.0913        +0.3499         0.2559         4.97     1.37
POOLED  1676   1.448  0.0354        +0.2122         0.0991

  MDE = (z_.975 + z_.80) x se, two-sided 5%. Weekly-series power (the quantity the claim is stated in):
  TRAIN  weeks  53 sd   6.41 se  0.88 observed  +2.44 R/wk  MDE  2.47 R/wk
  VAL    weeks  22 sd   7.81 se  1.66 observed  +5.97 R/wk  MDE  4.66 R/wk
  TEST   weeks  14 sd   7.65 se  2.04 observed  +6.80 R/wk  MDE  5.73 R/wk