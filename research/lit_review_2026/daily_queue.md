# RUNBOOK row 9 — M16 single-stock gap table (measurement, $5+, $2M+/day, 2025-01→2026-09)

Columns: n, P(close>open), mean and median open→close in bps, P(gap fully filled that day), P(half filled). Split TRAIN=2025 / VAL+TEST=2026.

## gap table 2025
                       n  p_up  mean_bps  med_bps  fill  half
gapb     dvb                                                 
<-20%    $2-10M      103  53.4       180      136   2.9  18.4
         $10-50M     129  57.4       238      187   1.6  20.2
         >$50M       152  48.0        19      -15   4.6  15.8
-20..-10 $2-10M      473  49.5       117        0  19.0  40.8
         $10-50M     579  52.7        88       49  18.1  41.6
         >$50M       755  49.3        89      -18  16.4  35.0
-10..-5  $2-10M     2209  54.1        73       52  30.2  54.8
         $10-50M    2713  50.1        26        3  30.2  55.7
         >$50M      3571  48.7         7      -15  24.3  47.1
-5..-2   $2-10M    12405  52.4        34       21  40.6  64.2
         $10-50M   13857  53.5        28       27  40.1  65.9
         >$50M     17927  54.1        29       32  37.1  63.8
-2..+2   $2-10M   348809  49.5         5        0  69.6  81.6
         $10-50M  304452  49.8         5        0  72.9  84.6
         >$50M    356146  50.6         4        2  72.8  85.4
+2..+5   $2-10M    11726  40.3       -62      -57  44.5  67.8
         $10-50M   13058  41.9       -56      -53  43.8  69.0
         >$50M     17943  43.3       -49      -41  39.7  64.3
+5..+10  $2-10M     2365  42.1       -78      -88  33.9  60.0
         $10-50M    2597  42.4       -80      -78  31.7  58.0
         >$50M      3758  42.4       -94      -79  26.8  52.7
+10..+20 $2-10M      594  42.4      -166     -154  21.5  54.9
         $10-50M     756  44.8       -91      -91  20.6  47.6
         >$50M       839  40.6      -152     -177  17.0  42.7
>+20%    $2-10M      152  48.0      -246      -18  11.2  34.2
         $10-50M     188  37.8      -355     -250   8.0  37.8
         >$50M       166  34.3      -341     -244   9.6  38.6

## gap table 2026
                       n  p_up  mean_bps  med_bps  fill  half
gapb     dvb                                                 
<-20%    $2-10M      110  48.2       -51      -50   4.5  18.2
         $10-50M     135  48.1        76      -70   3.0  20.0
         >$50M       133  51.9        98       10   2.3  15.0
-20..-10 $2-10M      487  47.0        -8      -26  17.9  39.6
         $10-50M     661  52.3        48       28  16.6  43.0
         >$50M       799  45.9       -35      -58  10.6  35.3
-10..-5  $2-10M     2242  52.0        27       29  29.3  53.9
         $10-50M    2404  54.0        54       56  30.2  56.5
         >$50M      3429  50.2         6        3  22.2  48.4
-5..-2   $2-10M    10979  53.4        22       27  40.1  63.6
         $10-50M   12805  54.6        33       33  42.0  67.7
         >$50M     18887  53.0        20       23  38.0  64.4
-2..+2   $2-10M   261779  48.9         4        0  66.3  78.6
         $10-50M  229043  49.3         4        0  70.7  82.7
         >$50M    281039  49.4         0        0  70.9  83.8
+2..+5   $2-10M    10970  49.1        16       -2  37.8  59.5
         $10-50M   12623  50.2        21        3  37.3  60.8
         >$50M     19489  50.1        15        1  34.1  59.9
+5..+10  $2-10M     2632  45.3       -32      -43  30.7  56.2
         $10-50M    2887  47.2        16      -18  26.9  54.3
         >$50M      4229  46.9         4      -24  21.3  48.4
+10..+20 $2-10M      699  46.2       -18      -49  18.9  50.5
         $10-50M     716  48.0        16      -27  20.1  50.6
         >$50M       975  47.8       -12      -40  13.1  41.8
>+20%    $2-10M      140  47.1      -261      -37   9.3  33.6
         $10-50M     203  47.8      -149      -18   4.4  38.4
         >$50M       168  43.5      -355      -43  10.1  36.9

cells with mean open→close >= +50 bps and n >= 200 in BOTH years: 0
(none)

# RUNBOOK row 14 — M29 cross-sectional overnight continuation, large caps ($50M+/day)

TRAIN top-decile  n  36779 gross  +13.4 bps median  +7.3 net   +7.4 t +5.12 hit 50.5%
TRAIN top-4 book  n    904 gross  +48.7 bps median  +6.8 net  +42.7 t +2.55 hit 50.4%
VAL   top-decile  n  24867 gross  +12.7 bps median  +5.4 net   +6.7 t +3.45 hit 49.9%
VAL   top-4 book  n    408 gross  +78.5 bps median +75.5 net  +72.5 t +2.34 hit 57.8%
TEST  top-decile  n  16927 gross   -7.4 bps median  +0.0 net  -13.4 t -5.48 hit 47.2%
TEST  top-4 book  n    268 gross  -34.4 bps median  -1.5 net  -40.4 t -0.82 hit 45.9%

# RUNBOOK row 6 refinement — M36 large-loser reversal by market-volatility regime (trailing 20-day SPY vol tercile, cut on 2025)

TRAIN low  n   273 gross  -179.3 bps median -122.2 net  -197.8 t -4.52 hit 39.2%
TRAIN mid  n   323 gross  -109.6 bps median  -74.4 net  -127.9 t -2.53 hit 43.3%
TRAIN high n   252 gross   +36.6 bps median  -11.7 net   +16.8 t +0.29 hit 46.0%
VAL   low  n   136 gross   -64.1 bps median -117.9 net   -83.4 t -1.01 hit 40.4%
VAL   mid  n   188 gross   -75.4 bps median  -52.0 net   -95.1 t -1.60 hit 45.2%
VAL   high n    79 gross  -187.4 bps median   +5.5 net  -206.0 t -1.46 hit 48.1%
TEST  low  n    42 gross   -21.0 bps median  +86.9 net   -39.4 t -0.26 hit 52.4%
TEST  mid  n   140 gross  +101.4 bps median  +35.5 net   +82.1 t +1.05 hit 50.7%
TEST  high n    70 gross  -125.0 bps median -123.2 net  -145.3 t -1.19 hit 41.4%