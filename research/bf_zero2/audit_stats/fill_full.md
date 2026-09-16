# the fill model, re-simulated on the whole F6 candidate pool (25,876 of 25,876 candidates had bars)

## how far above the trigger level is the price one minute later? (all candidates)
split   median bps  mean bps     p75     p90  share <= +60 bps
TRAIN           48        69     105     209             58.2%
VAL             45        66     104     211             59.6%
TEST            48        79     121     255             56.9%

## the 4-concurrent first-come book under each entry convention
convention                                                  TRAIN                        VAL                       TEST
A  fill at the touch (THE CLAIM)       +0.131R +2.47/wk t+2.95 n996 +0.326R +6.05/wk t+4.37 n408 +0.356R +6.91/wk t+3.90 n272
B  stop order, next bar open           -0.247R -4.64/wk t-6.46 n996 -0.160R -2.96/wk t-2.59 n408 -0.110R -2.13/wk t-1.46 n272
C  live capped limit, no chase         -0.277R -5.19/wk t-7.01 n994 -0.315R -5.84/wk t-4.78 n408 -0.295R -5.73/wk t-3.52 n272

## C, in detail — the live spec: what fraction of signals are even obtainable?
  TRAIN  candidates  13323  fillable at <= +0.6% 58.2%  filled-only meanR -0.112  book n 994
  VAL    candidates   7214  fillable at <= +0.6% 59.6%  filled-only meanR -0.066  book n 408
  TEST   candidates   5339  fillable at <= +0.6% 56.9%  filled-only meanR -0.157  book n 272

## the same book, weeks green
  A: TRAIN 34/53 | VAL 17/22 | TEST 12/14
  B: TRAIN 11/53 | VAL 8/22 | TEST 6/14
  C: TRAIN 11/53 | VAL 3/22 | TEST 2/14