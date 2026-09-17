# Counterfactuals on the real live trades

## orb (116 real trades)
                               rule   n  R_real  meanR_real  WR_real  R_closestop  meanR_closestop  R_buffer  meanR_buffer
                         all trades 116   -10.7      -0.092     0.38         -8.3           -0.072      -7.5        -0.065
                  vol_break >= 1.5x  45    -6.5      -0.145     0.42         -6.4           -0.143      -8.5        -0.189
vol_break < 1.5x (would be skipped)  71    -4.2      -0.059     0.35         -1.9           -0.027       1.0         0.014
                         gap >= +2% 116   -10.7      -0.092     0.38         -8.3           -0.072      -7.5        -0.065
       gap < +2% (would be skipped)   0     0.0         NaN      NaN          0.0              NaN       0.0           NaN
          vol >= 1.5x AND gap >= 2%  45    -6.5      -0.145     0.42         -6.4           -0.143      -8.5        -0.189

stopped trades: 54 — real -36.1R, close-stop -36.8R, buffer -35.9R; close-stop turns 9 of them positive, buffer 12

## bull_flag (53 real trades)
                               rule  n  R_real  meanR_real  WR_real  R_closestop  meanR_closestop  R_buffer  meanR_buffer
                         all trades 53   -10.4      -0.197     0.34        -20.0           -0.378     -23.2        -0.438
                  vol_break >= 1.5x 20     7.2       0.360     0.55         -6.4           -0.318      -4.2        -0.208
vol_break < 1.5x (would be skipped) 33   -17.6      -0.534     0.21        -13.7           -0.414     -19.1        -0.578
                         gap >= +2% 24     3.2       0.134     0.42         -1.7           -0.069      -8.7        -0.361
       gap < +2% (would be skipped) 29   -13.7      -0.471     0.28        -18.4           -0.633     -14.6        -0.503
          vol >= 1.5x AND gap >= 2% 12     9.3       0.772     0.58          4.2            0.347       5.2         0.435

stopped trades: 40 — real -19.6R, close-stop -25.9R, buffer -29.4R; close-stop turns 6 of them positive, buffer 4
