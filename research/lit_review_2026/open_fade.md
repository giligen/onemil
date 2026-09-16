# RUNBOOK row 10 — M18 open fade on prior-day attention names | rows 16105

mean return in bps from the 09:30 open to each clock time; "attention" = prior-day top-20 by abs(return) x volume ratio, "control" = ranks 100-120 the same day

## TRAIN
                      n  r0935  r1000  r1030  r1200  rclose
group     dvb                                              
attention $2-10M   1681    -39    -60    -74    -85     -74
          $10-50M  1569    -37    -48    -69    -73     -81
          >$50M    1463    -14    -27    -33    -47     -52
control   $2-10M   1430    -14    -11    -19    -18     -12
          $10-50M  1597    -21    -11     -8    -18     -11
          >$50M    1681    -15     -7    -14    -15      -1

## VAL
                     n  r0935  r1000  r1030  r1200  rclose
group     dvb                                             
attention $2-10M   742    -15    -31    -40    -41       0
          $10-50M  696     -9    -35    -11    -24      -8
          >$50M    569      4    -21    -27    -11      -4
control   $2-10M   603    -13    -31    -23    -21     -12
          $10-50M  647      1     -5     -9      9      35
          >$50M    757     11     11      6     26      40

## TEST
                     n  r0935  r1000  r1030  r1200  rclose
group     dvb                                             
attention $2-10M   476    -41    -51    -55    -68     -67
          $10-50M  460    -13     12     21     -7       4
          >$50M    399     -1     -3    -26    -80    -135
control   $2-10M   375      8    -26    -42    -56     -76
          $10-50M  436    -15      6     -9    -33     -42
          >$50M    524     -7      4      7    -14     -24

TRAIN: attention minus control, open→10:30 = -46.1 bps (t -4.92); attention alone -59.8 bps, n 4713
VAL: attention minus control, open→10:30 = -18.8 bps (t -1.10); attention alone -26.3 bps, n 2007
TEST: attention minus control, open→10:30 = -8.4 bps (t -0.37); attention alone -20.1 bps, n 1335