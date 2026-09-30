# RESULT_1675 -- short the HOD-break failure, sealed forward test

**Population window: 2026-06-01..2026-09-04 (universe.csv/nbbo.csv end 09-04; 09-05..09-26 not covered)** -- universe.csv and causal_filter/nbbo.csv both end 2026-09-04; their builder script could not be located within budget, so 2026-09-05..09-26 is NOT covered. Same population definition (cell 1,438), not a substitute.

## TRAIN-H2 model -- week table (P>=0.6), n=97 fired shorts, 14 weeks
| wk                    |   shorts |   etb_share |   hit_rate |   mean_net_R |      sum_R |   dollars_at_150 |   worst_day |   max_concurrent |   shorts_under_cap |
|:----------------------|---------:|------------:|-----------:|-------------:|-----------:|-----------------:|------------:|-----------------:|-------------------:|
| 2026-05-30/2026-06-05 |        4 |         nan |   0.25     |  -0.638398   | -2.55359   |       -383.039   |   -1.16605  |                1 |                  4 |
| 2026-06-06/2026-06-12 |        8 |         nan |   0.5      |  -0.190636   | -1.52509   |       -228.764   |   -2.22959  |                4 |                  8 |
| 2026-06-13/2026-06-19 |        9 |         nan |   0.666667 |  -0.00518743 | -0.0466869 |         -7.00303 |   -1.11099  |                4 |                  9 |
| 2026-06-20/2026-06-26 |       11 |         nan |   0.545455 |  -0.064265   | -0.706915  |       -106.037   |   -4.57611  |                6 |                 11 |
| 2026-06-27/2026-07-03 |       10 |         nan |   0.4      |  -0.279008   | -2.79008   |       -418.513   |   -2.13052  |                4 |                 10 |
| 2026-07-04/2026-07-10 |       13 |         nan |   0.461538 |  -0.117685   | -1.52991   |       -229.486   |   -1.04219  |                5 |                 13 |
| 2026-07-11/2026-07-17 |        6 |         nan |   0.166667 |  -0.653874   | -3.92324   |       -588.486   |   -1.17208  |                2 |                  6 |
| 2026-07-18/2026-07-24 |        3 |         nan |   0.333333 |  -0.574842   | -1.72453   |       -258.679   |   -2.23695  |                2 |                  3 |
| 2026-07-25/2026-07-31 |        7 |         nan |   0.428571 |  -0.807188   | -5.65031   |       -847.547   |   -6.15535  |                6 |                  7 |
| 2026-08-01/2026-08-07 |       10 |         nan |   0.4      |  -0.247892   | -2.47892   |       -371.838   |   -5.06745  |                4 |                 10 |
| 2026-08-08/2026-08-14 |        6 |         nan |   0.333333 |  -0.603169   | -3.61902   |       -542.853   |   -2.16848  |                3 |                  6 |
| 2026-08-15/2026-08-21 |        3 |         nan |   1        |  10.3524     | 31.0572    |       4658.58    |   15.277    |                2 |                  3 |
| 2026-08-22/2026-08-28 |        3 |         nan |   0        |  -1.13785    | -3.41356   |       -512.034   |   -2.27731  |                2 |                  3 |
| 2026-08-29/2026-09-04 |        4 |         nan |   0.75     |   0.0608635  |  0.243454  |         36.5181  |   -0.226746 |                2 |                  4 |

* TRAIN-H2 model, no ETB filter: n=97, mean net R=+0.0138, week-t=0.47 (n_weeks=14), green 2/14, worst week=-5.65R, ETB share=nan, shorts/wk under cap=6.93, pessimistic-gap mean=+0.0138 -> **NO-GO**
* TRAIN-H2 model, WITH ETB filter: n=97, mean net R=+0.0138, week-t=0.47 (n_weeks=14), green 2/14, worst week=-5.65R, ETB share=nan, shorts/wk under cap=6.93, pessimistic-gap mean=+0.0138 -> **NO-GO**
* Cap split (first 12/day): under-cap n=97 mean R=+0.0138 | over-cap n=0 mean R=+nan
* Stop bucket 1.5-3%: n=88 mean R=-0.1227
* Stop bucket >=3%: n=9 mean R=+1.3485
* Gapped-through share of resolving exits: 15.5%; pessimistic-gap pooled mean = +0.0138 vs optimistic -0.0067
* Reversal variant (cut+short, long units): mean=-0.6517 vs holding the long mean=-0.5345 (n=97)

## VAL model -- week table (P>=0.6), n=57 fired shorts, 12 weeks
| wk                    |   shorts |   etb_share |   hit_rate |   mean_net_R |      sum_R |   dollars_at_150 |   worst_day |   max_concurrent |   shorts_under_cap |
|:----------------------|---------:|------------:|-----------:|-------------:|-----------:|-----------------:|------------:|-----------------:|-------------------:|
| 2026-05-30/2026-06-05 |        3 |         nan |   0.666667 |    0.118402  |   0.355206 |          53.281  |   -1.04997  |                1 |                  3 |
| 2026-06-06/2026-06-12 |        6 |         nan |   0.5      |   -0.133398  |  -0.800387 |        -120.058  |   -1.31772  |                3 |                  6 |
| 2026-06-13/2026-06-19 |       10 |         nan |   0.7      |   -0.0194076 |  -0.194076 |         -29.1115 |   -2.03154  |                5 |                 10 |
| 2026-06-20/2026-06-26 |        5 |         nan |   0.6      |   -0.346926  |  -1.73463  |        -260.194  |   -1.10872  |                2 |                  5 |
| 2026-06-27/2026-07-03 |        8 |         nan |   0.375    |   -0.50489   |  -4.03912  |        -605.868  |   -2.05356  |                4 |                  8 |
| 2026-07-04/2026-07-10 |       10 |         nan |   0.5      |   -0.133839  |  -1.33839  |        -200.759  |   -1.3934   |                5 |                 10 |
| 2026-07-11/2026-07-17 |        3 |         nan |   0.666667 |    0.337302  |   1.01191  |         151.786  |   -1.07047  |                2 |                  3 |
| 2026-07-18/2026-07-24 |        2 |         nan |   0        |  -12.1733    | -24.3466   |       -3651.99   |  -23.1168   |                1 |                  2 |
| 2026-07-25/2026-07-31 |        5 |         nan |   0.4      |   -0.217954  |  -1.08977  |        -163.465  |   -1.59481  |                4 |                  5 |
| 2026-08-01/2026-08-07 |        3 |         nan |   0.333333 |   -0.586867  |  -1.7606   |        -264.09   |   -1.1135   |                2 |                  3 |
| 2026-08-08/2026-08-14 |        1 |         nan |   1        |    0.621049  |   0.621049 |          93.1574 |    0.621049 |                1 |                  1 |
| 2026-08-29/2026-09-04 |        1 |         nan |   1        |    0.441779  |   0.441779 |          66.2669 |    0.441779 |                1 |                  1 |

* VAL model, no ETB filter: n=57, mean net R=-0.5767, week-t=-1.03 (n_weeks=12), green 4/12, worst week=-24.35R, ETB share=nan, shorts/wk under cap=4.75, pessimistic-gap mean=-0.5767 -> **NO-GO**
* VAL model, WITH ETB filter: n=57, mean net R=-0.5767, week-t=-1.03 (n_weeks=12), green 4/12, worst week=-24.35R, ETB share=nan, shorts/wk under cap=4.75, pessimistic-gap mean=-0.5767 -> **NO-GO**
* Cap split (first 12/day): under-cap n=57 mean R=-0.5767 | over-cap n=0 mean R=+nan
* Stop bucket 1.5-3%: n=46 mean R=-0.1814
* Stop bucket >=3%: n=11 mean R=-2.2299
* Gapped-through share of resolving exits: 15.8%; pessimistic-gap pooled mean = -0.5767 vs optimistic -0.2744
* Reversal variant (cut+short, long units): mean=-0.7558 vs holding the long mean=-0.7995 (n=57)

## GO / NO-GO (frozen rule, BOTH models + ETB filter required): **NO-GO**