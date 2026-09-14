# HOD-break spread study — 6,847 signals with quotes ({'TEST': 2303, 'VAL': 2278, 'TRAIN': 2266}); TRAIN quintile edges of spread/R: [0.073, 0.133, 0.238, 0.458]

## TRAIN
|   q |   n |   meanR |   meanR_after_cost |     WR |   spread_bps |   sfr |
|----:|----:|--------:|-------------------:|-------:|-------------:|------:|
|   1 | 454 |   0.339 |              0.295 | 51.101 |        9.86  | 0.045 |
|   2 | 455 |   0.318 |              0.218 | 49.451 |       18.721 | 0.1   |
|   3 | 451 |   0.239 |              0.059 | 47.228 |       36.631 | 0.179 |
|   4 | 453 |   0.368 |              0.034 | 51.656 |       62.184 | 0.325 |
|   5 | 453 |   0.251 |             -0.659 | 47.02  |      143.369 | 0.735 |

## VAL
|   q |   n |   meanR |   meanR_after_cost |     WR |   spread_bps |   sfr |
|----:|----:|--------:|-------------------:|-------:|-------------:|------:|
|   1 | 399 |   0.238 |              0.192 | 47.368 |        9.479 | 0.048 |
|   2 | 405 |   0.319 |              0.217 | 48.642 |       19.057 | 0.102 |
|   3 | 488 |   0.32  |              0.139 | 47.746 |       37.094 | 0.179 |
|   4 | 498 |   0.172 |             -0.154 | 44.578 |       59.347 | 0.312 |
|   5 | 488 |   0.29  |             -0.626 | 48.566 |      146.407 | 0.736 |

## TEST
|   q |   n |   meanR |   meanR_after_cost |     WR |   spread_bps |   sfr |
|----:|----:|--------:|-------------------:|-------:|-------------:|------:|
|   1 | 384 |   0.447 |              0.401 | 52.865 |       10.186 | 0.047 |
|   2 | 493 |   0.287 |              0.183 | 47.667 |       20.833 | 0.104 |
|   3 | 510 |   0.25  |              0.07  | 46.667 |       39.093 | 0.179 |
|   4 | 514 |   0.184 |             -0.143 | 44.358 |       64.612 | 0.318 |
|   5 | 402 |   0.17  |             -0.805 | 44.527 |      138.413 | 0.701 |

gate spread/R <= 10%: TRAIN: keep 681/2266 meanR kept +0.369 dropped +0.275 | after-cost kept +0.311 | VAL: keep 590/2278 meanR kept +0.252 dropped +0.272 | after-cost kept +0.193 | TEST: keep 598/2303 meanR kept +0.442 dropped +0.199 | after-cost kept +0.381
gate spread/R <= 15%: TRAIN: keep 1004/2266 meanR kept +0.348 dropped +0.268 | after-cost kept +0.269 | VAL: keep 896/2278 meanR kept +0.262 dropped +0.270 | after-cost kept +0.181 | TEST: keep 981/2303 meanR kept +0.330 dropped +0.212 | after-cost kept +0.245
gate spread/R <= 20%: TRAIN: keep 1224/2266 meanR kept +0.321 dropped +0.283 | after-cost kept +0.225 | VAL: keep 1139/2278 meanR kept +0.281 dropped +0.253 | after-cost kept +0.180 | TEST: keep 1246/2303 meanR kept +0.316 dropped +0.198 | after-cost kept +0.212
gate spread/R <= 30%: TRAIN: keep 1529/2266 meanR kept +0.316 dropped +0.275 | after-cost kept +0.191 | VAL: keep 1510/2278 meanR kept +0.286 dropped +0.228 | after-cost kept +0.150 | TEST: keep 1589/2303 meanR kept +0.302 dropped +0.172 | after-cost kept +0.167

by price band (all splits):                       n  spread_bps    sfr  meanR  after
  price                                                   
  (5.0, 10.0]        2092      39.526  0.191  0.182 -0.167
  (10.0, 20.0]       2142      39.005  0.197  0.189 -0.140
  (20.0, 50.0]       1782      38.206  0.181  0.349  0.052
  (50.0, 1000000.0]   824      38.534  0.176  0.597  0.326