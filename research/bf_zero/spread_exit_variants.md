# Exit variants under real spreads — 1,908 signals, costs charged identically

## V0
| split   |    n |   meanR |     WR |   target_rate |   stop_rate |   usd_per_100 |
|:--------|-----:|--------:|-------:|--------------:|------------:|--------------:|
| TRAIN   | 2147 |   0.119 | 44.993 |        29.53  |      49.511 |        11.905 |
| VAL     | 2135 |   0.076 | 42.436 |        30.023 |      51.194 |         7.613 |
| TEST    | 2154 |   0.069 | 42.34  |        30.223 |      52.228 |         6.898 |

## V1
| split   |    n |   meanR |     WR |   target_rate |   stop_rate |   usd_per_100 |
|:--------|-----:|--------:|-------:|--------------:|------------:|--------------:|
| TRAIN   | 2147 |   0.132 | 44.341 |        26.968 |      50.07  |        13.239 |
| VAL     | 2135 |   0.074 | 41.405 |        26.417 |      51.85  |         7.406 |
| TEST    | 2154 |   0.073 | 41.365 |        27.623 |      52.878 |         7.283 |

## V2
| split   |    n |   meanR |     WR |   target_rate |   stop_rate |   usd_per_100 |
|:--------|-----:|--------:|-------:|--------------:|------------:|--------------:|
| TRAIN   | 2147 |   0.192 | 48.3   |        26.735 |      42.431 |        19.238 |
| VAL     | 2135 |   0.137 | 45.621 |        25.714 |      43.185 |        13.748 |
| TEST    | 2154 |   0.136 | 45.404 |        27.623 |      45.682 |        13.552 |

Note: V2 changes R itself (stop below the structure), so its R-multiples are on a larger unit — compare usd_per_100 (same $100 risk) across variants.