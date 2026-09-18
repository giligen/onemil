# Stage K — phase A tables

## Signal frequency (pre-book, primary universe)

| family | TRAIN signals | /week | VAL | TEST |
|---|---:|---:|---:|---:|
| K2 | 1,076 | 20.7 | 1,174 | 526 |

## TRAIN — the 20 pre-registered cells (primary universe, PREREG cost model)

| cell | n | tr/wk | gross bps | **net bps** | t | WR% | wk mean bps | wk green% | stop% | net R | MDE/trade bps | MDE/week bps | p_adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `K2_h10_n10` | 159 | 5.9 | 37.1 | **-11.6** | -0.11 | 45.3 | -6.8 | 55.6 | 28.3 | -0.017 | 204.5 | 119.0 | 0.762 |
| `K2_h10_n20` | 308 | 11.0 | 30.4 | **-18.4** | -0.27 | 45.8 | -10.1 | 46.4 | 29.5 | -0.026 | 133.9 | 84.5 | 0.801 |
| `K2_h5_n10` | 281 | 10.4 | -43.0 | **-91.5** | -1.56 | 45.2 | -95.2 | 40.7 | 24.6 | -0.131 | 117.2 | 143.9 | 0.977 |
| `K2_h5_n20` | 509 | 18.9 | -24.5 | **-70.4** | -1.76 | 44.6 | -66.4 | 40.7 | 21.0 | -0.101 | 80.0 | 109.7 | 0.968 |

## VAL — the 20 pre-registered cells (primary universe, PREREG cost model)

| cell | n | tr/wk | gross bps | **net bps** | t | WR% | wk mean bps | wk green% | stop% | net R | MDE/trade bps | MDE/week bps | p_adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `K2_h10_n10` | 127 | 5.8 | 67.3 | **19.2** | 0.12 | 37.8 | 11.1 | 40.9 | 37.8 | 0.027 | 314.1 | 203.2 | 0.844 |
| `K2_h10_n20` | 250 | 10.9 | 202.2 | **154.3** | 1.17 | 43.6 | 83.9 | 56.5 | 32.8 | 0.220 | 264.1 | 157.5 | 0.413 |
| `K2_h5_n10` | 227 | 10.3 | 25.0 | **-22.9** | -0.33 | 47.1 | -23.6 | 45.5 | 23.8 | -0.033 | 140.8 | 183.8 | 0.933 |
| `K2_h5_n20` | 421 | 19.1 | 35.9 | **-10.5** | -0.22 | 48.2 | -10.0 | 45.5 | 23.0 | -0.015 | 95.8 | 100.0 | 0.922 |

## Gate verdicts

G1 (TRAIN): mean net > 0, t >= 2.0, >= 5 trades/week. G2 (VAL): mean net > 0, t >= 1.0, >= 55% weeks green, weekly mean above 0 SE (0 cells passed G1 -> 0//10 = 0).

| cell | G1 | G2 |
|---|---|---|
| `K2_h10_n10` | fail | fail |
| `K2_h10_n20` | fail | fail |
| `K2_h5_n10` | fail | fail |
| `K2_h5_n20` | fail | fail |

**G1 survivors: 0 of 4. G2 survivors: 0.**

## Tails and the cost model (primary universe)

### TRAIN

| cell | net bps | no top 1% | no top 5% | winners capped | net bps (daily-band costs) | net bps (auction costs) | gross bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K2_h10_n10` | -11.6 | -78.5 | -178.0 | -48.3 | 27.7 | 27.1 | 37.1 |
| `K2_h10_n20` | -18.4 | -79.6 | -181.6 | -47.4 | 21.0 | 20.4 | 30.4 |
| `K2_h5_n10` | -91.5 | -130.0 | -206.1 | -111.5 | -52.3 | -53.0 | -43.0 |
| `K2_h5_n20` | -70.4 | -111.8 | -190.6 | -88.5 | -33.4 | -34.5 | -24.5 |

### VAL

| cell | net bps | no top 1% | no top 5% | winners capped | net bps (daily-band costs) | net bps (auction costs) | gross bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K2_h10_n10` | 19.2 | -129.6 | -272.2 | -67.2 | 58.0 | 57.3 | 67.3 |
| `K2_h10_n20` | 154.3 | -30.2 | -164.7 | 3.0 | 192.9 | 192.2 | 202.2 |
| `K2_h5_n10` | -22.9 | -96.3 | -191.0 | -75.3 | 15.8 | 15.0 | 25.0 |
| `K2_h5_n20` | -10.5 | -72.9 | -158.3 | -53.9 | 27.0 | 25.9 | 35.9 |

## Capacity (1% of the 20-day median dollar volume per position)

One position size for the whole book (PREREG sizes equal-$ per position): the median over the cell's trades of 1% of the name's 20-day median dollar volume.

| cell | split | position $ (median) | position $ (p25) | book $ at N slots | $/month (mean) | worst month $ | %book/month | months |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `K2_h10_n10` | TRAIN | 424,908 | 243,440 | 4,249,080 | -11,202 | -311,495 | -0.26% | 7 |
| `K2_h10_n20` | TRAIN | 410,574 | 220,339 | 8,211,480 | -33,192 | -747,705 | -0.40% | 7 |
| `K2_h5_n10` | TRAIN | 424,908 | 212,707 | 4,249,080 | -156,026 | -588,594 | -3.67% | 7 |
| `K2_h5_n20` | TRAIN | 555,100 | 248,038 | 11,102,000 | -284,270 | -1,142,990 | -2.56% | 7 |
| `K2_h10_n10` | VAL | 441,795 | 192,036 | 4,417,950 | 17,950 | -664,500 | 0.41% | 6 |
| `K2_h10_n20` | VAL | 447,631 | 204,676 | 8,952,630 | 287,817 | -528,791 | 3.21% | 6 |
| `K2_h5_n10` | VAL | 413,262 | 197,454 | 4,132,620 | -35,786 | -283,336 | -0.87% | 6 |
| `K2_h5_n20` | VAL | 513,892 | 230,196 | 10,277,800 | -37,723 | -567,517 | -0.37% | 6 |

## Survivorship control — secondary universe (stock OR not-in-map, i.e. only positively identified wrappers dropped), declared hold, 10 slots

| cell | split | n | net bps | t | wk green% |
|---|---|---:|---:|---:|---:|
| `K2_h10_n10_sec` | TRAIN | 156 | 26.2 | 0.25 | 55.6 |
| `K2_h10_n10_sec` | VAL | 124 | 131.1 | 0.68 | 45.5 |

