# N3 — stocks-in-play ORB, 2019-2023, XNAS.ITCH tape, our simulator (atr_scale=1.0)

## A  paper fill + paper commission

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,063 | -0.439 | 11.6 | -1782 | -8.0 |
| 2020 | 4,144 | -0.332 | 12.8 | -1375 | -4.5 |
| 2021 | 4,211 | -0.292 | 13.3 | -1230 | -5.0 |
| 2022 | 4,141 | -0.117 | 14.7 | -485 | -2.2 |
| 2023 | 4,100 | -0.242 | 12.9 | -994 | -4.2 |
| ALL | 20,659 | -0.284 | 13.1 | -5866 | -10.6 |

## A0 paper fill, zero cost

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,063 | -0.375 | 11.6 | -1525 | -6.9 |
| 2020 | 4,144 | -0.275 | 12.8 | -1141 | -3.7 |
| 2021 | 4,211 | -0.236 | 13.3 | -996 | -4.1 |
| 2022 | 4,141 | -0.064 | 14.8 | -267 | -1.2 |
| 2023 | 4,100 | -0.184 | 13.0 | -754 | -3.2 |
| ALL | 20,659 | -0.227 | 13.1 | -4682 | -8.4 |

## B0 our fill (next-bar open), zero cost

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,060 | +0.176 | 13.8 | +715 | +2.8 |
| 2020 | 4,142 | +0.021 | 14.1 | +85 | +0.4 |
| 2021 | 4,211 | +0.213 | 14.9 | +896 | +3.2 |
| 2022 | 4,139 | +0.240 | 16.8 | +994 | +4.1 |
| 2023 | 4,099 | +0.215 | 14.9 | +883 | +3.6 |
| ALL | 20,651 | +0.173 | 14.9 | +3573 | +6.4 |

## B1 our fill + 10 bps spread contract

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,060 | -0.020 | 13.7 | -81 | -0.3 |
| 2020 | 4,142 | -0.112 | 14.0 | -462 | -2.2 |
| 2021 | 4,211 | +0.079 | 14.9 | +331 | +1.2 |
| 2022 | 4,139 | +0.111 | 16.8 | +459 | +1.9 |
| 2023 | 4,099 | +0.046 | 14.7 | +188 | +0.8 |
| ALL | 20,651 | +0.021 | 14.8 | +434 | +0.8 |

## B2 our fill + 40 bps spread contract

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,060 | -0.608 | 13.3 | -2470 | -9.4 |
| 2020 | 4,142 | -0.508 | 13.8 | -2104 | -9.6 |
| 2021 | 4,211 | -0.324 | 14.7 | -1363 | -4.8 |
| 2022 | 4,139 | -0.277 | 16.4 | -1145 | -4.6 |
| 2023 | 4,099 | -0.463 | 14.5 | -1899 | -7.5 |
| ALL | 20,651 | -0.435 | 14.5 | -8981 | -15.7 |

## B3 our fill + OUR banded cost contract

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,060 | -0.855 | 13.2 | -3471 | -13.0 |
| 2020 | 4,142 | -0.650 | 13.8 | -2693 | -12.1 |
| 2021 | 4,211 | -0.485 | 14.5 | -2042 | -7.1 |
| 2022 | 4,139 | -0.427 | 16.4 | -1766 | -7.0 |
| 2023 | 4,099 | -0.685 | 14.2 | -2807 | -10.9 |
| ALL | 20,651 | -0.619 | 14.4 | -12779 | -22.1 |

## A3 paper fill + OUR banded cost contract

| year | trades | R/trade | WR % | total R | t |
|---|---|---|---|---|---|
| 2019 | 4,063 | -2.150 | 10.2 | -8736 | -36.3 |
| 2020 | 4,144 | -1.425 | 11.8 | -5904 | -18.6 |
| 2021 | 4,211 | -1.438 | 12.5 | -6056 | -23.5 |
| 2022 | 4,141 | -1.218 | 13.8 | -5042 | -21.5 |
| 2023 | 4,100 | -1.734 | 11.6 | -7110 | -28.2 |
| ALL | 20,659 | -1.590 | 12.0 | -32849 | -55.9 |

## mechanism

- median R = 0.10 x ATR14 = **0.403% of the 09:30 price** (40 bps) — one 40 bps round trip is 0.99 R of cost.
- paper fill stopped out inside its own trigger bar: **34.5%** of fills; our next-bar-open fill: 21.3%.
- stop rate: paper 86.2%, ours 84.4%.

## R/trade by relative volume

| RV bucket | trades | A  paper fill + commission | B0 our fill, zero cost | B1 our fill + 10 bps |
|---|---|---|---|---|
| 1-2x | 35 | -0.275 | -0.481 | -0.672 |
| 2-3x | 120 | -0.038 | +0.034 | -0.142 |
| 3-5x | 2,661 | -0.002 | +0.156 | -0.013 |
| 5-10x | 10,736 | -0.166 | +0.216 | +0.064 |
| 10-30x | 5,863 | -0.517 | +0.127 | -0.019 |
| >30x | 1,244 | -0.835 | +0.086 | -0.053 |

## daily portfolio hit ratio (equal risk per trade)

| arm | sessions | % days total R > 0 |
|---|---|---|
| A  paper fill + paper commission | 1,258 | 31.2 |
| A0 paper fill, zero cost | 1,258 | 33.4 |
| B0 our fill (next-bar open), zero cost | 1,258 | 48.4 |
| B1 our fill + 10 bps spread contract | 1,258 | 43.1 |
| B2 our fill + 40 bps spread contract | 1,258 | 28.7 |
| B3 our fill + OUR banded cost contract | 1,258 | 24.2 |
| A3 paper fill + OUR banded cost contract | 1,258 | 8.4 |

## tail dependence

| variant | A  paper fill + commission | B0 our fill, zero cost | B1 our fill + 10 bps |
|---|---|---|---|
| all | -0.284 | +0.173 | +0.021 |
| ex top 1% | -0.521 | -0.075 | -0.227 |
| ex top 5% | -0.954 | -0.549 | -0.704 |
| winners capped at +5R | -0.667 | -0.309 | -0.453 |
| winners capped at +10R | -0.452 | -0.040 | -0.189 |

## side split

| side | trades | A  paper fill + commission | B0 our fill, zero cost | B1 our fill + 10 bps |
|---|---|---|---|---|
| long | 10,274 | -0.286 | +0.156 | -0.002 |
| short | 10,385 | -0.281 | +0.190 | +0.044 |

## price-scale guard

- picks whose 09:30 open differs from the daily-file prior close by more than 50% (split-like): **158** of 25,135.
  - A  paper fill + commission: ex-jump -0.275 vs jump rows -1.690
  - B0 our fill, zero cost: ex-jump +0.175 vs jump rows -0.189
  - B1 our fill + 10 bps: ex-jump +0.023 vs jump rows -0.296

