# Stage H / F6 — step 0: parity anchors

generated 2026-09-17 12:39:57

## universe P — 12,018 F6 rows, weeks TRAIN=53 VAL=22 TEST=14

| cell | n | tr/wk | net R | gross R | t | WR | green |
|---|---:|---:|---:|---:|---:|---:|---:|
| hold floor=1 TRAIN | 1242 | 23.4 | **+0.0510** | +0.0837 | 1.34 | 44.4 | 0.57 |
| &nbsp;&nbsp;expected |  | | +0.0510 | | 1.34 | | |
| hold floor=1 VAL | 536 | 24.4 | **+0.1629** | +0.1973 | 2.46 | 47.8 | 0.68 |
| &nbsp;&nbsp;expected |  | | +0.1630 | | 2.46 | | |
| hold floor=0 TRAIN | 1242 | 23.4 | **+0.0510** | +0.0837 | 1.34 | 44.4 | 0.57 |
| hold floor=0 VAL | 536 | 24.4 | **+0.1629** | +0.1973 | 2.46 | 47.8 | 0.68 |
| stopm1 floor=1 TRAIN | 1290 | 24.3 | **+0.0245** | +0.0558 | 0.98 | 46.4 | 0.53 |
| stopm1 floor=1 VAL | 559 | 25.4 | **+0.0893** | +0.1194 | 2.22 | 51.0 | 0.68 |
| stopm1 floor=0 TRAIN | 1290 | 24.3 | **+0.0245** | +0.0558 | 0.98 | 46.4 | 0.53 |
| stopm1 floor=0 VAL | 559 | 25.4 | **+0.0893** | +0.1194 | 2.22 | 51.0 | 0.68 |

## universe Q — 32,922 F6 rows, weeks TRAIN=51 VAL=22 TEST=14

| cell | n | tr/wk | net R | gross R | t | WR | green |
|---|---:|---:|---:|---:|---:|---:|---:|
| hold floor=1 TRAIN | 1034 | 20.3 | **+0.0896** | +0.1182 | 2.03 | 45.1 | 0.55 |
| &nbsp;&nbsp;expected | 1034 | | +0.0896 | | 2.03 | | |  dn=+0
| hold floor=1 VAL | 512 | 23.3 | **+0.2088** | +0.2398 | 2.80 | 46.7 | 0.77 |
| &nbsp;&nbsp;expected | 512 | | +0.2088 | | 2.80 | | |  dn=+0
| hold floor=0 TRAIN | 1846 | 36.2 | **-0.0871** | +0.0134 | -1.58 | 32.1 | 0.33 |
| hold floor=0 VAL | 784 | 35.6 | **+0.0389** | +0.1416 | 0.50 | 35.8 | 0.41 |
| stopm1 floor=1 TRAIN | 1084 | 21.3 | **+0.0510** | +0.0775 | 1.77 | 47.9 | 0.53 |
| stopm1 floor=1 VAL | 539 | 24.5 | **+0.0803** | +0.1078 | 1.87 | 49.5 | 0.73 |
| stopm1 floor=0 TRAIN | 1965 | 38.5 | **-0.1315** | -0.0434 | -4.90 | 39.1 | 0.29 |
| stopm1 floor=0 VAL | 825 | 37.5 | **+0.0288** | +0.1174 | 0.69 | 45.3 | 0.55 |

### the floor twin — each half booked on its own (E §8.4)

| half | split | n | net R | t | stop% |
|---|---|---:|---:|---:|---:|
| passed floor | TRAIN | 1034 | **+0.0896** | 2.03 | 28.2 |
| passed floor | VAL | 512 | **+0.2088** | 2.80 | 30.1 |
| below floor | TRAIN | 1807 | **-0.1144** | -2.05 | 58.9 |
| below floor | VAL | 777 | **-0.0457** | -0.61 | 55.9 |
