# Stage K diagnostic D5 — raw family vs slot limit vs declared ranking key

Diagnostic only; cannot promote anything. Counted in the report's cell count.

| family | book | split | n | gross bps | net bps | net bps (auction) | t |
|---|---|---|---:|---:|---:|---:|---:|
| K1 h5 | all (no book) | TRAIN | 328 | -57.5 | -105.1 | -67.5 | -1.47 |
| K1 h5 | all (no book) | VAL | 188 | -91.9 | -139.5 | -101.9 | -1.75 |
| K1 h5 | declared key, 10 slots | TRAIN | 261 | -75.4 | -122.6 | -85.4 | -1.43 |
| K1 h5 | declared key, 10 slots | VAL | 133 | +16.6 | -30.7 | +6.6 | -0.29 |
| K1 h5 | random key, 10 slots | TRAIN | 260 | -54.0 | -101.2 | -64.0 | -1.17 |
| K1 h5 | random key, 10 slots | VAL | 138 | -37.2 | -83.6 | -47.2 | -0.83 |
| K1 h5 | reversed key, 10 slots | TRAIN | 258 | -66.4 | -113.5 | -76.4 | -1.31 |
| K1 h5 | reversed key, 10 slots | VAL | 138 | -114.2 | -161.6 | -124.2 | -1.89 |
| K2 h10 | all (no book) | TRAIN | 1,647 | +69.5 | +23.8 | +59.5 | +0.79 |
| K2 h10 | all (no book) | VAL | 1,174 | +108.2 | +64.1 | +98.2 | +1.44 |
| K2 h10 | declared key, 10 slots | TRAIN | 237 | -30.8 | -78.8 | -40.8 | -1.16 |
| K2 h10 | declared key, 10 slots | VAL | 127 | +67.3 | +19.2 | +57.3 | +0.12 |
| K2 h10 | random key, 10 slots | TRAIN | 228 | -5.0 | -53.9 | -15.0 | -0.68 |
| K2 h10 | random key, 10 slots | VAL | 123 | +60.0 | +13.6 | +50.0 | +0.12 |
| K2 h10 | reversed key, 10 slots | TRAIN | 215 | +46.8 | +0.6 | +36.8 | +0.01 |
| K2 h10 | reversed key, 10 slots | VAL | 115 | +136.1 | +92.0 | +126.1 | +0.80 |
| K3 h3 | all (no book) | TRAIN | 15,342 | -16.3 | -61.5 | -26.3 | -10.60 |
| K3 h3 | all (no book) | VAL | 7,264 | +15.6 | -29.0 | +5.6 | -3.39 |
| K3 h3 | declared key, 10 slots | TRAIN | 800 | +62.9 | +17.4 | +52.9 | +0.51 |
| K3 h3 | declared key, 10 slots | VAL | 340 | +17.5 | -29.0 | +7.5 | -0.52 |
| K3 h3 | random key, 10 slots | TRAIN | 800 | +81.0 | +36.2 | +71.0 | +1.37 |
| K3 h3 | random key, 10 slots | VAL | 340 | -29.2 | -74.6 | -39.2 | -1.75 |
| K3 h3 | reversed key, 10 slots | TRAIN | 800 | +69.4 | +22.4 | +59.4 | +0.93 |
| K3 h3 | reversed key, 10 slots | VAL | 340 | +3.7 | -39.7 | -6.3 | -1.11 |
| K4 h1 | all (no book) | TRAIN | 49,243 | -7.4 | -51.4 | -17.4 | -29.15 |
| K4 h1 | all (no book) | VAL | 24,017 | +17.9 | -24.3 | +7.9 | -9.41 |
| K4 h1 | declared key, 10 slots | TRAIN | 2,320 | -32.8 | -80.1 | -42.8 | -5.41 |
| K4 h1 | declared key, 10 slots | VAL | 1,020 | -15.0 | -62.4 | -25.0 | -3.00 |
| K4 h1 | random key, 10 slots | TRAIN | 2,320 | -11.7 | -55.2 | -21.7 | -7.00 |
| K4 h1 | random key, 10 slots | VAL | 1,020 | -5.3 | -48.0 | -15.3 | -4.14 |
| K4 h1 | reversed key, 10 slots | TRAIN | 2,320 | -5.7 | -49.3 | -15.7 | -8.16 |
| K4 h1 | reversed key, 10 slots | VAL | 1,020 | +11.9 | -30.1 | +1.9 | -3.16 |
| K5 h5 | all (no book) | TRAIN | 3,903 | -5.0 | -49.9 | -15.0 | -3.29 |
| K5 h5 | all (no book) | VAL | 2,509 | +84.2 | +39.8 | +74.2 | +1.82 |
| K5 h5 | declared key, 10 slots | TRAIN | 576 | -46.7 | -92.1 | -56.7 | -2.03 |
| K5 h5 | declared key, 10 slots | VAL | 267 | +1.0 | -44.9 | -9.0 | -0.59 |
| K5 h5 | random key, 10 slots | TRAIN | 540 | +7.3 | -38.6 | -2.7 | -1.00 |
| K5 h5 | random key, 10 slots | VAL | 247 | +34.8 | -10.0 | +24.8 | -0.16 |
| K5 h5 | reversed key, 10 slots | TRAIN | 514 | +55.2 | +9.4 | +45.2 | +0.25 |
| K5 h5 | reversed key, 10 slots | VAL | 235 | -39.2 | -85.1 | -49.2 | -1.23 |

The declared ranking key is the pre-registered tie-break; where "declared" is far below "random" on BOTH splits, the key itself is adverse — the strongest signal by that measure is the worst trade, and the book is spending its ten slots on it.
