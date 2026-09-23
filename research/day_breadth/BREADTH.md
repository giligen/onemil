# BREADTH.md — find the winners (PREREG.md, cells 1,406–1,411)

Kept = measure ≥ TRAIN-H1 median. Stats: trade-weighted mean, t = day-clustered trade-weighted. SLOTTED = first 12/day, 4 concurrent, applied after the filter.

## Book: base (6247 trades, TRAIN 3958 / VAL 2289)

| cell | measure | cut | H2 kept (t) | H2 dropped | VAL kept (t) | VAL dropped | H2 / VAL slotted | fills/wk | terciles H1 / H2 / VAL | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1406 | NG | 53.000 | -0.044 (-0.47) | -0.013 | +0.165 (+1.29) | +0.016 | -0.029 / +0.006 | 16.0 | [-0.044, 0.116, -0.338] / [0.091, -0.044, -0.099] / [-0.074, 0.027, 0.258] | False |
| 1407 | BR | 0.535 | +0.000 (+0.00) | -0.052 | +0.171 (+1.29) | +0.027 | -0.090 / +0.029 | 19.1 | [-0.231, -0.119, 0.071] / [-0.028, -0.086, 0.084] / [-0.043, 0.075, 0.245] | False |
| 1408 | HH | 0.297 | -0.031 (-0.32) | -0.031 | +0.164 (+1.23) | +0.037 | +0.001 / +0.105 | 20.4 | [-0.101, -0.152, -0.024] / [0.001, -0.041, -0.062] / [0.004, 0.019, 0.246] | False |

### Winner-day anatomy, TRAIN only (base): top 10 % of days = -191% of TRAIN R

| group | days | mean day R | NG | BR 10:30 | HH 10:30 | SPY 09:30→10:30 % | weekday mix |
|---|---|---|---|---|---|---|---|
| top 10 % days | 25 | +18.24 | 57.6 | 0.565 | 0.425 | +0.135 | {'Wednesday': 0.24, 'Thursday': 0.2, 'Monday': 0.2, 'Friday': 0.2, 'Tuesday': 0.16} |
| other days | 225 | -3.09 | 48.9 | 0.437 | 0.297 | +0.001 | {'Tuesday': 0.21, 'Wednesday': 0.2, 'Friday': 0.2, 'Thursday': 0.19, 'Monday': 0.19} |

## Book: HOD (12010 trades, TRAIN 7315 / VAL 4695)

| cell | measure | cut | H2 kept (t) | H2 dropped | VAL kept (t) | VAL dropped | H2 / VAL slotted | fills/wk | terciles H1 / H2 / VAL | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1409 | NG | 44.000 | -0.276 (-4.43) | -0.226 | -0.239 (-5.12) | -0.458 | -0.336 / -0.219 | 30.4 | [-0.275, -0.465, 0.128] / [-0.213, -0.314, -0.238] / [-0.489, -0.242, -0.28] | False |
| 1410 | BR | 0.526 | -0.211 (-2.89) | -0.276 | -0.329 (-7.24) | -0.283 | -0.312 / -0.307 | 28.5 | [-0.45, -0.33, 0.185] / [-0.242, -0.271, -0.18] / [-0.289, -0.274, -0.507] | False |
| 1411 | HH | 0.661 | -0.249 (-3.53) | -0.253 | -0.269 (-3.81) | -0.328 | -0.322 / -0.264 | 26.7 | [-0.527, -0.061, -0.008] / [-0.239, -0.198, -0.311] / [-0.324, -0.318, -0.273] | False |

### Winner-day anatomy, TRAIN only (HOD): top 10 % of days = -40% of TRAIN R

| group | days | mean day R | NG | BR 10:30 | HH 10:30 | SPY 09:30→10:30 % | weekday mix |
|---|---|---|---|---|---|---|---|
| top 10 % days | 25 | +26.08 | 80.1 | 0.508 | 0.379 | +0.191 | {'Tuesday': 0.28, 'Wednesday': 0.24, 'Thursday': 0.24, 'Friday': 0.16, 'Monday': 0.08} |
| other days | 217 | -10.46 | 46.9 | 0.442 | 0.3 | -0.006 | {'Monday': 0.21, 'Friday': 0.2, 'Tuesday': 0.2, 'Wednesday': 0.2, 'Thursday': 0.18} |

