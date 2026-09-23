# TIMESTOP.md — cells 1,403–1,405 (PREREG_TIMESTOP.md). Primary = SLOTTED book

## Time to first close above the high (breakers only), TRAIN+VAL

breakers 4903 of 6275 (78%); within 15 min: 56%, within 30 min: 70%, within 60 min: 82%, within 120 min: 91%

- P(eventual break | no break by 15 min) = 61% (n=3547)
- P(eventual break | no break by 30 min) = 52% (n=2840)
- P(eventual break | no break by 60 min) = 39% (n=2258)

## Cells

| cell | N | TRAIN slotted | VAL slotted | TRAIN all | VAL all | halves (slotted) | ex-top-5 % TR/VAL | fills/wk VAL | D1 VAL | exit mix VAL | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1403 | 15 | -0.055 (-2.04) | -0.052 (-1.24) | -0.104 | +0.018 | -0.046 / -0.062 | -0.159 / -0.156 | 44.3 | -0.052 | {'nobreak': 0.46, 'stop': 0.21, 'target': 0.2, 'eod': 0.12} | False |
| 1404 | 30 | -0.069 (-2.28) | +0.015 (+0.34) | -0.102 | +0.058 | -0.105 / -0.035 | -0.173 / -0.087 | 40.1 | +0.003 | {'stop': 0.31, 'nobreak': 0.28, 'target': 0.25, 'eod': 0.16} | False |
| 1405 | 60 | -0.049 (-1.46) | +0.048 (+0.95) | -0.085 | +0.088 | -0.085 / -0.017 | -0.153 / -0.052 | 35.7 | +0.031 | {'stop': 0.38, 'target': 0.28, 'eod': 0.19, 'nobreak': 0.14} | False |
