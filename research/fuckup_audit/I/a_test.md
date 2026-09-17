# Stage I Part A — TEST (read once, per FREEZE.md; descriptive, no selection)

| stack | exit | split | n | tr/wk | mean net R | gross | t | WR% | wkR | weeks green | worst wk | MDD | months green | ex-top5% | cap+3R | family mix |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| S0 | hold | TEST | 376 | 26.9 | +0.104 | +0.138 | 0.74 | 41.5 | +2.79 | 0.43 | -13.3 | -26.9 | 2/4 | -0.223 | -0.061 | F6:376 |
| S1 | hold | TEST | 386 | 27.6 | +0.085 | +0.120 | 0.62 | 41.7 | +2.36 | 0.50 | -14.3 | -25.5 | 2/4 | -0.238 | -0.070 | F6:362 F14:24 |
| S2 | hold | TEST | 393 | 28.1 | -0.049 | -0.013 | -0.75 | 41.7 | -1.37 | 0.36 | -11.9 | -32.5 | 1/4 | -0.258 | -0.092 | F6:336 F11:33 F14:24 |
| S3 | hold | TEST | 390 | 27.9 | -0.089 | -0.053 | -1.41 | 39.2 | -2.47 | 0.29 | -11.4 | -39.7 | 1/4 | -0.293 | -0.127 | F6:271 F8N30:85 F11:29 F14:5 |

Reference (already public, `H/F6/f6_pdr_book.md` exit hold | PDR>=8, TEST): n 376, +0.104 R, t 0.74, 26.9 tr/wk, 43% weeks green, worst week -13.3, MDD -26.9.

## Tail tests on the FROZEN cell (S0 = F6 alone, hold)

| split | n | mean net R | top-1% removed | top-5% removed | winners capped at +3R | share of total R in the top 5% of trades |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 1122 | +0.078 | -0.016 | -0.161 | -0.005 | 296% |
| VAL | 519 | +0.251 | +0.123 | -0.036 | +0.114 | 114% |
| TEST | 376 | +0.104 | -0.069 | -0.223 | -0.061 | 305% |

Frozen per-trade CSV: `research/fuckup_audit/I/frozen_stack_trades.csv` (2017 booked trades, all three splits).

monthly net R (frozen cell): 2025-01 +3.4 (83) | 2025-02 +10.1 (79) | 2025-03 +11.1 (93) | 2025-04 -0.6 (105) | 2025-05 +1.5 (97) | 2025-06 -5.3 (92) | 2025-07 +25.9 (94) | 2025-08 +12.7 (89) | 2025-09 +7.1 (86) | 2025-10 +4.8 (113) | 2025-11 +16.0 (91) | 2025-12 +0.7 (100) | 2026-01 +19.1 (102) | 2026-02 +3.1 (94) | 2026-03 +35.5 (105) | 2026-04 +50.6 (108) | 2026-05 +21.8 (110) | 2026-06 -28.9 (125) | 2026-07 -4.7 (121) | 2026-08 +69.3 (111) | 2026-09 +3.4 (19)
months green 17/21

