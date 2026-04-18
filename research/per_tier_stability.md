# S2 variants — monthly stability + max drawdown

Per-month P&L across 2025 full + Q1 2026. Shows whether S2 lift is stable across months or concentrated in outliers.

## Monthly P&L (all tiers, all splits)

| Month | Baseline | S2-cons | S2-mid | S2-max | S2-mid Δ |
|---|---:|---:|---:|---:|---:|
| 2025-01 | $  +4,610 | $  +2,795 | $  +1,981 | $  +1,644 | **$ -2,629** |
| 2025-02 | $ +16,717 | $ +16,008 | $ +16,959 | $ +17,099 | **$   +242** |
| 2025-03 | $  +4,224 | $  +9,032 | $ +10,316 | $ +11,411 | **$ +6,092** |
| 2025-04 | $  +4,452 | $  +8,003 | $  +7,846 | $  +7,739 | **$ +3,394** |
| 2025-05 | $  +3,357 | $  +6,425 | $  +6,229 | $  +6,033 | **$ +2,872** |
| 2025-06 | $  +1,900 | $    -676 | $    -788 | $    -899 | **$ -2,688** |
| 2025-07 | $    +891 | $  +2,154 | $  +3,021 | $  +3,779 | **$ +2,129** |
| 2025-08 | $  +1,647 | $  +1,332 | $  +1,273 | $  +1,213 | **$   -374** |
| 2025-09 | $  +8,324 | $ +10,008 | $ +10,297 | $ +11,428 | **$ +1,973** |
| 2025-10 | $    +623 | $  -1,133 | $  -1,274 | $  -1,382 | **$ -1,897** |
| 2025-11 | $  +3,080 | $  +7,028 | $  +8,256 | $  +9,451 | **$ +5,176** |
| 2025-12 | $  +1,755 | $  +3,727 | $  +3,727 | $  +3,727 | **$ +1,972** |
| 2026-01 | $  -3,790 | $  -3,143 | $  -3,002 | $  -2,550 | **$   +788** |
| 2026-02 | $  -3,428 | $    -723 | $    -523 | $    -322 | **$ +2,906** |
| 2026-03 | $ +16,575 | $ +25,523 | $ +25,759 | $ +25,988 | **$ +9,185** |
| **TOTAL** | $+60,937 | $+86,360 | $+90,077 | $+94,360 | |

## Max drawdown per tier per variant

| Variant | A-tier MDD | E-tier MDD | Combined MDD |
|---|---:|---:|---:|
| Baseline | $-10,696 | $-20,000 | $-13,882 |
| S2-cons | $-10,696 | $-11,801 | $-10,010 |
| S2-mid | $-10,495 | $-11,801 | $-10,010 |
| S2-max | $-10,295 | $-11,801 | $-10,010 |

## Win rate by variant (all trades)

| Variant | n | WR |
|---|---:|---:|
| Baseline | 625 | 41.9% |
| S2-cons | 406 | 44.6% |
| S2-mid | 412 | 44.2% |
| S2-max | 412 | 44.2% |

## Sample-size sanity check

Per-tier × MACD bucket trade counts (baseline, conv>=1.4):

| Tier | MACD 1.0 | MACD 1.5 | total |
|---|---:|---:|---:|
| A | 85 | 92 | 177 |
| E | 219 | 223 | 442 |
| edge | 7 | 5 | 12 |

### Critical sample size: E-tier MACD 1.5 bucket (the +$16K goldmine)

- **VAL (2025 Aug-Dec)**: 10 trades, 40.0% WR, mean R +0.393, PnL $+2,144
- **HOQ1 (2026 Q1)**: 42 trades, 59.5% WR, mean R +0.449, PnL $+13,890

### Critical sample size: E-tier MACD 1.0 bucket (the -$14K landmine)

- **TRAIN (2025 Jan-Jul)**: 79 trades, 35.4% WR, mean R -0.004, PnL $-7,590
- **VAL (2025 Aug-Dec)**: 77 trades, 36.4% WR, mean R -0.021, PnL $-4,103
- **HOQ1 (2026 Q1)**: 63 trades, 39.7% WR, mean R -0.160, PnL $-3,040
