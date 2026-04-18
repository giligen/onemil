# Tier redesign post-hoc sim

Read-only: apply alt tier tables to cached trades (conv_mult >= 1.4).

PnL scaling mirrors batch_backtest.py Stage-2 exactly (divide out MACD when tier matches; cap at 3.0x combined).

## Baseline (current)

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+31,905 | $-15,545 | $+0 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+14,710 | $-14,365 | $+0 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+7,589 | $-12,919 | $+0 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+15,035 | $-2,053 | $+0 |

## A: add T3 <$5 @1.5x

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+32,718 | $-15,898 | $+813 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+15,118 | $-15,376 | $+408 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+8,324 | $-12,638 | $+735 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+15,228 | $-2,053 | $+193 |

## B: demote T1, <$5 @2.0x

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+33,940 | $-16,010 | $+2,035 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+13,504 | $-14,764 | $-1,206 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+7,884 | $-11,639 | $+295 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+15,858 | $-2,053 | $+823 |

## C: broad T3 <$5 any-vol @1.5

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+31,088 | $-16,206 | $-817 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+15,720 | $-14,539 | $+1,009 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+7,252 | $-12,251 | $-337 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+15,625 | $-2,053 | $+591 |

## D: B + $10-15 <500K orphan rescue

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+34,568 | $-16,040 | $+2,662 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+12,371 | $-14,649 | $-2,339 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+8,596 | $-11,525 | $+1,007 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+15,441 | $-2,053 | $+407 |

## E: A + rescue $10-15<500K @1.5

| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+33,346 | $-15,928 | $+1,441 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+13,985 | $-15,261 | $-725 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+9,036 | $-12,524 | $+1,447 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+14,812 | $-2,053 | $-223 |

## Bucket audit (current state; all splits combined)

| Price | Vol | n | WR | Total PnL |
|---|---|---:|---:|---:|
| $10-15 | 500K-5M | 77 | 44.2% | $+2,809 |
| $10-15 | <500K | 92 | 48.9% | $+14,867 |
| $15-23 | 500K-5M | 69 | 37.7% | $+7,688 |
| $15-23 | 5M+ | 1 | 100.0% | $+1,197 |
| $15-23 | <500K | 90 | 36.7% | $-3,285 |
| $23+ | 500K-5M | 4 | 0.0% | $-3,289 |
| $23+ | <500K | 8 | 12.5% | $-2,644 |
| $5-10 | 500K-5M | 123 | 43.1% | $+2,981 |
| $5-10 | 5M+ | 7 | 28.6% | $-1,296 |
| $5-10 | <500K | 172 | 40.1% | $+6,191 |
| <$5 | 500K-5M | 98 | 50.0% | $+38,771 |
| <$5 | 5M+ | 30 | 46.7% | $+2,488 |
| <$5 | <500K | 172 | 36.6% | $+2,003 |

## Ranking by HOLDOUT gain

Baseline HOLDOUT total: $+22,624

| Variant | HOLDOUT total | Δ vs base |
|---|---:|---:|
| D: B + $10-15 <500K orphan rescue | $+24,037 | $+1,413 |
| E: A + rescue $10-15<500K @1.5 | $+23,848 | $+1,224 |
| B: demote T1, <$5 @2.0x | $+23,742 | $+1,118 |
| A: add T3 <$5 @1.5x | $+23,552 | $+928 |
| C: broad T3 <$5 any-vol @1.5 | $+22,877 | $+254 |
