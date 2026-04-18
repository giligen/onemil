# Audit-fix post-hoc simulator (drops rules 3/5/7)

Note: this is a first-order approximation. Re-scales pnl linearly with the new conviction_mult. Tier scaling + MACD divide-out are NOT re-applied, so absolute numbers are slightly off from a full rebuild — but the DELTA trend is reliable.

## Baseline (audit_fix OFF, min_threshold=1.4)

| Split | n | WR | Total PnL |
|---|---:|---:|---:|
| TRAIN (2025 Jan-Jul) | 356 | 42.7% | $+28,962 |
| VAL (2025 Aug-Dec) | 276 | 36.6% | $+14,328 |
| HOLDOUT Q1 2026 | 262 | 40.8% | $+8,213 |
| HOLDOUT Apr 1-17 | 49 | 61.2% | $+16,977 |

## Audit-fix ON, threshold sweep

| threshold | Split | n | WR | Total PnL | Δ vs base |
|---|---|---:|---:|---:|---:|
| 0.8 | TRAIN (2025 Jan-Jul) | 430 | 41.9% | $+22,324 | $-6,638 |
| 0.8 | VAL (2025 Aug-Dec) | 343 | 35.9% | $+16,392 | $+2,064 |
| 0.8 | HOLDOUT Q1 2026 | 311 | 38.9% | $+7,062 | $-1,151 |
| 0.8 | HOLDOUT Apr 1-17 | 53 | 58.5% | $+12,355 | $-4,621 |
| 0.9 | TRAIN (2025 Jan-Jul) | 388 | 43.6% | $+22,225 | $-6,738 |
| 0.9 | VAL (2025 Aug-Dec) | 274 | 37.6% | $+15,676 | $+1,347 |
| 0.9 | HOLDOUT Q1 2026 | 281 | 42.0% | $+7,890 | $-324 |
| 0.9 | HOLDOUT Apr 1-17 | 53 | 58.5% | $+12,355 | $-4,621 |
| 1.0 | TRAIN (2025 Jan-Jul) | 369 | 43.4% | $+20,824 | $-8,139 |
| 1.0 | VAL (2025 Aug-Dec) | 258 | 37.6% | $+17,074 | $+2,746 |
| 1.0 | HOLDOUT Q1 2026 | 268 | 42.2% | $+7,017 | $-1,196 |
| 1.0 | HOLDOUT Apr 1-17 | 51 | 58.8% | $+12,262 | $-4,715 |
| 1.1 | TRAIN (2025 Jan-Jul) | 297 | 43.4% | $+19,082 | $-9,880 |
| 1.1 | VAL (2025 Aug-Dec) | 195 | 37.9% | $+19,682 | $+5,354 |
| 1.1 | HOLDOUT Q1 2026 | 204 | 42.2% | $+6,223 | $-1,990 |
| 1.1 | HOLDOUT Apr 1-17 | 43 | 60.5% | $+12,235 | $-4,742 |
| 1.2 | TRAIN (2025 Jan-Jul) | 275 | 44.0% | $+19,923 | $-9,039 |
| 1.2 | VAL (2025 Aug-Dec) | 159 | 40.9% | $+15,291 | $+963 |
| 1.2 | HOLDOUT Q1 2026 | 196 | 43.4% | $+7,382 | $-831 |
| 1.2 | HOLDOUT Apr 1-17 | 43 | 60.5% | $+12,235 | $-4,742 |
| 1.3 | TRAIN (2025 Jan-Jul) | 255 | 42.7% | $+15,843 | $-13,119 |
| 1.3 | VAL (2025 Aug-Dec) | 140 | 40.7% | $+13,544 | $-784 |
| 1.3 | HOLDOUT Q1 2026 | 170 | 41.8% | $-149 | $-8,362 |
| 1.3 | HOLDOUT Apr 1-17 | 39 | 61.5% | $+12,574 | $-4,403 |
| 1.4 | TRAIN (2025 Jan-Jul) | 145 | 40.7% | $+10,135 | $-18,828 |
| 1.4 | VAL (2025 Aug-Dec) | 79 | 46.8% | $+15,293 | $+965 |
| 1.4 | HOLDOUT Q1 2026 | 97 | 46.4% | $+1,364 | $-6,849 |
| 1.4 | HOLDOUT Apr 1-17 | 19 | 57.9% | $+6,910 | $-10,067 |

## TRAIN+VAL sweet spot (leakage-clean threshold pick)

| threshold | TRAIN PnL | VAL PnL | T+V sum | Δ vs base (T+V) |
|---|---:|---:|---:|---:|
| 0.8 | $+22,324 | $+16,392 | $+38,717 | $-4,574 |
| 0.9 | $+22,225 | $+15,676 | $+37,900 | $-5,390 |
| 1.0 | $+20,824 | $+17,074 | $+37,898 | $-5,392 |
| 1.1 | $+19,082 | $+19,682 | $+38,764 | $-4,526 |
| 1.2 | $+19,923 | $+15,291 | $+35,215 | $-8,076 |
| 1.3 | $+15,843 | $+13,544 | $+29,387 | $-13,903 |
| 1.4 | $+10,135 | $+15,293 | $+25,428 | $-17,863 |
| 1.5 | $+11,242 | $+12,678 | $+23,920 | $-19,370 |

**Best TRAIN+VAL threshold (audit_fix ON): 1.1  → Δ $-4,526**

## One-shot HOLDOUT validation (audit_fix ON @ T=1.1)

HOLDOUT baseline total: $+25,190
HOLDOUT audit_fix total: $+18,458
HOLDOUT Δ: $-6,732
