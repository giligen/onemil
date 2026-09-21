# Independent 1328: Union of Pools

## Task A — Seed Definition Verification
| Metric | Count |
|--------|-------|
| Wide CSV seed (criteria met) | 12882 |
| COMB features (input) | 12882 |
| Only in wide | 0 |
| Only in COMB | 0 |
| In both | 12882 |

## Task B — Union Book Results

### Production vs Union by Split

| Split | Book | n | Mean R | Total $ | Fills/Week | Weekly Max DD (R) |
|-------|------|---|--------|---------|------------|------------------|
| TRAIN | Production | 85 | 0.206 | $6561 | 2.50 | -1.407 |
| TRAIN | Union | 162 | 0.151 | $9190 | 3.38 | -2.891 |
| VAL | Production | 42 | 0.406 | $6398 | 2.33 | -1.319 |
| VAL | Union | 110 | 0.247 | $10207 | 5.24 | -1.155 |

### Added Rows (Addon Picks Not in Production)

| Split | n | Mean R | Total $ | Mean R (ex-top 5%) |
|-------|---|--------|---------|-------------------|
| TRAIN | 77 | 0.091 | $2629 | 0.020 |
| VAL | 68 | 0.149 | $3809 | 0.084 |

## Summary
Union book raised total $ on both splits. TRAIN: $+2629, VAL: $+3809.
