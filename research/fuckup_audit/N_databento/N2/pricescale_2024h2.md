# N2 step 1c — price-scale check, 2024H2 Databento daily vs Alpaca `daily_bars`

Generated 2026-09-18T05:50:50+00:00 by `research/fuckup_audit/N_databento/N2/pricescale_2024h2.py`.

200 keys matched out of 11,331 sampled panel rows (cache.db holds 27,079 daily rows in 2024H2 — the bull-flag universe only, so most panel rows have no Alpaca counterpart). `diff = databento / alpaca - 1`, percent.

| field | within 0.01% | within 0.5% | within 1% | median abs | p95 abs | max abs | **off by > 0.5%** |
|---|---:|---:|---:|---:|---:|---:|---:|
| open | 99.0% | 99.5% | 99.5% | 0.0000% | 0.0000% | 100.000% | **0.5%** |
| high | 99.0% | 99.5% | 99.5% | 0.0000% | 0.0000% | 100.000% | **0.5%** |
| low | 99.0% | 99.5% | 99.5% | 0.0000% | 0.0000% | 100.000% | **0.5%** |
| close | 99.0% | 99.5% | 99.5% | 0.0000% | 0.0000% | 100.000% | **0.5%** |
| vol | 99.0% | 100.0% | 100.0% | 0.0000% | 0.0000% | 0.150% | **0.0%** |

Keys off by more than 0.5% on ANY of OHLC: **1 of 200 (0.5%)**.

| symbol | day | db close | alpaca close | diff % |
|---|---|---:|---:|---:|
| ALDF | 2024-12-31 | 0.0000 | 9.9300 | -100.000% |

Rows: `pricescale_2024h2.csv`.
