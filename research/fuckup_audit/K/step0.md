# Stage K step 0 — price scale + asset-class map coverage

Generated 2026-09-17T20:52:58+00:00 by `research/fuckup_audit/K/step0_pricescale.py`.

## Price-scale check (PLAN.md §1 / CLAUDE.md check 3)

200 random (symbol, day) keys present in BOTH the Databento daily panel and `data/cache.db::daily_bars` (Alpaca, the prices the live account sees). `diff = panel / alpaca - 1`, in percent.

| field | within 0.01% | within 0.1% | within 1% | median abs | p95 abs | max abs |
|---|---:|---:|---:|---:|---:|---:|
| open | 97.5% | 97.5% | 98.5% | 0.0000% | 0.0000% | 100.000% |
| high | 97.5% | 97.5% | 98.5% | 0.0000% | 0.0000% | 100.000% |
| low | 97.5% | 97.5% | 98.5% | 0.0000% | 0.0000% | 100.000% |
| close | 97.5% | 97.5% | 98.5% | 0.0000% | 0.0000% | 100.000% |
| vol | 83.6% | 92.8% | 99.0% | 0.0000% | 0.1326% | 1.970% |

Keys disagreeing by more than 1% on the CLOSE: 3 of 200 (1.5%).

| symbol | day | panel close | alpaca close | diff % |
|---|---|---:|---:|---:|
| ODVWZ | 2025-10-07 | 0.0000 | 0.2900 | -100.00% |
| LCFYW | 2025-07-15 | 0.0000 | 11.5000 | -100.00% |
| TVACU | 2026-03-27 | 0.0000 | 10.5401 | -100.00% |

Rows: `K/pricescale.csv`.

## Asset-class map coverage (PREREG "common stock only")

`data/research/orb_asset_class_map_20260711.csv`: 33,246 symbols (26,552 stock, 6,136 wrapper, 558 other).

The map is a **2026-07-11 dump of live Alpaca assets**. A symbol that delisted in 2025 cannot be in it, so *requiring* map membership is itself a survivorship filter. Coverage is measured on the liquid slice this stage trades (20-day median dollar volume >= $10M, close >= $5).

| split | symbol-days | distinct symbols | stock | wrapper | other-in-map | not in map |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 697,706 | 3,919 | 73.2% | 23.4% | 0.1% | 3.2% |
| VAL | 329,266 | 3,904 | 73.2% | 25.9% | 0.1% | 0.8% |
| TEST | 227,657 | 3,852 | 74.0% | 25.7% | 0.0% | 0.2% |
| ALL | 1,254,629 | 4,610 | 73.3% | 24.5% | 0.1% | 2.1% |

Symbols in the liquid slice that are NOT in the map: 255 (25,794 symbol-days). Of those, 218 have no panel bar after 2026-07-11 (i.e. they are gone by the dump date) — this is the survivorship channel the rule opens. `K/classmap_missing.csv` lists them.

**Decision taken here and carried into the scoring (pre-registered before any P&L was computed):** the PREREG rule (`asset_class == stock`) is the PRIMARY universe; a secondary universe `stock or not-in-map` (drop only POSITIVELY identified wrappers) is carried alongside every cell so the survivorship cost of the primary rule is visible rather than assumed away. Neither is tuned on a result.

