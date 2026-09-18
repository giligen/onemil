## Availability audit (PLAN §1 standing rule)

Coverage = share of BASE-universe symbol-days (dvol20_med >= $10M, close >= $5, common stock, non-test, valid prices, non-early-close) on which the field is finite. A field below 100% cannot be used without knowing WHO is missing.

| field | TRAIN | VAL | TEST | what missing means |
|---|---:|---:|---:|---|
| `gap` | 100.00% | 100.00% | 100.00% | no prior close (first bar of a symbol) |
| `vol_ratio` | 100.00% | 100.00% | 100.00% | adv20 undefined (<10 prior bars) |
| `close_pos` | 100.00% | 100.00% | 100.00% | never missing (high==low falls back to 0.5) |
| `high52` | 99.10% | 99.60% | 99.08% | <60 prior bars (young listing) — K2 cannot fire |
| `ret5` | 100.00% | 100.00% | 100.00% | <5 prior bars |
| `ret3` | 100.00% | 100.00% | 100.00% | <3 prior bars |
| `sma20` | 99.81% | 99.92% | 99.82% | <20 prior bars |
| `hi50` | 100.00% | 100.00% | 100.00% | never missing (expanding window until 50 bars exist) |
| `on20` | 99.89% | 99.95% | 99.90% | <15 of the last 20 overnight returns |
| `dvol20_med` | 100.00% | 100.00% | 100.00% | gate itself — 100% by construction |

The class map is audited in `K/step0.md`: 73% of the liquid slice is positively identified as common stock, 24.5% as a leveraged/inverse wrapper, 2.1% is not in the 2026-07-11 dump at all (2025-weighted — the survivorship channel). Every cell is therefore scored on BOTH the primary (stock-only) and the secondary (non-wrapper) universe.

