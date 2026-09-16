# Stage D1 — premarket dollar volume per candidate symbol-day

Generated 2026-09-16T21:11:48+00:00 by `research/fuckup_audit/D/pm_backfill.py`.

**What this is.** One row per (day, symbol) of the Stage-D candidate table (`D/table.csv`, F6/F8, entry_m >= 600): premarket 04:00-09:29 ET aggregates from the 1-min SIP tape. `pm_dollar_vol = sum(close x volume)` (the live helper `trading/orb_pm_mult.compute_pm_dollar_vol` prefers bar vwap and falls back to close; `bars_sip.db` carries no vwap column, so CLOSE is used for BOTH provenances here so they are comparable). 09:30 ET boundaries are built per day with `zoneinfo`, so DST is exact.

## Coverage

| src | meaning | keys | share |
|---|---|---:|---:|
| `sip_store` | already in research/bf_zero/bars_sip.db (no API call) | 21,179 | 34.2% |
| `alpaca` | fetched from Alpaca SIP, >= 1 premarket bar | 32,252 | 52.1% |
| `none` | Alpaca returned nothing — genuine no-premarket-trades day | 8,471 | 13.7% |
| **total** | rows in `pm` | **61,902** | |

Unresolved fetch failures still in `fetch_log(status='error')`: **0** (these have NO `pm` row; re-running the script retries them).

## pm_dollar_vol distribution (keys with >= 1 premarket bar)

n = 53,431

| quantile | pm_dollar_vol ($) |
|---|---:|
| p0 | 242 |
| p10 | 7,439 |
| p25 | 42,602 |
| p50 | 342,686 |
| p75 | 2,587,094 |
| p90 | 14,858,626 |
| p95 | 38,628,351 |
| p99 | 232,731,141 |
| p100 | 5,422,997,651 |

**Share above the ORB cut ($5,816,688): 9,135 / 53,431 = 17.10%** of keys with premarket trades (14.76% of all 61,902 keys).

| src | n with pm bars | above cut | share | median | p90 |
|---|---:|---:|---:|---:|---:|
| `sip_store` | 21,179 | 3,253 | 15.36% | 248,557 | 13,024,217 |
| `alpaca` | 32,252 | 5,882 | 18.24% | 428,743 | 15,778,045 |

## Spot check — 20 store-served keys re-fetched fresh from Alpaca

| symbol | day | n bars (store) | n bars (alpaca) | verdict |
|---|---|---:|---:|---|
| MUZ | 2026-08-25 | 257 | 257 | BAR-EXACT |
| WGMI | 2025-01-02 | 10 | 10 | BAR-EXACT |
| ROKU | 2025-03-19 | 68 | 68 | BAR-EXACT |
| OTIS | 2025-07-23 | 74 | 74 | BAR-EXACT |
| ASTS | 2025-11-04 | 179 | 179 | BAR-EXACT |
| ASTX | 2025-10-13 | 83 | 83 | BAR-EXACT |
| VSCO | 2025-04-15 | 24 | 24 | BAR-EXACT |
| AVL | 2026-09-03 | 216 | 216 | BAR-EXACT |
| GDXD | 2026-09-02 | 204 | 204 | BAR-EXACT |
| REAL | 2025-02-18 | 15 | 15 | BAR-EXACT |
| CCUP | 2025-09-25 | 101 | 101 | BAR-EXACT |
| BKV | 2026-06-12 | 7 | 7 | BAR-EXACT |
| CGEM | 2026-04-24 | 8 | 8 | BAR-EXACT |
| AGCO | 2026-05-05 | 13 | 13 | BAR-EXACT |
| CON | 2026-05-08 | 12 | 12 | BAR-EXACT |
| IOT | 2026-02-04 | 18 | 18 | BAR-EXACT |
| HIMX | 2026-07-07 | 43 | 43 | BAR-EXACT |
| BMY | 2025-07-31 | 133 | 133 | BAR-EXACT |
| AMBQ | 2026-07-08 | 3 | 3 | BAR-EXACT |
| ROOT | 2026-05-20 | 1 | 1 | BAR-EXACT |

**20/20 bar-exact** (identical timestamp set and identical h/l/c/v to 1e-6 relative).

## Independent check (CLAUDE.md item 1)

`research/fuckup_audit/D/pm_verify.py` recomputes every field of randomly drawn `pm` rows
straight from a fresh Alpaca pull, with its own aggregation code (it does not import
`pm_backfill.py`). Run 2026-09-16: **12/12 rows reproduce exactly** (4 `alpaca`, 4 `sip_store`,
4 `none`), including the `none` rows, for which the fresh pull also returns zero bars.

Structural checks on the finished table (all pass): 61,902 rows = exactly the 61,902 distinct
(day, symbol) of `D/table.csv` with `fam in (F6, F8)` and `entry_m >= 600` (0 missing, 0 extra);
0 rows in `fetch_log`; every `src='none'` row has `n_pm_bars=0` and NULL aggregates; every other
row has `n_pm_bars >= 1` and non-NULL aggregates; `pm_low <= pm_vwap <= pm_high` on every row;
no row with bars and zero volume.

## Caveats

- `src='none'` is "Alpaca served no 04:00-09:29 ET bar for this symbol-day". For a thin small-cap that is the normal case, not an error; a fetch error is recorded separately and never becomes a `none`.

- Keys already in the store were NOT re-fetched, so their tape is whatever `bars_sip.db` holds (SIP, per the 2026-09-15 parity review). The spot check above is the evidence for treating the two provenances as one series.

- Bars are `adjustment=raw`. A split between the bar date and today makes pm_high/low/last raw-price, consistent with the intraday tape used elsewhere in this tree, but NOT comparable to an adjusted daily file (PLAN.md §1 price-scale rule).

