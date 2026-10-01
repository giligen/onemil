# RESULT 1,689a — recovery sub-pools 19'/19/20: the population production's 500K floor drops
(PREREG_1684.md amendment 1b; RESULT_1685.md scoped this population out — "no existing source
covers the 100K-500K slice in either window" — built fresh here, per the owner's direct ask)

**Verdict: 0/3 pools pass. 19' (whole slice) in-regime meanR +0.042 (dc_t 0.56, exTop5 −0.035, both
miss the bar); 19 (F1) and 20 (F2) are net NEGATIVE in both windows. Union would make weekly P10 and
worst-week WORSE than production alone in every case. Union = production alone, unchanged.**

## Population and method
Candidates: gap≥5% (yesterday's close → the official 09:30 open from the minute bars — study_orb_
features.py's own `extract_features()`, same code production uses; the daily-bar `open` is used only
to build the fetch list, buffered to ≥4.0% so a vendor-rounding miss at the margin doesn't drop a
true ≥5.0% name before minute bars exist to true it up — the "share that differs," verified directly
on the feature CSVs: in-regime 1,068/4,526 feature rows sat in the [4,5)% daily-open buffer, of which
only 10 were promoted into pool 19' by the true minute-bar gap (the other 1,058 stayed correctly
excluded), while 20 rows daily-open called a clean ≥5% were DROPPED once trued up (gap<5% on the
actual 09:30 minute-bar open); out-regime: 241 buffer rows / 3 promoted / 4 dropped. The buffer
caught 10+3=13 true positives a strict ≥5.0% fetch screen would have missed, at the cost of fetching
~1,300 extra symbol-days that ultimately stayed below 5%), price (daily open) $3-30, prior-day volume
[100K,500K). Admission: data/cache.db
daily_bars (read-only, source of record on overlap) UNION databento EQUS.SUMMARY (delisted
included) — in-regime: cache.db primary + equs_daily_2025_2026.parquet cross-check (covers
2025-01-02..2026-09-04, ~3wk short of this cell's 09-26 end — stated); out-regime: equs_daily_2024H2
primary (cell 1,684/1,685 convention) + cache.db cross-check. Minute bars appended to bars_sip.db
ONLY via backfill_bars_sip.py through a scratchpad wrapper with its own STATE file (never touches
cache.db). Pools, each its OWN selection chain (study_orb_pipeline_static_lock.py run separately per
pool so ranking/8-slot cap competes within that pool's own population, not the slice at large):
* **19' = the whole slice, no further gate.**
* **19 = slice × F1** relative volume at 09:35: `(range_total_volume/avg_daily_volume_20d)/0.049862
  ≥ 3.0`; 0.049862 = cell 1,685's own TRAIN(2025) median of that ratio on the non-production
  wide-seed population (1685_subpools.log) — REUSED VERBATIM, never refit on this slice.
* **20 = slice × F2** pre-market $ volume: `sum(volume×(h+l+c)/3)` over 04:00-09:30 ET bars ≥ $5M.

LIVE config: ORB_CATALYST_VETO=0, 8 slots (`max_concurrent`), 300bps spread gate, Q1 skip, 15:45
close — read from orb.yaml unchanged. Windows: in-regime 2025-01-01..2026-09-26 (halves by calendar
year), out-regime 2024H2. R = $375. Stats/union/cadence machinery reused verbatim from 1684_score.py.

## Stage 1 — daily-bar candidate slice (cache.db × databento cross-check)
| Window | Candidate rows | Days | Rows/day mean/median/p90/max | [4,5)% buffer rows | Databento-only symbols |
|---|---|---|---|---|---|
| in_regime | 10,185 | 434 | 23.5 / 18 / 41 / 271 | 3,004 (29.5%) | 200 |
| out_regime (2024H2) | 2,175 | 128 | 17.1 / 13 / 29 / 253 | 665 (30.6%) | 2 |

This is "pool 19'" at the daily-bar (no-minute-bar-needed) level — the population size, independent
of fetch success. (1685's own floor-dropped/day and production-tercile reads are cited below, not
rebuilt, except the tercile split which IS independently rebuilt per the owner's repeat-with-exact-
numbers ask.)

## Stage 2 — minute-bar backfill (research/bf_zero/backfill_bars_sip.py, own STATE file)
**2,210,965 bars appended over 561 distinct calendar days** that needed any new fetch (bars_sip.db
already carried 393,346 symbol-day pairs from earlier, unrelated populations before this ran).
**9/561 days (1.6%) were entirely skipped** — 8 from persistent Alpaca "invalid symbol" rejections on
a handful of preferred/warrant-style tickers (BW-A, AXIA-C, VACI=, KRSP=) that poison their whole
day's batch request, 1 from a `UNIQUE constraint` collision (bars.symbol,day,t) — both logged
ERROR/WARNING by the unmodified production script, days left unmarked-done for a future resume.

## Stage 3 — features, pool split, coverage
| Window | Candidates | Feature rows (any range built) | Zero-bar symbol-days | True gap≥5% (pool 19') | 19 (F1) | 20 (F2) | Premkt bar coverage |
|---|---|---|---|---|---|---|---|
| in_regime | 10,185 | 4,526 (44.4%) | 541 (5.3%) | 3,448 | 1,395 | 374 | 3,409/3,448 (98.9%) |
| out_regime | 2,175 | 1,028 (47.3%) | — | 786 | 414 | 90 | 742/786 (94.4%) |

The gap between "has some minute bars" (≈95% once fetched) and "has a complete, continuous 5-min
09:30-09:35 range" (44-47%) is this LOW-volume population's own sparse-tape reality, not a fetch
defect — exactly the kind of name the 500K floor is designed to keep out.

## Reads (own population, entered-only; R in units of $375)
| Pool | Window | n | fills/wk | meanR | iid_t | dc_t | exTop5 | MDE | wkP10 | worstWk | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 19' | in 2025 | 101 | 2.59 | +0.007 | 0.15 | −0.59 | −0.078 | 0.127 | −0.74R | −1.48R | +260 |
| 19' | in 2026 | 136 | 3.89 | +0.069 | 1.57 | 1.14 | −0.012 | 0.123 | −0.63R | −1.22R | +3,505 |
| 19' | in FULL | 237 | 3.20 | +0.042 | 1.34 | 0.56 | −0.035 | 0.089 | −0.72R | −1.48R | +3,765 |
| 19' | out (2024H2) | 28 | 1.47 | −0.085 | −1.37 | −1.26 | −0.133 | 0.174 | −0.57R | −1.57R | −892 |
| 19 (F1) | in FULL | 40 | 1.43 | −0.093 | −2.09 | −2.00 | −0.132 | 0.124 | −0.46R | −0.93R | −1,388 |
| 19 (F1) | out | 3 | 1.50 | −0.206 | −0.84 | −0.23 | −0.453 | 0.690 | −0.79R | −0.91R | −232 |
| 20 (F2) | in FULL | 11 | 1.00 | −0.073 | −0.79 | −0.79 | −0.122 | 0.259 | −0.43R | −0.49R | −301 |
| 20 (F2) | out | 3 | 1.00 | −0.027 | −0.11 | −0.11 | −0.261 | 0.718 | −0.37R | −0.44R | −31 |

Production reference (unchanged, same numbers as 1685/1684): in-regime n=482; out-regime n=59.

## Union with production
| Pool/win | Union n | frequency gain | union meanR | union wkP10 | prod-alone wkP10 | union worstWk | prod-alone worstWk | C1 gap median (both) | shared worst day |
|---|---|---|---|---|---|---|---|---|---|
| 19'/in | 713 | +231 fills | +0.083 | −1.16R | −0.90R | −3.32R | −2.39R | 12.0 wk (fail, both) | 2026-07-02: pool −1.20R, prod +0.00R that day |
| 19'/out | 87 | +28 fills | −0.006 | −0.73R | −0.51R | −2.35R | −0.78R | n/a | 2024-12-12: pool −0.91R, prod −0.27R (SHARED) |
| 19/in | 522 | +40 fills | +0.090 | −0.86R | −0.90R | −3.32R | −2.39R | 12.0 wk (fail, both) | 2025-12-02: pool −0.56R, prod −0.81R (SHARED) |
| 19/out | 62 | +3 fills | +0.019 | −0.52R | −0.51R | −1.69R | −0.78R | n/a | 2024-12-12 (SHARED) |
| 20/in | 493 | +11 fills | +0.102 | −0.89R | −0.90R | −2.50R | −2.39R | 12.0 wk (fail, both) | 2025-02-04: pool −0.49R, prod −0.56R (SHARED) |
| 20/out | 62 | +3 fills | +0.028 | −0.57R | −0.51R | −1.23R | −0.78R | n/a | 2024-12-12 (SHARED) |

Raw overlap with production = 0.0% for every pool by construction (gap≥5% is shared, but the 500K
floor makes these mutually exclusive by volume) — frequency gain is real, never double-counted.
Every union's weekly P10 and worst-week are AS BAD OR WORSE than production alone except 19/in's
wkP10 (−0.86R vs −0.90R, a rounding-level, not meaningful, improvement) — consistent with 1,684/
1,685's finding that admission-widening pools drag cadence, not help it. 2024-12-12 is a shared bad
day across every out-regime pool AND production — a real shared-tail risk, not pool-specific noise.

## Pass bar verdict (own meanR≥+0.05 & dc_t≥2.0 in-regime, ≥0 out-of-regime, exTop5>0)
19PRIME: in n=237 meanR=+0.042 dc_t=0.56 exTop5=−0.035 | out n=28 meanR=−0.085 → **FAIL**
19: in n=40 meanR=−0.093 dc_t=−2.00 exTop5=−0.132 | out n=3 meanR=−0.206 → **FAIL**
20: in n=11 meanR=−0.073 dc_t=−0.79 exTop5=−0.122 | out n=3 meanR=−0.027 → **FAIL**
0/3 pass → stage 2 (pairs) NOT RUN (pre-declared, passers only). **Union = production alone.**

## Production tercile repeat (independent rebuild, confirms RESULT_1685.md exactly)
In-regime (n=482): low n=161 meanR=+0.100, mid n=160 meanR=+0.119, high n=161 meanR=+0.098 — flat.
Out-regime (n=59): low n=20 meanR=+0.272, mid n=19 meanR=−0.031, high n=20 meanR=−0.152 — inverted,
thin (n≈20/tercile). Independently rebuilt from scratch (fresh cache.db+databento panel load, fresh
script) and matches 1685's published numbers to 3 decimals — the floor's own selectivity still shows
no quality signal within the already-≥500K population, and now we also know the population it drops
(this cell's slice) is itself edgeless-to-negative, so the 500K floor is not obviously discarding
edge by being volume-absolute rather than relative.

## Files
`1689a_slice.py` (candidates/poolsplit/pipeline/score/tercile stages), `1689a_features.py`
(study_orb_features.py loader-seam build, run per window), `pools_1689a_lib.py` (shared constants +
daily-panel loaders), `1689a_slice.log`, `1689a_candidates.csv`/`1689a_reads.csv` (12,360 daily-bar
rows), `1689a_fetch_list.csv`, `1689a_pool_books.csv` (322 scored trade rows, tagged pool/window),
`subpools_1689a/` (per-pool features/true/log, incl. `*_true.log` pipeline logs). Scratchpad:
`1689a_backfill_fill_days.py`, `1689a_backfill.log`, `1689a_features_run.log`, `1689a_poolsplit.log`,
`1689a_pipeline.log`, `1689a_score.log`, `1689a_tercile.log`.
