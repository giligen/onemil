# RESULT 1,689a — recovery sub-pools 19'/19/20: the population production's 500K floor drops
(PREREG_1684.md amendment 1b; RESULT_1685.md scoped this population out -- "no existing source
covers the 100K-500K slice in either window" -- built fresh here, per the owner's direct ask)

**Verdict: PENDING -- backfill/pipeline/score stages running.**

## Population and method
Candidates: gap>=5% (yesterday's close -> the official 09:30 open from the minute bars, computed by
study_orb_features.py's own `extract_features()` exactly as production does it; the daily-bar `open`
is used ONLY to build the fetch list, buffered to >=4.0% so a vendor-rounding miss at the margin
doesn't drop a true >=5.0% name before minute bars exist to true it up, and as an explicit, flagged
fallback for the rare symbol-day where the minute-bar fetch itself failed), price (today's daily
open) $3-30, prior-day volume [100K,500K). Admission source: data/cache.db::daily_bars (read-only,
source of record on overlap) UNION data/research/databento EQUS.SUMMARY parquet (delisted included)
-- in-regime: cache.db primary + equs_daily_2025_2026.parquet cross-check (covers 2025-01-02..
2026-09-04 only, ~3wk short of this cell's 2026-09-26 admission end -- stated, not hidden);
out-regime: equs_daily_2024H2.parquet primary (cell 1,684/1,685's own convention) + cache.db
cross-check/20d-lookback stub. Minute bars appended to research/bf_zero/bars_sip.db ONLY via
research/bf_zero/backfill_bars_sip.py, called through a scratchpad wrapper with its own STATE file
(never touches cache.db; never deletes/rewrites a row).

Pools, each its own selection chain (study_orb_pipeline_static_lock.py run SEPARATELY per pool so
its ranking/8-slot cap competes within that pool's own population, cell 1,684/1,685 convention):
* **19' = the whole slice, no further gate.**
* **19 = slice x F1** relative volume at 09:35: `(range_total_volume/avg_daily_volume_20d) /
  0.049862 >= 3.0`. 0.049862 is cell 1,685's own TRAIN(2025) cross-sectional median of that raw
  ratio on the non-production wide-seed population (1685_subpools.log) -- REUSED VERBATIM, never
  refit on this slice (refitting a normalization on the population being scored is its own leak).
* **20 = slice x F2** pre-market dollar volume: `sum(volume*(h+l+c)/3)` over each symbol-day's
  04:00-09:30 ET bars in bars_sip.db >= $5,000,000.

LIVE config: ORB_CATALYST_VETO=0 (veto OFF), 8 slots, spread gate, Q1 skip, 15:45 close -- read from
orb.yaml/config unchanged (never written). Windows: in-regime 2025-01-01..2026-09-26 (halves by
calendar year: 2025, 2026); out-regime 2024H2 (2024-07-01..2024-12-31) -- the 2023-01..2024-06
minute-bar store no longer exists on disk (cell 1,684's finding), not rebuilt here. R = $375.
Scoring machinery (stats/MDE/union/cadence) reused verbatim from 1684_score.py.

## Stage 1: daily-bar candidate slice (data/cache.db x databento cross-check)
| Window | Candidate rows | Calendar days in window | Rows/day (mean) | Rows/day (median) | Rows/day (p90) | Rows/day (max) | [4,5)% buffer-zone rows | Symbols ONLY via databento (delisted) |
|---|---|---|---|---|---|---|---|---|
| in_regime (2025-01-01..2026-09-26) | 10,185 | 434 | 23.47 | 18.0 | 41.0 | 271 | 3,004 (29.5%) | 200 |
| out_regime (2024H2) | 2,175 | 128 (127 with >=1 row) | 17.13 | 13.0 | 29.4 | 253 | 665 (30.6%) | 2 |
| **Total** | **12,360** | | | | | | | |

Cross-check direction differs by window by construction: in-regime uses cache.db as the admission
source of record (databento recovers names cache.db alone would have missed to delisting -- 200
symbols, all days); out-regime uses databento (EQUS.SUMMARY, delisted included) as the source of
record, cache.db is the cross-check (669 symbols appear ONLY in cache.db across the full 2024H2
daily panel load, reported by `pools_1689a_lib.py`'s own logging -- most of those are outside this
slice's price/volume/gap gates; the 2-symbol number above is restricted to the admitted slice rows).
This is the "pool 19'" daily-bar universe size, independent of whether minute bars exist yet.

## Stage 2: minute-bar backfill
PENDING -- `1689a_backfill.log` (scratchpad), state `backfill_state_1689a.json`. bars_sip.db already
carried 393,346 distinct symbol-day pairs (133.2M bars, 2024-07-01..2026-09-28) from earlier,
unrelated populations' fetches before this cell ran; `backfill_bars_sip.py`'s own `missing_pairs()`
auto-skips any of this slice's 12,360 symbol-days already covered. Symbol-days requested / already
covered / newly fetched / bars appended: **PENDING**.

## Stage 3: features, pool split, pipeline, scoring
PENDING.

## Production tercile repeat
PENDING (repeat of RESULT_1685.md's read: in-regime n=482 -- low n=161 meanR=+0.100, mid n=160
meanR=+0.119, high n=161 meanR=+0.098 [flat]; out-regime n=59 -- low n=20 meanR=+0.272, mid n=19
meanR=-0.031, high n=20 meanR=-0.152 [inverted, thin]).

## Files
`1689a_slice.py` (candidates/poolsplit/pipeline/score/tercile stages), `1689a_features.py`
(study_orb_features.py loader-seam build, run per window), `pools_1689a_lib.py` (shared constants +
daily-panel loaders), `1689a_slice.log`, `1689a_candidates.csv`, `1689a_fetch_list.csv`,
`1689a_reads.csv`, `1689a_pool_books.csv`, `subpools_1689a/` (per-pool features/true/log).
