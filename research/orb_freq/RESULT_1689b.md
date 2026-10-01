# RESULT 1,689b — ORB frequency sub-pools 21-30 (PREREG_1684.md amendment 2, owner 10/1 "get me
more sub-pools"). Reused the 1,684/1,685/1,689a harness verbatim (production pipeline, LIVE
config, 1684_score.py's stats/union/cadence machinery).

**Verdict: 0/8 live pools pass. 2 VOID for no data source (22 earnings, 29 sector). Of the 8, only
6 pool/window cells have a real read at all — a shared-box resource conflict (another job
hammering `bars_sip.db`, then a 10+min 0%-CPU hang touching the LIVE service's own `cache.db`)
cut this cell short: every in-regime cheap-pool read and BOTH windows of the 3 build pools
(21/27/28) are VOID for DATA AVAILABILITY, not measured as negative. This is a partial cell,
stated plainly, not a closure.**

## What broke and what survived (read before trusting any number below)
`bars_sip.db` coverage of the in-regime WIDE seed (gap[3,5)%) is only 20.8% (vs 100% out-regime) —
checked directly before building anything. The live scanner's own `data/cache.db::intraday_bars_1min`
has the missing bars (98.5-100% coverage, verified per-pool), so pools 24/25/26/30's in-regime leg
and pool 23's WIDE-sourced in-regime half were built to use `ORB_BT_BARS_DB` cache.db DEFAULTS —
production's own convention (`1685_subpools.py::_pipeline_env_for`). That leg then hung for 10+
minutes at ~0% CPU on TWO independent attempts (one with a fast local daily-source parquet
substituted, ruling out a slow daily-bars re-scan) right after "Database initialized: data/cache.db"
— `onemil-trader` was confirmed ACTIVE on this box throughout, so this reads as a write-lock wait
on the live service's own DB, not a slow read. Per CLAUDE.md (never touch/disrupt the live
service), this cell did not retry it: **that leg is VOID, not attempted a third time.**
Separately, backfilling pools 21/27/28's fresh candidates into `bars_sip.db` collided with an
unrelated, already-running external job (`/tmp/.../scratchpad/backfill_fill_days.py`, not ours)
also writing `bars_sip.db` — repeated `database is locked` retries. Stopped (safe: backfill
commits per day, resumable) after **290/561 days** (2024-07-02..2025-09-05), **~3.59M bars**,
**8,142/20,377 (40.0%) candidate symbol-days** covered: **out-regime 95.8%** (3,193 pairs, clears
the 80% rail), **in-regime only 29.6%** (17,184 pairs, fails it) — the external job claimed the
remainder of the queue. Per CLAUDE.md's 80% availability rail, pools 21/27/28's in-regime leg is
VOID for coverage; their out-regime leg never got a feature build at all (the loader-seam script
also hung, same cache.db symptom, before reaching out-regime) — **VOID both windows.**
Exit menu: only the LIVE exit was run on every pool (own selection chain only) — stated per this
cell's explicit budget allowance, not hidden.

## Seeds used (no new fetch for pools 23-26,30)
"2-5% gapper seed" = `research/orb_seed_wide/out/*.csv` (in-regime) + `out_out_regime/orb_features_
20261001_0757.csv` (2024H2) — empirically gap[2,3) is a <150-row sliver in both windows (same
artifact 1685 found for band A), so pools 24/25/26/30 use gap[3,5)%, stated. "≥5% 100K-500K slice"
= cell 1,689a's own 19PRIME population, unioned into pool 23 only (its own spec has no gap band).
Pool 23's extra condition (yesterday's own return/range-position) and pools 24/25/26's pre-market/
range aggregates came from a direct `bars_sip.db`/`cache.db` query (own-range 09:30-09:35,
pre-market 04:00-09:30), never a guessed column.

## Reads (own population, entered-only; R in units of $375; production reference in=482, out=59)
| Pool | Def | Window | n | fills/wk | meanR | dc_t | exTop5 | MDE | wkP10 | worstWk |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 large-cap gap | — | in/out | 0/0 | — | — | — | — | — | — | **VOID (no feature build)** |
| 22 earnings-day | — | — | — | — | — | — | — | — | — | **VOID (no earnings calendar on disk)** |
| 23 y'day strong close | 1689a-half only | in | 18 | 1.20 | −0.067 | −0.72 | −0.104 | 0.190 | −0.39R | −0.53R |
| 23 y'day strong close | both halves | out | 7 | 1.40 | −0.082 | −2.62 | −0.105 | 0.088 | −0.17R | −0.17R |
| 24 premkt-high break | — | in | 0 | — | — | — | — | — | — | **VOID (cache.db hang)** |
| 24 premkt-high break | — | out | 20 | 1.67 | +0.158 | 1.39 | +0.109 | 0.305 | −0.35R | −0.50R |
| 25 premkt turnover | — | in | 0 | — | — | — | — | — | — | **VOID (cache.db hang)** |
| 25 premkt turnover | — | out | 6 | 1.00 | +0.084 | 0.46 | −0.091 | 0.514 | −0.22R | −0.29R |
| 26 compression | — | in | 0 | — | — | — | — | — | — | **VOID (cache.db hang)** |
| 26 compression | — | out | 0 | — | — | — | — | — | — | 0 entered (233 candidates, none selected) |
| 27 price-floor recovery | — | in/out | 0/0 | — | — | — | — | — | — | **VOID (no feature build)** |
| 28 price-cap recovery | — | in/out | 0/0 | — | — | — | — | — | — | **VOID (no feature build)** |
| 29 sector sympathy | — | — | — | — | — | — | — | — | — | **VOID (no sector map on disk)** |
| 30 opening drive | — | in | 0 | — | — | — | — | — | — | **VOID (cache.db hang)** |
| 30 opening drive | — | out | 6 | 1.00 | +0.259 | 1.18 | +0.094 | 0.616 | −0.21R | −0.30R |

Pool 23 is NEGATIVE in both measurable windows. Pools 24/25/30 out-regime are positive point
estimates but t<2.0 on n=6-20 — not statistically distinguishable from the null at this size, and
their in-regime leg (the window that would confirm or refute) is exactly the one VOIDed.

## Union with production (out-regime only — the only window with >=1 real read per pool; pool 23
also has an in-regime union since its 1689a-half ran)
| Pool/win | Union n | freq gain | union meanR | union wkP10 | prod-alone wkP10 | union worstWk | prod worstWk | C4 green (union/null) | shared worst day |
|---|---|---|---|---|---|---|---|---|---|
| 23/in | 500 | +18 | +0.099 | −0.91R | −0.90R | −2.76R | −2.39R | 64%/50% pass | 2026-08-04 both negative (−0.53/−0.36) |
| 23/out | 66 | +7 | +0.019 | −0.59R | −0.51R | −0.78R | −0.78R | 56%/49% **fail** (prod-alone passes, union doesn't) | 2024-08-09 pool-only |
| 24/out | 79 | +20 | +0.063 | −0.57R | −0.51R | −0.97R | −0.78R | 64%/62% pass | 2024-12-18 pool-only |
| 25/out | 65 | +6 | +0.036 | −0.60R | −0.51R | −0.78R | −0.78R | 55%/50% fail | 2024-12-20 pool-only |
| 26/out | — | +0 | — | — | — | — | — | unchanged (0 fills) | — |
| 30/out | 65 | +6 | +0.052 | −0.51R | −0.51R | −0.78R | −0.78R | 70%/49% pass | 2024-09-09 pool-only, OPPOSITE sign |
Raw overlap with production = 0.0% for every pool (by construction). Every union's weekly P10 and
worst-week are AS BAD OR WORSE than production alone except 30/out (tied) — same pattern 1684/
1685/1689a already found: admission-widening pools drag cadence, they don't help it. Pool 23's
out-regime union actually turns a passing C4 (prod-alone) into a failing one.

## Pass bar verdict (own meanR>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime, exTop5>0)
**0/8 pass.** 23 FAIL (negative both windows). 24/25/30 FAIL (in-regime VOID -> bar unmet by
construction; out-regime t<2.0 regardless). 26 FAIL (0 fills). 21/27/28 FAIL (VOID, no feature
build either window). 22/29 VOID. **Union = production alone, unchanged, for every pool.**

## What this cell does NOT license
Do not read "0/8 pass" as "ideas 21-30 have no edge" — 6 of 16 pool/window cells were never
measured (VOID for data availability, not scored negative), and the 2 pools that got close to a
full read (24, 30) have single-digit-to-20 out-regime fills, far below the frequency needed to
reject or accept. A re-run that (a) avoids cache.db-default mode entirely (reuse bars_sip.db +
a fresh, targeted backfill run alone, no concurrent job) or (b) waits for a quiet window on this
shared box would give pools 21/24/25/26/27/28/30 their first real in-regime read.

## Files
`research/orb_freq/pools_1689b_lib.py` (constants, daily-panel lag loader, WIDE-seed loader,
`bar_aggregates` with the cache.db-vs-bars_sip.db dispatch), `1689b_pools.py` (stages: prep,
backfill, build_feats, pipeline, score), `1689b_features.py` (loader-seam build for 21/27/28,
never ran to completion), `1689b_pools.log`, `1689b_reads.csv` (16 rows), `1689b_pool_books.csv`
(57 scored trade rows, tagged pool/window), `subpools_1689b/` (every pool's features/true/log
per window, `build_candidates_*.csv`, `1689b_fetch_list.csv`, `1689b_backfill_state.json`),
`daily_source_1689b_{in,out}_regime.parquet`.
