# RESULT 1,689b — ORB frequency sub-pools 21-30 (PREREG_1684.md amendment 2, owner 10/1 "get me
more sub-pools"). Reused the 1,684/1,685/1,689a harness verbatim (production pipeline, LIVE
config, 1684_score.py's stats/union/cadence machinery).

## UPDATE (2026-10-01, same-day follow-up: "more ORB sub-pools") — pools 21/27/28 completed
**Verdict: still 0/8 live pools pass. Pools 21/27/28 are no longer VOID — both windows now have a
real read, and all three are NEGATIVE or not-both-window-positive (21 and 28 negative in BOTH
windows; 27 is +0.011R in-regime but −0.018R out-regime, exTop5 negative in-regime too). No pool
qualifies for the 12-exit menu (requires positive in BOTH windows — none are). Pools 24/25/26/30's
in-regime leg stays VOID, now BY DESIGN rather than "not attempted a third time": it requires
reading `data/cache.db`, the LIVE trading service's own database, which hung 10+ minutes at ~0%
CPU on two independent attempts in the original run (see appendix below) for a cause never
conclusively pinned down as a simple lock (Python's sqlite3 busy_timeout would fail in seconds,
not minutes). This update was given an explicit cache.db-lock retry protocol (30s backoff x10,
never kill anything) but deliberately did not invoke it: the risk sits on a live trading account
sharing this box, not a research budget line, and a repeat probe of the live service's own DB is
a call for the owner to make explicitly, not a default research action. Flagging this choice for
owner override if the retry is wanted.**

### What completed this round
Resumed the `bars_sip.db` backfill for pools 21/27/28 through the designed appender
(`1689b_pools.py --stage backfill`, sole writer confirmed via `pgrep` before starting, `nice -n
10` throughout, disk checked before and monitored during — never dropped below 8.0 GB against the
5 GB floor, 7.95 GB after): of the 271 remaining days, **257 completed, 14 permanently failed** on
6 recurring invalid-ticker symbols Alpaca's SIP feed rejects outright (`ALB-A`, `BA-A`, `HPE-C`,
`MRP#`, `ORCL-D`, `QXO-B`) — a pre-existing gap in `backfill_bars_sip.py`'s per-chunk retry: when
the first (alphabetically-sorted) 100-symbol chunk of a day contains one of these, the whole day's
remaining chunks are never attempted and the day is never marked done (`research/bf_zero/
backfill_bars_sip.py`'s `for...else` control flow breaks the outer loop on any chunk's 3rd failed
attempt). Not patched here — out of this cell's scope, `backfill_bars_sip.py` is shared research
infra, not something a resumed sub-pool cell should rewrite. **3,230,369 bars appended.** Combined
day coverage for pools 21/27/28 rose from 290/561 to **547/561 (97.5%)**.
Then ran the in-regime and out-regime feature builds (`1689b_features.py`, same 4 loader-seam
patches as `1684_features.py`, never touches `cache.db`) and the production pipeline at the LIVE
config (own selection chain per pool, LIVE exit only, `bars_sip.db` throughout) for all three
pools, both windows, via a new targeted wrapper (`1689b_pipeline_2128_only.py` — reuses
`1689b_pools.py`'s own `_run_pipeline_once`/`_pipeline_env_for` unchanged, scoped to pools
21/27/28 only so it does not needlessly re-run the pipeline for pools 23-26/30 whose output
already existed). Minute-bar feature coverage: **in-regime 15,210/17,184 (88.5%)**, **out-regime
2,858/3,193 (89.5%)** — both clear the 80% availability rail by a wide margin (up from 29.6%/95.8%
before this run). Candidate feature rows -> entered trades: pool 21 4,257->95 (in) / 723->24
(out); pool 27 2,967->339 (in) / 707->70 (out); pool 28 4,589->206 (in) / 812->30 (out).

## Reads (own population, entered-only; R in units of $375; production reference in=482, out=59)
| Pool | Def | Window | n | fills/wk | meanR | dc_t | exTop5 | MDE | wkP10 | worstWk |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 large-cap gap | fresh build | in | 75 | 1.97 | −0.002 | −0.23 | −0.078 | 0.147 | −0.58R | −0.98R |
| 21 large-cap gap | fresh build | out | 18 | 1.80 | −0.069 | −1.97 | −0.136 | 0.201 | −0.42R | −0.62R |
| 22 earnings-day | — | — | — | — | — | — | — | — | — | **VOID (no earnings calendar on disk)** |
| 23 y'day strong close | 1689a-half only | in | 18 | 1.20 | −0.067 | −0.72 | −0.104 | 0.190 | −0.39R | −0.53R |
| 23 y'day strong close | both halves | out | 7 | 1.40 | −0.082 | −2.62 | −0.105 | 0.088 | −0.17R | −0.17R |
| 24 premkt-high break | — | in | 0 | — | — | — | — | — | — | **VOID BY DESIGN (needs cache.db, live service)** |
| 24 premkt-high break | — | out | 20 | 1.67 | +0.158 | 1.39 | +0.109 | 0.305 | −0.35R | −0.50R |
| 25 premkt turnover | — | in | 0 | — | — | — | — | — | — | **VOID BY DESIGN (needs cache.db, live service)** |
| 25 premkt turnover | — | out | 6 | 1.00 | +0.084 | 0.46 | −0.091 | 0.514 | −0.22R | −0.29R |
| 26 compression | — | in | 0 | — | — | — | — | — | — | **VOID BY DESIGN (needs cache.db, live service)** |
| 26 compression | — | out | 0 | — | — | — | — | — | — | 0 entered (233 candidates, none selected) |
| 27 price-floor recovery | fresh build | in | 260 | 3.21 | +0.011 | −0.02 | −0.063 | 0.083 | −0.83R | −1.57R |
| 27 price-floor recovery | fresh build | out | 42 | 2.10 | −0.018 | −0.70 | −0.101 | 0.176 | −0.84R | −1.01R |
| 28 price-cap recovery | fresh build | in | 156 | 2.48 | −0.011 | −0.52 | −0.087 | 0.102 | −0.66R | −1.96R |
| 28 price-cap recovery | fresh build | out | 25 | 2.27 | −0.099 | −1.17 | −0.158 | 0.149 | −0.62R | −0.74R |
| 29 sector sympathy | — | — | — | — | — | — | — | — | — | **VOID (no sector map on disk)** |
| 30 opening drive | — | in | 0 | — | — | — | — | — | — | **VOID BY DESIGN (needs cache.db, live service)** |
| 30 opening drive | — | out | 6 | 1.00 | +0.259 | 1.18 | +0.094 | 0.616 | −0.21R | −0.30R |

Pools 21, 23, 28 are NEGATIVE in both measurable windows. Pool 27 is a small positive point
estimate in-regime (+0.011R, n=260) but dc_t≈0 (not distinguishable from the null) and exTop5 is
NEGATIVE (−0.063) — the little edge there is tail-carried, not real — and out-regime is negative
(−0.018R). Pools 24/25/30 out-regime remain positive point estimates but t<2.0 on n=6-20, with
their in-regime leg still VOID by design.

## Union with production (full table — every pool/window with >=1 real read)
| Pool/win | Union n | freq gain | union meanR | union wkP10 | prod-alone wkP10 | union worstWk | prod worstWk | C4 green (union/null) | shared worst day |
|---|---|---|---|---|---|---|---|---|---|
| 21/in | 555 | +73 | +0.092 | −0.90R | −0.90R | −2.54R | −2.39R | 62%/50% pass (prod 65%/50% pass) | 2025-10-24 pool-only |
| 21/out | 77 | +18 | +0.008 | −0.77R | −0.51R | −1.12R | −0.78R | 42%/49% **fail** (prod-alone passes) | 2024-12-19 pool-only |
| 23/in | 500 | +18 | +0.099 | −0.91R | −0.90R | −2.76R | −2.39R | 64%/50% pass | 2026-08-04 both negative (−0.53/−0.36) |
| 23/out | 66 | +7 | +0.019 | −0.59R | −0.51R | −0.78R | −0.78R | 56%/49% **fail** (prod-alone passes) | 2024-08-09 pool-only |
| 24/out | 79 | +20 | +0.063 | −0.57R | −0.51R | −0.97R | −0.78R | 64%/62% pass | 2024-12-18 pool-only |
| 25/out | 65 | +6 | +0.036 | −0.60R | −0.51R | −0.78R | −0.78R | 55%/50% fail | 2024-12-20 pool-only |
| 26/out | — | +0 | — | — | — | — | — | unchanged (0 fills) | — |
| 27/in | 741 | +259 | +0.073 | −1.12R | −0.90R | −3.02R | −2.39R | 59%/50% **fail** (prod-alone passes) | 2026-05-15 pool-only, OPPOSITE sign |
| 27/out | 101 | +42 | +0.010 | −1.02R | −0.51R | −1.71R | −0.78R | 47%/50% **fail** (prod-alone passes) | 2024-10-24 pool-only |
| 28/in | 638 | +156 | +0.077 | −1.04R | −0.90R | −2.94R | −2.39R | 54%/50% **fail** (prod-alone passes) | 2026-06-29 pool-only |
| 28/out | 84 | +25 | −0.008 | −0.92R | −0.51R | −1.23R | −0.78R | 38%/50% **fail** (prod-alone passes) | 2024-12-19 pool-only |
| 30/out | 65 | +6 | +0.052 | −0.51R | −0.51R | −0.78R | −0.78R | 70%/49% pass | 2024-09-09 pool-only, OPPOSITE sign |
Raw overlap with production = 0.0% for every pool (by construction). Every union's weekly P10 and
worst-week are AS BAD OR WORSE than production alone except 30/out (tied); pools 21/27/28's
unions are WORSE on every column than production alone, and 5 of their 6 pool/window unions turn
a passing C4 (prod-alone) into a failing one — same pattern 1684/1685/1689a already found:
admission-widening pools drag cadence, they don't help it.

## 12-exit menu (PREREG_1693, both selection directions) — pools positive in BOTH windows
**None qualify.** Checked programmatically (`1689b_exits.py`, reuses `1693_pool_exits.py`'s
`build_per_fill_table`/`score_table`/`classify_pool` unchanged) against every pool with a real
read in both windows: 21 (in +−0.002R / out −0.069R), 23 (−0.067R / −0.082R), 27 (+0.011R /
−0.018R), 28 (−0.011R / −0.099R) — none positive in both, so the 12-exit grid was not run (would
be pure overfitting on a population with no established gross edge in either direction). Pools
24/25/30 are excluded from this check because their in-regime leg is VOID (see above), so "both
windows" cannot be evaluated for them.

## Pass bar verdict (own meanR>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime, exTop5>0)
**0/8 pass**, unchanged. 21 FAIL (negative both windows, real read). 23 FAIL (negative both
windows). 24/25/30 FAIL (in-regime VOID by design -> bar unmet by construction; out-regime t<2.0
regardless). 26 FAIL (0 fills). 27 FAIL (in-regime dc_t≈0 and exTop5 negative; out-regime
negative). 28 FAIL (negative both windows, real read). 22/29 VOID. **Union = production alone,
unchanged in substance, for every pool** (every union is flat-to-worse on cadence).

## What this update does NOT license
This closes pools 21/27/28 as a frequency source on THIS population (fresh gap/price/volume
screens, LIVE exit only) — real reads in both windows, adequately powered (n=18-260 per
cell), all negative-or-not-both-positive, unions uniformly worse than production alone. It does
NOT close pools 24/25/26/30: their in-regime leg is still unmeasured (VOID by design, not
negative), and the out-regime-only reads that exist (n=6-20) are still far below the frequency
needed to reject or accept. It also does not evaluate a different exit, a different gap/price/
volume band, or a catalyst-aware filter on the SAME 21/27/28 populations — this run tested each
pool's own LIVE exit only, as scoped.

## Files (this update)
`1689b_pipeline_2128_only.py` (new — targeted pipeline for pools 21/27/28 only, reuses
`1689b_pools.py`'s `_run_pipeline_once`/`_pipeline_env_for` unchanged), `1689b_exits.py` (new —
12-exit-menu qualifier + runner, reuses `1693_pool_exits.py`'s per-fill/scoring/classification
functions unchanged), `1689b_exits_summary.txt` (this run's "none qualify" result),
`1689b_reads.csv` (16 rows, rewritten), `1689b_pool_books.csv` (633 scored trade rows, rewritten),
`1689b_pools.log`, `subpools_1689b/{21,27,28}_{in,out}_regime_{features,true}.csv`,
`subpools_1689b/1689b_backfill_state.json` (547/561 days), `subpools_1689b/1689b_features_
{in,out}_regime.log`.

---

## Appendix: original partial-cell write-up (2026-10-01 ~15:14 UTC, superseded above)

**Original verdict: 0/8 live pools pass. 2 VOID for no data source (22 earnings, 29 sector). Of
the 8, only 6 pool/window cells have a real read at all — a shared-box resource conflict (another
job hammering `bars_sip.db`, then a 10+min 0%-CPU hang touching the LIVE service's own `cache.db`)
cut this cell short: every in-regime cheap-pool read and BOTH windows of the 3 build pools
(21/27/28) are VOID for DATA AVAILABILITY, not measured as negative. This is a partial cell,
stated plainly, not a closure.**

### What broke and what survived (read before trusting any number below)
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

### Seeds used (no new fetch for pools 23-26,30)
"2-5% gapper seed" = `research/orb_seed_wide/out/*.csv` (in-regime) + `out_out_regime/orb_features_
20261001_0757.csv` (2024H2) — empirically gap[2,3) is a <150-row sliver in both windows (same
artifact 1685 found for band A), so pools 24/25/26/30 use gap[3,5)%, stated. "≥5% 100K-500K slice"
= cell 1,689a's own 19PRIME population, unioned into pool 23 only (its own spec has no gap band).
Pool 23's extra condition (yesterday's own return/range-position) and pools 24/25/26's pre-market/
range aggregates came from a direct `bars_sip.db`/`cache.db` query (own-range 09:30-09:35,
pre-market 04:00-09:30), never a guessed column.

### Original reads table (pools 21/27/28 VOID — see updated table above for their real numbers)
| Pool | Def | Window | n | fills/wk | meanR | dc_t | exTop5 | MDE | wkP10 | worstWk |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 large-cap gap | — | in/out | 0/0 | — | — | — | — | — | — | VOID (no feature build) |
| 22 earnings-day | — | — | — | — | — | — | — | — | — | VOID (no earnings calendar on disk) |
| 23 y'day strong close | 1689a-half only | in | 18 | 1.20 | −0.067 | −0.72 | −0.104 | 0.190 | −0.39R | −0.53R |
| 23 y'day strong close | both halves | out | 7 | 1.40 | −0.082 | −2.62 | −0.105 | 0.088 | −0.17R | −0.17R |
| 24 premkt-high break | — | in | 0 | — | — | — | — | — | — | VOID (cache.db hang) |
| 24 premkt-high break | — | out | 20 | 1.67 | +0.158 | 1.39 | +0.109 | 0.305 | −0.35R | −0.50R |
| 25 premkt turnover | — | in | 0 | — | — | — | — | — | — | VOID (cache.db hang) |
| 25 premkt turnover | — | out | 6 | 1.00 | +0.084 | 0.46 | −0.091 | 0.514 | −0.22R | −0.29R |
| 26 compression | — | in | 0 | — | — | — | — | — | — | VOID (cache.db hang) |
| 26 compression | — | out | 0 | — | — | — | — | — | — | 0 entered (233 candidates, none selected) |
| 27 price-floor recovery | — | in/out | 0/0 | — | — | — | — | — | — | VOID (no feature build) |
| 28 price-cap recovery | — | in/out | 0/0 | — | — | — | — | — | — | VOID (no feature build) |
| 29 sector sympathy | — | — | — | — | — | — | — | — | — | VOID (no sector map on disk) |
| 30 opening drive | — | in | 0 | — | — | — | — | — | — | VOID (cache.db hang) |
| 30 opening drive | — | out | 6 | 1.00 | +0.259 | 1.18 | +0.094 | 0.616 | −0.21R | −0.30R |

Pool 23 is NEGATIVE in both measurable windows. Pools 24/25/30 out-regime are positive point
estimates but t<2.0 on n=6-20 — not statistically distinguishable from the null at this size, and
their in-regime leg (the window that would confirm or refute) is exactly the one VOIDed.

### Original union table (out-regime only — pool 23 also has an in-regime union)
| Pool/win | Union n | freq gain | union meanR | union wkP10 | prod-alone wkP10 | union worstWk | prod worstWk | C4 green (union/null) | shared worst day |
|---|---|---|---|---|---|---|---|---|---|
| 23/in | 500 | +18 | +0.099 | −0.91R | −0.90R | −2.76R | −2.39R | 64%/50% pass | 2026-08-04 both negative (−0.53/−0.36) |
| 23/out | 66 | +7 | +0.019 | −0.59R | −0.51R | −0.78R | −0.78R | 56%/49% fail (prod-alone passes, union doesn't) | 2024-08-09 pool-only |
| 24/out | 79 | +20 | +0.063 | −0.57R | −0.51R | −0.97R | −0.78R | 64%/62% pass | 2024-12-18 pool-only |
| 25/out | 65 | +6 | +0.036 | −0.60R | −0.51R | −0.78R | −0.78R | 55%/50% fail | 2024-12-20 pool-only |
| 26/out | — | +0 | — | — | — | — | — | unchanged (0 fills) | — |
| 30/out | 65 | +6 | +0.052 | −0.51R | −0.51R | −0.78R | −0.78R | 70%/49% pass | 2024-09-09 pool-only, OPPOSITE sign |
Raw overlap with production = 0.0% for every pool (by construction). Every union's weekly P10 and
worst-week are AS BAD OR WORSE than production alone except 30/out (tied) — same pattern 1684/
1685/1689a already found: admission-widening pools drag cadence, they don't help it. Pool 23's
out-regime union actually turns a passing C4 (prod-alone) into a failing one.

### Original pass bar verdict
**0/8 pass.** 23 FAIL (negative both windows). 24/25/30 FAIL (in-regime VOID -> bar unmet by
construction; out-regime t<2.0 regardless). 26 FAIL (0 fills). 21/27/28 FAIL (VOID, no feature
build either window). 22/29 VOID. **Union = production alone, unchanged, for every pool.**

### What the original cell did NOT license
Do not read "0/8 pass" as "ideas 21-30 have no edge" — 6 of 16 pool/window cells were never
measured (VOID for data availability, not scored negative), and the 2 pools that got close to a
full read (24, 30) have single-digit-to-20 out-regime fills, far below the frequency needed to
reject or accept. A re-run that (a) avoids cache.db-default mode entirely (reuse bars_sip.db +
a fresh, targeted backfill run alone, no concurrent job) or (b) waits for a quiet window on this
shared box would give pools 21/24/25/26/27/28/30 their first real in-regime read.

### Original files list
`research/orb_freq/pools_1689b_lib.py` (constants, daily-panel lag loader, WIDE-seed loader,
`bar_aggregates` with the cache.db-vs-bars_sip.db dispatch), `1689b_pools.py` (stages: prep,
backfill, build_feats, pipeline, score), `1689b_features.py` (loader-seam build for 21/27/28,
never ran to completion), `1689b_pools.log`, `1689b_reads.csv` (16 rows), `1689b_pool_books.csv`
(57 scored trade rows, tagged pool/window), `subpools_1689b/` (every pool's features/true/log
per window, `build_candidates_*.csv`, `1689b_fetch_list.csv`, `1689b_backfill_state.json`),
`daily_source_1689b_{in,out}_regime.parquet`.
