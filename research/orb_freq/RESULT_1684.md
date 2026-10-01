# RESULT 1,684 — ORB admission pools: ideas 1, 2, 10, 11 (PREREG_1684.md)

All four pools built on `study_orb_pipeline_static_lock.py` at the LIVE config (veto OFF, 8 slots,
Q1 skip, spread gate, 15:45 close), per-pool selection chain as cell 1,328 (own ranking, own 8-slot
cap, excluded from the day's production picks; raw overlap with production measured, not assumed).
**Verdict: all four FAIL the pre-registered pass bar. None is added to the union.**

## Scope actually built (read before the numbers)
* **In-regime** = 2025-01-02..2026-09-18 (the PREREG's window end, bounded by the latest wide-seed
  build on disk). **Out-of-regime** = **2024-07-01..2024-12-31 (2024H2) only**, not the PREREG's full
  2023-01..2024-12: `research/bf_zero/bars_sip.db` (the shared minute-bar store, appended via its
  designated wrapper per this cell's instructions) starts 2024-07-01, and the `research/orb_2023/bars.db`
  that built the 2023-01..2024-06 production reference no longer exists on disk — rebuilding a
  2023H1-2024H1 minute-bar store from scratch was outside this cell's budget. The existing production
  reference for that half (`research/orb_2023/book_1418_liveexit.csv`) is reported below for CONTEXT
  only; no new pool was built against it.
* **idea1** admission = gap_pct in **[3,5)%** (not the PREREG's unbounded "<5%"): the >=3% slice reuses
  the already-built wide-seed features (`research/orb_seed_wide/out/orb_features_202609{20_2142,21_1842}.csv`,
  gap>=3%, joined to `data/cache.db::daily_bars` for the actual open/prev_close/prev_high the wide CSV
  doesn't carry); the **<3% extension is untested** (budget). idea1's own range-move test:
  `(range_high - prev_close)/prev_close*100 >= 5.0` by 09:35, using `entry_price` (=range_high, the
  pipeline's own breakout-fill price) as the probe.
* **idea2** (gap vs prior-day HIGH >= 3%) is algebraically a SUBSET of gap-vs-CLOSE >= 3% (prior_high
  >= prior_close always), so it is also built from the wide-seed CSVs — no new minute bars needed.
* **idea10** (gap <= -5%) and **idea11** (yesterday's gap >= 10%, any gap today) are genuinely new
  populations (no existing gap-up build covers them): candidates from `data/cache.db::daily_bars`
  (in-regime) / `data/research/databento/equs_daily_2024H2.parquet` (out-of-regime, point-in-time,
  delisted included, per this cell's data instructions), fed through `study_orb_features.py` with its
  4 loader seams patched (the `research/orb_2023/build_features_2023.py` pattern) to source minute
  bars + SPY from `bars_sip.db`. Missing symbol-days (11,789 in-regime, 6,585 out-of-regime, mostly
  idea10/idea11 names bars_sip.db's existing HOD-break population never needed) were appended via
  `research/bf_zero/backfill_bars_sip.py` through the scratchpad wrapper — append-only, never touched
  `data/cache.db`. Post-backfill availability: in-regime 15,809/15,987 pairs covered (98.9%), out-of-regime
  6,502/6,585 (98.7%) — both clear the 80% rail. All admission fields (today's gap, yesterday's gap,
  price, prior volume) are known at or before the day's open: causal by construction.
* **Validity check**: my from-scratch production reconstruction (filtering the wide CSV to gap>=5%,
  $3-30 and running it through today's pipeline) gives n=482, mean +0.106R, t=3.82 for 2025-01..2026-09
  — matching CLAUDE.md's standing figure (n=473, +0.105R, t=3.31) closely, which validates this cell's
  harness before trusting the new-pool numbers built the same way.
* **Known data flaw, small**: 232/22,606 wide-seed rows (~1%) have gap_pct vs. a re-derived gap_pct
  from `daily_bars` differing by >1pp (unadjusted split/corporate-action class); out_regime/idea2's
  pipeline run crashed on a pre-existing zero-row edge case in `study_orb_pipeline_static_lock.py`
  (every candidate vetoed out, 0 entered) — scored here as n=0, not a build failure of this cell's code.

## Per-pool reads (R in units of $375; t = iid / day-clustered)
| Pool | Window | n | fills/wk | mean R | iid t | dc t | ex-top-5% | MDE | wk P10 | worst wk |
|---|---|---|---|---|---|---|---|---|---|---|
| production | in-regime 2025 | 215 | 4.39 | +0.101 | 2.66 | 2.22 | +0.012 | 0.106 | -0.71R | -2.39R |
| production | in-regime 2026 | 267 | 7.42 | +0.109 | 2.77 | 2.60 | +0.001 | 0.111 | -0.94R | -2.07R |
| production | in-regime FULL | 482 | 5.67 | +0.106 | 3.82 | 3.40 | +0.006 | 0.077 | -0.91R | -2.39R |
| **idea1** | in-regime 2025 | 178 | 3.49 | +0.041 | 1.14 | 0.18 | -0.032 | 0.101 | -0.90R | -1.38R |
| **idea1** | in-regime 2026 | 216 | 5.68 | +0.051 | 1.61 | 1.45 | -0.017 | 0.089 | -1.24R | -2.04R |
| **idea1** | in-regime FULL | 394 | 4.48 | +0.047 | 1.96 | 1.12 | -0.024 | 0.067 | -1.06R | -2.04R |
| **idea1** | out-regime 2024H2 | 40 | 2.11 | +0.141 | 1.45 | 1.58 | +0.033 | 0.272 | -0.50R | -0.70R |
| **idea2** | in-regime 2025 | 2 | 1.00 | +0.846 | 1.70 | 1.70 | +0.349 | 1.391 | +0.45R | +0.35R |
| **idea2** | in-regime 2026 | 11 | 2.20 | -0.073 | -1.09 | -1.15 | -0.121 | 0.187 | -0.43R | -0.50R |
| **idea2** | in-regime FULL | 13 | 1.86 | +0.068 | 0.55 | 0.93 | -0.038 | 0.348 | -0.39R | -0.50R |
| **idea2** | out-regime 2024H2 | 0 | — | — | — | — | — | — | — | — |
| **idea10** | in-regime 2025 | 265 | 5.10 | -0.103 | -4.35 | -3.98 | -0.165 | 0.066 | -1.55R | -2.38R |
| **idea10** | in-regime 2026 | 322 | 8.94 | +0.035 | 1.05 | 0.62 | -0.071 | 0.092 | -1.55R | -2.76R |
| **idea10** | in-regime FULL | 587 | 6.67 | -0.027 | -1.30 | -1.66 | -0.115 | 0.059 | -1.57R | -2.76R |
| **idea10** | out-regime 2024H2 | 75 | 3.00 | +0.069 | 0.58 | 0.54 | -0.114 | 0.333 | -0.77R | -1.30R |
| **idea11** | in-regime 2025 | 258 | 5.06 | -0.029 | -1.14 | -0.29 | -0.092 | 0.070 | -1.24R | -2.42R |
| **idea11** | in-regime 2026 | 229 | 6.19 | +0.041 | 1.08 | 0.84 | -0.064 | 0.106 | -1.41R | -2.83R |
| **idea11** | in-regime FULL | 487 | 5.53 | +0.004 | 0.18 | 0.47 | -0.082 | 0.062 | -1.35R | -2.83R |
| **idea11** | out-regime 2024H2 | 88 | 3.52 | +0.123 | 1.17 | 1.04 | -0.055 | 0.294 | -0.96R | -1.17R |
| production (context) | 2023-01..2024-06 | 106 | 1.80 | +0.015 | 0.30 | 0.47 | -0.077 | 0.139 | -0.52R | -1.55R |

## Union with production (per-pool, independent_1328.py's method: pool rows not already in the
day's production picks, admitted by `_composite` desc up to 8 minus that day's production count)
| Pool / window | raw overlap | union n | union fills/wk (Δ vs prod) | union mean R | union wk P10 | shared worst day |
|---|---|---|---|---|---|---|
| idea1 / in-regime | 0.0% | 873 | 9.70 (+71%) | +0.080 | -1.13R (worse than -0.91R) | 2026-07-08 R=-1.76 (prod alone same day -0.29) |
| idea2 / in-regime | 0.0% | 495 | 5.82 (+3%) | +0.105 | -0.91R (flat) | 2026-09-18 R=-1.49 (prod alone -1.49, no add) |
| idea10 / in-regime | 0.0% | 1,064 | 11.82 (+109%) | +0.032 | -1.73R (worse) | 2025-01-14 R=-3.08 (prod alone -1.37) |
| idea11 / in-regime | 0.0% | 967 | 10.74 (+89%) | +0.055 | -1.38R (worse) | 2026-09-18 R=-2.18 (prod alone -1.49) |
| idea1 / out-regime | 0.0% | 99 | 3.96 (+54%) | +0.075 | -0.62R (worse than -0.51R) | 2024-09-25 R=-0.79 (prod alone +0.00) |
| idea10 / out-regime | 0.0% | 134 | 5.15 (+100%) | +0.052 | -1.28R (worse) | 2024-08-16 R=-1.10 (prod alone -0.60) |
| idea11 / out-regime | 0.0% | 147 | 5.44 (+111%) | +0.086 | -1.11R (worse) | 2024-12-13 R=-0.98 (prod alone -0.21) |

Raw overlap with the production admission band is 0.0% for every pool by construction (each pool's
daily-bar filter explicitly excludes gap_pct>=5 & $3-30 that day) — frequency gain is real, not
double-counted. Cadence bar (scripts/cadence_bar.py's own C1/C4 on each window's actual [lo,hi], since
its CLI's hardcoded TRAIN/VAL/TEST ranges can't express either of this cell's windows): production
in-regime C1 (strong-week gap) already FAILS alone (median 12.0wk, P90 20.8wk vs the 3.0/6.0wk bar) and
every union makes the gap longer or the C4 green-vs-null check flip to FAIL (idea10, idea11) — no pool
improves cadence; several make it worse.

## Pass-bar verdicts (own mean R >= +0.05 & dc t >= 2.0 in-regime, >= 0 out-of-regime, ex-top-5% > 0,
union wk P10 & strong-week gap not worse than production's)
* **idea1 — FAIL.** In-regime dc t 1.12 (<2.0), ex-top-5% -0.024 (<0); 2025/2026 halves both weak
  positive (no sign flip, the best-behaved of the four) but neither clears the bar alone. Union wk P10
  and gap both worse than production's. Out-of-regime leg (+0.141R, n=40) is the single most promising
  number here but can't pass on its own against the in-regime failure.
* **idea2 — FAIL.** n=13 in-regime (2025 n=2, uninterpretable), n=0 out-of-regime (vetoed to nothing).
  Confirms the PREREG's own algebraic point: this population is a strict subset of the gap 3-5%, $3-30
  stratum already closed as "DRY-ONLY, p30 negative out of regime" (idea 5) — adds no usable frequency.
* **idea10 — FAIL, clearest reject.** In-regime mean R negative overall (-0.027) and SIGN-FLIPS across
  halves (2025 -0.103R t -3.98 vs 2026 +0.035R t 0.62) — the kind of era-instability this codebase
  treats as disqualifying on its own. Union drags weekly P10 from -0.91R to -1.73R and turns the C4
  green-vs-null check from pass to fail; the 2025-01-14 shared-worst-day more than doubles that day's
  production loss (-1.37R -> -3.08R) — the "gap-down morning hits every pool" tail the PREREG flagged.
* **idea11 — FAIL.** In-regime mean R ~0 (+0.004, 2025 half negative), dc t 0.47. Out-of-regime is the
  best single number among the fresh pools (+0.123R, n=88, dc t 1.04) but still short of t>=2 and the
  union still worsens production's weekly P10 and C4 check in both windows.

No pool is added to the union. The 2024H2-only out-of-regime build (not the full 2023-01..2024-12) and
idea1's gap>=3% floor are this cell's main open threads if the programme wants to revisit; the idea2
and idea10 findings look closed enough not to need a 2023H1 rebuild.

## Files
`1684_pool_books.csv` (2,331 scored trade rows, tagged window/pool), `1684_reads.csv` (39,001 candidate
rows, all 4 idea flags), `pools_1684_lib.py`/`1684_build.py` (candidates), `1684_fastpath.py` (idea1/2 +
production, in-regime), `1684_features.py`/`1684_pipeline.py` (idea10/11 in-regime, all-4 out-regime),
`1684_score.py`/`1684_finalize.py` (stats/union/cadence), `1684_pools.log` (consolidated run log);
per-pool features/`_true.csv`/pipeline logs under `fastpath/`, `fresh_in_regime/`, `fresh_out_regime/`,
`out_in_regime/`, `out_out_regime/`.
