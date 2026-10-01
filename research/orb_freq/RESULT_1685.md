# RESULT 1,685 — ORB admission sub-pools: 20 pools, each its own rules (PREREG_1684.md amendments 1, 1b)

**Verdict: 0/11 scored sub-pools pass the pre-registered bar. Stage 2 (pairs) correctly NOT RUN
(pre-declared: passers only). Union = production alone, unchanged.** 9/20 sub-pools are SCOPED OUT
(stated below, not hidden) because no existing feature build covers their population and a fresh
one didn't fit this cell's budget on top of the other 11 + the owner's two priority reads.

## Scope deviations (read before the numbers)
* **Exit: every scored pool uses the LIVE-RULE exit only**, not a TRAIN-selected exit from the
  3-item menu. The menu's other two exits need the 1,679 walker (687 lines, raw-minute-bar re-walk,
  BarStore/f1668/f1670/f1678 state) re-run per sub-pool — outside this cell's budget. This is a
  stated deviation from the amendment, not a silent one.
* **Band A (gap [2,3)%) × {F1,F2,F3,F4,F5}: SCOPED OUT.** No existing feature build covers the
  general 2-3% gap population (wide-seed's own floor is ≥3%); building one fresh (study_orb_features
  + backfill, as cell 1,684 did for idea10/idea11) is a 6th new minute-bar population. Band A's own
  daily-bar admission FREQUENCY is still reported (cheap, no minute bars needed).
* **F2 (pre-market $ volume) for every band: SCOPED OUT.** No existing feature build carries
  pre-market (04:00-09:30) bars — study_orb_features computes the 09:30-09:35 opening range only.
* **Recovery sub-pools 19/20 (gap≥5%, $3-30, prior-day volume 100K-500K): SCOPED OUT, both legs.**
  Tested the working assumption that the wide-seed CSV carries no volume floor at its own admission
  — **false**: 0/22,606 wide-seed rows have prev_volume<500K, i.e. wide-seed's own historical scan
  already enforced ~500K. No existing source (wide-seed, idea1/2/10/11) covers the 100K-500K slice
  in either window. The two numbers the owner asked for regardless (floor-dropped count, production
  tercile split) do NOT need this population and are reported in full below.
* Bands B/C (3-4%, 4-5%) reuse the ALREADY-BUILT wide-seed CSV (in-regime, price/vol-rejoined here)
  and cell 1,684's `fresh_out_regime/idea1_features.csv` (out-regime) — zero new minute bars. F6
  (any band) reuses cell 1,684's `idea11_features.csv` (both windows). Out-regime bands B/C are
  therefore implicitly volume≥500K (idea1's own floor), not the amendment's 300K — a byproduct of
  reuse, stated plainly.
* Selection chain = feeding each pool through `study_orb_pipeline_static_lock.py` gives it the same
  per-pool ranking/8-slot cap (`_composite` desc, cell 1,328) cell 1,684 relied on — no extra code.
* F1 formula used: `(range_total_volume/avg_daily_volume_20d) / TRAIN(2025)-median-of-that-ratio
  ≥ 3.0` — this cell's own reading of "the 1,665 definition B"; cell 1,665 itself was not re-read
  under this budget. F3 = `range_vwap_distance_pct>0 AND range_close_position≥0.5`. F4 = `open ≥
  0.95×(trailing-252-session high of daily `high`, shifted 1)`, stitched from
  `xnas_daily_2023_2024H1.parquet`+cache.db (in-regime) / +EQUS 2024H2 (out-regime) for full lookback.
  F5 = `(prev_high-prev_low) ≥ 1.5×ATR14`, ATR14 via the SHARED `build_atr14_lookup`
  (`study_orb_pipeline_static_lock.py`, same helper production uses). F6 = prior-day gap≥10%.

## Scored sub-pools — in-regime (2025-01-01..2026-09-18, halves by year) and out-regime (2024H2)
R in units of $375; t = iid / day-clustered. production (for reference): in-regime FULL n=482
meanR=+0.106 dc_t=3.40; out-regime n=59 (this cell's own production-tercile pull; RESULT_1684
reported the 2024H2 union baseline, not a standalone row — consistent n).

| Pool (band+feature) | win | n | fills/wk | meanR | iid_t | dc_t | exTop5% | MDE | wkP10 | worstWk |
|---|---|---|---|---|---|---|---|---|---|---|
| A+F6 (day-2≥10% gapper) | in | 28 | 1.27 | +0.134 | 1.31 | 1.22 | +0.022 | 0.287 | -0.23R | -0.28R |
| A+F6 | out | 10 | 1.43 | +0.218 | 1.04 | 1.04 | +0.076 | 0.586 | -0.19R | -0.30R |
| B+F1 (rel. vol 09:35) | in | 16 | 1.14 | -0.070 | -0.62 | -0.33 | -0.172 | 0.316 | -0.48R | -0.57R |
| B+F1 | out | 3 | 1.00 | -0.101 | -0.85 | -0.85 | -0.207 | 0.331 | -0.26R | -0.30R |
| B+F3 (VWAP+top-half) | in | 222 | 2.77 | -0.022 | -0.77 | -1.46 | -0.102 | 0.081 | -0.87R | -1.13R |
| B+F3 | out | 26 | 1.62 | +0.239 | 1.99 | 1.77 | +0.119 | 0.336 | -0.20R | -0.50R |
| B+F4 (within 5% 52w-hi) | in | 5 | 1.00 | -0.196 | -7.60 | -7.60 | -0.210 | 0.072 | -0.26R | -0.28R |
| B+F4 | out | 3 | 1.00 | -0.006 | -0.11 | -0.11 | -0.062 | 0.157 | -0.06R | -0.06R |
| B+F5 (range≥1.5×ATR14) | in | 59 | 1.40 | +0.010 | 0.19 | 0.15 | -0.052 | 0.150 | -0.42R | -0.61R |
| B+F5 | out | 12 | 1.71 | +0.197 | 0.93 | 1.00 | +0.009 | 0.596 | -0.34R | -0.39R |
| B+F6 | in | 30 | 1.25 | -0.165 | -2.87 | -2.46 | -0.231 | 0.161 | -0.49R | -0.63R |
| B+F6 | out | 3 | 1.00 | +1.025 | 1.75 | 1.75 | +0.481 | 1.639 | +0.26R | +0.11R |
| C+F1 | in | 25 | 1.39 | +0.003 | 0.03 | 0.04 | -0.085 | 0.259 | -0.47R | -1.69R |
| C+F1 | out | 3 | 1.00 | -0.096 | -1.43 | -1.43 | -0.160 | 0.187 | -0.18R | -0.19R |
| C+F3 | in | 181 | 2.59 | +0.030 | 0.82 | 1.05 | -0.043 | 0.101 | -0.79R | -1.69R |
| C+F3 | out | 19 | 1.58 | -0.133 | -2.31 | -3.53 | -0.174 | 0.161 | -0.46R | -0.85R |
| C+F4 | in | 2 | 1.00 | -0.016 | -0.09 | -0.09 | -0.194 | 0.500 | -0.16R | -0.19R |
| C+F4 | out | 2 | 1.00 | -0.160 | -5.73 | -5.73 | -0.188 | 0.078 | -0.18R | -0.19R |
| C+F5 | in | 43 | 1.48 | +0.038 | 0.55 | 0.28 | -0.037 | 0.192 | -0.38R | -0.44R |
| C+F5 | out | 10 | 1.25 | -0.153 | -4.38 | -3.94 | -0.173 | 0.098 | -0.35R | -0.46R |
| C+F6 | in | 23 | 1.05 | +0.006 | 0.08 | -0.22 | -0.052 | 0.182 | -0.36R | -0.38R |
| C+F6 | out | 1 | 1.00 | -0.159 | nan | nan | nan | nan | -0.16R | -0.16R |

Three pools clear in-regime meanR≥+0.03 (the report threshold, below the formal +0.05/dc_t≥2.0
bar): **A+F6** (+0.134, dc_t 1.22, n=28; out-regime +0.218 n=10 — same sign both windows, the
single most consistent pool here, still short of t≥2), **C+F3** (+0.030, dc_t 1.05, n=181; out-regime
**-0.133**, sign-flips), **C+F5** (+0.038, dc_t 0.28, n=43; out-regime -0.153, sign-flips). Every
other pool is flat-to-negative in-regime, several sign-flip across 2025/2026 halves (full halves in
`1685_pool_books.csv`), none reaches dc_t≥2.0 in-regime with the right sign.

## Scoped-out sub-pools (9/20, not pipeline-scored — reason, not a null result)
| Pool | Reason |
|---|---|
| A+F1, A+F2, A+F3, A+F4, A+F5 | Band A (2-3%) general population has no minute-bar feature build |
| B+F2, C+F2 | No pre-market bars in any existing feature build (any band) |
| REC-19 (F1) | Wide-seed already floors prev_volume≥500K at admission (0/22,606 rows <500K) |
| REC-20 (F2) | Same floor gap as REC-19, plus no pre-market data |

Band A's own daily-bar admission frequency (context only, price $3-30 & vol≥300K, same filters as
the scored pools): **25,710 candidate rows in-regime, 6,089 out-regime** — i.e. band A is NOT a
small population; it is the single largest un-scored gap in this program.

## Owner's amendment-1b asks
**Floor-dropped count** (gap≥5%, $3-30, by prior-day volume; daily-bar candidate rows/day, not
fills): in-regime **production(≥500K)=14,540 rows (33.9/day) vs dropped(<500K)=23,052 rows
(53.7/day)** — the floor drops MORE candidates than it keeps. Out-regime: **production=2,754
(21.5/day) vs dropped=13,807 (107.9/day)** — even more lopsided out of regime. Of the dropped rows,
6,660 (in) / 1,513 (out) sit specifically in the 100K-500K recovery band; the rest are <100K.

**Production book's own fills split by prior-day volume tercile** (does the floor track quality
among names it ALREADY admits?): in-regime n=482 — low tercile n=161 meanR=+0.100, mid n=160
meanR=+0.119, high n=161 meanR=+0.098 — **flat, no monotonic relationship**. Out-regime n=59 — low
n=20 meanR=+0.272, mid n=19 meanR=-0.031, high n=20 meanR=**-0.152** — **inverted** (lowest-volume
tercile does best), though n≈20/tercile is thin. Neither window shows volume (within the
already-≥500K population) tracking trade quality; the floor's own selectivity is not obviously
doing quality work, independent of whether the 100K-500K recovery slice itself has edge (unmeasured,
scoped out above).

## Stage 2 (pairs, pre-declared: ONLY for stage-1 passers)
0/11 pass stage 1 (full pass-bar table and FAIL reasons in `1685_subpools.log`; every in-regime
dc_t misses +2.0, several have the wrong sign or exTop5%<0 even where meanR is positive). **Stage 2
NOT RUN** — correct per PREREG, not a shortcut.

## Union
No sub-pool passes → **the union is production alone, unchanged.** (Per-pool hypothetical unions
were still computed for diagnostic purposes in `1685_subpools.log`: adding any of these pools to
production never improves weekly P10 and typically drags the C4 green-vs-null cadence check from
pass to fail out-of-regime — consistent with 1,684's finding that admission-widening pools tend to
make cadence worse, not better.) Raw overlap with production is 0.0% for every pool by construction
(each pool's gap band explicitly excludes gap≥5%) — frequency gain, had any pool passed, would have
been real, not double-counted.

## Files
`1685_subpools.py` (all stages: `daily`/`pools`/`pipeline`/`score`), `1685_subpools.log` (full run
log incl. the AF6/BF6/CF6 in-regime env bug — first attempt pointed at cache.db instead of
bars_sip.db, caught by the pipeline's own >2%-missing-bars gate, rc=1 — and its fix), `1685_reads.csv`
(365,525 daily-bar rows: band membership, floor/recovery/production flags, F4/F6, both windows),
`1685_pool_books.csv` (726 scored trade rows, tagged pool/window), `subpools_1685/` (per-pool
features/true/log, `manifest.csv`).
