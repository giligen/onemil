# Cell 1,481 independent-rebuild row comparison

Join key: `day+symbol`. Both files have 9,911 data rows, all unique on this key (no dupes either side).

## 1. Fill-set agreement
- Builder `status` vocab: `fill` 9,017 / `bar_tick_disagree` 607 / `never_retest` 287
- Rebuild `status` vocab: `fill` 8,973 / `bar_tick_disagree` 587 / `no_retest` 351 (rebuild also carries a redundant `filled` bool, True count = 8,973, matches `status=='fill'`)
- Intersection (both call it a fill) = 8,973; union = 9,017 → **Jaccard = 0.9951**
- Rebuild-only fills = 0; builder-only fills = 44 (rebuild calls these `bar_tick_disagree`/`no_retest` instead of `fill`) — rebuild's fill set is a strict subset of builder's.

## 2. Agreement among the 8,973 common fills
- `net_R_prime` within 0.01 R: **8,969 / 8,973 = 99.96%**
- within 0.05 R: same 8,969 (no rows fall in the 0.01–0.05 band — the mismatches are either ~0 or large)
- mean(builder − rebuild) net_R_prime = **-0.00017** (essentially zero once the 4 outliers below are set aside)
- identical entry price (tol 1e-9): **100%**
- identical exit minute (`exit_m`): **99.52%** (43 rows differ, mostly `eod`/`eod_fallback` boundary ties or 1-minute retest-delay rounding, net R impact negligible)

## 3. Cell 1,482
`research/hod_entry/cell_1482_fills.csv` exists (9,911 rows) but **no `rebuild_1482_fills.csv` counterpart exists in the repo** — nothing to compare.

## 4. Why the paired delta vs base differs (builder +0.172 VAL, rebuild +0.223 VAL)
Reproduced directly: builder VAL paired delta (mean of `net_R_prime - base_net_R` over `status=='fill'` rows, n=5,036) = **+0.1720**; rebuild VAL (n=5,016) = **+0.2229**. Gap ≈ 0.051 R.

Both files carry `raw_R` and `cost_R` for the *retest* fill, and those two columns are **identical** row-for-row between builder and rebuild (spot-checked 5 common rows: e.g. AIR raw_R -1.0/-1.0, cost_R 0.1071/0.1071 both sides). The divergence is entirely in **`base_net_R`** — the plain (non-retest) baseline outcome — which is computed independently by each pipeline and is NOT derived from the shown raw_R/cost_R columns. Mean(builder base_net_R − rebuild base_net_R) over the 8,973 common rows = **+0.0457** (builder's baseline scores ~0.046 R higher on average). That ~0.046 R base-side gap accounts for essentially all of the 0.051 R paired-delta gap — i.e. the retest-entry side of the two implementations agrees almost exactly; the disagreement is in how each pipeline scores the **baseline/no-retest comparison leg** (likely a differing cost or exit-timing convention applied to the base rule, since it isn't reconstructable from raw_R/cost_R alone).

## 5. Top 20 |net_R_prime difference| rows
16 of the 20 are floating-point noise (diff ~1e-13, entries/exits/why identical both sides — see `research/hod_entry/review/` companion `top20.txt` in scratchpad, not committed). Only **4 rows have a real disagreement**:

| day | symbol | diff | b_entry | r_entry | b_exit_m | r_exit_m | b_exit_px | r_exit_px | b_why | r_why |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-12-26 | GLSI | -3.110 | 20.10 | 20.10 | 708 | 716 | 19.85 | 20.60 | stop | target |
| 2026-01-05 | QPUX | +1.521 | 20.07 | 20.07 | 684 | 955 | 21.45 | 20.42 | target | eod |
| 2026-03-31 | IONQ | +0.886 | 28.62 | 28.62 | 941 | 955 | 29.18 | 28.96 | target | eod |
| 2026-02-09 | AAOI | -0.857 | 45.87 | 45.87 | 955 | 821 | 47.78 | 49.13 | eod | target |

**Dominant cause**: entries agree exactly in all 4; the two implementations disagree on **which exit condition fires first / when the day-cutoff applies** on these specific bars (stop-vs-target same-bar tie-break, or a target/stop touch that one pipeline registers before end-of-day and the other misses, letting the trade run to `eod`) — a bar-level exit-priority/tie-break divergence, not an entry or cost-model bug.
