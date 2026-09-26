# FEATURES_1489 — feature matrix for cells 1,489 (retest-instant BUY) / 1,490 (retest-instant SHORT)

Builder: `research/hod_entry/build_features_1489.py`. Output: `research/hod_entry/features_1489.csv`
(8,973 rows x 88 columns — one row per 1,481-rebuild fill with `status == fill`; TRAIN 3,957 / VAL 5,016;
TEST absent, not fabricated). Label `Y = net_R_prime > 0`, base rate 0.349 (VAL/TRAIN pooled).

**Minute convention** (verified, not assumed): `fill_min` / `retest_minute` are float minutes since ET
midnight of the `day` column. Proof: AAP 2025-07-01 fill_min=605.31 -> bar `2025-07-01T14:05:00+00:00` in
`bars_fills_1478.db` (14:05 UTC = 10:05 ET) has close 49.0273, exactly `features_1478_A.close_j`. sip_cache
filenames key on `int(retest_minute)`, confirmed against `AMN_2025-07-01_676.pkl` (retest_minute=676.0
exactly). Tape `ts` is **nanosecond epoch UTC** (confirmed: ts=1751378755591949416 -> 2025-07-01 14:05:55.59
UTC, the same fill).

**HARD RULE enforced in code**: every tape feature filters `trades`/`quotes` to `ts <= retest_ts` (strictly
`< retest_ts` for the "before the fill print" counts). Every bar feature uses the LAST bar at-or-before a
boundary minute (never nearest-by-distance) so a data gap can never pull in a future bar — this was a bug
caught and fixed during the build (see caveats).

## Group 1 — arm-bar inheritance (56 columns, prefix `arm_`)
All of `features_1478_A.csv`, `features_1478_B.csv` (minus its own decoys) and `features_1478_C.csv`,
plus `model_1478_L3_predictions.hgb_prob_L3`, joined on exact `(day, symbol, fill_min)` — merge verified
100.0% exact match (float diff 0.0) against the population, 0 duplicate day+symbol+level groups. Every one
of these columns already carries its own timestamp proof (`<= bar j`, `j` strictly before `fill_min`) in
`FEATURES_A.md` / `FEATURES_B.md` / `FEATURES_C.md` (this directory) — restated there per-feature rather
than duplicated row-by-row here, since bar j precedes fill_min which precedes retest_ts by construction.
Coverage: 100% (`arm_hgb_prob_L3` 100%, all A/B/C columns 100% except pre-existing NaNs already documented
in those files, e.g. `arm_float_shares` per the 1,478 disclosure). Two columns keep a doubled prefix
(`arm_arm_m`, `arm_arm_minute`) because the source columns were already named `arm_m`/`arm_minute`; they are
redundant with `fill_min` (an identity index, not a signal) and are flagged for the model script to drop or
ignore.

| name | source | timestamp bound | coverage | NaN policy |
|---|---|---|---|---|
| `arm_*` (56 cols) | features_1478_A/B/C.csv, model_1478_L3_predictions.csv | <= bar j (< fill_min < retest_ts); proof in FEATURES_A/B/C.md | 100% (join), per-column NaNs as documented in those files | inherited: NaN where the 1,478 builder found no data (e.g. float snapshot), reported there |

## Group 2 — the break (all causal at fill_min / retest_ts, no tape needed)

| name | source | timestamp bound | coverage | NaN policy |
|---|---|---|---|---|
| `brk_minutes_fill_to_tr` | `ns_to_et_minute(retest_ts) - fill_min` | bound = retest_ts (the fill print itself); min observed −8e-9 (float noise for same-instant), never negative | 100% | none |
| `brk_fill_dist_above_level_bps` | `(fill/level - 1)*1e4` | fill and level are both properties of the base fill, which precedes retest_ts | 100% | none |
| `brk_bar_vol_rel_mean_through_j` | last bar volume at-or-before floor(fill_min) / mean volume of all bars at-or-before floor(fill_min), `bars_fills_1478.db` | bars with `et_minute <= floor(fill_min) <= fill_min < retest_ts` | 100% | none (bars_fills_1478.db has full-day coverage for every fill symbol-day, verified 0/8973 missing) |
| `brk_high_pct_of_level` | max high over bars in `[floor(fill_min), int(retest_minute)]` inclusive, as % of level | window's right edge is the retest MINUTE (retest_ts's own minute), never past it | 100% | none |
| `dip_speed_min_high_to_tr` | `retest_minute_precise − (minute of that max-high bar)` | same window as above | 100% | none |

Caveat: 1-minute bar resolution cannot separate "before the fill print" from "after it, same minute" —
when `brk_minutes_fill_to_tr < 1` (median 0.85 min, 75th pct 1.19 min — most retests happen within the
SAME minute or the next one) the "window" is just the break bar itself, so `brk_high_pct_of_level` may
reflect price action at or slightly after the fill within that single minute. This is a genuine resolution
floor, not a lookahead: nothing past `int(retest_minute)` is ever read.

## Group 3 — the dip, from the tape (`sip_cache_1481/`, fallback `sip_cache_1480/`)

| name | source | timestamp bound | coverage | NaN policy |
|---|---|---|---|---|
| `dip_bar_vol_rel_break_bar` | last bar volume at-or-before `int(retest_minute)` / break-bar volume, `bars_fills_1478.db` | window capped at `int(retest_minute)` | 100% | none |
| `dip_n_prints_leq_level_before_fill` | count of trades with `ts < retest_ts` and `price <= level` | strict < retest_ts | 100% | none (0 is a valid count) |
| `dip_lowest_print_leq_tr_bps` | min(price) of trades with `ts <= retest_ts` and `price <= level`, vs level in bps | <= retest_ts | 58.5% | NaN when zero qualifying prints exist at or before t_r (the fill print's own price already satisfies `<= level` by the 1,481 entry convention `level − $0.01`, so this is near-total; residual NaNs are `bar_tick_disagree`-adjacent rows where the exact tick used by the rebuild's fill logic differs from what the raw tape shows — reported, not imputed) |
| `dip_odd_lot_share_before_tr` | share of `size < 100` among trades `ts < retest_ts, price <= level` | strict < retest_ts | 58.2% | NaN when the "before" count (above) is 0 |
| `dip_mean_size_before_tr` | mean `size` of the same set | strict < retest_ts | 58.2% | same as above |
| `dip_nbbo_spread_bps_at_tr` | `(ask−bid)/mid*1e4` of the last quote with `ts <= retest_ts` | <= retest_ts | 79.5% | NaN when no quote print exists in the tape window at or before t_r (thin-quote symbols; sip_cache window itself is present for 100% of rows, so this is a within-window sparsity, not a missing-file gap) |
| `dip_bid_stepped_down_thru_level_5s` | bool: first bid in `[retest_ts−5s, retest_ts]` was `> level` AND last bid in that window `<= level` | window right edge = retest_ts | 53.2% (of which True 710 / False 4,062) | NaN when fewer than 2 quotes fall in the 5-second window (cannot detect a step with one observation) |

## Group 4 — context

| name | source | timestamp bound | coverage | NaN policy |
|---|---|---|---|---|
| `ctx_spy_ret_fill_to_tr` | SPY close of the last 1-min bar at-or-before `fill_min`, vs the last bar at-or-before `retest_minute_precise`, `data/cache.db` (read-only) `intraday_bars_1min` | both bounds <= their respective boundary; retest-side bound <= retest_ts's own minute | 70.0% (6,283/8,973) | NaN for every row on or after 2026-03-21: `cache.db`'s SPY 1-min series ends 2026-03-20, and `bars_fills_1478.db` carries ZERO SPY rows (verified: it was fetched only for the 9,911 fill symbols, not the benchmark) — the PREREG's own fallback chain (cache.db -> bars_fills_1478.db -> NaN) is exhausted, so this is disclosed NaN, not an error |
| `ctx_n_prior_retests_same_level` | rank of this retest within `(day, symbol, level)`, ordered by `retest_minute` | uses only the population's own key columns, no forward information | 100% | **always 0 in this population** — 0 of 8,973 rows share a `(day, symbol, level)` group with another row, so this feature has zero variance and cannot inform the model; disclosed here, not silently dropped |
| breadth at the retest minute (features_1478_B) | **not implemented as specified** | — | — | `features_1478_B.csv` is one row per FILL at the arm minute only (columns `breadth_count_j`, `breadth_share_j` etc.), not a full per-minute breadth matrix for the universe — there is no cheap source for breadth at the retest minute specifically (recomputing it would mean scanning the whole HOD universe's price at ~8,973 distinct minute instants, out of this task's budget). `arm_breadth_share_j` / `arm_breadth_count_j` (group 1) stand in as the causal proxy — breadth at the arm bar, not at t_r — median gap 0.85 min so the proxy is close in time but is NOT the literal feature the PREREG describes. Flagged for the model script and for a refuter to check. |

## Group 5 — decoys (metadata-only model; PREREG requires VAL AUC <= 0.55 on these alone or VOID)

| name | source | timestamp bound | coverage | NaN policy |
|---|---|---|---|---|
| `decoy_store_served_1438` | features_1478_A.csv (the bar store that served cell 1,438) | n/a — metadata about data provenance, not a market feature | 100% | none |
| `decoy_rth_bar_count_1438` | features_1478_A.csv | n/a | 100% | none |
| `decoy_tick_window_has_bar_j` | features_1478_A.csv | n/a | 100% | none |
| `decoy_window_found_1478C` | features_1478_C.csv (`window_found`) | n/a | 100% | none |
| `decoy_has_prebreak_1478C` | features_1478_C.csv (`has_prebreak`) | n/a | 100% | none |

## Bookkeeping (not features — for 1,490 scoring only)
`c1490_short_net_R` / `c1490_shortable` / `c1490_ssr`, joined from `rebuild_1479_1480.csv` on
`(day, symbol, fill_min)`: coverage **4.5% (406/8,973)** — cell 1,480's short leg was built on a sample,
not the full 9,911-fill population; the 1,490 model must be scored only on the matched subset, and this
gap must be reported alongside any 1,490 number, not filled in.

## Bugs found and fixed during the build (disclosed per the independent-check protocol)
1. Early version selected the break/dip/SPY reference bar by **nearest absolute distance** to the boundary
   minute, which — across a data gap — could silently select a bar strictly AFTER the boundary (a small,
   real lookahead risk for `brk_bar_vol_rel_mean_through_j`, `dip_bar_vol_rel_break_bar`,
   `brk_high_pct_of_level`/`dip_speed_min_high_to_tr` window edges, and `ctx_spy_ret_fill_to_tr`). Fixed to
   "last bar at-or-before the boundary" everywhere before the reported run; the reported `features_1489.csv`
   is from the fixed code (`build_features_1489.py`, current version in the repo).
2. An initial window definition (bars strictly between `fill_min` and `retest_minute_precise`) produced
   39.5% NaN on `brk_high_pct_of_level` purely from 1-minute resolution (same-minute retests have no bar
   strictly between); redefined to the inclusive break-bar-to-retest-bar window (documented above under
   Group 2's caveat) which drops that NaN rate to 0% at the cost of coarser resolution, disclosed.

## Smoke test
`--smoke 200` run first (first 200 rows by day/symbol/level/retest_minute sort, all from 2025-07-01):
100% tape coverage, 0% bars-missing, verified column-by-column before the full 8,973-row run.
