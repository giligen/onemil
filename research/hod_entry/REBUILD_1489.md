# REBUILD_1489 — independent rebuild of PREREG_1489 (cells 1,489 BUY / 1,490 SHORT)

Built from `PREREG_1489.md` and the protocol section of `PREREG_1478.md` prose ONLY. Did not open
`build_features_1489.py`, `cell_1489.py`, `features_1489.csv`, `FEATURES_1489.md`,
`model_1489_predictions.csv` or `RESULT_1489.md`. Code: `rebuild_1489.py`. Outputs:
`rebuild_1489_predictions.csv` (8,973 rows), `rebuild_1489_report.json`.

## Population
`rebuild_1481_fills.csv`, `status == 'fill'` → **n = 8,973** (TRAIN 3,957 / VAL 5,016; TEST absent —
not fabricated). Y = `net_R_prime > 0`.

## Verdict
**1,489 BUY: FAILS every VAL pass-bar clause.** **1,490 SHORT: UNDECIDABLE on VAL — not a result** (see
coverage gap below); TRAIN-only numbers are reported for completeness but are not a pass/fail readout.

## Per-cell x holdout table

| cell | split | n_kept | kept mean R | day-clustered t | ex-top-5% mean | dropped mean | AUC (HGB) | AUC (LR) | placebo AUC | decoy AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1489 BUY | TRAIN | 1,319 | **+1.316** | 38.16 | +1.312 | −0.828 | 0.956 | — | — | — |
| 1489 BUY | VAL | 1,841 | **−0.049** | **−1.92** | −0.056 | −0.112 | **0.516** | 0.497 | 0.500 | 0.543 |
| 1490 SHORT | TRAIN | 69 | +0.160 | 0.82 | +0.048 | −0.552 | (same model) | — | — | — |
| 1490 SHORT | VAL | **0** | NaN | NaN | NaN | NaN | — | — | — | — |

Thresholds (from TRAIN only, single model scores both books): top-tercile p ≥ 0.4029 (buy), bottom-tercile
p ≤ 0.2237 (short). Best HGB grid point: `max_depth=5, learning_rate=0.03, max_iter=600,
min_samples_leaf=200` (5-fold CV inside TRAIN, seed 1489).

Additional pass-bar checks (VAL, 1489):
- **Paired ΔR vs base (same fills):** base (unfiltered) VAL mean = −0.089 R; kept mean = −0.049 R →
  **ΔR = +0.040**, far below the required ≥ +0.10.
- **Kept cache-only share (`store_served_1438`)** vs population: VAL 18.1% vs pop 17.4% (Δ0.7pp, inside
  the 5pp band); **TRAIN 24.3% vs pop 19.2% (Δ5.1pp, marginally outside)** — a soft flag, moot since VAL
  already fails on AUC/mean/t.
- **Frequency:** VAL kept = 1,841 fills / 22 weeks ≈ 84/wk (population itself already runs far above
  12/4 fills-per-week — the gate never binds on this population).
- **TRAIN-H2 same sign as VAL:** TRAIN kept mean **+1.32** vs VAL kept mean **−0.05** — opposite signs.
  Combined with the TRAIN/VAL AUC gap (0.956 → 0.516) this is the standard signature of the model
  memorizing TRAIN rather than learning a generalizable retest signal.

## Cell 1,490 coverage gap (found, not fabricated)
`rebuild_1479_1480.csv` (the preferred 1,480 short-leg source per the task's own instructions) has the
`shortable`/`ssr`/`short_net_R` fields populated for only **10.3% of TRAIN** fills and **0% of VAL**
fills (0 of 5,016). Every non-null `shortable` row in the whole 8,973-row population falls in TRAIN;
none in VAL. This means cell 1,490 cannot be scored on the holdout the pass bar is keyed to — it is not
that the short book failed, it is that **the input file this task points to as "preferred" has no VAL
short-leg data yet**. Per the project's phrasing rule, this is reported as a coverage void, not a
negative result: 1,490 needs a short-leg rebuild that actually covers the VAL date range before any
pass/fail claim is possible. (`cell_1480_fills.csv`, the fallback source, is also worth checking for VAL
coverage before concluding 1,490 is closed — out of this run's step budget.)

## Known gap: breadth AT the retest instant (disclosed, not fabricated)
PREREG_1489 §4 asks for "breadth at the retest minute (`features_1478_B`'s breadth matrix at that
minute)". Inspection shows `features_1478_B.csv` carries **one** breadth reading per fill, at that
fill's own arm minute (`arm_minute == floor(fill_min)`) — it is not a per-day, per-minute matrix
addressable at an arbitrary later minute t_r. Rebuilding full-universe breadth at t_r from scratch
(bars_sip.db + cache.db across the whole universe, for every retest minute, ~8,973 distinct
symbol-day-minutes) is outside this run's 40-call/step budget. `breadth_at_tr` is therefore **not
computed** (would need to be NaN-filled, not fabricated); only the arm-bar breadth
(`breadth_count_j`/`breadth_share_j`/`breadth_5d`/`spy_ret_5d`, all ≤ bar j) is inherited via group 1.
This is a real hole in the "context at t_r" feature group and should be flagged to whoever reviews the
other (unread) 1,489 implementation, since it may have solved this differently.

## Timestamp-bound / causality table (per feature group)

| group | features | source | bound proof |
|---|---|---|---|
| 1. Arm-bar inheritance | all `features_1478_A/B/C` columns, `hgb_prob_L3`, `lr_prob_L3` | joined on exact `(day,symbol,fill_min)` float match (9,911/9,911 verified) | by PREREG_1478 construction, every column is computed "at the close of arm bar j" — j ≤ fill_min ≤ t_r always (retest is strictly after the base fill) |
| 2. The break | `fill_vs_level_bps`, `minutes_fill_to_tr`, `break_bar_vol_vs_mean_j`, `post_break_high_pct_of_level`, `dip_speed_min` | `rebuild_1481_fills.csv` (fill, level, fill_min, retest_minute) + `bars_fills_1478.db` bars in `[floor(fill_min), floor(retest_minute)]` | SQL range is bounded above by `floor(retest_minute)` — the retest bar itself, never later; verified on LABU 2025-07-01 (fill_min 709.00133 → 15:49:00 UTC bar; retest_minute 722.0 → 16:02:00 UTC bar; retest_ts epoch-ns → 16:02:55 UTC, inside minute 722) |
| 3. The dip (tape) | `n_prints_le_level`, `mean_size_le_level`, `odd_lot_share_le_level`, `lowest_print_bps`, `spread_bps_tr`, `bid_stepped_down` | `sip_cache_1481/SYMBOL_DAY_M.pkl` (fallback `sip_cache_1480/`), `M = floor(retest_minute)` | trades/quotes filtered to `ts <= retest_ts` (epoch **nanoseconds** — verified: `1751378755591949416/1e9` → 2025-07-01, matching the pkl's own day); bid-step feature additionally uses `ts <= retest_ts − 5e9` (5s earlier), still ≤ t_r |
| 3. The dip (bars) | `dip_vol_vs_break_vol`, `n_path_bars` | same `bars_fills_1478.db` range as group 2 | dip bars = strictly after the post-break-high bar, up to and including the retest bar — never past `floor(retest_minute)` |
| 4. Context | `spy_ret_fill_to_tr`, `n_prior_retests_same_level` | `data/cache.db` (read-only) `intraday_bars_1min`, symbol SPY; `rebuild_1489.py`'s own running counter ordered by `retest_ts` | SPY closes taken `<= ` the fill-minute and `<=` the retest-minute UTC timestamps respectively (last-known-value join, never a future bar); prior-retest counter only counts rows with a strictly earlier `retest_ts` in the same (day,symbol,level) group |
| 4. Context (NOT built) | `breadth_at_tr` | — | see "Known gap" above — omitted rather than approximated from the wrong minute |
| 5. Decoys (decoy-only model) | `store_served_1438`, `rth_bar_count_1438`, `tick_window_has_bar_j` | `features_1478_A.csv` | metadata about which BT store served bar j — excluded from the real model, fit alone in the decoy model per PREREG_1478's amendment |

## Diagnostics
- Decoy VAL AUC = **0.543** (< 0.55 VOID gate — passes, but close; the metadata-leak channel is not
  clearly closed on this rebuild and is worth a second look before trusting the real model's 0.516 as
  clean).
- Placebo (label-shuffled) VAL AUC = **0.500** — behaves exactly as it should, so the real model's
  overfitting (train 0.956 → val 0.516) looks like ordinary variance/overfit on 74 features × ~4,000
  TRAIN rows rather than a shuffle-invariant leak.
- Tape/bar/SPY coverage on the full population: `tape_window_found` 100%, `spy_ret_fill_to_tr` available
  100%, `n_path_bars_zero_share` 0% — no coverage excuse for the null; the population's own base rate is
  already negative in both holdouts (TRAIN −0.11 R, VAL −0.09 R), consistent with the programme's other
  1,478/1,479/1,480 findings on this fill population.

## Not allowed — compliance
No feature reads past the retest fill print (`ts <= retest_ts` everywhere tape is touched; bar ranges
capped at `floor(retest_minute)`); thresholds set from TRAIN only; TEST never read (absent from the
input file, not fabricated); the 1481 builder's fill-at-print convention was not used — this rebuild
reads `rebuild_1481_fills.csv`'s own `entry`/`fill`/`retest_ts` columns (the INDEPENDENT rebuild's
limit-price convention) as instructed.

## Bottom line for the reviewer
1,489 (BUY) is a clean FAIL on VAL: AUC at chance (0.516), kept VAL mean negative (−0.05 R, t −1.9),
paired ΔR (+0.04) far under the +0.10 bar, and a TRAIN/VAL sign flip that is the textbook shape of
overfitting rather than signal. 1,490 (SHORT) cannot be read at all on VAL from the pointed-to input
file — that is a data-coverage question to resolve (rebuild the 1,480 short leg over the VAL date range,
or locate why `rebuild_1479_1480.csv` was only ever computed for TRAIN days) before anyone reports a
verdict on the short book.
