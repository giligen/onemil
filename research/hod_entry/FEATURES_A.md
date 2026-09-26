# FEATURES_A — Feature Set A for cells 1,478-1,480 (PREREG_1478.md + the 2026-09-26 amendment)

Builder: `research/hod_entry/build_features_1478_A.py`. Output: `features_1478_A.csv` (9,911 rows,
one per `causal_arming_causal.csv` fill; 41 feature columns + day/symbol/fill_min/split). Every
bar-derived quantity below is computed ONLY from `bars_fills_1478.db` (Alpaca SIP 1-minute bars,
04:00-20:00 ET, fetched fresh per the amendment) using bars strictly before `fill_min` (arm bar j =
the last such RTH bar) — never `bars_sip.db` or `data/cache.db`. Non-bar features reuse the
already-reviewed causal joins in `cell_1445.py` / `cell_1457.py` (daily panel shift(1)/shift(1).
rolling(K) — the signal day itself is excluded from its own trailing terms by construction).

## Item 1 — bar-derived (recomputed from the new store)
| Feature | One-line timestamp proof |
|---|---|
| `range_to_j_pct` | `(max(h[0..j]) - min(l[0..j])) / min(l[0..j])`, bars 0..j only (`m < fill_min`) |
| `dollar_vol_to_j` | `sum(v[0..j]*c[0..j])`, bars 0..j only |
| `bar_density_j` | `len(bars 0..j) / (arm_m - 570 + 1)`, denominator is bar j's own minute |
| `dollar_vol_to_j`, `cum_volume_j` | cumulative sums over bars 0..j only |
| `dist_from_open_pct` | `(close_j - open[bar 0]) / open[bar 0]`, bar 0 = 09:30 bar, both ≤ j |
| `close_j`, `n_bars_j`, `arm_m` | bar j's own close / count / minute — the arm bar itself |
| `arm_index` | count over `i in [0, j-1]` of `h[i+1] >= running_hod(0..i) + 0.01`; `i+1 <= j` always, so only bars 0..j are read |
| `pm_dollar_vol` | bars with `m in [240, 570)` (04:00-09:29 ET), strictly before RTH open ≤ j |
| `rv_j` | `rv_profile(cum_volume_j, adv20, arm_m)` (`trading/hod_break.rv_profile`) — `cum_volume_j` ≤ j, `adv20` is a trailing same-day-of snapshot from `research/bf_zero/universe.csv` (production feature, pre-existing) |

## Item 2 — the stock's own state
| Feature | One-line timestamp proof |
|---|---|
| `consol_vol_slope` | OLS slope of `v[j-K+1..j]` vs bar index, `K=min(5, j+1)` (`HodBreakParams().consol_bars`) |
| `consol_vol_ratio` | `mean(v[j-K+1..j]) / mean(v[0..j])` |
| `higher_lows_count` | `sum(diff(l[j-K+1..j]) > 0)` |
| `level_touches` | count of bars in `0..j` whose `[l,h]` range enters `level * [1-0.002, 1+0.002]` — `level` is the pre-set breakout level from the base book, known at arm time |
| `pullback_depth_pct` | `(max(h[0..j-K]) - min(l[j-K+1..j])) / max(h[0..j-K])`, both windows ≤ j |
| `vwap_dist_pct`, `vwap_slope` | cumulative session VWAP (typical price × volume, `sum/sum`) through bar j and its last-15-bar tail, all ≤ j |
| `halt_proxy` | `any(diff(bar minutes[0..j]) >= 5)` |
| `prev_day_volume`, `prev_day_rvol` | Databento daily panel, `groupby(instrument_id).volume.shift(1)` (prior session) and `shift(2).rolling(20,min_periods=20).mean()` (ADV20 ending the session BEFORE that) — the signal day and its immediate predecessor never enter their own denominator |
| `dow` | `pd.Timestamp(day).dayofweek` — a property of the calendar date, not future data |

## Item 1's non-bar carryover (cell_1445/1457's daily panel — reused, not rebuilt)
| Feature | One-line timestamp proof |
|---|---|
| `float_shares` | `data/cache.db universe.float_shares` — **CURRENT snapshot, not point-in-time** (same disclosed treatment as cell_1445/1457; coverage 50.9%, matches the ~51% documented there) |
| `atr14_pct`, `prior_range_pct` | `cell_1457.build_daily_panel_ext`: True Range from shift(1) prev_close, then `shift(1).rolling(14)` over the 14 PRIOR sessions; `prior_range_pct` from shift(1) high/low |
| `prev_close`, `prev_high`, `high20` | `cell_1445.build_daily_panel`: `shift(1)` and `shift(1).rolling(20)` — the signal day's own high/close never contributes |
| `gap_vs_prior_close_pct` | `(level - prev_close) / prev_close` — `level` (pre-set breakout level, known at arm time) vs the PRIOR day's close |
| `level_vs_prior_high_pct`, `level_vs_high20_pct` | `level` vs `prev_high` / `high20`, both shift(1)-only |
| `half_entry`, `spread_frac_at_fill` | `cell_1445.corrected_cost()`: recovered from the fill's own recorded `cost_R/R/exit_price/exit_half_src` and the (day,symbol) NBBO mean spread — the cost realized AT the fill, reused unchanged from cells 1445/1457 |
| `time_of_day_min` | `fill_min` itself |
| `R_pct` | `R / fill * 100` — R as a % of price (per the "R must exceed the spread" standing rule) |

## Item 5 — symbol persistence (DEVIATION disclosed per the task's own instruction)
`symbol_persistence`: the PREREG defines this from `bars_sip.db`/`cache.db` intraday state at 11:00
ET over the prior 60 SESSIONS — those prior sessions are outside the fill store (`bars_fills_1478.db`
holds only the fill day itself) and are not obtainable from any store this task is scoped to use.
Computed instead from the PIT daily parquet as `share of the prior 60 sessions (shift(1).rolling(60,
min_periods=10)) where (high-open)/open >= 5% AND close > open*1.05` — a whole-day proxy for the
intraday 11:00 rule. **This changes what the feature measures** (whole-day breakout-and-hold vs.
an 11:00-anchored intraday persistence) and should be treated as a distinct, coarser signal, not a
faithful reproduction of the PREREG's rule.

## Decoy columns (metadata-only leak-detection model, amendment §1)
| Feature | One-line timestamp proof / definition |
|---|---|
| `store_served_1438` | `1` if `(symbol,day)` has ZERO rows in `bars_sip.db` (batched per-day COUNT, read-only), else `0` |
| `rth_bar_count_1438` | if `store_served_1438==1`: RTH-window row count in `data/cache.db intraday_bars_1min`; else: RTH-window row count in `bars_sip.db` — both via a UTC-hour string match (`13`-`19`) on the timestamp, a coarse DST-unaware RTH filter adequate ONLY as a decoy control input, never disclosed as a modelling feature |
| `tick_window_has_bar_j` | copied from `features_1478_C.csv.has_prebreak` (joined on day, symbol, fill_min) |

## Coverage (from the build log, `research/hod_entry/build_features_1478_A.log`)
100% on every bar-derived feature (0/9,911 fills had no RTH bar before `fill_min` in the new store —
the completeness gate on the fetch itself passed at 0.41% lost, `fetch_bars_1478.log`). Non-bar:
`float_shares` 50.9% (disclosed snapshot gap, matches cell 1445/1457), `atr14_pct` 98.7%,
`high20`/`level_vs_high20_pct` 98.0%, `prev_day_rvol` 97.9%, `symbol_persistence` 99.4% — every
other column 100%. No column is coverage-VOID by the cell 1445/1457 bar (≥ some materially lower
threshold); `float_shares` is disclosed exactly as those cells disclosed it, not re-litigated here.

## CAUTION for the model-fitting step (NOT resolved by this build — flag for cell 1,478's own PREREG discipline)
`ask_distance_proxy_pct` (`(fill - level) / level`) is included in the CSV for reporting only. It is
**NOT verified causal at the close of arm bar j**: `fill` is the realized fill price, known only once
the resting order actually triggers and fills (at or after bar j+1 under the resting stop-limit spec,
`docs/hod_resting_entry_spec_20260925.md`), not at bar j itself. For the live rule `fill ≈ level ×
(1 + entry_limit_pct)` (a fixed 15 bps constant known at arm time) so most of this column's variance
is likely a known constant, not new information — but any additional chase/slippage embedded in a
particular fill is NOT knowable at bar j. Per the PREREG's "Not allowed" clause ("any feature using
data after bar j"), this column should be excluded from cell 1,478's actual model matrix unless the
independent rebuilder confirms a strictly pre-fill definition (e.g., the resting limit price itself,
not the realized fill) produces materially the same values. Same caution applies transitively to any
feature computed from `fill` (`R_pct`, `half_entry`, `spread_frac_at_fill`) — these were nonetheless
part of the PREREG's own item-1 list reused unchanged from cells 1445/1457, which already accepted
this cost/quality framing; this build did not re-litigate that acceptance, only flags it here for the
independent check the PREREG requires before any model number is fit.
