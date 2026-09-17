# Stage D1 — pre-registration (written 2026-09-16 23:06 UTC, BEFORE any D1 run)

H4 (PLAN §3) re-run on `B/candidates4.csv` instead of `bf_zero2/candidates3.csv`. D0 answered the same question on
candidates3 and found no ship candidate; D0's own hand-off (LOG 2026-09-16, Stage D0 "NEXT") lists what candidates4
adds and candidates3 could not hold: the signal bar's OHLCV, the level's pre-signal history (n_touches, consol_bars,
consol_vol_ratio), cum $ volume, VWAP distance, close-confirm, prev-day range, asset class, premarket dollars, and
~290K signal rows candidates3 never contained. The pipeline is D0's, file for file (`d0_table -> d0_features ->
d0_parity -> d0_model -> d0_eval trainpred val -> d0_perm -> freeze -> d0_eval test`), re-pointed and extended.

Carried in from Stage C (C/REPORT.md, unchanged here, not re-litigated): next-open fill is the reference fill (H2
rejected), the touch stop is the reference stop (H1 not adopted), hold-to-close is the default exit (H8), base shapes
F6 {} / F8 N=30 / F14 / F11-on-F6, F12 and F13 dropped, and the PRIMARY window is ALL-DAY with the entry minute as a
feature (Stage A's >= 10:00 window did not generalise).

---

## 0.1 Population (fixed, applied identically to every cell)

Source `research/fuckup_audit/C/pop_c.csv` — a LOSSLESS row-subset of `B/candidates4.csv` (C/REPORT.md §0 verified
identity, max abs diff 0.0 over all 107x25 numeric cells). Rows kept:

    next_entry is present (the next-open fill exists)
    next_entry >= 5                      # "price >= 5". NOTE: in candidates4 `price` IS THE LEVEL, not a fill
                                         # (B/build_candidates4.py header). Stage A/C both express the $5 floor on
                                         # the FILL of the model being scored; D1 keeps that so its cells are
                                         # comparable to Stage C's cell for cell. Reported: how many rows the two
                                         # readings differ on.
    next_entry_m <= 841                  # 14:01 ET
    r_pct of the variant being scored >= 1.0
    range_so_far_pct >= 5                # causal membership guarantee, bars strictly before the signal
                                         # (F6 implies it; asserted, not assumed, for F8/F11/F14)
    ALL-DAY: no minimum entry minute. `f_mins_since_open` is a feature.

Families (the base shapes Stage C carried forward): `F6 {}`, `F8 {"N": 30}`, `F14 {"N": 15}`, `F11 {"base": "F6"}`.
Declared EXTRA, used ONLY in the univariate two-leg cell of §0.7 (never in the model cells): `F8 {"N": 5}` (the ORB
entry).

Splits by day, fixed by PLAN §1: TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11.
TEST is read ONCE, after the VAL table and the selection rule are frozen in writing in D1/REPORT.md.

## 0.2 Target

Net R under Stage A/C cost contract (c), computed exactly as `C/score5c.py::net_r` for the `next` fill:

    half = 0.5 * (spread_cc_bps / 100) / max(r_pct_variant, 0.05)
    net  = rr - 0.25 * half - half * {stop: .875, lock: .875, eod: .412, target: .875, none: .875}[why]

Two stop/exit variants ("2 stops" of the 48-cell grid), each with its OWN R and its own population:

  * **primary**   `next_rr_hold` / `next_why_hold` / `next_exit_m_hold`, R = `next_r_pct` — hold-to-close, touch stop.
  * **secondary** `next_rr_2r_stopm1` / `next_why_2r_stopm1` / `next_exit_m_2r_stopm1`, R = `next_r_pct_m1` — the
    stop moved 1% of price below the structural stop, R redefined. DISCLOSED DEVIATION: candidates4 builds the
    stop-1% variant only crossed with the +2R close target (B/build_candidates4.py emits `rr_2r_stopm1`, there is no
    `rr_hold_stopm1`), so the secondary is "stop-1% AND +2R target", not a pure stop swap. It is Stage C's closest
    miss cell (`F6 {}` next-open `2R stop-1%`, TRAIN +0.031 / VAL +0.063) and is carried for that reason.

Secondary outcome reported but never gated: P(target before stop) = the `target` share of the booked exit mix.

## 0.3 Features (every one computed from data at or before the SIGNAL minute; asserted in `d1_features.py`)

`price`, `dist_open_pct`, `vwap_dist_pct`, `range_so_far_pct` etc. of candidates4 are LEVEL-based, i.e. already
causal (unlike candidates3, where they were fill-derived and D0 had to rebuild twins). `r_pct` is still fill-derived,
so the causal twin `f_r_pct_sig = (level - stop)/level*100` is used and the fill-derived one is NOT a feature.

| feature | definition | live engine source |
|---|---|---|
| f_log_price | log10(level) | the running level in `hod_break.py` / `orb_engine` |
| f_pb | price band code | same |
| f_r_pct_sig | (level - stop)/level*100 | level and the consolidation/range low, both known at the bar close |
| f_dist_open_pct | (level/open09:30 - 1)*100 | `scanner/realtime_scanner.py` true 09:30 open |
| f_range_so_far | high-low so far / open, % | streamed bars |
| f_rv_adv | volume so far / ADV20 | `load_adv20_from_daily_bars` + streamed volume |
| f_gap_pct | open vs prev close | daily_bars |
| f_log_adv20 | log10(ADV20) | daily_bars |
| f_prev_day_range_pct | prev day (high-low)/close *100 | daily_bars (the ORB PDR feature) |
| f_mins_since_open | sig_m - 570 | clock |
| f_dow | weekday | clock |
| f_spread_cc_bps | cost-curve median NBBO for (price band x time band) | live: the real NBBO at the signal minute |
| f_sp_over_r | spread / R | same |
| f_sig_o_rel, f_sig_h_rel, f_sig_l_rel, f_sig_c_rel | signal bar o/h/l/c vs level, in % | the closing bar |
| f_sig_range_pct | (h-l)/level*100 | the closing bar |
| f_sig_body | (c-o)/(h-l) | the closing bar |
| f_log_sig_v | log10(1+signal-bar volume) | the closing bar |
| f_sig_dollar | log10(1 + sig_c * sig_v) | the closing bar |
| f_n_touches | prior bars whose high came within 0.2% of the level | the engine's own bar array |
| f_consol_bars | bars since the level was set | same |
| f_consol_vol_ratio | signal-bar volume / consolidation mean volume | same |
| f_log_cum_dollar | log10(1 + cum $ volume so far) | same |
| f_vwap_dist_pct | (level/VWAP - 1)*100 | same |
| f_close_confirm | 1 if the signal bar closed above the level | same |
| f_asset_class | stock / wrapper / unknown, coded | `trading/orb_asset_class.py` (symbol-level, NOT point-in-time — disclosed) |
| f_log_pm_dollar | log10(1 + premarket 04:00-09:29 $ volume) | `D/pm_bars.db` (1-min SIP premarket); live: the premarket aggregate the ORB engine already computes |
| f_pm_missing | 1 where no premarket source covers the key | — |
| f_pm_hi | pm $ > $5,816,688 (the shipped ORB cut) | same |
| f_news_pre | # articles prev-day 15:00 -> 09:30 | Alpaca/Benzinga news, the live ORB prefetch |
| f_news_intraday | # articles 09:30 -> strictly before sig_m | same |
| f_news_missing | 1 where the news pull does not cover the key | — |
| f_spy_gap … f_iwm_vol20, f_regime | `research/fuckup_audit/day_features.csv` (09:30-knowable) | daily_bars + `trading/regime_helpers.py` |
| f_spy_ret, f_iwm_ret, f_spy_sign, f_iwm_sign | index 09:30 open -> the SIGNAL bar's close, % | live: the streamed SPY/IWM minute bars |
| f_breadth, f_n_pop_before | A3's breadth-so-far: share of the day's earlier population candidates trading above their 09:30 open, and how many there were | the scanner's own mover list |
| f_n_fam_before | the family's signal count so far that day | the engine's own state |

DEVIATION FROM THE BRIEF, deliberate and stricter: the intraday index state is read at the **signal** minute's close,
not at the entry minute. The decision is taken at the signal bar's close; the entry minute's index return is one
minute of look-ahead. Same convention as D0.

Missing data: `pm` and `news` coverage is reported and each carries its own `_missing` indicator column; HistGradient-
Boosting handles NaN natively, and the transparent baseline scores a NaN at mid-rank (D0's rule). News is the union of
`D/news_presence.csv` and `E/news_presence_e.csv` as far as they exist when `d1_features.py` runs; coverage is reported
per split.

## 0.4 Models (D0's exact config, no tuning)

* `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`, `max_depth=4, max_iter=200,
  learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0, random_state=0`.
* Walk-forward: train on ALL months strictly before the predicted month; first predicted month **2025-10**; monthly
  refit through 2026-09. Nothing is refit on VAL or TEST rows.
* Transparent baseline: per family, the top-3 features by TRAIN decile monotonicity (|rho| >= 0.6, ranked by decile
  spread), rank-sum, fitted ONCE on months < 2025-10 and frozen forever. This is the thing that could actually be
  shipped; PLAN §3 H4 makes it a hard requirement.
* Tapes, all three run with the identical pipeline: `real`; `rev` = the **Nagel reversed tape** (every target's sign
  flipped); `shuf` = targets permuted within each day.

## 0.5 Selection and book

* S1: per day, top 12 by the predicted value (tie-break symbol), then `trading.hod_break.run_book(12, 4)`.
* S2: per-row gate — reg > 0, clf P > 0.5, baseline score >= the TRAIN median — then `run_book(12, 4)`.
* Comparators: FCFS (no selection, the same book) and RAND12 (12 random candidates a day, mean over 20 seeds).

## 0.6 Declared cells

**48 model cells** = 4 families x (HGB reg + HGB clf + transparent baseline) x (S1 + S2) x (primary + secondary
stop/exit). Each is additionally run on the `rev` and `shuf` tapes (those are gates, not new cells).
Comparators FCFS/RAND12 (4 families x 2 stops x 2 = 16 rows) are controls, not cells.

**5 univariate two-leg cells** (§0.7): the ORB rule `has_news AND pm_dollar_vol > $5,816,688` vs the rest, per family,
hold exit, ALL-DAY — F6 {}, F8 N=30, F14, F11(F6) and the declared extra F8 N=5.

Total declared: **53**. Every additional number in the report is a diagnostic on these cells, and the final cell count
in REPORT.md counts everything actually looked at.

## 0.7 The two-leg ORB rule (the D0b re-run, all-day)

D0b found the combo bucket WORSE after 10:00 on the >=5%-range universe. Here the same rule is measured ALL-DAY and by
time band (09:30-10:00 / 10:00-12:00 / 12:00+), per family, per split, on the hold exit, contract (c): per-trade mean
net R of `combo` vs the rest, Welch t, plus the booked combo-only book. **No adoption**: this cell is reported, and
would only be carried into a later stage if TRAIN diff >= +0.05R with t >= 2 AND the VAL sign agrees.

## 0.8 Gates and mandatory checks (PLAN §1)

* **G1 (TRAIN)**: mean net R > 0, t >= 2.0, >= 5 trades/week. On the walk-forward cells TRAIN can only be scored on
  the PREDICTED TRAIN months (2025-10..2025-12) — a weak, 3-month gate; it is reported as such, exactly as D0 did.
* **G2 (VAL)**: mean net R > 0, t >= 1.0, >= 55% of weeks green; the bar is raised by 1 SE of weekly R per 10 cells
  that passed G1.
* **G3 (TEST)**: read ONCE, after the selection rule is frozen in writing, and reported whatever it says.
* **Reversed-tape gate (mandatory, PLAN §3 H4)**: a cell whose selected book is profitable on the `rev` tape FAILS,
  whatever it does on the real tape.
* **Transparent baseline**: a ship candidate requires the baseline to clear G1/G2 too.
* **Decile calibration** monotone on VAL (rho reported per family x model x period).
* **Permutation importance** and its stability across refits (primary target, real tape).
* **Tail test**: top 1% and top 5% of booked trades removed, winners capped at +3R.
* **Per-month table** for every G2 survivor.
* **Search-adjusted permutation p** across the declared cells (within-day permutation of the realised outcomes, the
  statistic = the max weekly R over all cells).

## 0.9 Decision rule, frozen here

A D1 ship candidate requires ALL of: (1) G1 on the predicted TRAIN months, (2) G2 on VAL, (3) the reversed-tape book
NOT profitable, (4) the transparent baseline clearing G1 and G2 for the same family and stop variant, (5) VAL decile
calibration rho > 0, (6) the cell still positive with the top 5% of trades removed. Only then is TEST read for it.
If nothing satisfies (1)-(6), TEST is read once anyway for the best VAL cell per family and reported as a descriptive
number, clearly labelled as such, and the stage's answer is NO CANDIDATE.

## 0.10 Resources

`nice -n 10`, `ulimit -v 1800000`, ONE python process at a time, runs > 2 min detached with a log and polled.
`pop_c.csv` (310 MB) is read with `usecols` + `chunksize` + `keep_default_na=False`. `data/cache.db` and
`research/lit_review_2026/etf_1min.db` are opened `file:...?mode=ro`. Nothing outside `research/fuckup_audit/D1/`
is written.
