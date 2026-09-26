# 1489/1490 refuter 1: look-ahead and timestamp check (boundary = retest fill print t_r)

Verdict: **DEFECT FOUND.** Several features read data after t_r, so the "nothing after t_r" claim in
FEATURES_1489.md is false. The leak is small. It can only make the model look better on held-out data, never
worse, so the recorded VAL AUC 0.515 is an upper bound. The FAIL verdict still stands and is if anything
stronger. What changes: the TRAIN numbers, the importance ranking and "the AUC on record" are not clean
measurements.

Scripts: `1489_refuter_1_chk1.py`, `1489_refuter_1_chk2.py`, `1489_refuter_1_chk3.py` (this directory). All run
read-only on bars_fills_1478.db and sip_cache_1481.

## Facts measured
- `retest_minute` M is an integer bar label. The retest print t_r falls between 55 s and 120 s after M:00
  (median 58.6 s). In **57.4 % of rows (5,154 of 8,973; TRAIN 55.7 %, VAL 58.8 %)** t_r is inside bar M, and bar M
  runs on past t_r by a median 3.9 s (at most 5.0 s).
- Arm bar j is defined as `m < fill_min`, and fill_min is a fractional minute. So bar j is **the fill's own
  minute bar**: `arm_m == floor(fill_min)` in 98.4 % of rows, and `arm_m == retest_minute` in 80.3 %.
- Bars are labelled by the start of the minute. `arm_close_j` equals the last tape print of minute M in 39 % of
  sampled rows. In **23 %** of sampled rows that print is timestamped **after t_r**. So bar j's close is a price
  printed after the fill.
- In a sample of 600 same-minute rows, 51 % have prints after t_r inside bar M. Bar M's LOW is set by one of those
  prints in 16 % of rows and its HIGH in 12 %. The post-t_r share of bar volume is median 0.03 %, mean 3.4 %.

## Features by permutation importance
| feature | causal at t_r? | defect |
|---|---|---|
| arm_range_to_j_pct (#1) | NO | uses bar M's high and low, which can be set up to 5 s after t_r (recomputed: identical to the feature only when bar M is included) |
| brk_high_pct_of_level (#2) | NO | the window runs through the whole bar `retest_minute_int` |
| dip_speed_min_high_to_tr (#3) | NO | its reference point is the minute of the maximum-high bar in that same window |
| ctx_spy_ret_fill_to_tr (#4) | NO, **every non-NaN row** | c1 is the close of the SPY bar containing t_r ("at-or-before retest_minute_precise"), which prints up to 60 s after t_r. It is non-zero in 42 % of same-minute rows, because t_r is in M+1 there |
| arm_* bar-j features: close_j, running high/low, dollar_vol_to_j, vwap, rv (#5 atr14 and #7 spy_ret_open_to_j included if they use bar j) | NO where t_r < M+60 s | bar j is the fill-minute bar, used whole |
| dip_bar_vol_rel_break_bar | NO | the whole volume of the retest bar (always 1.0 in same-minute rows) |
| arm_float_shares, arm_sic2, arm_spread_bps_at_arm, arm_trigger_print_size | not verified here | static, or taken at the arm instant; outside the bar-j defect |
| tape group (dip_n_prints, lowest print, odd-lot share, mean size, NBBO spread, bid step 5 s) | YES | filters on `ts < / <= retest_ts`; the last quote at or before t_r is used. The tape window starts at about M+55 s, so the counts are truncated, not leaked |
| ctx_n_prior_retests_same_level | YES | constant 0 |
| breadth | proxy at bar j | inherits the bar-j defect |

## Label
- The label `net_R_prime` is taken directly from rebuild_1481_fills.csv. That walk starts at the retest print: the
  same-minute tape uses `ts > entry_ts`, then bars with `m > m_retest`. `exit_m >= retest_minute` holds in 100 % of
  rows, and entry equals level − $0.01 in 100 % of rows. OK.
- Minor: when t_r falls in M+1 and the tape does not resolve the trade, the bar path starts at M+1 and uses that bar
  whole. That can include prints before the entry. It only matters for the target, since prints before t_r are
  above the limit, so it is negligible.
- 1,490: the 1,480 short searches from the END of the break bar (`window_ns(m_break)[1]`), so it starts at the same
  instant or later. OK. It cannot be scored on VAL in any case (0 of 5,016 rows matched).

## Does the leak change the verdict?
The leak is at most 5 s of the stock's bar and at most 60 s of SPY. In the sampled rows, Y does not differ between
rows whose bar-M low was set after t_r and the rest (0.379 vs 0.372). A leak can only lift the held-out AUC, and
the held-out AUC is at chance (0.515). The FAIL on 1,489 stands under this check.

Required before this cell, or any successor (for example a live "retest score" gate), is used:
- define bar j as `m < floor(fill_min)`, or cut it at t_r;
- cap every bar window at `floor(t_r) − 1`;
- use the SPY bar that ENDS at or before t_r.
