#!/usr/bin/env python3
"""Build the feature matrix for PREREG_1489/1490 (the retest-instant classifier).

Population: fills of `rebuild_1481_fills.csv` with status == 'fill' (the independent,
limit-price-convention rebuild of the 1,481 retest long). One row per base-fill /
retest event (day, symbol) -> (level, base fill, retest_ts, net_R_prime, ...).

HARD RULE: nothing after the retest fill print (`retest_ts`, nanosecond epoch UTC)
enters any feature. Every tape-derived feature filters trades/quotes to ts <= retest_ts
(strictly < retest_ts for the "before the fill print" counts), and every bar-derived
feature uses only bars at or before the retest minute.

Sources
-------
- research/hod_entry/rebuild_1481_fills.csv   population + label + retest_ts (ns epoch, UTC)
- research/hod_entry/sip_cache_1481/SYMBOL_DAY_M.pkl (fallback sip_cache_1480/, same key)
      M = int(retest_minute); pkl = (trades_df[ts,price,size], quotes_df[ts,bid,ask]); ts = ns epoch UTC
- research/hod_entry/bars_fills_1478.db  table bars(symbol,day,t,o,h,l,c,v); t = ISO8601 UTC, 1-min, 04:00-20:00 ET
- research/hod_entry/features_1478_A.csv / _B.csv / _C.csv  arm-bar (bar j) features, joined on day+symbol
- research/hod_entry/model_1478_L3_predictions.csv  column hgb_prob_L3, joined on day+symbol
- data/cache.db (READ-ONLY, sqlite URI ?mode=ro)  table intraday_bars_1min, symbol='SPY', for the SPY-return context feature
- research/hod_entry/rebuild_1479_1480.csv  cell 1,480 short leg (short_net_R et al.), joined on day+symbol, for 1,490 bookkeeping

Minute-of-day convention (verified against bars_fills_1478.db, see FEATURES_1489.md):
`fill_min` / `retest_minute` are FLOAT MINUTES SINCE ET MIDNIGHT of the `day` column (NOT UTC midnight,
NOT a bar row-index). E.g. AAP 2025-07-01 fill_min=605.31 -> ET 10:05:1x -> UTC bar 2025-07-01T14:05:00Z,
whose close (49.0273) matches features_1478_A's close_j exactly. sip_cache pkl filenames key on
`int(retest_minute)`, confirmed against AMN_2025-07-01_676.pkl (retest_minute=676.0 exactly).

Usage: python3 build_features_1489.py [--smoke 200] [--out features_1489.csv]
"""
import argparse
import glob
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REBUILD_1481 = os.path.join(HERE, "rebuild_1481_fills.csv")
SIP_1481 = os.path.join(HERE, "sip_cache_1481")
SIP_1480 = os.path.join(HERE, "sip_cache_1480")
BARS_DB = os.path.join(HERE, "bars_fills_1478.db")
FEAT_A = os.path.join(HERE, "features_1478_A.csv")
FEAT_B = os.path.join(HERE, "features_1478_B.csv")
FEAT_C = os.path.join(HERE, "features_1478_C.csv")
MODEL_L3 = os.path.join(HERE, "model_1478_L3_predictions.csv")
CACHE_DB = os.path.join(REPO, "data", "cache.db")
CELL_1480 = os.path.join(HERE, "rebuild_1479_1480.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def midnight_et(day_str):
    y, m, d = (int(x) for x in day_str.split("-"))
    return datetime(y, m, d, tzinfo=ET)


def et_minute_to_utc(day_str, minute_float):
    """ET-midnight-relative minute (float) -> tz-aware UTC datetime."""
    return (midnight_et(day_str) + timedelta(minutes=float(minute_float))).astimezone(UTC)


def ns_to_et_minute(ts_ns, day_str):
    """Nanosecond epoch UTC timestamp -> float minutes since ET midnight of `day_str`."""
    dt_utc = datetime.utcfromtimestamp(ts_ns / 1e9).replace(tzinfo=UTC)
    dt_et = dt_utc.astimezone(ET)
    return (dt_et - midnight_et(day_str)).total_seconds() / 60.0


def load_bars_cache(conn, symbol, day):
    """Return a DataFrame of 1-min bars for (symbol, day) with an et_minute column, sorted."""
    df = pd.read_sql_query(
        "select t,o,h,l,c,v from bars where symbol=? and day=? order by t",
        conn, params=(symbol, day),
    )
    if df.empty:
        return df
    dt_utc = pd.to_datetime(df["t"], utc=True)
    dt_et = dt_utc.dt.tz_convert(ET)
    mid = midnight_et(day)
    df["et_minute"] = (dt_et - mid).dt.total_seconds() / 60.0
    return df


def load_tape(symbol, day, M):
    """Load (trades, quotes) for the retest minute; primary 1481, fallback 1480. Returns (trades,quotes,source) or (None,None,None)."""
    for tag, root in (("1481", SIP_1481), ("1480", SIP_1480)):
        path = os.path.join(root, f"{symbol}_{day}_{M}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    trades, quotes = pickle.load(fh)
                return trades, quotes, tag
            except Exception as e:
                log(f"WARNING tape load failed {path}: {e}")
    return None, None, None


def build_row_features(row, bars_df, trades, quotes, tape_src, spy_df, prior_retest_count):
    """Compute the causal (<= retest_ts) feature set for one fill row. Returns dict."""
    feats = {}
    day = row["day"]
    symbol = row["symbol"]
    level = row["level"]
    fill_min = row["fill_min"]
    retest_minute_int = int(row["retest_minute"])
    retest_ts = int(row["retest_ts"])
    fill = row["fill"]

    # ---- group 2: the break ----
    retest_minute_precise = ns_to_et_minute(retest_ts, day)
    feats["brk_minutes_fill_to_tr"] = retest_minute_precise - fill_min
    feats["brk_fill_dist_above_level_bps"] = (fill / level - 1.0) * 1e4

    if not bars_df.empty:
        break_bar_minute = np.floor(fill_min)
        through_j = bars_df[bars_df.et_minute <= break_bar_minute + 1e-6]
        # Bar resolution is 1-minute, so the break bar and the retest bar are each identified by
        # their whole ET minute (floor); the window spans both bars inclusive. When the retest
        # falls in the SAME or the immediately adjacent minute as the base fill, this window is
        # exactly {break bar} or {break bar, next bar} — the only resolution available below
        # tick data. Nothing here uses bars after the retest bar (retest_minute_int).
        window = bars_df[(bars_df.et_minute >= break_bar_minute - 1e-6) & (bars_df.et_minute <= retest_minute_int + 1e-6)]
        # Causal by construction: the break bar is the LAST bar at-or-before fill_min (never the
        # nearest by absolute distance, which could silently pick a bar AFTER fill_min across a
        # data gap). Likewise the dip bar is the last bar at-or-before the retest bar boundary
        # (the tail of `window`, itself capped at retest_minute_int).
        break_bar_vol = float(through_j["v"].iloc[-1]) if not through_j.empty else np.nan
        mean_vol_through_j = float(through_j["v"].mean()) if not through_j.empty else np.nan
        feats["brk_bar_vol_rel_mean_through_j"] = (
            break_bar_vol / mean_vol_through_j if mean_vol_through_j not in (0, np.nan) and not pd.isna(mean_vol_through_j) else np.nan
        )
        dip_bar_vol = float(window["v"].iloc[-1]) if not window.empty else np.nan
        feats["dip_bar_vol_rel_break_bar"] = (
            dip_bar_vol / break_bar_vol if break_bar_vol not in (0, np.nan) and not pd.isna(break_bar_vol) else np.nan
        )
        if not window.empty:
            imax = window["h"].idxmax()
            post_break_high = float(window.loc[imax, "h"])
            post_break_high_minute = float(window.loc[imax, "et_minute"])
            feats["brk_high_pct_of_level"] = (post_break_high / level - 1.0) * 100.0
            feats["dip_speed_min_high_to_tr"] = retest_minute_precise - post_break_high_minute
        else:
            feats["brk_high_pct_of_level"] = np.nan
            feats["dip_speed_min_high_to_tr"] = np.nan
    else:
        feats["brk_bar_vol_rel_mean_through_j"] = np.nan
        feats["dip_bar_vol_rel_break_bar"] = np.nan
        feats["brk_high_pct_of_level"] = np.nan
        feats["dip_speed_min_high_to_tr"] = np.nan

    # ---- group 3: the dip, from the tape (strictly ts < retest_ts for "before the fill print") ----
    feats["tape_coverage"] = tape_src is not None
    feats["tape_source"] = tape_src if tape_src is not None else ""
    if trades is not None and not trades.empty:
        before = trades[trades.ts < retest_ts]
        leq_level_before = before[before.price <= level]
        feats["dip_n_prints_leq_level_before_fill"] = len(leq_level_before)
        at_or_before = trades[trades.ts <= retest_ts]
        leq_level_atbefore = at_or_before[at_or_before.price <= level]
        feats["dip_lowest_print_leq_tr_bps"] = (
            (leq_level_atbefore.price.min() / level - 1.0) * 1e4 if not leq_level_atbefore.empty else np.nan
        )
        if not leq_level_before.empty:
            feats["dip_odd_lot_share_before_tr"] = float((leq_level_before["size"] < 100).mean())
            feats["dip_mean_size_before_tr"] = float(leq_level_before["size"].mean())
        else:
            feats["dip_odd_lot_share_before_tr"] = np.nan
            feats["dip_mean_size_before_tr"] = np.nan
    else:
        feats["dip_n_prints_leq_level_before_fill"] = np.nan
        feats["dip_lowest_print_leq_tr_bps"] = np.nan
        feats["dip_odd_lot_share_before_tr"] = np.nan
        feats["dip_mean_size_before_tr"] = np.nan

    if quotes is not None and not quotes.empty:
        q_before = quotes[quotes.ts <= retest_ts]
        if not q_before.empty:
            last_q = q_before.iloc[-1]
            mid = (last_q.bid + last_q.ask) / 2.0
            feats["dip_nbbo_spread_bps_at_tr"] = (last_q.ask - last_q.bid) / mid * 1e4 if mid else np.nan
        else:
            feats["dip_nbbo_spread_bps_at_tr"] = np.nan
        window_5s = q_before[q_before.ts >= retest_ts - 5_000_000_000]
        if len(window_5s) >= 2:
            first_bid = window_5s.iloc[0].bid
            last_bid = window_5s.iloc[-1].bid
            feats["dip_bid_stepped_down_thru_level_5s"] = bool(first_bid > level and last_bid <= level)
        else:
            feats["dip_bid_stepped_down_thru_level_5s"] = np.nan
    else:
        feats["dip_nbbo_spread_bps_at_tr"] = np.nan
        feats["dip_bid_stepped_down_thru_level_5s"] = np.nan

    # ---- group 4: context ----
    if spy_df is not None and not spy_df.empty:
        # Causal: last SPY bar AT OR BEFORE each boundary (never nearest-by-distance, which could
        # cross into a future bar over a data gap) — same convention as the break/dip bars above.
        spy_thru_fill = spy_df[spy_df.et_minute <= fill_min + 1e-6]
        spy_thru_tr = spy_df[spy_df.et_minute <= retest_minute_precise + 1e-6]
        if not spy_thru_fill.empty and not spy_thru_tr.empty:
            c0 = float(spy_thru_fill["c"].iloc[-1])
            c1 = float(spy_thru_tr["c"].iloc[-1])
            feats["ctx_spy_ret_fill_to_tr"] = c1 / c0 - 1.0 if c0 else np.nan
        else:
            feats["ctx_spy_ret_fill_to_tr"] = np.nan
    else:
        feats["ctx_spy_ret_fill_to_tr"] = np.nan

    feats["ctx_n_prior_retests_same_level"] = prior_retest_count

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run on first N rows only")
    ap.add_argument("--out", default=os.path.join(HERE, "features_1489.csv"))
    args = ap.parse_args()

    log("loading rebuild_1481_fills.csv (population)")
    pop = pd.read_csv(REBUILD_1481)
    pop = pop[pop.status == "fill"].reset_index(drop=True)
    log(f"population = {len(pop)} fill rows")

    # prior-retest count within (day,symbol,level), by retest order
    pop = pop.sort_values(["day", "symbol", "level", "retest_minute"]).reset_index(drop=True)
    pop["ctx_n_prior_retests_same_level"] = pop.groupby(["day", "symbol", "level"]).cumcount()

    if args.smoke:
        pop = pop.head(args.smoke).copy()
        log(f"SMOKE MODE: {len(pop)} rows")

    log("loading features_1478_A/B/C.csv and model_1478_L3_predictions.csv")
    A = pd.read_csv(FEAT_A)
    B = pd.read_csv(FEAT_B)
    C = pd.read_csv(FEAT_C)
    L3 = pd.read_csv(MODEL_L3)
    # store_served_1438 / rth_bar_count_1438 / tick_window_has_bar_j are metadata-only decoys
    # (PREREG group 5), sourced from features_1478_A per the task spec ("store_served_1438 from
    # features_1478_A if present") — pulled out BEFORE the blanket arm_ prefix so they land under
    # decoy_ only, never masquerading as a genuine arm-bar feature.
    DECOY_COLS_A = ["store_served_1438", "rth_bar_count_1438", "tick_window_has_bar_j"]
    decoy_from_A = A[["day", "symbol", "fill_min"] + [c for c in DECOY_COLS_A if c in A.columns]].copy()
    decoy_from_A = decoy_from_A.rename(columns={c: f"decoy_{c}" for c in DECOY_COLS_A if c in A.columns})
    A_feat = A.drop(columns=[c for c in DECOY_COLS_A if c in A.columns])
    DECOY_COLS_C = ["window_found", "has_prebreak"]
    decoy_from_C = C[["day", "symbol", "fill_min"] + [c for c in DECOY_COLS_C if c in C.columns]].copy()
    decoy_from_C = decoy_from_C.rename(columns={"window_found": "decoy_window_found_1478C", "has_prebreak": "decoy_has_prebreak_1478C"})
    C_feat = C.drop(columns=[c for c in DECOY_COLS_C if c in C.columns])

    A_pref = A_feat.add_prefix("arm_").rename(columns={"arm_day": "day", "arm_symbol": "symbol", "arm_fill_min": "fill_min", "arm_split": "split"})
    B_pref = B.add_prefix("arm_").rename(columns={"arm_day": "day", "arm_symbol": "symbol", "arm_fill_min": "fill_min", "arm_split": "split"})
    C_pref = C_feat.add_prefix("arm_").rename(columns={"arm_day": "day", "arm_symbol": "symbol", "arm_fill_min": "fill_min"})
    L3_keep = L3[["day", "symbol", "fill_min", "hgb_prob_L3"]].rename(columns={"hgb_prob_L3": "arm_hgb_prob_L3"})

    merged = pop.merge(A_pref, on=["day", "symbol", "fill_min"], how="left", suffixes=("", "_Adup"))
    merged = merged.merge(B_pref.drop(columns=["split"], errors="ignore"), on=["day", "symbol", "fill_min"], how="left")
    merged = merged.merge(C_pref, on=["day", "symbol", "fill_min"], how="left")
    merged = merged.merge(L3_keep, on=["day", "symbol", "fill_min"], how="left")
    merged = merged.merge(decoy_from_A, on=["day", "symbol", "fill_min"], how="left")
    merged = merged.merge(decoy_from_C, on=["day", "symbol", "fill_min"], how="left")

    # cell 1,480 short leg, joined for 1,490 bookkeeping (not a feature, kept for downstream scoring only)
    if os.path.exists(CELL_1480):
        short = pd.read_csv(CELL_1480)
        short_keep = short[["day", "symbol", "fill_min", "short_net_R", "shortable", "ssr"]].rename(
            columns={"short_net_R": "c1490_short_net_R", "shortable": "c1490_shortable", "ssr": "c1490_ssr"}
        )
        merged = merged.merge(short_keep, on=["day", "symbol", "fill_min"], how="left")
        log(f"joined 1,480 short leg: {merged['c1490_short_net_R'].notna().sum()}/{len(merged)} matched")
    else:
        log("WARNING rebuild_1479_1480.csv not found; c1490_* columns absent")

    log("opening bars_fills_1478.db (read-only) and data/cache.db (read-only, SPY only)")
    bars_conn = sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True)
    cache_conn = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)

    spy_cache = {}
    bars_cache = {}
    missing_tape = 0
    missing_bars = 0
    rows_out = []
    n = len(merged)
    for i, row in merged.iterrows():
        day = row["day"]
        symbol = row["symbol"]
        key = (symbol, day)
        if key not in bars_cache:
            bars_cache[key] = load_bars_cache(bars_conn, symbol, day)
        bars_df = bars_cache[key]
        if bars_df.empty:
            missing_bars += 1

        # SPY bars keyed only by day (symbol fixed = SPY), loaded from cache.db (read-only)
        if day not in spy_cache:
            spy_df = pd.read_sql_query(
                "select timestamp as t, open as o, high as h, low as l, close as c, volume as v "
                "from intraday_bars_1min where symbol='SPY' and bar_date=?",
                cache_conn, params=(day,),
            )
            if not spy_df.empty:
                dt_utc = pd.to_datetime(spy_df["t"], utc=True)
                dt_et = dt_utc.dt.tz_convert(ET)
                mid = midnight_et(day)
                spy_df["et_minute"] = (dt_et - mid).dt.total_seconds() / 60.0
            spy_cache[day] = spy_df
        spy_df = spy_cache[day]

        M = int(row["retest_minute"])
        trades, quotes, tape_src = load_tape(symbol, day, M)
        if tape_src is None:
            missing_tape += 1

        feats = build_row_features(row, bars_df, trades, quotes, tape_src, spy_df, row["ctx_n_prior_retests_same_level"])
        out = {
            "day": day, "symbol": symbol, "split": row["split"], "retest_ts": row["retest_ts"],
            "Y": int(row["net_R_prime"] > 0), "net_R_prime": row["net_R_prime"], "base_net_R": row["base_net_R"],
            "store_served_1438": row.get("decoy_store_served_1438", np.nan),
        }
        out.update(feats)
        # carry arm_* / decoy_* / c1490_* columns already computed by the merge
        for col in merged.columns:
            if col.startswith("arm_") or col.startswith("decoy_") or col.startswith("c1490_"):
                out[col] = row[col]
        rows_out.append(out)

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            log(f"processed {i+1}/{n} rows (tape missing so far: {missing_tape}, bars missing: {missing_bars})")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out, index=False)
    cov = 1.0 - missing_tape / max(n, 1)
    log(f"DONE: wrote {len(out_df)} rows x {len(out_df.columns)} cols to {args.out}")
    log(f"tape coverage: {cov:.4f} ({n - missing_tape}/{n}); bars missing for {missing_bars}/{n} (symbol,day) keys")
    print(f"ROWS={len(out_df)} COLS={len(out_df.columns)} TAPE_COVERAGE={cov:.4f} MISSING_TAPE={missing_tape} MISSING_BARS={missing_bars}")


if __name__ == "__main__":
    main()
