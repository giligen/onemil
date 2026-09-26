"""
Independent rebuild of cells 1,489 (retest-instant BUY classifier) and 1,490 (retest-instant SHORT
book) from PREREG_1489.md prose ONLY. This script does not read build_features_1489.py, cell_1489.py,
features_1489.csv, FEATURES_1489.md, model_1489_predictions.csv or RESULT_1489.md.

Population: rebuild_1481_fills.csv rows with status == 'fill' (the independent 1,481 rebuild, correct
limit-price convention). Label Y = net_R_prime > 0. Holdouts present: TRAIN (TRAIN-H2 of the programme)
and VAL only -- TEST is sealed and is never fabricated here.

Feature groups (each with a timestamp proof, see REBUILD_1489.md):
  1. Arm-bar inheritance    -- every 1478 feature (<= bar j), the 1478-L3 HGB/LR probabilities.
  2. The break itself       -- minutes fill->t_r, fill-vs-level bps, break-bar volume vs the day's
                                mean bar volume through j, post-break high as % of level.
  3. The dip                -- tape at t_r (sip_cache_1481) + minute bars between fill and t_r
                                (bars_fills_1478.db): print counts/sizes below level, dip speed and
                                relative volume, NBBO spread and bid step-down at t_r.
  4. Context at t_r          -- SPY return fill->t_r (cache.db, read-only), count of the day's prior
                                retests of the same level. (Breadth AT t_r is NOT available -- see
                                "Known gap" below; only the arm-bar breadth is inherited in group 1.)
  5. Decoys                 -- store served in 1438, its RTH bar count, tick-window coverage flag --
                                fit ONLY in a metadata-only decoy model, excluded from the real model.

Known gap (disclosed, not fabricated): PREREG_1489 asks for "breadth at the retest minute (features_
1478_B's breadth matrix at that minute)". features_1478_B.csv carries ONE breadth reading per fill, at
that fill's OWN arm minute (arm_minute == floor(fill_min)) -- it is not a per-day, per-minute matrix
addressable at an arbitrary later minute (t_r). Rebuilding full-universe breadth at t_r from scratch
(bars_sip.db + cache.db across the whole universe, for every retest minute) is out of this script's
step budget. breadth_at_tr is therefore left NaN and reported as a coverage gap; the arm-bar breadth
(breadth_count_j / breadth_share_j / breadth_5d / spy_ret_5d) is still inherited via group 1.

Usage:
  python3 rebuild_1489.py --smoke        # first 200 population rows, verbose
  python3 rebuild_1489.py                # full population
"""
import argparse
import sqlite3
import sys
import time
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

BASE = Path(__file__).parent
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
    "max_iter": [200, 600],
    "min_samples_leaf": [50, 200],
}
SEED = 1489


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def minute_et_to_utc_str(day_str, minute_float):
    """day_str='YYYY-MM-DD', minute_float = minutes since midnight US/Eastern (float, floor to the
    minute). Returns the UTC 'YYYY-MM-DDTHH:MM:SS+00:00' string matching bars_fills_1478.db's t column.
    Timestamp proof: verified against LABU 2025-07-01 fill_min=709.00133 (-> 11:49 ET -> 15:49 UTC) and
    retest_minute=722.0 (-> 12:02 ET -> 16:02 UTC), both bars present at exactly those UTC minutes, and
    retest_ts (epoch ns) -> 16:02:55 UTC, inside minute 722. """
    m = int(np.floor(minute_float))
    h, mm = divmod(m, 60)
    d = datetime.strptime(day_str, "%Y-%m-%d")
    local = datetime(d.year, d.month, d.day, h, mm, tzinfo=ET)
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def load_population():
    """rebuild_1481_fills.csv, status == fill only. Y = net_R_prime > 0."""
    fills = pd.read_csv(BASE / "rebuild_1481_fills.csv")
    pop = fills[fills["status"] == "fill"].copy().reset_index(drop=True)
    pop["Y"] = (pop["net_R_prime"] > 0).astype(int)
    assert set(pop["split"].unique()) <= {"TRAIN", "VAL"}, "TEST must stay sealed/absent"
    return pop


def join_1478_features(pop):
    """Group 1: arm-bar inheritance. Join on day+symbol+fill_min (verified exact-float match,
    9911/9911 rows on the full 1481 population)."""
    key = ["day", "symbol", "fill_min"]
    A = pd.read_csv(BASE / "features_1478_A.csv").drop(columns=["split"])
    B = pd.read_csv(BASE / "features_1478_B.csv").drop(columns=["split"])
    C = pd.read_csv(BASE / "features_1478_C.csv")
    L3 = pd.read_csv(BASE / "model_1478_L3_predictions.csv")[
        ["day", "symbol", "fill_min", "hgb_prob_L3", "lr_prob_L3"]
    ]
    df = pop.merge(A, on=key, how="left", validate="one_to_one")
    df = df.merge(B, on=key, how="left", validate="one_to_one")
    df = df.merge(C, on=key, how="left", validate="one_to_one")
    df = df.merge(L3, on=key, how="left", validate="one_to_one")
    return df


def join_1480_short(df):
    """Cell 1,490's short leg outcome, from the preferred rebuild_1479_1480.csv (day+symbol unique)."""
    short = pd.read_csv(BASE / "rebuild_1479_1480.csv")[
        ["day", "symbol", "short_net_R", "shortable", "ssr"]
    ]
    assert short.groupby(["day", "symbol"]).size().max() == 1
    return df.merge(short, on=["day", "symbol"], how="left")


def load_pkl(symbol, day, minute_floor):
    """sip_cache_1481/SYMBOL_DAY_M.pkl, fallback sip_cache_1481 -> sip_cache_1480 with the same key.
    Returns (trades_df, quotes_df) or (None, None) if neither store has the minute cached."""
    for d in ("sip_cache_1481", "sip_cache_1480"):
        p = BASE / d / f"{symbol}_{day}_{minute_floor}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                trades, quotes = pickle.load(f)
            return trades, quotes
    return None, None


def path_bar_features(con, symbol, day, fill_min, retest_minute):
    """Group 2/3 (bar-derived): break-bar volume, post-break high (% of level is computed by the
    caller once `level` is known), post-break-high time, dip-bar volume sum. All bars strictly between
    the break bar (at floor(fill_min)) and the retest bar (at floor(retest_minute)), inclusive of both
    ends -- causal: nothing after t_r (floor(retest_minute)) is touched."""
    t_lo = minute_et_to_utc_str(day, fill_min)
    t_hi = minute_et_to_utc_str(day, retest_minute)
    rows = con.execute(
        "SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? AND t BETWEEN ? AND ? ORDER BY t",
        (symbol, day, t_lo, t_hi),
    ).fetchall()
    if not rows:
        return dict(break_bar_vol=np.nan, post_break_high=np.nan, post_break_high_t=None,
                    dip_vol_sum=np.nan, n_path_bars=0)
    break_bar_vol = rows[0][5]  # bar at floor(fill_min)
    highs = [r[2] for r in rows]
    i_max = int(np.argmax(highs))
    post_break_high = highs[i_max]
    post_break_high_t = rows[i_max][0]
    dip_vol_sum = sum(r[5] for r in rows[i_max + 1:])  # bars strictly after the post-break high
    return dict(break_bar_vol=break_bar_vol, post_break_high=post_break_high,
                post_break_high_t=post_break_high_t, dip_vol_sum=dip_vol_sum, n_path_bars=len(rows))


def tape_features(symbol, day, retest_minute, retest_ts_ns, level):
    """Group 3 (tape-derived), from the retest-minute pkl only (nothing after retest_ts_ns is read)."""
    out = dict(n_prints_le_level=np.nan, mean_size_le_level=np.nan, odd_lot_share_le_level=np.nan,
               lowest_print_bps=np.nan, spread_bps_tr=np.nan, bid_stepped_down=np.nan,
               tape_window_found=0)
    minute_floor = int(np.floor(retest_minute))
    trades, quotes = load_pkl(symbol, day, minute_floor)
    if trades is None:
        return out
    out["tape_window_found"] = 1
    t_r = retest_ts_ns
    tr = trades[trades["ts"] <= t_r]
    below = tr[tr["price"] <= level]
    if len(below) > 0:
        out["n_prints_le_level"] = len(below)
        out["mean_size_le_level"] = below["size"].mean()
        out["odd_lot_share_le_level"] = float((below["size"] < 100).mean())
        lowest = below["price"].min()
        out["lowest_print_bps"] = (level - lowest) / level * 1e4
    q = quotes[quotes["ts"] <= t_r]
    if len(q) > 0:
        last = q.iloc[-1]
        bid, ask = last["bid"], last["ask"]
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            out["spread_bps_tr"] = (ask - bid) / mid * 1e4
        q_before = quotes[quotes["ts"] <= t_r - 5_000_000_000]  # 5s before t_r, same ns clock
        if len(q_before) > 0:
            bid_before = q_before.iloc[-1]["bid"]
            out["bid_stepped_down"] = int(bid_before > level and bid <= level)
    return out


def spy_return(con_cache, day, fill_min, retest_minute):
    """SPY return, base fill -> t_r, from cache.db (read-only) intraday_bars_1min."""
    t_fill = minute_et_to_utc_str(day, fill_min)
    t_r = minute_et_to_utc_str(day, retest_minute)
    row_fill = con_cache.execute(
        "SELECT close FROM intraday_bars_1min WHERE symbol='SPY' AND timestamp<=? ORDER BY timestamp DESC LIMIT 1",
        (t_fill,),
    ).fetchone()
    row_r = con_cache.execute(
        "SELECT close FROM intraday_bars_1min WHERE symbol='SPY' AND timestamp<=? ORDER BY timestamp DESC LIMIT 1",
        (t_r,),
    ).fetchone()
    if row_fill and row_r and row_fill[0]:
        return row_r[0] / row_fill[0] - 1.0
    return np.nan


def build_features(df, verbose_every=200):
    """Rows-at-a-time enrichment (groups 2/3/4). df already carries groups 1 (1478 join) and short leg."""
    con = sqlite3.connect(str(BASE / "bars_fills_1478.db"))
    con_cache = sqlite3.connect(f"file:{(BASE.parent.parent / 'data' / 'cache.db')}?mode=ro", uri=True)

    n = len(df)
    out_rows = []
    prior_retest_count = {}  # (day,symbol,round(level,2)) -> running count, ordered by retest_ts
    df_sorted_idx = df.sort_values("retest_ts").index
    order_rank = {idx: i for i, idx in enumerate(df_sorted_idx)}

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % verbose_every == 0:
            log(f"  feature build {i}/{n}")
        level = row["level"]
        fill_min = row["fill_min"]
        retest_minute = row["retest_minute"]
        retest_ts = row["retest_ts"]
        day, symbol = row["day"], row["symbol"]

        rec = {}
        rec["fill_vs_level_bps"] = (row["fill"] - level) / level * 1e4
        rec["minutes_fill_to_tr"] = retest_minute - fill_min

        pb = path_bar_features(con, symbol, day, fill_min, retest_minute)
        mean_bar_vol_j = (row.get("cum_volume_j", np.nan) / row["n_bars_j"]
                          if row.get("n_bars_j", 0) not in (0, np.nan) else np.nan)
        rec["break_bar_vol_vs_mean_j"] = (pb["break_bar_vol"] / mean_bar_vol_j
                                          if mean_bar_vol_j and mean_bar_vol_j > 0 else np.nan)
        rec["post_break_high_pct_of_level"] = (
            (pb["post_break_high"] - level) / level * 100.0 if pd.notna(pb["post_break_high"]) else np.nan
        )
        if pb["post_break_high_t"] is not None:
            t_high = datetime.fromisoformat(pb["post_break_high_t"])
            t_r_utc = datetime.strptime(minute_et_to_utc_str(day, retest_minute), "%Y-%m-%dT%H:%M:%S%z")
            rec["dip_speed_min"] = (t_r_utc - t_high).total_seconds() / 60.0
        else:
            rec["dip_speed_min"] = np.nan
        rec["dip_vol_vs_break_vol"] = (
            pb["dip_vol_sum"] / pb["break_bar_vol"]
            if pb["break_bar_vol"] and pb["break_bar_vol"] > 0 and pd.notna(pb["dip_vol_sum"]) else np.nan
        )
        rec["n_path_bars"] = pb["n_path_bars"]

        tf = tape_features(symbol, day, retest_minute, retest_ts, level)
        rec.update(tf)

        rec["spy_ret_fill_to_tr"] = spy_return(con_cache, day, fill_min, retest_minute)

        key = (day, symbol, round(level, 2))
        rec["n_prior_retests_same_level"] = prior_retest_count.get(key, 0)
        prior_retest_count[key] = prior_retest_count.get(key, 0) + 1

        out_rows.append(rec)

    con.close()
    con_cache.close()
    feat = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df, feat], axis=1)


DECOY_COLS = ["store_served_1438", "rth_bar_count_1438", "tick_window_has_bar_j"]

# Real-model feature columns: every 1478 feature (minus decoys, minus join keys/labels), plus the new
# 1489 features. hgb_prob_L3/lr_prob_L3 included per "the cell 1,478-L3 probability (HGB) ... inherited".
NON_FEATURE_COLS = {
    "day", "symbol", "split", "wk", "fill", "stop", "level", "fill_min", "base_why", "base_net_R",
    "limit", "status", "filled", "retest_minute", "retest_ts", "dip_low", "entry", "stop_used",
    "target", "Rp", "r_pct_price", "retest_delay_min", "exit_m", "exit_price", "why", "raw_R",
    "cost_R", "net_R_prime", "Y", "short_net_R", "shortable", "ssr", "sic2",
    "outcome_R", "L3", "hgb_kept_L3", "lr_kept_L3",
}


def feature_columns(df):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c not in DECOY_COLS]
    return cols


def fit_hgb(X_train, y_train, seed):
    base = HistGradientBoostingClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(base, GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def fit_lr(X_train, y_train, seed):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    pipe.fit(X_train, y_train)
    return pipe


def day_clustered_t(values, days):
    """Day-clustered t-stat for the mean of `values`, clustering by `days`."""
    s = pd.Series(values.values, index=days.values if hasattr(days, "values") else days)
    day_means = s.groupby(level=0).mean()
    n_days = len(day_means)
    if n_days < 2:
        return np.nan
    m = day_means.mean()
    se = day_means.std(ddof=1) / np.sqrt(n_days)
    return m / se if se > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="first 200 population rows only")
    args = ap.parse_args()

    log("loading population (rebuild_1481_fills.csv, status==fill)")
    pop = load_population()
    log(f"population n={len(pop)} (TRAIN={sum(pop.split=='TRAIN')}, VAL={sum(pop.split=='VAL')})")

    if args.smoke:
        pop = pd.concat([pop[pop.split == "TRAIN"].head(100),
                         pop[pop.split == "VAL"].head(100)]).copy()
        log(f"SMOKE MODE: n={len(pop)} (100 TRAIN + 100 VAL, so both holdouts exercise the pipeline)")

    log("joining 1478 arm-bar features (group 1)")
    df = join_1478_features(pop)
    log("joining 1480 short-leg outcomes")
    df = join_1480_short(df)

    log("building retest-instant features (groups 2/3/4) -- row by row, verbose")
    df = build_features(df, verbose_every=50 if args.smoke else 500)

    feat_cols = feature_columns(df)
    log(f"n real-model features = {len(feat_cols)}")

    train = df[df.split == "TRAIN"]
    val = df[df.split == "VAL"]
    X_train, y_train = train[feat_cols], train["Y"]
    X_val, y_val = val[feat_cols], val["Y"]

    log("fitting real HGB model (5-fold CV inside TRAIN, fixed grid, seed 1489)")
    hgb, best_params = fit_hgb(X_train, y_train, SEED)
    log(f"best params: {best_params}")
    p_train = hgb.predict_proba(X_train)[:, 1]
    p_val = hgb.predict_proba(X_val)[:, 1]
    auc_train = roc_auc_score(y_train, p_train)
    auc_val = roc_auc_score(y_val, p_val)
    log(f"HGB AUC train={auc_train:.3f} val={auc_val:.3f}")

    log("fitting LR (beside)")
    lr = fit_lr(X_train, y_train, SEED)
    lr_p_val = lr.predict_proba(X_val)[:, 1]
    lr_auc_val = roc_auc_score(y_val, lr_p_val)
    log(f"LR AUC val={lr_auc_val:.3f}")

    log("fitting label-shuffled placebo (same grid/seed)")
    rng = np.random.RandomState(SEED)
    y_train_shuf = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    placebo, _ = fit_hgb(X_train, y_train_shuf, SEED)
    placebo_p_val = placebo.predict_proba(X_val)[:, 1]
    placebo_auc_val = roc_auc_score(y_val, placebo_p_val)
    log(f"placebo AUC val={placebo_auc_val:.3f}")

    log("fitting metadata-only decoy model")
    Xd_train, Xd_val = train[DECOY_COLS], val[DECOY_COLS]
    decoy, _ = fit_hgb(Xd_train, y_train, SEED)
    decoy_p_val = decoy.predict_proba(Xd_val)[:, 1]
    decoy_auc_val = roc_auc_score(y_val, decoy_p_val)
    log(f"decoy AUC val={decoy_auc_val:.3f} (VOID if > 0.55)")

    thr_top = np.quantile(p_train, 2 / 3)  # TRAIN top tercile threshold (buy)
    thr_bot = np.quantile(p_train, 1 / 3)  # TRAIN bottom tercile threshold (short)
    log(f"thresholds from TRAIN: top-tercile={thr_top:.4f} bottom-tercile={thr_bot:.4f}")

    p_all = pd.Series(index=df.index, dtype=float)
    p_all.loc[train.index] = p_train
    p_all.loc[val.index] = p_val
    df["p_hgb"] = p_all

    df["kept_1489"] = df["p_hgb"] >= thr_top
    df["kept_1490"] = df["p_hgb"] <= thr_bot

    # ---- cell 1,489 (BUY): net_R_prime on kept fills ----
    # ---- cell 1,490 (SHORT): 1,480 short_net_R on kept fills, shortable & not SSR ----
    def cell_summary(sub, value_col, label):
        rows = []
        for split_name, g in sub.groupby("split"):
            kept = g[g["_kept"]]
            dropped = g[~g["_kept"]]
            vals = kept[value_col].dropna()
            n = len(vals)
            kept_mean = vals.mean() if n else np.nan
            t = day_clustered_t(vals, kept.loc[vals.index, "day"]) if n else np.nan
            ex_top5 = np.nan
            if n >= 20:
                cap = vals.quantile(0.95)
                ex_top5 = vals[vals <= cap].mean()
            dropped_mean = dropped[value_col].dropna().mean() if len(dropped) else np.nan
            rows.append(dict(cell=label, split=split_name, n_kept=n, kept_mean=kept_mean,
                             day_clustered_t=t, ex_top5pct_mean=ex_top5, dropped_mean=dropped_mean))
        return rows

    df_1489 = df.copy()
    df_1489["_kept"] = df_1489["kept_1489"]
    rows_1489 = cell_summary(df_1489, "net_R_prime", "1489_BUY")

    df_1490 = df.copy()
    df_1490["_kept"] = df_1490["kept_1490"] & (df_1490["shortable"] == True) & (df_1490["ssr"] != True)
    rows_1490 = cell_summary(df_1490, "short_net_R", "1490_SHORT")

    summary = pd.DataFrame(rows_1489 + rows_1490)
    log("\n" + summary.to_string(index=False))

    out_pred = df[["day", "symbol", "split", "p_hgb", "kept_1489", "kept_1490", "net_R_prime",
                  "short_net_R"]].copy()
    pred_path = BASE / ("rebuild_1489_predictions_smoke.csv" if args.smoke else "rebuild_1489_predictions.csv")
    out_pred.to_csv(pred_path, index=False)
    log(f"wrote {pred_path} ({len(out_pred)} rows)")

    # tape/breadth coverage report
    coverage = dict(
        tape_window_found_share=float(df["tape_window_found"].mean()),
        spy_ret_available_share=float(df["spy_ret_fill_to_tr"].notna().mean()),
        n_path_bars_zero_share=float((df["n_path_bars"] == 0).mean()),
    )
    log(f"coverage: {coverage}")

    report = dict(
        auc_train=auc_train, auc_val=auc_val, lr_auc_val=lr_auc_val,
        placebo_auc_val=placebo_auc_val, decoy_auc_val=decoy_auc_val,
        thr_top=thr_top, thr_bot=thr_bot, best_params=best_params, coverage=coverage,
        summary=summary,
    )
    import json
    report_path = BASE / ("rebuild_1489_report_smoke.json" if args.smoke else "rebuild_1489_report.json")
    with open(report_path, "w") as f:
        json.dump({k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
                   for k, v in report.items()}, f, indent=2, default=str)
    log(f"wrote {report_path}")
    log("DONE")


if __name__ == "__main__":
    main()
