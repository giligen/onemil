"""
Independent rebuild of cell 1,478 (PREREG_1478.md + amendments, read whole, NOT the
builder's cell_1478.py / features_1478_A.py / FEATURES_*.md / RESULT_1478.md / tests).

Rebuilds the feature matrix from prose, fits the same model family (HistGradientBoosting
+ logistic regression, seed 1478, fixed grid, TRAIN-H2 5-fold CV -> fit once -> apply once
to VAL, top-tercile-of-TRAIN threshold), and writes:
  research/hod_entry/rebuild_1478_features.csv
  research/hod_entry/rebuild_1478_predictions.csv
  research/hod_entry/rebuild_1478_compare.csv   (per label/model comparison vs the builder)
  research/hod_entry/rebuild_1478.log           (verbose progress)

Every fallback / skipped feature logs WARNING with the reason (item 5 symbol-persistence
requires a 60-prior-session per-symbol panel that is NOT present in the single fresh store
bars_fills_1478.db -- covers only the 9,911 fill symbol-days -- and is out of the 40-tool
budget to fetch fresh via Alpaca for ~thousands of symbol-days; left NaN, documented, HGB
handles NaN natively).
"""
import logging
import sys
import time
import warnings
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path("/home/ec2-user/onemil")
HOD = ROOT / "research/hod_entry"
LOG = HOD / "rebuild_1478.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rebuild_1478")

SEED = 1478
K = 10          # consolidation lookback bars
K_VWAP = 15     # vwap slope lookback bars
RTH_OPEN_MIN = 570   # 09:30 ET
RTH_CLOSE_MIN = 960  # 16:00 ET
PM_START_MIN = 240   # 04:00 ET

GRID = [
    dict(max_depth=md, learning_rate=lr, max_iter=mi, min_samples_leaf=ml)
    for md in (3, 5)
    for lr in (0.03, 0.1)
    for mi in (200, 600)
    for ml in (50, 200)
]


def t():
    return time.time()


# ---------------------------------------------------------------------------
# 1. Base book
# ---------------------------------------------------------------------------
def load_base_book():
    log.info("loading base book causal_arming_causal.csv")
    df = pd.read_csv(HOD / "causal_arming_causal.csv")
    df = df[df["status"] == "fill"].copy()
    log.info("base book fills: %d", len(df))
    assert len(df) == 9911, f"expected 9911 fills, got {len(df)}"
    # split normalisation: TRAIN (==TRAIN-H2) / VAL, per task text
    df["split_norm"] = np.where(df["split"] == "VAL", "VAL", "TRAIN-H2")
    log.info("split counts: %s", df["split_norm"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# 2. Bar-derived features, recomputed fresh from the single-source store
#    (amendment: rv_at_j, range_to_j, arm_index, bar_density, dvol_to_j, pm_dvol)
#    plus item-2 NEW stock-state features (all bar-derived, same store).
# ---------------------------------------------------------------------------
def load_bars():
    log.info("loading bars_fills_1478.db (this is large: ~4.46M rows)")
    con = sqlite3.connect(f"file:{HOD / 'bars_fills_1478.db'}?mode=ro", uri=True)
    bars = pd.read_sql_query("SELECT symbol, day, t, o, h, l, c, v FROM bars", con)
    con.close()
    log.info("bars loaded: %d rows, %d symbol-days", len(bars), bars[["symbol", "day"]].drop_duplicates().shape[0])
    ts = pd.to_datetime(bars["t"], utc=True).dt.tz_convert("America/New_York")
    bars["m"] = ts.dt.hour * 60 + ts.dt.minute
    bars = bars.sort_values(["symbol", "day", "m"]).reset_index(drop=True)
    return bars


def ols_slope(y):
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    yc = np.asarray(y, dtype=float) - np.mean(y)
    denom = (x * x).sum()
    if denom == 0:
        return np.nan
    return float((x * yc).sum() / denom)


def bar_features_for_fill(g_rth, g_pm, m_fill, level):
    """g_rth: RTH bars for the (symbol, day) sorted by m. g_pm: premarket bars.
    m_fill: fill_min. level: breakout level. Returns dict of features, arm bar j index."""
    before = g_rth[g_rth["m"] < m_fill]
    if before.empty:
        return None  # no arm bar available (fill at/near the open) -- reported as NaN row
    j = before.index[-1]  # position within g_rth (integer index after reset)
    up_to_j = g_rth.loc[:j]  # g_rth is reset_index'd so .loc[:j] == first j+1 rows
    open_px = g_rth["o"].iloc[0]

    # --- recomputed item-1 bar-derived (6) ---
    closes = up_to_j["c"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(closes))
    rv_at_j = float(np.nanstd(logret) * 100) if len(logret) > 1 else np.nan
    range_to_j = float((up_to_j["h"].max() - up_to_j["l"].min()) / open_px * 100) if open_px else np.nan
    arm_index = int(len(up_to_j))
    span_min = (up_to_j["m"].iloc[-1] - up_to_j["m"].iloc[0] + 1)
    bar_density = float(len(up_to_j) / span_min) if span_min > 0 else np.nan
    dvol_to_j = float((up_to_j["v"] * up_to_j["c"]).sum())
    pm_dvol = float((g_pm["v"] * g_pm["c"]).sum()) if not g_pm.empty else 0.0

    # --- item 2 NEW: stock's own state (all bar-derived, same fresh store) ---
    lastK = up_to_j.tail(K)
    vol_slope = ols_slope(lastK["v"].values)
    mean_lastK_vol = float(lastK["v"].mean())
    mean_day_vol = float(up_to_j["v"].mean())
    vol_ratio = mean_lastK_vol / mean_day_vol if mean_day_vol else np.nan
    lows = lastK["l"].values
    higher_lows = int(np.sum(np.diff(lows) > 0)) if len(lows) > 1 else np.nan
    tol = 0.002
    touches = int(((up_to_j["l"] <= level * (1 + tol)) & (up_to_j["h"] >= level * (1 - tol))).sum()) if pd.notna(level) else np.nan
    pre_cons = up_to_j.iloc[: max(len(up_to_j) - K, 0)]
    if len(pre_cons) >= 3:
        run_high = pre_cons["h"].max()
        min_low_after = up_to_j.iloc[max(len(up_to_j) - K, 0):]["l"].min()
        pullback_depth = float((run_high - min_low_after) / run_high * 100) if run_high else np.nan
    else:
        pullback_depth = np.nan
    tp = (up_to_j["h"] + up_to_j["l"] + up_to_j["c"]) / 3.0
    cum_pv = (tp * up_to_j["v"]).cumsum()
    cum_v = up_to_j["v"].cumsum()
    vwap_series = (cum_pv / cum_v.replace(0, np.nan)).values
    vwap_now = vwap_series[-1] if len(vwap_series) else np.nan
    vwap_distance = float((closes[-1] - vwap_now) / vwap_now * 100) if vwap_now and not np.isnan(vwap_now) else np.nan
    vwap_slope = ols_slope(vwap_series[-K_VWAP:]) if len(vwap_series) >= 2 else np.nan
    mdiffs = np.diff(up_to_j["m"].values)
    halt_proxy = int(np.any(mdiffs >= 5)) if len(mdiffs) else 0

    return dict(
        rv_at_j=rv_at_j, range_to_j=range_to_j, arm_index=arm_index, bar_density=bar_density,
        dvol_to_j=dvol_to_j, pm_dvol=pm_dvol,
        cons_vol_slope=vol_slope, cons_vol_ratio=vol_ratio, higher_lows_ct=higher_lows,
        level_touches=touches, pullback_depth_pct=pullback_depth,
        vwap_distance_pct=vwap_distance, vwap_slope=vwap_slope, halt_proxy=halt_proxy,
        open_px=open_px,
    ), int(up_to_j["m"].iloc[-1])


def label_features(bars, day_high_after):
    """Also computes L3 raw ingredient: day's high strictly after arm bar j (all bars, same store)."""
    pass


def build_bar_derived(base, bars):
    log.info("grouping bars by (symbol, day)")
    groups = {k: v.reset_index(drop=True) for k, v in bars.groupby(["symbol", "day"])}
    log.info("groups: %d", len(groups))
    rows = []
    n = len(base)
    t0 = t()
    for i, r in enumerate(base.itertuples(index=False)):
        key = (r.symbol, r.day)
        g = groups.get(key)
        rec = dict(day=r.day, symbol=r.symbol, fill_min=r.fill_min)
        if g is None:
            log.warning("NO BARS for fill %s %s -- all bar features NaN", r.symbol, r.day)
            rows.append(rec)
            continue
        g_rth = g[(g["m"] >= RTH_OPEN_MIN) & (g["m"] < RTH_CLOSE_MIN)].reset_index(drop=True)
        g_pm = g[(g["m"] >= PM_START_MIN) & (g["m"] < RTH_OPEN_MIN)]
        if g_rth.empty:
            log.warning("NO RTH BARS for fill %s %s -- all bar features NaN", r.symbol, r.day)
            rows.append(rec)
            continue
        out = bar_features_for_fill(g_rth, g_pm, r.fill_min, r.level)
        if out is None:
            log.warning("NO ARM BAR (fill before first RTH bar) for %s %s fill_min=%.2f", r.symbol, r.day, r.fill_min)
            rows.append(rec)
            continue
        feats, m_j = out
        rec.update(feats)
        # L3 ingredient: day's high strictly after bar j, ANY session bar (04:00-20:00 store)
        after = g[g["m"] > m_j]
        rec["post_j_high"] = float(after["h"].max()) if not after.empty else np.nan
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            log.info("bar features: %d/%d done (%.1fs)", i + 1, n, t() - t0)
    log.info("bar-derived features done: %d rows in %.1fs", len(rows), t() - t0)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. PIT daily: labels L1/L2, prior-day volume/rvol, ATR%, prior-day range, float(n/a),
#    gap vs prior close, level vs prior-day high and 20-session high.
# ---------------------------------------------------------------------------
def load_pit_daily():
    log.info("loading PIT daily parquet")
    df = pd.read_parquet(ROOT / "data/research/databento/equs_daily_2025_2026.parquet")
    df = df.dropna(subset=["symbol"]).copy()
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    df = df.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    df["range_pct"] = (df["high"] - df["low"]) / df["open"] * 100
    df["prior_close"] = df.groupby("symbol")["close"].shift(1)
    df["prior_volume"] = df.groupby("symbol")["volume"].shift(1)
    df["prior_range_pct"] = df.groupby("symbol")["range_pct"].shift(1)
    df["adv20"] = df.groupby("symbol")["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    df["prior_rvol"] = df["prior_volume"] / df["adv20"]
    df["gap_pct"] = (df["open"] - df["prior_close"]) / df["prior_close"] * 100
    df["prior_day_high"] = df.groupby("symbol")["high"].shift(1)
    df["high_20s"] = df.groupby("symbol")["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    # ATR% (14-day, Wilder-simplified true range / close)
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
    df["atr_pct"] = df["atr14"] / df["close"] * 100
    log.info("PIT daily rows: %d, symbols: %d", len(df), df["symbol"].nunique())
    return df


def join_pit(base, pit):
    b = base.copy()
    b["bar_date"] = pd.to_datetime(b["day"])
    p = pit[["symbol", "bar_date", "open", "high", "low", "close", "range_pct",
             "prior_volume", "prior_rvol", "prior_range_pct", "gap_pct",
             "prior_day_high", "high_20s", "atr_pct"]].rename(columns={
                 "open": "d_open", "high": "d_high", "low": "d_low", "close": "d_close"})
    out = b.merge(p, on=["symbol", "bar_date"], how="left")
    missing = out["d_close"].isna().sum()
    if missing:
        log.warning("PIT daily join missing for %d / %d fills (symbol/day not in PIT parquet)", missing, len(out))
    out["L1"] = (out["range_pct"] >= 10).astype(float)
    out["L2"] = ((out["L1"] == 1) & (out["d_close"] >= 10) & (out["prior_volume"] >= 1_000_000)).astype(float)
    out.loc[out["d_close"].isna(), ["L1", "L2"]] = np.nan
    out["level_vs_prior_high_pct"] = (out["level"] - out["prior_day_high"]) / out["prior_day_high"] * 100
    out["level_vs_20s_high_pct"] = (out["level"] - out["high_20s"]) / out["high_20s"] * 100
    return out


# ---------------------------------------------------------------------------
# 4. Assemble the full feature matrix
# ---------------------------------------------------------------------------
def assemble(base):
    log.info("=== step: bar-derived (recompute 6 + item2 NEW) ===")
    bars = load_bars()
    bar_df = build_bar_derived(base, bars)
    del bars

    log.info("=== step: PIT daily (labels + item1 non-bar + item2 prior-day) ===")
    pit = load_pit_daily()
    m = join_pit(base, pit)

    log.info("=== step: merge bar-derived ===")
    m = m.merge(bar_df, on=["day", "symbol", "fill_min"], how="left", suffixes=("", "_bar"))
    m["L3"] = (m["post_j_high"] >= m["level"] * 1.05).astype(float)
    m.loc[m["post_j_high"].isna(), "L3"] = np.nan

    log.info("=== step: item1 remaining flags from cell_1445/1457 (non-recomputed) ===")
    c45 = pd.read_csv(HOD / "cell_1445_features.csv")
    c57 = pd.read_csv(HOD / "cell_1457_features.csv")
    # flags 1445..1462 map 1:1 (17 total) onto item-1's 17 named features, in prose order;
    # the 6 recomputed (rv_at_j=1446, range_to_j=1451, arm_index=1452, bar_density=1453,
    # dvol_to_j=1454, pm_dvol=1455) are DROPPED here and replaced by the fresh columns above.
    # This flag<->feature order mapping is an INFERENCE (count matches 17==17, and the
    # recomputed-name positions land exactly inside cell_1445_features.csv's flag span) --
    # not confirmed against the builder's code, which this rebuild does not read. Flagged
    # as an assumption in the caveats.
    keep_1445 = ["flag_1445", "flag_1447", "flag_1448", "flag_1449", "flag_1450"]
    keep_1457 = ["flag_1457", "flag_1458", "flag_1459", "flag_1460", "flag_1461", "flag_1462"]
    rename_1445 = {"flag_1445": "f_dist_from_open", "flag_1447": "f_ask_distance",
                   "flag_1448": "f_spread_at_fill", "flag_1449": "f_time_of_day", "flag_1450": "f_R_pct"}
    rename_1457 = {"flag_1457": "f_atr_pct_old", "flag_1458": "f_prior_day_range_old",
                   "flag_1459": "f_float", "flag_1460": "f_gap_old", "flag_1461": "f_level_vs_pdh_old",
                   "flag_1462": "f_level_vs_20s_old"}
    c45s = c45[["day", "symbol", "fill_min"] + keep_1445].rename(columns=rename_1445)
    c57s = c57[["day", "symbol", "fill_min"] + keep_1457].rename(columns=rename_1457)
    m = m.merge(c45s, on=["day", "symbol", "fill_min"], how="left")
    m = m.merge(c57s, on=["day", "symbol", "fill_min"], how="left")

    log.info("=== step: item3 crowd (features_1478_B.csv, as given) ===")
    b = pd.read_csv(HOD / "features_1478_B.csv")
    b_cols = ["breadth_count_j", "breadth_share_j", "n_universe_j", "sector_peers_j",
              "spy_ret_open_to_j", "breadth_5d", "spy_ret_5d"]
    m = m.merge(b[["day", "symbol", "fill_min"] + b_cols], on=["day", "symbol", "fill_min"], how="left")

    log.info("=== step: item4 tape (features_1478_C.csv, as given) ===")
    c = pd.read_csv(HOD / "features_1478_C.csv")
    c_cols = ["has_prebreak", "trigger_print_odd_lot", "trigger_print_size", "pre_break_odd_lot_share",
              "pre_break_mean_trade_size", "pre_break_print_count", "pre_break_buy_share", "spread_bps_at_arm"]
    m = m.merge(c[["day", "symbol", "fill_min"] + c_cols], on=["day", "symbol", "fill_min"], how="left")

    log.info("=== step: item5 symbol persistence -- SKIPPED (see docstring) ===")
    m["symbol_persistence_60s"] = np.nan
    log.warning("item5 symbol_persistence_60s left NaN for all %d rows: needs a 60-prior-session "
                "per-symbol 11:00ET panel not present in bars_fills_1478.db (fill-days only) and "
                "out of the tool/time budget to fetch fresh via Alpaca for ~thousands of symbol-days", len(m))

    m["day_of_week"] = pd.to_datetime(m["day"]).dt.dayofweek

    return m


FEATURE_COLS = [
    "rv_at_j", "range_to_j", "arm_index", "bar_density", "dvol_to_j", "pm_dvol",
    "cons_vol_slope", "cons_vol_ratio", "higher_lows_ct", "level_touches", "pullback_depth_pct",
    "vwap_distance_pct", "vwap_slope", "halt_proxy", "day_of_week",
    "f_dist_from_open", "f_ask_distance", "f_spread_at_fill", "f_time_of_day", "f_R_pct",
    "f_atr_pct_old", "f_prior_day_range_old", "f_float", "f_gap_old", "f_level_vs_pdh_old", "f_level_vs_20s_old",
    "prior_volume", "prior_rvol", "atr_pct", "prior_range_pct", "gap_pct",
    "level_vs_prior_high_pct", "level_vs_20s_high_pct",
    "breadth_count_j", "breadth_share_j", "n_universe_j", "sector_peers_j",
    "spy_ret_open_to_j", "breadth_5d", "spy_ret_5d",
    "has_prebreak", "trigger_print_odd_lot", "trigger_print_size", "pre_break_odd_lot_share",
    "pre_break_mean_trade_size", "pre_break_print_count", "pre_break_buy_share", "spread_bps_at_arm",
    "symbol_persistence_60s",
]


# ---------------------------------------------------------------------------
# 5. Model fit per label, HGB (grid CV inside TRAIN-H2) + LR, top-tercile threshold
# ---------------------------------------------------------------------------
def fit_label(m, label, feature_cols):
    log.info("--- fitting label %s ---", label)
    d = m.dropna(subset=[label]).copy()
    tr = d[d["split_norm"] == "TRAIN-H2"]
    va = d[d["split_norm"] == "VAL"]
    log.info("%s: TRAIN-H2 n=%d (pos rate %.3f), VAL n=%d (pos rate %.3f)",
             label, len(tr), tr[label].mean(), len(va), va[label].mean())
    X_tr = tr[feature_cols].values
    y_tr = tr[label].values
    X_va = va[feature_cols].values
    y_va = va[label].values

    # --- HGB: 5-fold CV over the fixed grid inside TRAIN-H2, pick best mean CV AUC ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_auc, best_params = -1, None
    for params in GRID:
        aucs = []
        for tri, tei in skf.split(X_tr, y_tr):
            clf = HistGradientBoostingClassifier(random_state=SEED, **params)
            clf.fit(X_tr[tri], y_tr[tri])
            p = clf.predict_proba(X_tr[tei])[:, 1]
            aucs.append(roc_auc_score(y_tr[tei], p))
        mean_auc = float(np.mean(aucs))
        log.info("  %s grid %s -> CV AUC %.4f", label, params, mean_auc)
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params
    log.info("%s best grid: %s (CV AUC %.4f)", label, best_params, best_auc)
    hgb = HistGradientBoostingClassifier(random_state=SEED, **best_params)
    hgb.fit(X_tr, y_tr)
    p_tr_hgb = hgb.predict_proba(X_tr)[:, 1]
    p_va_hgb = hgb.predict_proba(X_va)[:, 1]
    thr_hgb = float(np.quantile(p_tr_hgb, 2 / 3))  # top TERCILE of TRAIN-H2
    auc_va_hgb = roc_auc_score(y_va, p_va_hgb) if len(np.unique(y_va)) > 1 else np.nan

    # --- LR: impute + scale, same features ---
    lr_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    lr_pipe.fit(X_tr, y_tr)
    p_tr_lr = lr_pipe.predict_proba(X_tr)[:, 1]
    p_va_lr = lr_pipe.predict_proba(X_va)[:, 1]
    thr_lr = float(np.quantile(p_tr_lr, 2 / 3))
    auc_va_lr = roc_auc_score(y_va, p_va_lr) if len(np.unique(y_va)) > 1 else np.nan

    log.info("%s VAL AUC: hgb=%.4f lr=%.4f | thresholds hgb=%.4f lr=%.4f",
             label, auc_va_hgb, auc_va_lr, thr_hgb, thr_lr)

    out = m[["day", "symbol", "fill_min", "split_norm"]].copy()
    out[f"{label}_hgb_prob"] = np.nan
    out[f"{label}_lr_prob"] = np.nan
    idx_tr, idx_va = tr.index, va.index
    out.loc[idx_tr, f"{label}_hgb_prob"] = p_tr_hgb
    out.loc[idx_va, f"{label}_hgb_prob"] = p_va_hgb
    out.loc[idx_tr, f"{label}_lr_prob"] = p_tr_lr
    out.loc[idx_va, f"{label}_lr_prob"] = p_va_lr
    out[f"{label}_hgb_kept"] = out[f"{label}_hgb_prob"] >= thr_hgb
    out[f"{label}_lr_kept"] = out[f"{label}_lr_prob"] >= thr_lr
    out.loc[out[f"{label}_hgb_prob"].isna(), f"{label}_hgb_kept"] = np.nan
    out.loc[out[f"{label}_lr_prob"].isna(), f"{label}_lr_kept"] = np.nan

    stats = dict(label=label, val_auc_hgb=auc_va_hgb, val_auc_lr=auc_va_lr,
                 thr_hgb=thr_hgb, thr_lr=thr_lr, best_grid=str(best_params), cv_auc=best_auc,
                 n_train=len(tr), n_val=len(va))
    return out, stats


def main():
    t0 = t()
    base = load_base_book()
    m = assemble(base)
    m.to_csv(HOD / "rebuild_1478_features.csv", index=False)
    log.info("wrote rebuild_1478_features.csv (%d rows, %d cols) in %.1fs", len(m), m.shape[1], t() - t0)

    all_stats = []
    preds = m[["day", "symbol", "fill_min", "split_norm", "net_R"]].copy()
    for label in ["L1", "L2", "L3"]:
        out, stats = fit_label(m, label, FEATURE_COLS)
        preds = preds.merge(out.drop(columns=["split_norm"]), on=["day", "symbol", "fill_min"], how="left")
        all_stats.append(stats)

    preds.to_csv(HOD / "rebuild_1478_predictions.csv", index=False)
    pd.DataFrame(all_stats).to_csv(HOD / "rebuild_1478_model_stats.csv", index=False)
    log.info("wrote rebuild_1478_predictions.csv (%d rows) and rebuild_1478_model_stats.csv", len(preds))

    # --- comparison vs the builder's model_1478_predictions.csv ---
    log.info("=== comparing against model_1478_predictions.csv ===")
    builder = pd.read_csv(HOD / "model_1478_predictions.csv")
    cmp_rows = []
    for label in ["L1", "L2"]:  # builder's file predates the L3 amendment: no L3 columns there
        for model in ["hgb", "lr"]:
            bk = f"{model}_kept_{label}"
            mk = f"{label}_{model}_kept"
            if bk not in builder.columns:
                log.warning("builder file has no column %s -- skipping %s/%s comparison", bk, label, model)
                continue
            j = preds.merge(
                builder[["day", "symbol", "fill_min", "outcome_R", bk]],
                on=["day", "symbol", "fill_min"], how="inner",
            )
            j = j[j["split_norm"] == "VAL"]
            j = j.dropna(subset=[mk, bk])
            mine_kept = j[mk].astype(bool)
            build_kept = j[bk].astype(bool)
            inter = (mine_kept & build_kept).sum()
            union = (mine_kept | build_kept).sum()
            jac = inter / union if union else np.nan
            mine_mean = j.loc[mine_kept, "outcome_R"].mean()
            build_mean = j.loc[build_kept, "outcome_R"].mean()
            stat_row = next(s for s in all_stats if s["label"] == label)
            val_auc = stat_row[f"val_auc_{model}"]
            mean_gap = abs(mine_mean - build_mean) if pd.notna(mine_mean) and pd.notna(build_mean) else np.nan
            if jac >= 0.95 and mean_gap <= 0.03:
                verdict = "REPRODUCED"
            elif jac >= 0.85 and mean_gap <= 0.06:
                verdict = "PARTIAL"
            else:
                verdict = "NOT_REPRODUCED"
            log.info("%s/%s: jaccard=%.4f mine_kept_mean=%.4f builder_kept_mean=%.4f gap=%.4f my_val_auc=%.4f -> %s",
                     label, model, jac, mine_mean, build_mean, mean_gap, val_auc, verdict)
            cmp_rows.append(dict(label=label, model=model, n_joined=len(j), jaccard=jac,
                                  mine_kept_mean_R=mine_mean, builder_kept_mean_R=build_mean,
                                  mean_gap=mean_gap, my_val_auc=val_auc, verdict=verdict,
                                  n_mine_kept=int(mine_kept.sum()), n_builder_kept=int(build_kept.sum())))
    pd.DataFrame(cmp_rows).to_csv(HOD / "rebuild_1478_compare.csv", index=False)
    log.info("wrote rebuild_1478_compare.csv")
    log.info("ALL DONE in %.1fs", t() - t0)


if __name__ == "__main__":
    main()
