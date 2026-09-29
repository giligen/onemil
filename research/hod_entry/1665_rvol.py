"""Cell 1,665 -- relative volume to the arm minute vs the prior N sessions.

PREREG: research/hod_entry/PREREG_1665.md (FROZEN). Owner's cut: "how come 1663
doesn't look at average volume till that point compared to previous X days?"

Population: research/hod_entry/1663_features.csv (5,506 fills, stop >= 1.5% floor,
already joined fills_1658.csv x causal_arming_causal.csv with ATR/stop-bucket added
by cell 1,663 -- reused here, not re-derived). We bring in `level` (needed for the
arm-minute definition) and `fill_id`/`bucket_r` from the two source files via the
same (date,symbol) 1:1 join cell 1,663 used.

Feature definition A (primary, pre-declared):
  CV(d, m) = sum of volume of the symbol's 1-min bars on session d with
             09:30 <= bar start (ET) and bar END <= m (bars that CLOSED before
             the instant m -- fractional-minute rule, never the bar containing m).
  m_arm    = the START (ET minute-of-day, integer) of the first bar of session d,
             at or before the fill, whose high equals `level` within 1 cent.
             The fill bar is never used (search window ends one bar before the
             bar containing fill_min).
  RVOL_A(N) = CV(d, m_arm) / mean over the prior N *store* sessions k of
              CV(d-k, m_arm) (same clock minute, applied to each prior day).
              A prior session with no bars in [09:30, m_arm) counts as MISSING,
              not zero. N=20 primary, N=5 secondary.
  Availability: a symbol needs >=10 (N=20) / >=3 (N=5) prior store sessions
              before day d, else MISSING for that N.

bars_sip.db is opened read-only, one bounded range query PER SYMBOL
(`WHERE symbol=? AND day BETWEEN ? AND ?`), never a full scan (PK is
(symbol,day,t), so this is an index range scan). Bar timestamps are UTC;
converted to America/New_York per bar via zoneinfo (handles DST).

Run:
  python3 1665_rvol.py            # full run (resumes automatically if partial exists)
  python3 1665_rvol.py --resume   # explicit resume (same behaviour)
  python3 1665_rvol.py --mode=B   # force definition-B fallback path (daily panel)
"""
import argparse
import bisect
import logging
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = "/home/ec2-user/onemil"
BARS_DB = f"{ROOT}/research/bf_zero/bars_sip.db"
FILLS_1658 = f"{ROOT}/research/hod_entry/fills_1658.csv"
CAUSAL = f"{ROOT}/research/hod_entry/causal_arming_causal.csv"
FEATURES_1663 = f"{ROOT}/research/hod_entry/1663_features.csv"
PANEL_PARQUET = f"{ROOT}/research/overnight_high/panel_2024_2026.parquet"

OUT_FEATURES = f"{ROOT}/research/hod_entry/1665_features.csv"
OUT_PARTIAL = OUT_FEATURES + ".partial"
OUT_CHECKPOINT = f"{ROOT}/research/hod_entry/1665_features.checkpoint"
LOG_PATH = f"{ROOT}/research/hod_entry/1665_rvol.log"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN_MIN = 9 * 60 + 30  # 570

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("1665_rvol")


def build_population() -> pd.DataFrame:
    """Reuse cell 1,663's floored population + join in `level`, `fill_id`, `bucket_r`.

    Join key (date,symbol), 1:1, exactly as cell 1,663 did. Asserts full match
    (raises loudly rather than silently dropping rows) -- root-cause discipline.
    """
    feat = pd.read_csv(FEATURES_1663)
    feat = feat.rename(columns={"half": "split"})

    fills = pd.read_csv(FILLS_1658)[["fill_id", "date", "symbol", "bucket_r"]]
    causal = pd.read_csv(CAUSAL)
    causal = causal[causal["status"] == "fill"][["day", "symbol", "level", "fill_min"]]
    causal = causal.rename(columns={"day": "date", "fill_min": "fill_min_causal"})

    n0 = len(feat)
    merged = feat.merge(fills, on=["date", "symbol"], how="left", validate="one_to_one")
    n_miss_fillid = merged["fill_id"].isna().sum()
    merged = merged.merge(causal, on=["date", "symbol"], how="left", validate="one_to_one")
    n_miss_level = merged["level"].isna().sum()

    if n_miss_fillid or n_miss_level:
        log.error(
            "join gaps: %d/%d missing fill_id, %d/%d missing level -- "
            "population is not the clean 1:1 join cell 1,663 reported",
            n_miss_fillid, n0, n_miss_level, n0,
        )
    # fill_min: causal file is the source (1663 used it as arm-adjacent time);
    # sanity-check against 1663's own fill_min column (same source, should match).
    fmin_diff = (merged["fill_min"] - merged["fill_min_causal"]).abs()
    bad = (fmin_diff > 0.01).sum()
    if bad:
        log.warning("fill_min mismatch vs causal file on %d/%d rows (>0.01 min)", bad, n0)

    merged["day"] = merged["date"]
    log.info("population built: %d rows (expected 5,506)", len(merged))
    return merged


def et_minute(ts: str) -> int:
    """UTC ISO timestamp -> ET minute-of-day (integer, bar START)."""
    dt = pd.Timestamp(ts).tz_convert(ET)
    return dt.hour * 60 + dt.minute


def fetch_symbol_bars(cur, symbol: str, lo: str, hi: str) -> dict:
    """One bounded range query for this symbol. Returns {day: (minutes[], cumvol[])}
    with minutes sorted ascending and cumvol[i] = sum of v for bars[0..i] (both
    restricted to minute >= 570). cumvol lets CV(d,m) be a bisect + O(1) lookup.
    """
    rows = cur.execute(
        "SELECT day, t, h, v FROM bars WHERE symbol=? AND day BETWEEN ? AND ? ORDER BY day, t",
        (symbol, lo, hi),
    ).fetchall()
    by_day = {}
    for day, t, h, v in rows:
        m = et_minute(t)
        if m < MARKET_OPEN_MIN:
            continue
        by_day.setdefault(day, {"m": [], "h": [], "v": []})
        d = by_day[day]
        d["m"].append(m)
        d["h"].append(h)
        d["v"].append(v)
    out = {}
    for day, d in by_day.items():
        m = np.array(d["m"])
        order = np.argsort(m, kind="stable")
        m = m[order]
        h = np.array(d["h"])[order]
        v = np.array(d["v"])[order]
        cumv = np.cumsum(v)
        out[day] = (m, h, cumv)
    return out


def cv_before(day_data, m_ref: int):
    """CV(day, m_ref) = sum v for bars with 570<=start and end<=m_ref, i.e.
    start <= m_ref-1. Returns (cv, n_bars_in_window). n_bars_in_window==0 means
    MISSING (no bars before m_ref), distinct from a legitimate cv==0.
    """
    m, h, cumv = day_data
    # rightmost index with m[idx] <= m_ref - 1
    idx = bisect.bisect_right(m.tolist(), m_ref - 1) - 1
    if idx < 0:
        return 0.0, 0
    return float(cumv[idx]), idx + 1


def find_level_bar(day_data, level: float, upper_minute_excl: int):
    """First bar (ascending minute, start>=570, start < upper_minute_excl) whose
    high matches `level` within 1 cent. Returns m_arm (int) or None if not found.
    """
    m, h, _ = day_data
    for i in range(len(m)):
        if m[i] >= upper_minute_excl:
            break
        if abs(h[i] - level) <= 0.0101:
            return int(m[i])
    return None


def process_symbol(cur, symbol: str, sub: pd.DataFrame, mode: str) -> list:
    """Compute RVOL_A(20), RVOL_A(5) for every fill of one symbol. One bars_sip.db
    query for the whole symbol (bounded range), reused across all its fills and
    across all prior-session lookups.
    """
    lo = (pd.Timestamp(sub["date"].min()) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    hi = sub["date"].max()
    bars = fetch_symbol_bars(cur, symbol, lo, hi)
    store_days = sorted(bars.keys())

    out = []
    for _, r in sub.iterrows():
        d = r["date"]
        rec = {
            "fill_id": r["fill_id"], "day": d, "symbol": symbol, "split": r["split"],
            "r_pct": r["r_pct"], "bucket_r": r["bucket_r"], "net_R": r["net_R"],
            "rvol_a20": np.nan, "rvol_a5": np.nan, "rvol_b": np.nan,
            "m_arm": np.nan, "missing_level": True,
            "n_prior_store_days": 0, "missing_a20": True, "missing_a5": True,
        }
        if d not in bars:
            out.append(rec)
            continue
        day_data = bars[d]
        fill_min = r["fill_min"]
        upper = int(np.floor(fill_min))  # exclude the fill-containing bar
        m_arm = find_level_bar(day_data, r["level"], upper)
        prior_days = [x for x in store_days if x < d]
        rec["n_prior_store_days"] = len(prior_days)
        if m_arm is None:
            out.append(rec)
            continue
        rec["missing_level"] = False
        rec["m_arm"] = m_arm

        cv_today, n_today = cv_before(day_data, m_arm)
        if n_today == 0:
            # arm minute itself has no bars before it (shouldn't happen once a
            # level bar was found, but guard anyway)
            out.append(rec)
            continue

        for N, key in ((20, "rvol_a20"), (5, "rvol_a5")):
            thresh = 10 if N == 20 else 3
            window = prior_days[-N:]
            if len(window) < thresh:
                continue  # stays missing
            vals = []
            for pd_day in window:
                if pd_day not in bars:
                    continue  # no bars at all that day -> missing session
                cv_p, n_p = cv_before(bars[pd_day], m_arm)
                if n_p == 0:
                    continue  # no bars before m_arm that day -> missing, not zero
                vals.append(cv_p)
            miss_key = "missing_a20" if N == 20 else "missing_a5"
            if len(vals) == 0:
                continue
            mean_prior = float(np.mean(vals))
            if mean_prior <= 0:
                continue
            rec[key] = cv_today / mean_prior
            rec[miss_key] = False
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Statistics -- day_clustered_t ported verbatim from research/hod_entry/cell_1617.py
# (that file names cell_1445.py as the canonical source). MDE follows PREREG_1665's
# own wording literally: "the MDE at that n (two-sided, 80% power, the book's SD)"
# -- i.e. the iid sample SD of the read, NOT a clustered SE (some earlier cells,
# e.g. RESULT_1663, used a clustered-SE MDE; 1665's prose is explicit and controls).
# ---------------------------------------------------------------------------

def day_clustered_t(y, day):
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d.to_numpy()})
    return float(model.tvalues[0])


def iid_t(y):
    y = pd.Series(y).dropna()
    n = len(y)
    if n < 2:
        return np.nan
    sd = y.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(y.mean() / (sd / np.sqrt(n)))


def mde(y):
    """80% power, two-sided 5%, on the book's (this read's) own iid SD."""
    y = pd.Series(y).dropna()
    n = len(y)
    if n < 3:
        return np.nan
    return float(2.802 * y.std(ddof=1) / np.sqrt(n))


def ex_top5_mean(y):
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def row_stats(sub, wk_denom, label, half, cut_name):
    """One row of 1665_reads.csv for a (cut value, half) slice."""
    y = sub["net_R"]
    return dict(
        cut=cut_name, bucket=label, half=half, n=len(sub),
        mean_net_R=float(y.mean()) if len(y) else np.nan,
        iid_t=iid_t(y), day_clustered_t=day_clustered_t(y, sub["day"]),
        mde=mde(y), ex_top5_mean=ex_top5_mean(y),
        fills_per_week=len(sub) / wk_denom if wk_denom else np.nan,
    )


def generate_reads(feat: pd.DataFrame) -> pd.DataFrame:
    """Reads 1-4 (PREREG_1665 sec "Reads"), for RVOL_A(20) then repeated for
    RVOL_A(5) (read 5). Read 6 (coverage) is reported in RESULT_1665.md, not here.
    """
    rows = []
    floored = feat  # feat IS the floored (stop>=1.5%) book already
    wk_denom = {h: weeks_spanned(floored[floored["split"] == h]["day"]) for h in floored["split"].unique()}

    for N, rcol in ((20, "rvol_a20"), (5, "rvol_a5")):
        valid = floored[floored[rcol].notna()].copy()
        if len(valid) == 0:
            log.warning("RVOL_A(%d): zero valid rows, skipping reads for this N", N)
            continue
        # pooled-edge terciles/quintiles (cut choice happens on the whole floored+valid pool)
        valid["tercile"] = pd.qcut(valid[rcol], 3, labels=["T1(low)", "T2(mid)", "T3(high)"], duplicates="drop")
        valid["quintile"] = pd.qcut(valid[rcol], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

        for half in sorted(valid["split"].unique()):
            hv = valid[valid["split"] == half]
            wk = wk_denom.get(half, np.nan)

            # Read 1: terciles + quintiles
            for t in hv["tercile"].cat.categories:
                rows.append(row_stats(hv[hv["tercile"] == t], wk, str(t), half, f"read1_tercile_A{N}"))
            for q in hv["quintile"].cat.categories:
                rows.append(row_stats(hv[hv["quintile"] == q], wk, str(q), half, f"read1_quintile_A{N}"))

            # Read 2: top tercile vs rest (paired cut) -- ΔR via day-clustered OLS on a dummy
            top = hv[hv["tercile"] == "T3(high)"]
            rest = hv[hv["tercile"] != "T3(high)"]
            if len(top) >= 2 and len(rest) >= 2:
                yy = pd.concat([top["net_R"], rest["net_R"]])
                dd = pd.concat([top["net_R"] * 0 + 1, rest["net_R"] * 0])
                day = pd.concat([top["day"], rest["day"]])
                X = sm.add_constant(dd.to_numpy())
                model = sm.OLS(yy.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": day.to_numpy()})
                delta_r = float(model.params[1])
                t_delta = float(model.tvalues[1])
                delta_ex5 = ex_top5_mean(top["net_R"]) - ex_top5_mean(rest["net_R"])
                rows.append(dict(
                    cut=f"read2_top_vs_rest_A{N}", bucket="T3_minus_rest", half=half,
                    n=len(top) + len(rest), mean_net_R=delta_r, iid_t=np.nan,
                    day_clustered_t=t_delta, mde=mde(yy), ex_top5_mean=delta_ex5,
                    fills_per_week=len(top) / wk if wk else np.nan,
                ))

            # Read 3: stop bucket x RVOL tercile interaction
            for b in sorted(hv["bucket_r"].dropna().unique()):
                for t in hv["tercile"].cat.categories:
                    cell = hv[(hv["bucket_r"] == b) & (hv["tercile"] == t)]
                    rows.append(row_stats(cell, wk, f"{b}|{t}", half, f"read3_interaction_A{N}"))

            # Read 4: Spearman rho, RVOL_A(N) vs net_R
            if len(hv) >= 3:
                rho, p = spearmanr(hv[rcol], hv["net_R"])
            else:
                rho, p = np.nan, np.nan
            rows.append(dict(
                cut=f"read4_spearman_A{N}", bucket="rho", half=half, n=len(hv),
                mean_net_R=float(rho) if rho is not None else np.nan, iid_t=np.nan,
                day_clustered_t=np.nan, mde=float(p) if p is not None else np.nan,
                ex_top5_mean=np.nan, fills_per_week=len(hv) / wk if wk else np.nan,
            ))

    return pd.DataFrame(rows)


def print_coverage(feat: pd.DataFrame):
    n = len(feat)
    for N, mcol in ((20, "missing_a20"), (5, "missing_a5")):
        miss = feat[mcol].sum()
        cov = 1 - miss / n
        log.info("RVOL_A(%d) coverage: %d/%d (%.1f%%) have the feature", N, n - miss, n, cov * 100)
        winners = feat[feat["net_R"] > 0]
        losers = feat[feat["net_R"] <= 0]
        wm = winners[mcol].mean() if len(winners) else np.nan
        lm = losers[mcol].mean() if len(losers) else np.nan
        log.info("  winner missingness %.1f%% vs loser missingness %.1f%% (gap %.1fpp)",
                  wm * 100, lm * 100, abs(wm - lm) * 100)
    lvl_miss = feat["missing_level"].sum()
    log.info("level-bar match failures: %d/%d (%.1f%%)", lvl_miss, n, lvl_miss / n * 100)
    thin = feat[feat["n_prior_store_days"] < 10]
    log.info("symbols/fills with <10 prior store sessions: %d/%d (%.1f%%)", len(thin), n, len(thin) / n * 100)


# ---------------------------------------------------------------------------
# Definition B (fallback, only needed because A failed the availability rail --
# see RESULT_1665.md "Coverage line"). PREREG_1665.md:
#   RVOL_B = CV(d, m_arm) / (ADV20(d-1) x P(m_arm))
#   ADV20 from research/overnight_high/panel_2024_2026.parquet, prior sessions only.
#   P(m) = cross-sectional median of CV(d,m)/day-volume(d) by minute, TRAIN days only.
# Coordinator's 2026-09-29 follow-up: P(m) is built from the SAME bars-store days
# already loaded for A (the fills' own (symbol,day) pairs), not a fresh broad pull.
# ---------------------------------------------------------------------------
MARKET_CLOSE_MIN = 16 * 60  # 960, end of the regular session grid for day-volume/P(m)
MINUTE_GRID = list(range(MARKET_OPEN_MIN, MARKET_CLOSE_MIN + 1))  # 570..960 inclusive


def load_adv20_index(symbols):
    """symbol -> (sorted bar_date as int64 ns array, adv20 array), for asof lookups."""
    cols = ["symbol", "bar_date", "adv20"]
    df = pd.read_parquet(PANEL_PARQUET, columns=cols)
    df = df[df["symbol"].isin(set(symbols))].copy()
    df["bar_date"] = pd.to_datetime(df["bar_date"]).values.astype("int64")
    df = df.sort_values(["symbol", "bar_date"])
    idx = {}
    for sym, g in df.groupby("symbol"):
        idx[sym] = (g["bar_date"].to_numpy(), g["adv20"].to_numpy())
    return idx


def asof_adv20(adv20_idx, symbol, date_str):
    """ADV20 from the last panel row STRICTLY BEFORE date (prior sessions only --
    same causal convention cell 1663 used for ATR14: 'last panel bar strictly
    before the fill date')."""
    if symbol not in adv20_idx:
        return np.nan
    dates, vals = adv20_idx[symbol]
    d = pd.Timestamp(date_str).value
    i = bisect.bisect_left(dates, d) - 1
    if i < 0:
        return np.nan
    return float(vals[i])


def day_total_volume(day_data):
    """Regular-session total volume (09:30-16:00 ET) for one (symbol,day)."""
    cv, n = cv_before(day_data, MARKET_CLOSE_MIN + 1)  # end<=960+1 => start<=960, i.e. incl. 960
    return cv if n > 0 else 0.0


def run_definition_b(feat: pd.DataFrame) -> pd.DataFrame:
    """One more bounded per-symbol pass over bars_sip.db (same query shape as A) to
    get (a) CV(d, m_arm) for every fill with a known m_arm [B's numerator -- reuses
    A's own m_arm, no re-search for the level bar] and (b) TRAIN-H2 cumulative
    volume-fraction curves on the 570..960 minute grid [feeds the cross-sectional
    median P(m)]. Returns feat with an added/overwritten `rvol_b` column.
    """
    con = sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True)
    cur = con.cursor()
    symbols = sorted(feat["symbol"].unique())
    cv_at_marm = {}
    train_curves = []
    t0 = time.time()
    for i, sym in enumerate(symbols):
        sub = feat[feat["symbol"] == sym]
        lo = (pd.Timestamp(sub["day"].min()) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        hi = sub["day"].max()
        bars = fetch_symbol_bars(cur, sym, lo, hi)

        known = sub[sub["missing_level"] == False]  # noqa: E712 -- pandas bool column
        for _, r in known.iterrows():
            day_data = bars.get(r["day"])
            if day_data is None:
                continue  # shouldn't happen (m_arm came from this same bars source); log via count below
            cv, n = cv_before(day_data, int(r["m_arm"]))
            if n > 0:
                cv_at_marm[r["fill_id"]] = cv

        train_days = sub[sub["split"] == "TRAIN-H2"]["day"].unique()
        for day in train_days:
            day_data = bars.get(day)
            if day_data is None:
                continue
            dtot = day_total_volume(day_data)
            if dtot <= 0:
                continue
            curve = np.array([cv_before(day_data, m)[0] for m in MINUTE_GRID], dtype=float) / dtot
            train_curves.append(curve)

        if (i + 1) % 200 == 0:
            log.info("[defB] %d/%d symbols, elapsed %.0fs", i + 1, len(symbols), time.time() - t0)

    n_missing_cv = len(feat[feat["missing_level"] == False]) - len(cv_at_marm)  # noqa: E712
    log.info("[defB] DONE %d symbols in %.0fs; CV(d,m_arm) resolved for %d fills (%d unexpected re-fetch misses)",
              len(symbols), time.time() - t0, len(cv_at_marm), n_missing_cv)

    P = np.nanmedian(np.vstack(train_curves), axis=0) if train_curves else np.full(len(MINUTE_GRID), np.nan)
    log.info("[defB] P(m) built from %d TRAIN-H2 (symbol,day) curves", len(train_curves))

    adv20_idx = load_adv20_index(symbols)
    minute_to_idx = {m: i for i, m in enumerate(MINUTE_GRID)}

    rvol_b = []
    for _, r in feat.iterrows():
        fid = r["fill_id"]
        if fid not in cv_at_marm:
            rvol_b.append(np.nan)
            continue
        m = int(r["m_arm"])
        pidx = minute_to_idx.get(m)
        p_m = P[pidx] if pidx is not None else np.nan
        adv20 = asof_adv20(adv20_idx, r["symbol"], r["day"])
        denom = adv20 * p_m if (not np.isnan(adv20) and not np.isnan(p_m) and p_m > 0) else np.nan
        rvol_b.append(cv_at_marm[fid] / denom if denom and denom > 0 else np.nan)

    feat = feat.copy()
    feat["rvol_b"] = rvol_b
    n = len(feat)
    miss = feat["rvol_b"].isna().sum()
    cov = 1 - miss / n
    winners = feat[feat["net_R"] > 0]
    losers = feat[feat["net_R"] <= 0]
    wm = winners["rvol_b"].isna().mean()
    lm = losers["rvol_b"].isna().mean()
    log.info("[defB] RVOL_B coverage: %d/%d (%.1f%%); winner missing %.1f%% vs loser missing %.1f%% (gap %.1fpp)",
              n - miss, n, cov * 100, wm * 100, lm * 100, abs(wm - lm) * 100)
    return feat


def generate_reads_b(feat: pd.DataFrame) -> pd.DataFrame:
    """Reads 1 (terciles+quintiles) and 3 (stop-bucket x tercile interaction) for
    RVOL_B only -- the coordinator's 2026-09-29 follow-up scoped B to these two."""
    rows = []
    floored = feat
    wk_denom = {h: weeks_spanned(floored[floored["split"] == h]["day"]) for h in floored["split"].unique()}
    valid = floored[floored["rvol_b"].notna()].copy()
    if len(valid) == 0:
        log.warning("RVOL_B: zero valid rows, no reads generated")
        return pd.DataFrame(rows)
    valid["tercile"] = pd.qcut(valid["rvol_b"], 3, labels=["T1(low)", "T2(mid)", "T3(high)"], duplicates="drop")
    valid["quintile"] = pd.qcut(valid["rvol_b"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

    for half in sorted(valid["split"].unique()):
        hv = valid[valid["split"] == half]
        wk = wk_denom.get(half, np.nan)
        for t in hv["tercile"].cat.categories:
            rows.append(row_stats(hv[hv["tercile"] == t], wk, str(t), half, "read1_tercile_B"))
        for q in hv["quintile"].cat.categories:
            rows.append(row_stats(hv[hv["quintile"] == q], wk, str(q), half, "read1_quintile_B"))
        for b in sorted(hv["bucket_r"].dropna().unique()):
            for t in hv["tercile"].cat.categories:
                cell = hv[(hv["bucket_r"] == b) & (hv["tercile"] == t)]
                rows.append(row_stats(cell, wk, f"{b}|{t}", half, "read3_interaction_B"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mode", default="A", choices=["A", "B"])
    ap.add_argument("--stats-only", action="store_true", help="skip feature build, read existing 1665_features.csv")
    ap.add_argument("--defB", action="store_true", help="compute Definition B, append its reads, update rvol_b column")
    args = ap.parse_args()

    if args.defB:
        log.info("=== 1665_rvol --defB (coordinator follow-up 2026-09-29) ===")
        feat = pd.read_csv(OUT_FEATURES)
        feat = run_definition_b(feat)
        tmp = OUT_FEATURES + ".defB.tmp"
        feat.to_csv(tmp, index=False)
        os.replace(tmp, OUT_FEATURES)
        log.info("rvol_b column written into %s (atomic replace)", OUT_FEATURES)

        reads_b = generate_reads_b(feat)
        reads_path = f"{ROOT}/research/hod_entry/1665_reads.csv"
        existing = pd.read_csv(reads_path)
        combined = pd.concat([existing, reads_b], ignore_index=True)
        tmp2 = reads_path + ".tmp"
        combined.to_csv(tmp2, index=False)
        os.replace(tmp2, reads_path)
        log.info("appended %d Definition-B rows to %s (%d total)", len(reads_b), reads_path, len(combined))
        return

    if args.stats_only:
        log.info("=== 1665_rvol --stats-only ===")
        feat = pd.read_csv(OUT_FEATURES)
        print_coverage(feat)
        reads = generate_reads(feat)
        reads.to_csv(f"{ROOT}/research/hod_entry/1665_reads.csv", index=False)
        log.info("wrote 1665_reads.csv (%d rows)", len(reads))
        return

    log.info("=== 1665_rvol start, mode=%s resume=%s ===", args.mode, args.resume)
    pop = build_population()

    done_symbols = set()
    if (args.resume or os.path.exists(OUT_CHECKPOINT)) and os.path.exists(OUT_CHECKPOINT):
        with open(OUT_CHECKPOINT) as f:
            done_symbols = set(line.strip() for line in f if line.strip())
        log.info("resume: %d symbols already done", len(done_symbols))

    write_header = not (os.path.exists(OUT_PARTIAL) and done_symbols)
    con = sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True)
    cur = con.cursor()

    symbols = sorted(pop["symbol"].unique())
    t0 = time.time()
    n_done = 0
    with open(OUT_PARTIAL, "a") as fh, open(OUT_CHECKPOINT, "a") as ckf:
        for sym in symbols:
            if sym in done_symbols:
                continue
            sub = pop[pop["symbol"] == sym]
            rows = process_symbol(cur, sym, sub, args.mode)
            df = pd.DataFrame(rows)
            df.to_csv(fh, header=write_header, index=False)
            write_header = False
            fh.flush()
            ckf.write(sym + "\n")
            ckf.flush()
            n_done += 1
            if n_done % 200 == 0:
                log.info("%d/%d symbols, elapsed %.0fs", n_done, len(symbols), time.time() - t0)

    log.info("DONE %d symbols in %.0fs", n_done, time.time() - t0)
    os.replace(OUT_PARTIAL, OUT_FEATURES)
    log.info("wrote %s", OUT_FEATURES)

    feat = pd.read_csv(OUT_FEATURES)
    print_coverage(feat)
    reads = generate_reads(feat)
    reads.to_csv(f"{ROOT}/research/hod_entry/1665_reads.csv", index=False)
    log.info("wrote 1665_reads.csv (%d rows)", len(reads))


if __name__ == "__main__":
    main()
