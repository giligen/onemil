"""PREREG 1,734: does the HOD-break book depend on the SPY market regime?

Read-only on data/cache.db, data/trades.db and 1681_per_fill.csv. Writes 1734_fills.csv and 1734_buckets.csv.
Features are computed at the PRIOR session's close (causal). Run: python3 research/hod_entry/1734_score.py
"""
import sqlite3
import numpy as np
import pandas as pd

HERE = "/home/ec2-user/onemil/research/hod_entry/"
ROOT = "/home/ec2-user/onemil/"
N_SHUF = 500
SEED = 1734
FEATURES = ["spy_ret20", "spy_below_50", "spy_ret1"]
READS = ["TRAIN-H2", "VAL", "FORWARD", "POOLED"]


def load_spy() -> pd.DataFrame:
    """Return SPY daily closes (read-only) with the three causal features known at each session's own close."""
    con = sqlite3.connect(f"file:{ROOT}data/cache.db?mode=ro", uri=True)
    spy = pd.read_sql_query("select bar_date, close from daily_bars where symbol='SPY' order by bar_date", con)
    con.close()
    spy["bar_date"] = spy["bar_date"].astype(str).str[:10]
    c = spy["close"]
    spy["ret20"] = c / c.shift(20) - 1          # close[t-1]/close[t-21]-1 when read at session t
    spy["below50"] = (c < c.rolling(50).mean()).astype(float).where(c.rolling(50).count() == 50)
    spy["ret1"] = c / c.shift(1) - 1
    return spy


def prior_features(dates: pd.Series, spy: pd.DataFrame) -> pd.DataFrame:
    """Map each fill date t to the features of the last SPY session strictly before t."""
    idx = spy["bar_date"].values
    pos = np.searchsorted(idx, dates.values, side="left") - 1
    ok = pos >= 0
    out = pd.DataFrame(index=dates.index)
    out["spy_prior_date"] = np.where(ok, idx[np.clip(pos, 0, None)], None)
    out["spy_ret20"] = np.where(ok, spy["ret20"].values[np.clip(pos, 0, None)], np.nan)
    out["spy_below_50"] = np.where(ok, spy["below50"].values[np.clip(pos, 0, None)], np.nan)
    out["spy_ret1"] = np.where(ok, spy["ret1"].values[np.clip(pos, 0, None)], np.nan)
    return out


def load_base() -> pd.DataFrame:
    """Base book: de-duplicated 1681_per_fill.csv (live-rule NET R per fill)."""
    d = pd.read_csv(HERE + "1681_per_fill.csv").drop_duplicates("fill_id")
    d = d[["read", "fill_id", "date", "symbol", "base_R"]].copy()
    d["fill_id"] = d["fill_id"].astype(str)
    return d


def load_forward() -> pd.DataFrame:
    """Forward paper HOD book from trades.db, R = pnl / total_risk."""
    con = sqlite3.connect(f"file:{ROOT}data/trades.db?mode=ro", uri=True)
    f = pd.read_sql_query(
        "select id, trade_date, symbol, pnl, total_risk from trades where strategy='hod_break' and account='paper' "
        "and trade_date >= '2026-09-25' and pnl is not null", con)
    con.close()
    f = f[f["total_risk"] > 0]
    return pd.DataFrame({"read": "FORWARD", "fill_id": "F" + f["id"].astype(str), "date": f["trade_date"].astype(str).str[:10],
                         "symbol": f["symbol"], "base_R": f["pnl"] / f["total_risk"]})


def tercile_labels(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """bottom (<=lo) / mid / top (>hi) with frozen cut points; NaN stays NaN."""
    lab = pd.Series(np.where(x <= lo, "bottom", np.where(x > hi, "top", "mid")), index=x.index)
    return lab.where(x.notna())


def clustered_se(r: np.ndarray, days: np.ndarray) -> float:
    """SE of the mean with clusters = days: sqrt(sum_g (sum_i (r_i - m))^2) / n."""
    n = len(r)
    if n < 2:
        return np.nan
    dev = pd.Series(r - r.mean()).groupby(days).sum().values
    return float(np.sqrt((dev ** 2).sum()) / n)


def bucket_stats(g: pd.DataFrame) -> dict:
    """All per-bucket numbers named in the PREREG."""
    r = g["base_R"].values
    n = len(r)
    if n == 0:
        return dict(n=0)
    days = g["date"].values
    m = r.mean()
    se_c = clustered_se(r, days)
    sd = r.std(ddof=1) if n > 1 else np.nan
    k = int(np.ceil(0.05 * n))
    trimmed = np.sort(r)[: n - k] if n - k > 0 else np.array([np.nan])
    dsum = pd.Series(r).groupby(days).sum()
    return dict(n=n, n_days=len(dsum), mean_R=m, t_day=m / se_c if se_c and se_c > 0 else np.nan,
                t_iid=m / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan, ex_top5_mean=trimmed.mean(),
                capped3_mean=np.minimum(r, 3).mean(), green_day_share=(dsum > 0).mean(), sum_R=r.sum(), se_day=se_c)


def runs(labels: pd.Series, target) -> int:
    """Count maximal runs of consecutive sessions carrying `target` (regime episodes)."""
    v = (labels == target).values
    return int(((v[1:] & ~v[:-1]).sum()) + (1 if len(v) and v[0] else 0))


def delta_and_placebo(g: pd.DataFrame, a, b, rng) -> dict:
    """Delta = mean(a) - mean(b) with day-clustered SE and the placebo percentile (day-level label shuffles)."""
    ga, gb = g[g["bucket"] == a], g[g["bucket"] == b]
    if len(ga) < 2 or len(gb) < 2:
        return dict(delta=np.nan, delta_se=np.nan, delta_t=np.nan, placebo_pct=np.nan)
    d = ga["base_R"].mean() - gb["base_R"].mean()
    se = np.hypot(clustered_se(ga["base_R"].values, ga["date"].values), clustered_se(gb["base_R"].values, gb["date"].values))
    day_lab = g.groupby("date")["bucket"].first()
    day_idx = {dd: i for i, dd in enumerate(day_lab.index)}
    di = g["date"].map(day_idx).values
    r = g["base_R"].values
    labs = day_lab.values
    nulls = np.empty(N_SHUF)
    for s in range(N_SHUF):
        pl = rng.permutation(labs)[di]
        ma, mb = r[pl == a], r[pl == b]
        nulls[s] = (ma.mean() - mb.mean()) if len(ma) and len(mb) else np.nan
    pct = 100.0 * np.nanmean(nulls < d)
    return dict(delta=d, delta_se=se, delta_t=d / se if se > 0 else np.nan, placebo_pct=pct)


def main() -> None:
    """Build the fills table, score every feature x read x bucket, evaluate the pass bar, write the CSVs."""
    spy = load_spy()
    base, fwd = load_base(), load_forward()
    allf = pd.concat([base, fwd], ignore_index=True)
    allf = pd.concat([allf, prior_features(allf["date"], spy)], axis=1)
    print(f"base fills {len(base)} (expect 5506), days {base['date'].nunique()}, forward fills {len(fwd)} (expect ~311)")
    print(f"missing features: {allf[['spy_ret20','spy_below_50','spy_ret1']].isna().sum().to_dict()}")

    # frozen cut points: one per DAY of the base book
    bdays = allf[allf["read"].isin(["TRAIN-H2", "VAL"])].drop_duplicates("date")
    cuts = {f: (bdays[f].quantile(1 / 3), bdays[f].quantile(2 / 3)) for f in ["spy_ret20", "spy_ret1"]}
    print("frozen tercile cuts (day-level, whole base book):", cuts)
    for f, (lo, hi) in cuts.items():
        allf[f + "_bkt"] = tercile_labels(allf[f], lo, hi)
    allf["spy_below_50_bkt"] = pd.Series(np.where(allf["spy_below_50"] == 1, "bottom", "top"), index=allf.index).where(allf["spy_below_50"].notna())
    allf[["fill_id", "date", "symbol", "base_R", "read", "spy_prior_date"] + FEATURES + [f + "_bkt" for f in FEATURES]].to_csv(
        HERE + "1734_fills.csv", index=False)

    # SPY session calendars labelled with the same cuts (frequency + regime episodes)
    spy["spy_ret20"], spy["spy_ret1"], spy["spy_below_50"] = spy["ret20"], spy["ret1"], spy["below50"]
    prior = spy.copy()
    for f in FEATURES:   # feature as known the PRIOR session = shift(1) on the calendar of fill sessions
        prior[f] = prior[f].shift(1)
    for f, (lo, hi) in cuts.items():
        prior[f + "_bkt"] = tercile_labels(prior[f], lo, hi)
    prior["spy_below_50_bkt"] = pd.Series(np.where(prior["spy_below_50"] == 1, "bottom", "top"), index=prior.index).where(prior["spy_below_50"].notna())

    rows, rng = [], np.random.default_rng(SEED)
    passes = {}
    for f in FEATURES:
        bk = f + "_bkt"
        pair = ("bottom", "top")
        for rd in READS:
            if rd == "POOLED":
                g = allf[allf["read"].isin(["TRAIN-H2", "VAL"])]
            else:
                g = allf[allf["read"] == rd]
            g = g[g[bk].notna()].copy()
            g["bucket"] = g[bk]
            lo_d, hi_d = g["date"].min(), g["date"].max()
            cal = prior[(prior["bar_date"] >= lo_d) & (prior["bar_date"] <= hi_d)]
            dp = delta_and_placebo(g, *pair, rng)
            for b in sorted(g["bucket"].unique()):
                st = bucket_stats(g[g["bucket"] == b])
                ns = int((cal[bk] == b).sum())
                st.update(feature=f, read=rd, bucket=b, n_sessions_cal=ns, fills_per_session=st["n"] / ns if ns else np.nan,
                          fills_per_fillday=st["n"] / st["n_days"], runs=runs(cal[bk], b))
                if b == pair[0]:
                    st.update(dp)
                rows.append(st)
            passes[(f, rd)] = (bucket_stats(g[g["bucket"] == "bottom"]), dp)
    res = pd.DataFrame(rows)
    cols = ["feature", "read", "bucket", "n", "n_days", "mean_R", "t_day", "t_iid", "ex_top5_mean", "capped3_mean", "green_day_share",
            "sum_R", "se_day", "n_sessions_cal", "fills_per_session", "fills_per_fillday", "runs", "delta", "delta_se", "delta_t", "placebo_pct"]
    res = res[cols]
    res.to_csv(HERE + "1734_buckets.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.3f}")
    print(res.to_string(index=False))

    print("\nPASS BAR (bottom mean >= +0.10 & day-t >= 2.0 & ex-top5 >= 0 & placebo >= 95 in TRAIN-H2 AND VAL; forward same sign)")
    for f in FEATURES:
        ok = True
        for rd in ["TRAIN-H2", "VAL"]:
            st, dp = passes[(f, rd)]
            c = (st["mean_R"] >= 0.10, st["t_day"] >= 2.0, st["ex_top5_mean"] >= 0, dp["placebo_pct"] >= 95)
            print(f"  {f:13s} {rd:9s} mean {st['mean_R']:+.3f} t {st['t_day']:+.2f} ex5 {st['ex_top5_mean']:+.3f} plc {dp['placebo_pct']:.1f} -> {c}")
            ok &= all(c)
        fw = passes[(f, "FORWARD")][0]
        fsign = fw.get("mean_R", np.nan) > 0
        print(f"  {f:13s} FORWARD   mean {fw.get('mean_R', np.nan):+.3f} n {fw.get('n')} same sign (positive): {fsign}")
        print(f"  ==> {f}: {'PASS' if ok and fsign else 'FAIL'}")


if __name__ == "__main__":
    main()
