"""PREREG 1,772: overnight leg (MOC buy day D, MOO sell D+1) of the attention gappers.

Reads only on-disk data (read-only DB URI). Writes 1772_events.parquet, 1772_summary.csv,
1772_placebo.csv, 1772_index_frame.csv. All returns in bp. Executed exactly per PREREG_1772.md.
"""
import os, sqlite3, sys, time
import numpy as np, pandas as pd

ROOT = "/home/ec2-user/onemil"
HERE = f"{ROOT}/research/signal_search"
POP = f"{ROOT}/research/hod_entry/cell_1428_population.csv"
BARS = f"{ROOT}/research/hod_entry/1762_cache/bars"
DB = f"file:{ROOT}/data/cache.db?mode=ro"
SEC_TAF = 0.3          # bp on the sell
HAIRCUTS = (0, 10, 25)  # bp per side
POS = 150.0
rng = np.random.default_rng(1772)


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tclust(x, nights):
    """Mean, night-clustered t (cluster-robust SE of the mean) and iid t."""
    x = np.asarray(x, float); nights = np.asarray(nights)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(nights).sum().values
    se_c = np.sqrt((s ** 2).sum()) / n
    se_i = x.std(ddof=1) / np.sqrt(n)
    return m, (m / se_c if se_c > 0 else np.nan), (m / se_i if se_i > 0 else np.nan)


def load_daily():
    """Load daily_bars from mid-2025 onward in chunks (float32, categorical symbol) to bound memory."""
    c = sqlite3.connect(DB, uri=True)
    parts = []
    q = ("select symbol, bar_date, open, high, low, close, volume from daily_bars "
         "where bar_date >= '2025-06-01' and close>0 and open>0")
    for ch in pd.read_sql(q, c, chunksize=400000):
        for k in ("open", "high", "low", "close"):
            ch[k] = ch[k].astype("float32")
        ch["volume"] = ch.volume.astype("float32")
        parts.append(ch)
    c.close()
    d = pd.concat(parts, ignore_index=True)
    d["symbol"] = d.symbol.astype("category"); d["bar_date"] = d.bar_date.astype("category")
    return d


def build_events(daily, pop):
    """One row per population symbol-night with D/D+1 fields, SPY excess, capacity."""
    spy = daily[daily.symbol == "SPY"].copy(); spy["bar_date"] = spy.bar_date.astype(str); spy = spy.sort_values("bar_date").reset_index(drop=True)
    cal = spy.bar_date.tolist()
    nxt = dict(zip(cal[:-1], cal[1:])); prv = dict(zip(cal[1:], cal[:-1]))
    spy_on = {}
    for a, b in zip(spy.iloc[:-1].itertuples(), spy.iloc[1:].itertuples()):
        spy_on[a.bar_date] = (b.open / a.close - 1) * 1e4
    ev = pop.rename(columns={"day": "day"}).copy()
    ev["next_day"] = ev.day.map(nxt); ev["prev_day"] = ev.day.map(prv)
    dl = daily[daily.symbol.isin(pop.symbol.unique())].copy(); dl["symbol"] = dl.symbol.astype(str); dl["bar_date"] = dl.bar_date.astype(str)
    def pull(df, dcol, cols, suffix):
        m = dl[["symbol", "bar_date"] + cols].rename(columns={"bar_date": dcol, **{k: k + suffix for k in cols}})
        return df.merge(m, on=["symbol", dcol], how="left")
    ev = pull(ev, "day", ["open", "high", "low", "close", "volume"], "_D")
    ev = pull(ev, "next_day", ["open", "close", "volume"], "_N")
    ev = pull(ev, "prev_day", ["close"], "_prev")
    ev = ev.rename(columns={"close_prev": "close_prev"})
    ev["spy_on"] = ev.day.map(spy_on)
    for k in ev.columns:
        if str(ev[k].dtype) == "float32":
            ev[k] = ev[k].astype("float64")
    ev["r_on"] = (ev.open_N / ev.close_D - 1) * 1e4
    ev["excess"] = ev.r_on - ev.spy_on
    ev["gap_pct"] = (ev.open_D / ev.close_prev - 1) * 100
    ev["oc_D"] = (ev.close_D / ev.open_D - 1) * 1e4
    ev["close_loc"] = (ev.close_D - ev.low_D) / (ev.high_D - ev.low_D).replace(0, np.nan)
    return ev, cal, spy_on


def add_capacity(ev):
    """16:00 ET bar dollar volume on D from the minute-bar cache; D+1 daily dollar volume proxy."""
    cl = {}
    for day in sorted(ev.day.unique()):
        f = f"{BARS}/{day}.parquet"
        if not os.path.exists(f):
            continue
        b = pd.read_parquet(f)
        t = pd.to_datetime(b.t, utc=True).dt.tz_convert("America/New_York")
        b = b[(t.dt.hour == 16) & (t.dt.minute == 0)]
        for s, vv, cc in zip(b.symbol, b.v, b.c):
            cl[(day, s)] = vv * cc
    ev["close_print_usd"] = [cl.get((d, s), np.nan) for d, s in zip(ev.day, ev.symbol)]
    ev["dvol_N"] = ev.volume_N * ev.close_N
    ev["cap_ok"] = (POS <= 0.05 * ev.close_print_usd) & (POS <= 0.02 * ev.dvol_N)
    return ev


def stats(df, col="excess", extra_cost=0.0):
    """Primary-style stats for a frame: mean, night-t, iid t, ex-top5, capped, worst1%, weekly P10, n."""
    d = df.dropna(subset=[col])
    if len(d) < 3:
        return dict(n=len(d))
    x = d[col].values - extra_cost
    m, tc, ti = tclust(x, d.day.values)
    k = int(np.ceil(0.05 * len(x)))
    ex5 = np.sort(x)[:-k].mean()
    capped = np.minimum(x, 5000.0).mean()
    w = max(1, int(np.ceil(0.01 * len(x))))
    worst = np.sort(x)[:w].mean()
    nm = pd.Series(x).groupby(d.day.values).mean()
    wk = nm.groupby(pd.to_datetime(nm.index).to_period("W")).sum()
    return dict(n=len(x), nights=d.day.nunique(), mean=m, t_night=tc, t_iid=ti, ex_top5=ex5,
                capped50=capped, worst1pct_mean=worst, weekly_p10=np.percentile(wk.values, 10),
                median_names_night=float(d.groupby("day").size().median()))


def placebo(daily, ev, cal, spy_on):
    """(i) count-matched random universe draws on the same nights; (ii) same symbol +5 trading days."""
    log("placebo: building universe overnight table")
    idx = {d: i for i, d in enumerate(cal)}
    u = daily[(daily.symbol != "SPY") & (daily.close >= 1) & (daily.close * daily.volume >= 1e6)]
    u = pd.DataFrame(dict(symbol=u.symbol.values, di=u.bar_date.astype(str).map(idx).astype("float32").values,
                          close=u.close.values, open=u.open.values))
    full = daily[daily.symbol != "SPY"]
    nxo = pd.DataFrame(dict(symbol=full.symbol.values, di=(full.bar_date.astype(str).map(idx) - 1).astype("float32").values,
                            next_open=full.open.values))
    del full
    u = u.merge(nxo, on=["symbol", "di"], how="inner"); del nxo
    u["bar_date"] = u.di.astype(int).map(lambda i: cal[i])
    u["excess"] = (u.next_open.astype("float64") / u.close.astype("float64") - 1) * 1e4 - u.bar_date.map(spy_on)
    u = u.dropna(subset=["excess"]); u["symbol"] = u.symbol.astype(str)
    by_night = {d: x.excess.values for d, x in u.groupby("bar_date")}
    out = []
    ev = ev.dropna(subset=["excess"])
    for sp in ("TRAIN", "VAL"):
        e = ev[ev.split == sp]
        cnt = e.groupby("day").size()
        means = np.empty(200)
        for i in range(200):
            tot = 0.0; n = 0
            for d, k in cnt.items():
                pool = by_night.get(d)
                if pool is None or len(pool) < k:
                    continue
                tot += rng.choice(pool, k, replace=False).sum(); n += k
            means[i] = tot / n
        out.append(dict(split=sp, placebo="i_random_universe", mean=means.mean(), p95=np.percentile(means, 95),
                        sd=means.std(), draws=200, primary_mean=e.excess.mean()))
        # (ii) same symbol, 5 trading days later (SPY calendar index +5)
        idx = {d: i for i, d in enumerate(cal)}
        idx = {d: i for i, d in enumerate(cal)}
        e5 = e[["day", "symbol"]].copy()
        e5["d5"] = e5.day.map(lambda d: cal[idx[d] + 5] if d in idx and idx[d] + 5 < len(cal) else None)
        e5 = e5.dropna(subset=["d5"]).merge(u[["symbol", "bar_date", "excess"]].rename(columns={"bar_date": "d5"}),
                                           on=["symbol", "d5"], how="inner")
        vals, nts = e5.excess.values, e5.day.values
        m, tc, _ = tclust(vals, nts)
        se = abs(m / tc) if tc and tc == tc and tc != 0 else np.nan
        out.append(dict(split=sp, placebo="ii_same_name_plus5", mean=m, p95=m + 1.645 * se, sd=se,
                        draws=len(vals), primary_mean=e.excess.mean()))
    return pd.DataFrame(out)


def index_frame():
    """SPY/QQQ close->open vs open->close per year from 1-min parquets."""
    rows = []
    for tk in ("SPY", "QQQ"):
        b = pd.read_parquet(f"{ROOT}/research/index_orb/cache/{tk}_1min.parquet")
        ts = pd.to_datetime(b.timestamp, utc=True).dt.tz_convert("America/New_York")
        b = b.assign(d=ts.dt.date.astype(str), tm=ts.dt.strftime("%H:%M"))
        b = b[(b.tm >= "09:30") & (b.tm <= "15:59")].sort_values("timestamp")
        g = b.groupby("d")
        day = pd.DataFrame(dict(o=g.open.first(), c=g.close.last()))
        day["on"] = (day.o.shift(-1) / day.c - 1) * 1e4
        day["oc"] = (day.c / day.o - 1) * 1e4
        day["yr"] = day.index.str[:4]
        for y, z in day.groupby("yr"):
            for leg in ("on", "oc"):
                x = z[leg].dropna()
                rows.append(dict(ticker=tk, year=y, leg=leg, n=len(x), mean_bp=x.mean(),
                                 sharpe=x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    log("load population + daily_bars")
    pop = pd.read_csv(POP)
    daily = load_daily()
    ev, cal, spy_on = build_events(daily, pop)
    ev = add_capacity(ev)
    ev["has_D"] = ev.close_D.notna(); ev["has_N"] = ev.open_N.notna()
    ev["cov"] = ev.has_D & ev.has_N
    ev["winner"] = ev.oc_D > 0
    ev["tier"] = pd.cut(ev.close_D, [0, 5, 20, 1e9], labels=["1-5", "5-20", ">20"])
    ev["loc3"] = pd.cut(ev.close_loc, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["bottom", "middle", "top"])
    tr = ev[(ev.split == "TRAIN") & ev.gap_pct.notna()]
    qs = tr.gap_pct.quantile([1 / 3, 2 / 3]).values
    ev["gap_terc"] = pd.cut(ev.gap_pct, [-1e9, qs[0], qs[1], 1e9], labels=["low", "mid", "high"])
    ev["fade"] = np.where(ev.oc_D < 0, "fade", "held")
    ud = sorted(ev[ev.split == "TRAIN"].day.unique()); mid = ud[len(ud) // 2]
    ev["half"] = np.where(ev.split != "TRAIN", "", np.where(ev.day <= mid, "H1", "H2"))
    ev.to_parquet(f"{HERE}/1772_events.parquet")

    S = []
    def add(name, df, **kw):
        for h in HAIRCUTS:
            s = stats(df, extra_cost=2 * h + SEC_TAF); s.update(read=name, haircut_bp_side=h, **kw); S.append(s)
        s = stats(df, "r_on"); s.update(read=name + "|raw_r_on", haircut_bp_side=-1, **kw); S.append(s)

    log("gap distribution")
    gd = ev.gap_pct.describe(percentiles=[.1, .25, .5, .75, .9]).to_dict()
    log("gap_pct", {k: round(v, 2) for k, v in gd.items()})
    cov = {}
    for sp, e in list(ev.groupby("split")) + [("ALL", ev)]:
        dd = e[e.has_D]
        cov[sp] = dict(cov_all=e["cov"].mean(), cov_N_given_D=dd.has_N.mean(),
                       cov_winner=dd[dd.winner].has_N.mean(), cov_loser=dd[~dd.winner].has_N.mean(),
                       n=len(e))
    log("coverage", cov)
    c = cov["ALL"]; rail_ok = c["cov_N_given_D"] >= 0.8 and abs(c["cov_winner"] - c["cov_loser"]) <= 0.05
    exe = ev[ev["cov"] & ev.cap_ok]
    exec_share = {sp: (ev[ev["cov"] & (ev.split == sp)].cap_ok.mean()) for sp in ("TRAIN", "VAL")}
    log("availability rail", rail_ok, "exec share", exec_share)

    for sp in ("TRAIN", "VAL"):
        e_all = ev[(ev.split == sp) & ev["cov"]]; e_x = exe[exe.split == sp]
        add("primary_exec", e_x, split=sp); add("all_covered", e_all, split=sp)
        add("nonexec", e_all[~e_all.cap_ok], split=sp)
        for col, nm in (("fade", "a"), ("tier", "b"), ("loc3", "c"), ("gap_terc", "d")):
            for lv, z in e_x.groupby(col, observed=True):
                add(f"sec_{nm}_{col}={lv}", z, split=sp)
        add("sec_f_short_mirror(report-only)", e_x.assign(excess=-e_x.excess), split=sp)
    for hf in ("H1", "H2"):
        add(f"sec_e_TRAIN_{hf}", exe[(exe.split == "TRAIN") & (exe.half == hf)], split="TRAIN")
    summ = pd.DataFrame(S)
    summ.to_csv(f"{HERE}/1772_summary.csv", index=False)

    log("placebos")
    pl = placebo(daily, ev[ev["cov"] & ev.cap_ok], cal, spy_on)
    pl.to_csv(f"{HERE}/1772_placebo.csv", index=False)
    print(pl.to_string(), flush=True)
    log("index frame")
    ix = index_frame(); ix.to_csv(f"{HERE}/1772_index_frame.csv", index=False)
    print(ix.groupby(["ticker", "leg"]).agg(n=("n", "sum"), mean=("mean_bp", "mean")).to_string(), flush=True)

    pd.set_option("display.width", 250)
    cols = ["read", "split", "haircut_bp_side", "n", "nights", "mean", "t_night", "t_iid", "ex_top5", "capped50",
            "worst1pct_mean", "weekly_p10", "median_names_night"]
    print(summ[cols].round(2).to_string(), flush=True)
    # verdict
    v = summ[(summ.read == "primary_exec") & (summ.split == "VAL") & (summ.haircut_bp_side == 10)].iloc[0]
    h = {k: summ[(summ.read == f"sec_e_TRAIN_{k}") & (summ.haircut_bp_side == 10)].iloc[0]["mean"] for k in ("H1", "H2")}
    p_i = pl[(pl.split == "VAL") & (pl.placebo == "i_random_universe")].iloc[0]
    p_ii = pl[(pl.split == "VAL") & (pl.placebo == "ii_same_name_plus5")].iloc[0]
    # placebos are on the gross excess scale; compare gross primary excess to the p95s
    prim_gross = ev[(ev.split == "VAL") & ev["cov"] & ev.cap_ok].excess.mean()
    checks = dict(net_ge_30=v["mean"] >= 30, t_ge_2=v.t_night >= 2, halves_same_sign=(h["H1"] > 0) == (h["H2"] > 0) and h["H1"] > 0,
                  ex5_pos=v.ex_top5 > 0, beats_i=prim_gross > p_i.p95, beats_ii=prim_gross > p_ii.p95,
                  rail=rail_ok, exec_ge_60=exec_share["VAL"] >= 0.6, med_names_ge_3=v.median_names_night >= 3)
    log("CHECKS", checks, "PASS" if all(checks.values()) else "FAIL", "prim_gross", prim_gross)


if __name__ == "__main__":
    main()
