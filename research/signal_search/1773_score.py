"""PREREG 1,773: gap-fade SHORT, MOO short / MOC cover, no stop, equal-$ $300, HTB-locate cost sensitivity.

On-disk data only (read-only DB URI). Writes 1773_events.parquet, 1773_summary.csv, 1773_placebo.csv.
Short return r = open_D/close_D - 1 in bp (positive = fade paid). Executed per PREREG_1773.md.
"""
import os, sqlite3, time
import numpy as np, pandas as pd

ROOT = "/home/ec2-user/onemil"; HERE = f"{ROOT}/research/signal_search"
POP = f"{ROOT}/research/hod_entry/cell_1428_population.csv"
BARS = f"{ROOT}/research/hod_entry/1762_cache/bars"
DB = f"file:{ROOT}/data/cache.db?mode=ro"
LOC = f"{HERE}/1773_locate_quotes_20261114.csv"
FEES, LOCATE, BORROW = 4.0, 9.0, 83.0 * 0 + 100 / 360 * 1e4 / 100  # borrow 100%/360 = 27.8 bp
POS, RISK, GAPMAX, CAPSZ = 300.0, 150.0, 200.0, 10
rng = np.random.default_rng(1773)
NET0 = FEES + LOCATE + BORROW   # primary cost, bp


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tclust(x, days):
    """Mean, day-clustered t, iid t."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    m = x.mean(); s = pd.Series(x - m).groupby(np.asarray(days)).sum().values
    se_c = np.sqrt((s ** 2).sum()) / n; se_i = x.std(ddof=1) / np.sqrt(n)
    return m, (m / se_c if se_c > 0 else np.nan), (m / se_i if se_i > 0 else np.nan)


def load_daily():
    """daily_bars from 2025-05-01 (float32, categorical symbol)."""
    c = sqlite3.connect(DB, uri=True); parts = []
    q = ("select symbol, bar_date, open, high, low, close, volume from daily_bars "
         "where bar_date >= '2025-05-01' and close>0 and open>0")
    for ch in pd.read_sql(q, c, chunksize=400000):
        for k in ("open", "high", "low", "close", "volume"):
            ch[k] = ch[k].astype("float32")
        parts.append(ch)
    c.close(); d = pd.concat(parts, ignore_index=True)
    d["symbol"] = d.symbol.astype("category"); d["bar_date"] = d.bar_date.astype("category")
    return d


def build_events(daily, pop):
    """Population rows + D fields, prior-day features, SPY open->close."""
    spy = daily[daily.symbol == "SPY"].copy(); spy["bar_date"] = spy.bar_date.astype(str)
    spy_oc = dict(zip(spy.bar_date, (spy.close / spy.open - 1).astype("float64") * 1e4))
    dl = daily[daily.symbol.isin(pop.symbol.unique())].copy()
    dl["symbol"] = dl.symbol.astype(str); dl["bar_date"] = dl.bar_date.astype(str)
    dl = dl.sort_values(["symbol", "bar_date"])
    g = dl.groupby("symbol")
    dl["close_prev"] = g.close.shift(1); dl["pd_high"] = g.high.shift(1); dl["pd_low"] = g.low.shift(1)
    dl["pd_close"] = g.close.shift(1); dl["pd_close2"] = g.close.shift(2)
    ev = pop.merge(dl.rename(columns={"bar_date": "day"}), on=["symbol", "day"], how="left")
    for k in ev.columns:
        if str(ev[k].dtype) == "float32":
            ev[k] = ev[k].astype("float64")
    ev["gap_pct"] = (ev.open / ev.close_prev - 1) * 100
    ev["r_gross"] = (ev.open / ev.close - 1) * 1e4
    ev["spy_oc"] = ev.day.map(spy_oc)
    ev["pd_range"] = (ev.pd_high - ev.pd_low) / ev.pd_low * 100
    ev["pd_ret"] = (ev.pd_close / ev.pd_close2 - 1) * 100
    ev["filt_c"] = (ev.pd_range >= 7) & (ev.pd_ret >= 0)
    ev["tier"] = pd.cut(ev.open, [0, 5, 20, 1e9], labels=["1-5", "5-20", ">20"], right=False)
    return ev, spy_oc


def add_bars(ev):
    """09:30 and 16:00 bar $vol, 10:30 open; mark capacity."""
    o930, c16, p1030 = {}, {}, {}
    for day in sorted(ev.day.unique()):
        f = f"{BARS}/{day}.parquet"
        if not os.path.exists(f):
            continue
        syms = set(ev.loc[ev.day == day, "symbol"])
        b = pd.read_parquet(f); b = b[b.symbol.isin(syms)]
        t = pd.to_datetime(b.t, utc=True).dt.tz_convert("America/New_York")
        hm = t.dt.hour * 60 + t.dt.minute
        for key, tgt, dct, val in (("a", 570, o930, b.v * b.o), ("b", 960, c16, b.v * b.c), ("c", 630, p1030, b.o)):
            m = (hm == tgt).values
            for s, v in zip(b.symbol[m], val[m]):
                dct[(day, s)] = v
    k = list(zip(ev.day, ev.symbol))
    ev["usd_0930"] = [o930.get(x, np.nan) for x in k]; ev["usd_1600"] = [c16.get(x, np.nan) for x in k]
    ev["p1030"] = [p1030.get(x, np.nan) for x in k]
    ev["cap_ok"] = (POS <= 0.02 * ev.usd_0930) & (POS <= 0.02 * ev.usd_1600)
    ev["r_1030"] = (ev.open / ev.p1030 - 1) * 1e4
    return ev


def weekly_R(d, net):
    """Weekly sum of R (= $300*net/$150), <=10 seeded-random names/day."""
    d = d.assign(net=net)
    d = d.groupby("day", group_keys=False).apply(lambda x: x.sample(min(len(x), CAPSZ), random_state=1773))
    R = POS * d.net / 1e4 / RISK
    wk = R.groupby(pd.to_datetime(d.day).dt.to_period("W")).sum()
    return wk


def stats(d, col="r_gross", cost=NET0):
    """Mean net, day-t, iid t, ex-top5/1%, worst1/5%, capped, weekly P10, green share."""
    d = d.dropna(subset=[col])
    if len(d) < 3:
        return dict(n=len(d))
    x = d[col].values - cost; xs = np.sort(x); n = len(x)
    m, tc, ti = tclust(x, d.day.values)
    k5, k1 = int(np.ceil(.05 * n)), max(1, int(np.ceil(.01 * n)))
    wk = weekly_R(d, x)
    return dict(n=n, days=d.day.nunique(), mean=m, t_day=tc, t_iid=ti, ex_top5=xs[:-k5].mean(),
                ex_top1=xs[:-k1].mean(), worst1=xs[:k1].mean(), worst5=xs[:k5].mean(),
                capped50=np.minimum(x, 5000).mean(), weekly_p10_R=np.percentile(wk, 10),
                green_wk=(wk > 0).mean(), weeks=len(wk), med_names_day=float(d.groupby("day").size().median()))


def placebo(daily, ev):
    """(i) 200 count-matched random-universe draws; (ii) same name +5 trading days. Gross bp + green-week null."""
    log("placebo: universe table")
    u = daily[(daily.symbol != "SPY") & (daily.close >= 1) & (daily.close * daily.volume >= 1e6)]
    r = ((u.open.astype("float64") / u.close.astype("float64") - 1) * 1e4).values
    u = pd.DataFrame(dict(symbol=u.symbol.astype(str).values, day=u.bar_date.astype(str).values, r=r))
    by = {d: x.r.values for d, x in u.groupby("day")}
    cal = sorted(by); idx = {d: i for i, d in enumerate(cal)}; out = []
    for sp in ("TRAIN", "VAL"):
        e = ev[(ev.split == sp)].dropna(subset=["r_gross"]); cnt = e.groupby("day").size()
        means = np.empty(200); greens = np.empty(200)
        for i in range(200):
            tot = 0.0; n = 0; wkr = {}
            for d, k in cnt.items():
                p = by.get(d)
                if p is None or len(p) < k:
                    continue
                s = rng.choice(p, k, replace=False); tot += s.sum(); n += k
                s = s[:CAPSZ] - NET0
                w = pd.Timestamp(d).to_period("W"); wkr[w] = wkr.get(w, 0) + (POS * s / 1e4 / RISK).sum()
            means[i] = tot / n; greens[i] = (np.array(list(wkr.values())) > 0).mean()
        out.append(dict(split=sp, placebo="i_random_universe_gross", mean=means.mean(), p95=np.percentile(means, 95),
                        green_null=greens.mean(), green_null_p95=np.percentile(greens, 95), primary_gross=e.r_gross.mean()))
        e5 = e[["day", "symbol"]].copy()
        e5["d5"] = e5.day.map(lambda d: cal[idx[d] + 5] if d in idx and idx[d] + 5 < len(cal) else None)
        e5 = e5.dropna(subset=["d5"]).merge(u.rename(columns={"day": "d5"}), on=["symbol", "d5"], how="inner")
        m, tc, _ = tclust(e5.r.values, e5.day.values)
        se = abs(m / tc) if tc == tc and tc else np.nan
        out.append(dict(split=sp, placebo="ii_same_name_plus5_gross", mean=m, p95=m + 1.645 * se, green_null=np.nan,
                        green_null_p95=np.nan, primary_gross=e.r_gross.mean(), n=len(e5)))
    return pd.DataFrame(out)


def main():
    """Run the cell end to end."""
    pop = pd.read_csv(POP); log("population rows", len(pop))
    daily = load_daily(); log("daily loaded", len(daily))
    ev, _ = build_events(daily, pop)
    n0 = len(ev); have = ev.r_gross.notna()
    log("coverage open/close", have.mean(), "n", n0)
    big = ev.gap_pct > GAPMAX; log("dropped gap>200%:", int(big.sum()), ev.loc[big, ["day", "symbol", "gap_pct"]].values.tolist())
    ev["gap_ok"] = ~big
    # threshold-list proxy: symbols in locate CSV with empty fee_bp (no quote) -- only the 2 error rows are in the on-disk file
    loc = pd.read_csv(LOC); thr = set(loc.loc[loc.fee_bp.isna(), "symbol"])
    ev["thr"] = ev.symbol.isin(thr); log("threshold proxy symbols", sorted(thr), "rows", int(ev.thr.sum()))
    ev = add_bars(ev)
    ev["bars_cov"] = ev.usd_0930.notna() & ev.usd_1600.notna()
    # availability rail
    win = ev.r_gross > 0; ok = ev.r_gross.notna()
    miss = ev.loc[ok].groupby(win[ok]).bars_cov.apply(lambda s: 1 - s.mean())
    log("availability: coverage", have.mean(), "bars missing by winner", miss.to_dict())
    ev.to_parquet(f"{HERE}/1773_events.parquet")
    base = ev[ev.gap_ok & ok].copy(); rows = []
    def add(name, d, cost=NET0, col="r_gross"):
        s = stats(d, col, cost); s["cell"] = name; rows.append(s); return s
    val = base[base.split == "VAL"]; tr = base[base.split == "TRAIN"]
    add("PRIMARY_VAL", val); add("PRIMARY_TRAIN", tr)
    dd = sorted(tr.day.unique()); med = dd[len(dd) // 2]
    add("TRAIN_H1", tr[tr.day <= med]); add("TRAIN_H2", tr[tr.day > med])
    for nm, c in (("loc0_bor0", FEES), ("loc9_bor0", FEES + 9), ("loc38_bor100", FEES + 38 + 27.8), ("loc113_bor300", FEES + 113 + 83.3), ("loc9_bor100", NET0)):
        add("VAL_cost_" + nm, val, c); add("TRAIN_cost_" + nm, tr, c)
    add("VAL_ex_threshold_proxy", val[~val.thr]); add("VAL_capacity_ok", val[val.cap_ok])
    for sp, d in (("VAL", val), ("TRAIN", tr)):
        for t in ["1-5", "5-20", ">20"]:
            add(f"{sp}_tier_{t}", d[d.tier == t])
        q = d.gap_pct.quantile([1 / 3, 2 / 3]).values
        add(f"{sp}_gap_T1", d[d.gap_pct <= q[0]]); add(f"{sp}_gap_T2", d[(d.gap_pct > q[0]) & (d.gap_pct <= q[1])]); add(f"{sp}_gap_T3", d[d.gap_pct > q[1]])
        add(f"{sp}_filter_c", d[d.filt_c]); add(f"{sp}_not_filter_c", d[~d.filt_c])
        add(f"{sp}_gross_1030cover", d, 0.0, "r_1030")
        add(f"{sp}_gross_nocost", d, 0.0)
        s = d.dropna(subset=["spy_oc"]).assign(hed=lambda x: x.r_gross + x.spy_oc); add(f"{sp}_spy_hedged", s, NET0, "hed")
    summ = pd.DataFrame(rows); summ.to_csv(f"{HERE}/1773_summary.csv", index=False)
    pl = placebo(daily, base)
    # filter (c) count-matched placebo p95 on VAL: random subsets of VAL with same count
    f = val[val.filt_c].r_gross.values; pool = val.r_gross.values
    dr = np.array([rng.choice(pool, len(f), replace=False).mean() for _ in range(2000)])
    pl = pd.concat([pl, pd.DataFrame([dict(split="VAL", placebo="filter_c_countmatched_subset_gross", mean=dr.mean(),
                     p95=np.percentile(dr, 95), primary_gross=f.mean() if len(f) else np.nan, n=len(f))])])
    pl.to_csv(f"{HERE}/1773_placebo.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string()); print(pl.round(2).to_string())
    print("coverage", have.mean(), "n_total", n0, "n_used", len(base), "dropped_gap", int(big.sum()))


if __name__ == "__main__":
    main()
