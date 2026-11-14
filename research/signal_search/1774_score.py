"""PREREG 1,774 scoring: gap-fade short MOO, cover at measured ask at 10:30 ET (also 10:00 / 11:00, MID).

Reuses stats/tclust/weekly_R and cost lines from 1773_score.py. Writes 1774_events.parquet, 1774_summary.csv, 1774_placebo.csv.
Deviations (declared): quote price-sanity reference = 1762 bar open at 10:30 where present, else the day open; the bar cache holds
only population symbols, so placebo (i) draws count-matched random POPULATION events from the same split (gross 09:30 open ->
10:30 bar open minus the measured half-spread median), not the full universe.
"""
import glob, importlib.util, time
import numpy as np, pandas as pd

HERE = "/home/ec2-user/onemil/research/signal_search"
sp = importlib.util.spec_from_file_location("s73", f"{HERE}/1773_score.py"); s73 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s73)
NET0, FEES, CAPSZ = s73.NET0, s73.FEES, s73.CAPSZ
rng = np.random.default_rng(1774)
log = s73.log


def load_quotes():
    """Concatenate cached quote parquets."""
    q = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{HERE}/1774_cache/quotes/*.parquet"))], ignore_index=True)
    return q


def build():
    """Top-10/day executable rows with cleaned NBBO columns per window."""
    e = pd.read_parquet(f"{HERE}/1773_events.parquet")
    x = e[e.gap_ok & e.r_gross.notna()].sort_values(["day", "gap_pct"], ascending=[True, False]).groupby("day").head(CAPSZ).copy()
    q = load_quotes(); log("quote rows", len(q), "err", int(q.get("err", pd.Series(dtype=float)).fillna(0).sum()))
    cnt = {}
    for w in ("1000", "1030", "1100"):
        z = q[q.win == w][["day", "symbol", "bid", "ask", "n"]].copy()
        z["mid"] = (z.bid + z.ask) / 2; z["hs"] = (z.ask - z.bid) / 2 / z.mid * 1e4
        x = x.merge(z.rename(columns={"bid": "bid" + w, "ask": "ask" + w, "mid": "mid" + w, "hs": "hs" + w, "n": "nq" + w}), on=["day", "symbol"], how="left")
        ref = x.p1030.fillna(x.open) if w == "1030" else x.open
        have = x["ask" + w].notna(); raw = int(have.sum())
        bad = have & ~((x["ask" + w] > x["bid" + w]) & (x["bid" + w] > 0) & (x["hs" + w] <= 1000) & (x["mid" + w] >= .5 * ref) & (x["mid" + w] <= 2 * ref))
        cnt[w] = (len(x), raw, int(bad.sum()))
        for c in ("bid", "ask", "mid", "hs"):
            x.loc[bad, c + w] = np.nan
        x["r_ask" + w] = (x.open / x["ask" + w] - 1) * 1e4; x["r_mid" + w] = (x.open / x["mid" + w] - 1) * 1e4
    log("coverage (rows, raw quotes, dropped by cleaning):", cnt)
    return x, cnt


def pl_i(pool, e, hs_med, col="r_1030"):
    """200 count-matched random draws from pool (same split); gross-of-fees ask-equivalent mean + green-week null."""
    cnt = e.groupby("day").size(); r = pool[col].values - hs_med; means = np.empty(200); gr = np.empty(200)
    for i in range(200):
        tot, n, wk = 0.0, 0, {}
        for d, k in cnt.items():
            s = rng.choice(r, k, replace=False); tot += s.sum(); n += k
            w = pd.Timestamp(d).to_period("W"); wk[w] = wk.get(w, 0) + (300 * (s - NET0) / 1e4 / 150).sum()
        means[i] = tot / n; gr[i] = (np.array(list(wk.values())) > 0).mean()
    return means.mean(), np.percentile(means, 95), gr.mean(), np.percentile(gr, 95)


def main():
    """Run the cell."""
    x, cnt = build(); allev = pd.read_parquet(f"{HERE}/1773_events.parquet"); allev = allev[allev.gap_ok & allev.r_1030.notna()]
    x.to_parquet(f"{HERE}/1774_events.parquet")
    ok = x.r_ask1030.notna(); win = x.r_gross > 0
    cov = ok.mean(); miss = ok.groupby(win).apply(lambda s: 1 - s.mean()); gap = abs(miss.get(True, 0) - miss.get(False, 0)) * 100
    log("QUOTE COVERAGE", round(cov, 3), "missing by winner", miss.to_dict(), "gap pp", round(gap, 2), "rail", cov >= .8 and gap <= 5)
    base = x[ok]; val = base[base.split == "VAL"]; tr = base[base.split == "TRAIN"]; rows = []
    def add(name, d, col="r_ask1030", cost=NET0):
        s = s73.stats(d, col, cost); s["cell"] = name; s["hs_med"] = d["hs1030"].median() if "hs1030" in d else np.nan; rows.append(s); return s
    add("PRIMARY_VAL", val); add("PRIMARY_TRAIN", tr)
    dd = sorted(tr.day.unique()); med = dd[len(dd) // 2]
    add("TRAIN_H1", tr[tr.day <= med]); add("TRAIN_H2", tr[tr.day > med])
    for nm, d in (("VAL", val), ("TRAIN", tr)):
        for t in ["1-5", "5-20", ">20"]:
            add(f"{nm}_tier_{t}", d[d.tier == t])
        q = d.gap_pct.quantile([1 / 3, 2 / 3]).values
        add(f"{nm}_gap_T1", d[d.gap_pct <= q[0]]); add(f"{nm}_gap_T2", d[(d.gap_pct > q[0]) & (d.gap_pct <= q[1])]); add(f"{nm}_gap_T3", d[d.gap_pct > q[1]])
        add(f"{nm}_filter_c", d[d.filt_c])
        for w in ("1000", "1100"):
            dw = x[(x.split == nm) & x["r_ask" + w].notna()]; add(f"{nm}_cover_{w}", dw, "r_ask" + w)
        add(f"{nm}_cover_1030_MID_info", d, "r_mid1030")
    summ = pd.DataFrame(rows); summ.to_csv(f"{HERE}/1774_summary.csv", index=False)
    hs = base.groupby("tier", observed=True).hs1030.median().to_dict(); log("half-spread median by tier", hs, "all", base.hs1030.median())
    # placebos
    pl = []; cal = sorted(allev.day.unique()); idx = {d: i for i, d in enumerate(cal)}
    for spn, e in (("TRAIN", tr), ("VAL", val)):
        pool = allev[allev.split == spn]; hsm = e.hs1030.median()
        m, p95, g, g95 = pl_i(pool, e, hsm)
        pl.append(dict(split=spn, placebo="i_random_population_events_ask_equiv_gross", mean=m, p95=p95, green_null=g, green_null_p95=g95,
                       primary_gross_of_fees=e.r_ask1030.mean(), hs_med=hsm))
        e5 = e[["day", "symbol"]].copy(); e5["d5"] = e5.day.map(lambda d: cal[idx[d] + 5] if d in idx and idx[d] + 5 < len(cal) else None)
        e5 = e5.dropna(subset=["d5"]).merge(allev[["day", "symbol", "r_1030"]].rename(columns={"day": "d5"}), on=["symbol", "d5"])
        v = e5.r_1030.values - hsm; mm, tc, _ = s73.tclust(v, e5.day.values); se = abs(mm / tc) if tc == tc and tc else np.nan
        pl.append(dict(split=spn, placebo="ii_same_name_plus5_ask_equiv_gross", mean=mm, p95=mm + 1.645 * se, primary_gross_of_fees=e.r_ask1030.mean(), n=len(e5), hs_med=hsm))
    pl = pd.DataFrame(pl); pl.to_csv(f"{HERE}/1774_placebo.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string()); print(pl.round(2).to_string())
    f = val[val.filt_c].r_ask1030.values
    if len(f):
        pool = val.r_ask1030.values; dr = np.array([rng.choice(pool, len(f), replace=False).mean() for _ in range(2000)])
        print("filter_c VAL n", len(f), "mean gross-of-fees", f.mean(), "countmatched p95", np.percentile(dr, 95))
    print("cleaning counts (rows, raw, dropped)", cnt)


if __name__ == "__main__":
    main()
