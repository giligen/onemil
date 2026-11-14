"""PREREG 1,775 score: headline type vs the squeeze tail of the 1,773 gap-fade short (MOO short -> MOC cover).

Reads 1773_events.parquet, 1775_cache/news, 1775_labels.parquet. Writes 1775_events.parquet, 1775_summary.csv,
1775_placebo.csv. Net = r_gross - NET0 (1,773 primary cost lines), short return in bp, positive = fade paid.
"""
import glob, importlib.util, os, time
import numpy as np, pandas as pd

HERE = "/home/ec2-user/onemil/research/signal_search"
spec = importlib.util.spec_from_file_location("s1773", f"{HERE}/1773_score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
rng = np.random.default_rng(1775)
CATS = ["MA", "DILUTION", "FDA", "CONTRACT", "EARNINGS", "ANALYST", "PR_OTHER", "NONE"]


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build():
    """Symbol-day table: category of the strongest headline (ties -> latest), max strength, n, minutes to 09:30."""
    e = pd.read_parquet(f"{HERE}/1773_events.parquet"); e = e[e.gap_ok & e.r_gross.notna()].copy()
    news = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{HERE}/1775_cache/news/*.parquet"))], ignore_index=True)
    fail = news[news.id == "FAIL"][["day", "symbol"]].assign(fail=True)
    hl = news[news.id.notna() & ~news.id.isin(["", "FAIL"])].merge(pd.read_parquet(f"{HERE}/1775_labels.parquet"), on="id", how="left")
    log("headline rows", len(hl), "unlabelled", int(hl.category.isna().sum()))
    hl["t"] = pd.to_datetime(hl.created_at, utc=True)
    hl["open_t"] = [pd.Timestamp(f"{d} 09:30", tz="America/New_York").tz_convert("UTC") for d in hl.day]
    hl["mins"] = (hl.open_t - hl.t).dt.total_seconds() / 60
    hl = hl.sort_values(["day", "symbol", "strength", "t"])
    top = hl.groupby(["day", "symbol"]).tail(1)[["day", "symbol", "category", "strength", "source"]]
    agg = hl.groupby(["day", "symbol"]).agg(n_head=("id", "nunique"), mins_last=("mins", "min")).reset_index()
    queried = news[news.id == ""][["day", "symbol"]].drop_duplicates().assign(queried=True)
    e = e.merge(queried, on=["day", "symbol"], how="left").merge(fail, on=["day", "symbol"], how="left")
    e = e.merge(top, on=["day", "symbol"], how="left").merge(agg, on=["day", "symbol"], how="left")
    e["queried"] = e.queried.fillna(False).astype(bool); e["fail"] = e.fail.fillna(False).astype(bool)
    e["category"] = e.category.fillna("NONE"); e["n_head"] = e.n_head.fillna(0); e["strength"] = e.strength.fillna(0)
    e["net"] = e.r_gross - S.NET0
    return e


def worst_mask(x, frac=0.05):
    """Boolean mask of the worst ceil(frac*n) trades by net."""
    k = int(np.ceil(frac * len(x))); m = np.zeros(len(x), bool); m[np.argsort(x)[:k]] = True
    return m


def odds(ind, w):
    """Odds ratio (Haldane 0.5) of indicator in worst mask vs the rest."""
    a = (ind & w).sum() + .5; b = (~ind & w).sum() + .5; c = (ind & ~w).sum() + .5; d = (~ind & ~w).sum() + .5
    return (a / b) / (c / d)


def or_ci(d, ind, draws=500):
    """OR and day-clustered bootstrap 95 % CI for an indicator over frame d."""
    x = d.net.values; ind = np.asarray(ind); o = odds(ind, worst_mask(x))
    days = d.day.values; ud = np.unique(days); idx = {u: np.where(days == u)[0] for u in ud}; bs = []
    for _ in range(draws):
        pick = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
        bs.append(odds(ind[pick], worst_mask(x[pick])))
    return o, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def veto_book(d, veto, label, rows, pl):
    """Stats of the book with `veto` rows dropped vs base, and the count-matched random-drop placebo."""
    keep = d[~veto]; b = S.stats(d.assign(r_gross=d.r_gross), "r_gross"); a = S.stats(keep, "r_gross")
    nd = int(veto.sum()); xb = d.net.values; k5 = int(np.ceil(.05 * len(xb))); w5b = np.sort(xb)[:k5].mean()
    dm, dw = [], []
    for _ in range(1000):
        keepi = np.ones(len(d), bool); keepi[rng.choice(len(d), nd, replace=False)] = False
        xx = xb[keepi]; kk = int(np.ceil(.05 * len(xx)))
        dm.append(xx.mean() - xb.mean()); dw.append(np.sort(xx)[:kk].mean() - w5b)
    rows.append(dict(cell=f"VETO_{label}_base", **b)); rows.append(dict(cell=f"VETO_{label}_applied", n_dropped=nd,
                     retained=len(keep) / len(d), d_mean=a["mean"] - b["mean"], d_worst5=a["worst5"] - b["worst5"], **a))
    pl.append(dict(veto=label, n_dropped=nd, placebo_dmean_p5=np.percentile(dm, 5), placebo_dmean_mean=np.mean(dm),
                   placebo_dworst5_p95=np.percentile(dw, 95), placebo_dworst5_mean=np.mean(dw),
                   veto_dmean=a["mean"] - b["mean"], veto_dworst5=a["worst5"] - b["worst5"], retained=len(keep) / len(d)))
    return a, b


def main():
    """Run the cell."""
    e = build(); n0 = len(e); cov = e.queried.mean()
    log("symbol-days", n0, "coverage ok", cov, "fail", int(e.fail.sum()))
    e.to_parquet(f"{HERE}/1775_events.parquet")
    d = e[e.queried].copy(); tr = d[d.split == "TRAIN"]; va = d[d.split == "VAL"]
    # availability rail: winner/loser missingness gap
    miss = e.groupby(e.r_gross > 0).queried.apply(lambda s: 1 - s.mean()); log("missingness by winner", miss.to_dict())
    log("category counts all", d.category.value_counts().to_dict())
    rows, pl = [], []
    for nm, x in (("TRAIN", tr), ("VAL", va)):
        rows.append(dict(cell=f"{nm}_ALL", **S.stats(x, "r_gross")))
        w = worst_mask(x.net.values)
        for c in CATS:
            m = (x.category == c).values
            if m.sum() == 0:
                continue
            o, lo, hi = or_ci(x, m)
            rows.append(dict(cell=f"{nm}_cat_{c}", n=int(m.sum()), mean=x.net[m].mean(), share_book=m.mean(), share_worst5=(m & w).sum() / w.sum(),
                             n_worst5=int((m & w).sum()), OR=o, OR_lo=lo, OR_hi=hi, ex_top5=np.sort(x.net[m].values)[:max(1, len(x.net[m]) - int(np.ceil(.05 * m.sum())))].mean()))
        m = (x.strength >= 4).values; o, lo, hi = or_ci(x, m)
        rows.append(dict(cell=f"{nm}_strength_ge4", n=int(m.sum()), mean=x.net[m].mean(), share_book=m.mean(), share_worst5=(m & w).sum() / w.sum(),
                         n_worst5=int((m & w).sum()), OR=o, OR_lo=lo, OR_hi=hi))
        # (d) extra cuts
        hn = x[x.n_head > 0]
        for nmc, col, asc in (("nhead", "n_head", True), ("recency", "mins_last", True)):
            q = pd.qcut(hn[col].rank(method="first"), 3, labels=["T1", "T2", "T3"])
            for t in ("T1", "T2", "T3"):
                s = hn[q == t]; mm = x.index.isin(s.index)
                o, lo, hi = or_ci(x, mm)
                rows.append(dict(cell=f"{nm}_{nmc}_{t}", n=len(s), mean=s.net.mean(), share_worst5=(mm & w).sum() / w.sum(), OR=o, OR_lo=lo, OR_hi=hi))
    sm = pd.DataFrame(rows)
    # veto candidates
    trs = sm[sm.cell.str.startswith("TRAIN_cat_") & (sm.n_worst5 >= 5)]
    cand = trs[trs.OR > 2].sort_values("OR", ascending=False)
    log("TRAIN categories OR>2 (n_worst5>=5):", cand[["cell", "OR", "n_worst5"]].values.tolist())
    verdict = {}
    vetoes = []
    if len(cand):
        c = cand.iloc[0].cell.replace("TRAIN_cat_", ""); vetoes.append((f"cat_{c}", (va.category == c).values))
    else:
        log("no TRAIN category with OR>2 -> veto (i) has no candidate")
    vetoes.append(("strength_ge4", (va.strength >= 4).values))
    for lab, v in vetoes:
        v = np.asarray(v); a, b = veto_book(va, v, lab, rows, pl)
        o, lo, hi = or_ci(va, v)
        p = pl[-1]; retained = p["retained"]
        ok = (o > 2 and lo > 1 and p["veto_dworst5"] >= 500 and p["veto_dmean"] >= p["placebo_dmean_p5"] and retained >= .6)
        verdict[lab] = dict(PASS=bool(ok), OR=o, lo=lo, hi=hi, **p); log("VETO", lab, verdict[lab])
        rows.append(dict(cell=f"VAL_veto_{lab}_OR", OR=o, OR_lo=lo, OR_hi=hi, PASS=bool(ok)))
    f = f"{HERE}/1774_events.parquet"
    log("read (c) 1774 events:", "present" if os.path.exists(f) else "ABSENT at scoring time (not run)")
    pd.DataFrame(rows).to_csv(f"{HERE}/1775_summary.csv", index=False); pd.DataFrame(pl).to_csv(f"{HERE}/1775_placebo.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).round(2).to_string())
    print(pd.DataFrame(pl).round(1).to_string())
    log("coverage", cov, "n", n0)


if __name__ == "__main__":
    main()
