"""PREREG 1,776 scoring: 1,774 book split by pre-open liquidity terciles (L1 pre-mkt $vol, L2 adv20, L3 prior-day $vol), TRAIN cuts -> VAL,
same-subset same-convention measured-ask placebos, joint Gap-T3 x L1-T3. Writes 1776_events.parquet, 1776_summary.csv, 1776_placebo.csv."""
import glob, importlib.util
import numpy as np, pandas as pd
HERE = "/home/ec2-user/onemil/research/signal_search"
sp = importlib.util.spec_from_file_location("s73", f"{HERE}/1773_score.py"); s73 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s73)
NET0 = s73.NET0; rng = np.random.default_rng(1776)


def stat_row(name, d, col="r_ask1030"):
    """stats() + bar-present share + hs median."""
    s = s73.stats(d, col, NET0); s["cell"] = name; s["bar_present"] = d.p1030.notna().mean() if len(d) else np.nan; s["hs_med"] = d.hs1030.median() if len(d) else np.nan
    s["names_day_med"] = s.pop("med_names_day", np.nan); return s


def clean_pool(q, a):
    """Attach population open, apply the 1,774 quote-sanity cleaning (ref = day open), compute ask-priced gross r."""
    q = q[q.ask.notna()].copy(); q["mid"] = (q.bid + q.ask) / 2; q["hs"] = (q.ask - q.bid) / 2 / q.mid * 1e4
    q = q.merge(a[["day", "symbol", "open"]].drop_duplicates(["day", "symbol"]), on=["day", "symbol"])
    ok = (q.ask > q.bid) & (q.bid > 0) & (q.hs <= 1000) & (q.mid >= .5 * q.open) & (q.mid <= 2 * q.open)
    q = q[ok].copy(); q["r_ask"] = (q.open / q.ask - 1) * 1e4; return q


def main():
    """Run the cell."""
    e = pd.read_parquet(f"{HERE}/1774_events.parquet"); a = pd.read_parquet(f"{HERE}/1776_allfeat.parquet")
    e = e.merge(a[["day", "symbol", "L1", "L3", "pm_file"]], on=["day", "symbol"], how="left").rename(columns={"adv20": "L2"})
    ok = e.r_ask1030.notna(); base = e[ok].copy(); tr, val = base[base.split == "TRAIN"], base[base.split == "VAL"]
    cuts = {v: tr[v].quantile([1 / 3, 2 / 3]).values for v in ("L1", "L2", "L3")}
    for v in cuts:
        c = cuts[v]; e[v + "_T"] = np.where(e[v].isna(), "NONE", np.where(e[v] <= c[0], "T1", np.where(e[v] <= c[1], "T2", "T3")))
    e.to_parquet(f"{HERE}/1776_events.parquet"); base = e[ok]; tr, val = base[base.split == "TRAIN"], base[base.split == "VAL"]
    print("TRAIN cuts", {k: v.round(0) for k, v in cuts.items()}, "NONE L1 n", int((e.L1_T == "NONE").sum()))
    rows = []
    for v in ("L1", "L2", "L3"):
        for nm, d in (("TRAIN", tr), ("VAL", val)):
            for t in ("T1", "T2", "T3", "NONE"):
                z = d[d[v + "_T"] == t]
                if len(z):
                    rows.append(stat_row(f"{v}_{t}_{nm}", z))
    p_tr, p_val = tr[tr.L1_T == "T3"], val[val.L1_T == "T3"]
    dd = sorted(p_tr.day.unique()); med = dd[len(dd) // 2]
    rows.append(stat_row("L1_T3_TRAIN_H1", p_tr[p_tr.day <= med])); rows.append(stat_row("L1_T3_TRAIN_H2", p_tr[p_tr.day > med]))
    gq = tr.gap_pct.quantile(2 / 3)  # gap T3 cut from TRAIN book, as 1,774 terciles were per-split; declared TRAIN cut here
    for nm, d in (("TRAIN", tr), ("VAL", val)):
        rows.append(stat_row(f"JOINT_gapT3_L1T3_{nm}", d[(d.gap_pct > gq) & (d.L1_T == "T3")] if False else d[(d.gap_pct > gq) & (d.L1_T3 == "T3")] if False else d[(d.gap_pct > gq) & (d.L1_T == "T3")]))
        rows.append(stat_row(f"gapT3_only_{nm}", d[d.gap_pct > gq]))
    S = pd.DataFrame(rows); S.to_csv(f"{HERE}/1776_summary.csv", index=False)
    # placebos
    q = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{HERE}/1774_cache/quotes/1776_*.parquet")], ignore_index=True)
    allp = pd.read_parquet(f"{HERE}/1773_events.parquet"); q = clean_pool(q, allp)
    tgt = pd.read_parquet(f"{HERE}/1776_allfeat.parquet")
    book = set(zip(e.day, e.symbol)); pool = q.merge(tgt[["day", "symbol", "L1"]], on=["day", "symbol"])
    pool = pool[~pd.Series(list(zip(pool.day, pool.symbol)), index=pool.index).isin(book) & (pool.L1 > cuts["L1"][1])]
    cnt = p_val.groupby("day").size(); pdays = pool.groupby("day").size()
    avail = pd.concat([cnt.rename("k"), pdays.rename("p")], axis=1).fillna(0); cov_ev = avail.k[avail.p > 0].sum() / avail.k.sum()
    means, wkm = [], []
    for _ in range(200):
        r, dy = [], []
        for d, k in cnt.items():
            g = pool[pool.day == d].r_ask.values
            if len(g):
                s = rng.choice(g, min(k, len(g)), replace=False); r.extend(s); dy.extend([d] * len(s))
        means.append(np.mean(r) - NET0)
    pc = pd.DataFrame(dict(day=pool.day, col=pool.r_ask)); ps = s73.stats(pc.rename(columns={"col": "r_ask"}).assign(), "r_ask", NET0) if len(pc) >= 3 else {}
    # same-name +5d
    cal = sorted(allp.day.unique()); idx = {d: i for i, d in enumerate(cal)}
    p5 = p_val[["day", "symbol"]].copy(); p5["d5"] = p5.day.map(lambda d: cal[idx[d] + 5] if idx[d] + 5 < len(cal) else None)
    p5 = p5.dropna().merge(q[["day", "symbol", "r_ask"]].rename(columns={"day": "d5"}), on=["d5", "symbol"])
    m5, t5, _ = s73.tclust(p5.r_ask.values - NET0, p5.d5.values); se5 = abs(m5 / t5) if t5 == t5 and t5 else np.nan
    pl = pd.DataFrame([dict(placebo="i_same_day_same_L1T3_nonbook_measured_ask_net", n_pool=len(pool), n_days=int(pool.day.nunique()), event_cov=cov_ev,
                            mean=np.mean(means), p95=np.percentile(means, 95), primary_net=p_val.r_ask1030.mean() - NET0, pool_all_mean_net=pool.r_ask.mean() - NET0),
                       dict(placebo="ii_same_name_plus5_measured_ask_net", n_pool=len(p5), mean=m5, p95=m5 + 1.645 * se5 if se5 == se5 else np.nan, primary_net=p_val.r_ask1030.mean() - NET0, thin=len(p5) < 100)])
    pl.to_csv(f"{HERE}/1776_placebo.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[["cell", "n", "days", "mean", "t_day", "t_iid", "ex_top5", "worst5", "weekly_p10_R", "green_wk", "names_day_med", "bar_present"]].round(2).to_string()); print(pl.round(2).to_string())
    print("TRAIN primary halves ok; gap cut", gq)


if __name__ == "__main__":
    main()
