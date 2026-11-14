"""PREREG 1,778 scoring: long gap-and-go survivors (p1030 >= open) bought at ask1030*1.001, MOC exit at close.

Reuses stats/tclust/weekly_R from 1773_score.py (day-clustered t convention of 1774_score.py). R unit: NOTIONAL $3,000 x 5% = $150
(s73.POS overridden to 3000). Writes 1778_events.parquet, 1778_summary.csv, 1778_placebo.csv, 1778_freq.csv.
"""
import os, sys, glob, sqlite3, importlib.util, time
import numpy as np, pandas as pd
os.nice(19)
HERE = "/home/ec2-user/onemil/research/signal_search"
R = "/home/ec2-user/onemil/research"
sp = importlib.util.spec_from_file_location("s73", f"{HERE}/1773_score.py"); s73 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s73)
s73.POS = 3000.0
FEES, SLIP, STOP_PCT, STOP_SLIP = 4.0, 0.001, 0.05, 35e-4
rng = np.random.default_rng(1778)


def log(*a):
    """Flushed progress line."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_l1(days):
    """Pre-market $vol 04:00-09:25 ET per (day, symbol), as in 1776R_build.load_universe."""
    l1 = {}
    for day in days:
        f = f"{R}/hod_entry/1762_cache/premarket/{day}.parquet"
        if not os.path.exists(f): continue
        b = pd.read_parquet(f)
        t = pd.to_datetime(b.t, utc=True).dt.tz_convert("America/New_York"); end = t + pd.Timedelta(minutes=1); m = t.dt.normalize()
        ok = (t >= m + pd.Timedelta(hours=4)) & (end <= m + pd.Timedelta(hours=9, minutes=25))
        b = b[ok.values].assign(dv=lambda z: z.c * z.v)
        for s, v in b.groupby("symbol").dv.sum().items(): l1[(day, s)] = v
    return l1


def bar_features(ev):
    """Per event: VWAP 09:30-10:29 ET and stop-variant exit price from minute bars (1762 cache first, else bars_sip.db)."""
    db = sqlite3.connect(f"file:{R}/bf_zero/bars_sip.db?mode=ro", uri=True)
    out = []; srcs = []
    for day, g in ev.groupby("day"):
        f = f"{R}/hod_entry/1762_cache/bars/{day}.parquet"
        p = pd.read_parquet(f) if os.path.exists(f) else None
        for i, r in g.iterrows():
            b = None
            if p is not None:
                b = p[p.symbol == r.symbol]
                if len(b): srcs.append("1762")
            if b is None or not len(b):
                b = pd.read_sql("select t,o,h,l,c,v from bars where symbol=? and day=?", db, params=(r.symbol, day))
                if len(b): srcs.append("sip")
                else: srcs.append("none")
            vw, ex, stopped = np.nan, np.nan, np.nan
            if len(b):
                t = pd.to_datetime(b.t, utc=True).dt.tz_convert("America/New_York"); mid = t.dt.normalize()
                mins = (t - mid).dt.total_seconds().values / 60
                pre = (mins >= 570) & (mins < 630)
                if pre.sum() >= 5 and b.v[pre].sum() > 0:
                    tp = ((b.h + b.l + b.c) / 3)[pre]; vw = float((tp * b.v[pre]).sum() / b.v[pre].sum())
                aft = (mins >= 630)
                if aft.sum() >= 5:
                    stop = r.p1030 * (1 - STOP_PCT); bb = b[aft].reset_index(drop=True); hit = np.where(bb.l.values <= stop)[0]
                    if len(hit): ex, stopped = stop * (1 - STOP_SLIP), 1.0
                    else: ex, stopped = r.close, 0.0
            out.append((i, vw, ex, stopped))
        if day[-2:] == "15": log("bars", day)
    o = pd.DataFrame(out, columns=["i", "vwap", "stop_exit", "stopped"]).set_index("i")
    return o, pd.Series(srcs).value_counts().to_dict()


def build():
    """Event table with survivor flag, net returns, L1, tier, bars features."""
    e = pd.read_parquet(f"{HERE}/1774_events.parquet")
    n0 = len(e)
    e = e[e.ask1030.notna() & (e.nq1030 >= 1) & (e.gap_pct <= 200) & (e.open > 0) & e.close.notna() & e.p1030.notna()].copy()
    log("events", n0, "->", len(e), e.split.value_counts().to_dict())
    e["surv"] = e.p1030 >= e.open
    e["hold5"] = e.p1030 >= e.open * 1.05
    e["gross_moc"] = (e.close / (e.ask1030 * (1 + SLIP)) - 1) * 1e4
    l1 = load_l1(sorted(e.day.unique())); e["L1"] = [l1.get((a, b), np.nan) for a, b in zip(e.day, e.symbol)]
    log("L1 coverage", float(e.L1.notna().mean()))
    bf, srcs = bar_features(e); e = e.join(bf)
    e["gross_stop"] = (e.stop_exit / (e.ask1030 * (1 + SLIP)) - 1) * 1e4
    e["above_vwap"] = np.where(e.vwap.notna(), e.p1030 >= e.vwap, np.nan)
    log("bars sources", srcs, "vwap cov", float(e.vwap.notna().mean()), "stop cov", float(e.stop_exit.notna().mean()))
    return e, srcs


def cuts(e):
    """Pre-committed cut -> boolean mask dict (L1 terciles use TRAIN survivor cutoffs)."""
    s = e.surv; tr = e[s & (e.split == "TRAIN") & e.L1.notna()].L1
    q1, q2 = tr.quantile([1 / 3, 2 / 3]).values
    c = {"ALL": s, "L1_T1": s & (e.L1 <= q1), "L1_T2": s & (e.L1 > q1) & (e.L1 <= q2), "L1_T3": s & (e.L1 > q2),
         "HOLD5": s & e.hold5, "TIER_3-5": s & (e.tier == "1-5"), "TIER_5-20": s & (e.tier == "5-20"), "TIER_20-30": s & (e.tier == ">20"),
         "ABOVE_VWAP": s & (e.above_vwap == 1)}
    return c, (q1, q2)


def freq(d, alldays, allweeks):
    """Frequency read for a cut within one split: zero-filled over every population day / week of the split."""
    pdays = d.groupby("day").size().reindex(alldays, fill_value=0)
    wk = d.assign(w=pd.to_datetime(d.day).dt.to_period("W")).groupby("w").size().reindex(allweeks, fill_value=0)
    return dict(n=len(d), names_day_med=pdays.median(), names_day_mean=pdays.mean(), days_with=int((pdays > 0).sum()), days=len(pdays),
                names_wk_p10=np.percentile(wk, 10), names_wk_med=wk.median(), wk_ge3=(wk >= 3).mean(), wk_ge5=(wk >= 5).mean(), weeks=len(wk))


def placebo_i(d, pool, col, nd=200):
    """In-day placebo: per day, k = count of cut names, draw k from that day's non-survivors; 200 draws -> mean, p95 of mean net."""
    cnt = d.groupby("day").size(); by = {k: v[col].values for k, v in pool.groupby("day")}
    ms = np.empty(nd)
    for i in range(nd):
        tot, n = 0.0, 0
        for day, k in cnt.items():
            a = by.get(day)
            if a is None or len(a) == 0: continue
            kk = min(k, len(a)); tot += rng.choice(a, kk, replace=False).sum(); n += kk
        ms[i] = tot / n - FEES if n else np.nan
    return np.nanmean(ms), np.nanpercentile(ms, 95)


def placebo_ii(d, allev, col, nd=200):
    """Same-name placebo: same symbol, other population days within +-5 calendar days, any survivor status; 1 draw per event."""
    allev = allev.assign(dt=pd.to_datetime(allev.day)); by = {s: g for s, g in allev.groupby("symbol")}
    cand = []
    for _, r in d.iterrows():
        g = by[r.symbol]; dt = pd.Timestamp(r.day)
        m = g[(abs((g.dt - dt).dt.days) <= 5) & (g.day != r.day)][col].values
        cand.append(m)
    have = [c for c in cand if len(c)]
    if not have: return np.nan, np.nan, 0
    ms = np.array([np.mean([rng.choice(c) for c in have]) - FEES for _ in range(nd)])
    return ms.mean(), np.percentile(ms, 95), len(have)


def main():
    """Run the cell."""
    e, srcs = build(); e.to_parquet(f"{HERE}/1778_events.parquet")
    cs, qs = cuts(e); log("L1 TRAIN tercile cutoffs", qs)
    summ, plc, fq = [], [], []
    for exitn, col in (("MOC", "gross_moc"), ("STOP", "gross_stop")):
        cov = e[e.surv][col].notna().mean(); log("exit", exitn, "coverage among survivors", cov)
        for nm, m in cs.items():
            for sp_ in ("TRAIN", "VAL"):
                d = e[m & (e.split == sp_)]
                if exitn == "MOC" and True:
                    sub = e[e.split == sp_]; alld = sorted(sub.day.unique()); allw = sorted(pd.to_datetime(alld).to_period("W").unique())
                    fq.append(dict(cut=nm, split=sp_, **freq(d, alld, allw)))
                if exitn == "STOP" and nm in ("ABOVE_VWAP",) and False: pass
                dd = d.dropna(subset=[col])
                s = s73.stats(dd, col, FEES); s.update(cut=nm, split=sp_, exit=exitn, cov_cut=len(dd) / max(len(d), 1))
                s["med_bp"] = float(np.median(dd[col] - FEES)) if len(dd) else np.nan; s["win"] = float((dd[col] - FEES > 0).mean()) if len(dd) else np.nan
                if sp_ == "TRAIN" and len(dd) > 10:
                    days = sorted(dd.day.unique()); md = days[len(days) // 2]
                    s["h1"] = (dd[dd.day <= md][col] - FEES).mean(); s["h2"] = (dd[dd.day > md][col] - FEES).mean()
                if len(dd) >= 10:
                    pool = e[(~e.surv) & (e.split == sp_)].dropna(subset=[col])
                    pm, p95 = placebo_i(dd, pool, col); s["pl_i_mean"], s["pl_i_p95"] = pm, p95
                    sm = e[(e.split == sp_)].dropna(subset=[col]); pm2, p952, nn = placebo_ii(dd, e.dropna(subset=[col]), col)
                    s["pl_ii_mean"], s["pl_ii_p95"], s["pl_ii_n"] = pm2, p952, nn
                    plc.append(dict(cut=nm, split=sp_, exit=exitn, pl_i_mean=pm, pl_i_p95=p95, pl_ii_mean=pm2, pl_ii_p95=p952, pl_ii_events=nn))
                summ.append(s); log(exitn, nm, sp_, "n", s.get("n"), "mean", s.get("mean"), "t", s.get("t_day"))
    pd.DataFrame(summ).to_csv(f"{HERE}/1778_summary.csv", index=False); pd.DataFrame(plc).to_csv(f"{HERE}/1778_placebo.csv", index=False)
    pd.DataFrame(fq).to_csv(f"{HERE}/1778_freq.csv", index=False)
    log("done; bars sources", srcs)


if __name__ == "__main__":
    main()
