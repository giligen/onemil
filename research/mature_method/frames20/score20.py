#!/usr/bin/env python3
"""frames20 F57 — score the POWERED placebo test (S vs P1x5) of the passive mirror short.

PREREG frames20/PREREG.md (frozen; FREEZE.md carries the git hash).  S = frames18/grid18.csv at
k=0.010 (unchanged).  P1x5 = frames20/p120.csv.  TEST sealed.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import clust_t1, null_green, week_shape                   # noqa: E402

D = f'{ROOT}/research/mature_method/frames20'
D16 = f'{ROOT}/research/mature_method/frames16'
D17 = f'{ROOT}/research/mature_method/frames17'
D18 = f'{ROOT}/research/mature_method/frames18'
TEST_FROM = '2026-06-01'
K = 0.010
WS = (5, 10)


# ------------------------------------------------------------------ measured cost model (PREREG 2)
def cost_model():
    parts = []
    for p in (f'{D16}/nbbo16.csv', f'{D17}/nbbo17.csv', f'{D18}/nbbo18.csv'):
        q = pd.read_csv(p, dtype={'day': str, 'symbol': str})
        parts.append(q[['day', 'symbol', 'm', 'sp_med', 'mid_med', 'n_q']])
    q = pd.concat(parts, ignore_index=True)
    q = q[(q.n_q > 0) & np.isfinite(q.sp_med) & np.isfinite(q.mid_med) & (q.mid_med > 0)]
    q = q.drop_duplicates(['day', 'symbol', 'm'])
    sp = {(a, b, int(c)): float(v) for a, b, c, v in zip(q.day, q.symbol, q.m, q.sp_med)}
    nbb = {(a, b, int(c)): float(mm) - float(s) / 2.0
           for a, b, c, mm, s in zip(q.day, q.symbol, q.m, q.mid_med, q.sp_med)}
    q = q.assign(rel=q.sp_med / q.mid_med)
    edges = np.unique(q.mid_med.quantile(np.linspace(0, 1, 11)).values)
    lab = pd.cut(q.mid_med, edges, include_lowest=True)
    med = q.groupby(lab, observed=True).rel.median()
    print(f'[cost] pooled measured quotes {len(q):,}; decile median sp/mid '
          f'{[round(float(v)*1e4, 1) for v in med.values]} bps', flush=True)
    return sp, nbb, edges, med.values


def charge(d, sp, edges, med):
    got = np.array([sp.get((a, b, int(c)), np.nan) if c >= 0 else np.nan
                    for a, b, c in zip(d.day, d.symbol, d.exit_m)], float)
    ix = np.clip(np.searchsorted(edges, d.entry.values, side='left') - 1, 0, len(med) - 1)
    fb = np.where(np.isfinite(d.entry.values), med[ix] * d.entry.values, np.nan)
    sphat = np.where(np.isfinite(got), got, fb)
    return sphat, np.isfinite(got), (0.5 * sphat / d.entry.values) / d.rpct.values


# ------------------------------------------------------------------ stats
def iid_t(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return se, (x.mean() / se if se > 0 else np.nan)


def diff_clust(ys, days_s, yp, days_p):
    """S - P difference of means with a day-cluster-robust t (verbatim from frames19/score19)."""
    y = np.concatenate([np.asarray(ys, float), np.asarray(yp, float)])
    g = np.concatenate([np.asarray(days_s), np.asarray(days_p)])
    x = np.concatenate([np.ones(len(ys)), np.zeros(len(yp))])
    ok = np.isfinite(y)
    y, g, x = y[ok], g[ok], x[ok]
    if len(y) < 5 or x.sum() < 2 or (1 - x).sum() < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    meat = np.zeros((2, 2))
    for _, ix in pd.Series(np.arange(len(y))).groupby(g).indices.items():
        Xg, ug = X[ix], u[ix]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(V[1, 1]))
    return float(beta[1]), (float(beta[1] / se) if se > 0 else np.nan)


def diff_iid(ys, yp):
    a = np.asarray(ys, float); a = a[np.isfinite(a)]
    b = np.asarray(yp, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(se), (float((a.mean() - b.mean()) / se) if se > 0 else np.nan)


def ex5(rr, net):
    rr = np.asarray(rr, float)
    net = np.asarray(net, float)
    if len(rr) < 5:
        return np.nan
    q = np.nanquantile(rr, 0.95)
    return float(np.nanmean(net[rr <= q]))


def trim_top5(z):
    """PREREG §3: drop the top 5 % of rows by GROSS rr, within this population/split."""
    if len(z) < 5:
        return z
    q = np.nanquantile(z.rr.values, 0.95)
    return z[z.rr.values <= q]


# ------------------------------------------------------------------ populations
def load_S(sp, nbb, edges, med):
    d = pd.read_csv(f'{D18}/grid18.csv', dtype={'day': str, 'symbol': str})
    d = d[np.isclose(d.k, K)].copy()
    assert d.day.max() < TEST_FROM
    d['nbb_fill'] = [nbb.get((a, b, int(c)), np.nan) if c >= 0 else np.nan
                     for a, b, c in zip(d.day, d.symbol, d.fill_m)]
    touched = d.touch_off >= 0
    d['ssr_void'] = False
    m = touched & d.ssr_active.astype(bool)
    d.loc[m & ~np.isfinite(d.nbb_fill), 'ssr_void'] = True
    d.loc[m & np.isfinite(d.nbb_fill) & (d.limit <= d.nbb_fill), 'ssr_void'] = True
    d.loc[touched & d.ssr_undet.astype(bool), 'ssr_void'] = True
    d['pop'] = 'S'
    return d


def prep(d, sp, edges, med):
    sphat, meas, c = charge(d, sp, edges, med)
    d = d.copy()
    d['cost'] = c
    d['meas'] = meas
    d['net'] = d.rr - d.cost
    return d


def filled(d, w, use_ssr):
    t = (d.touch_off >= 0) & (d.touch_off < w)
    f = t & (~d.ssr_void.astype(bool) if use_ssr else True)
    return d[f].copy()


def report(name, f, full, split, out):
    n_rows = int((full.split == split).sum())
    z = f[f.split == split]
    if len(z) < 5:
        return dict(pop=name, split=split, n=len(z))
    mn, tn = clust_t1(z.net.values, z.day.values)
    mg, tg = clust_t1(z.rr.values, z.day.values)
    se_i, t_i = iid_t(z.net.values)
    r = dict(pop=name, split=split, n_rows=n_rows, n=len(z), fill=len(z) / max(n_rows, 1),
             gross=mg, gross_t=tg, net=mn, net_t=tn, net_se_iid=se_i, net_t_iid=t_i,
             ex5=ex5(z.rr.values, z.net.values),
             meas_cov=float(z.meas.mean()), ssr_act=float(z.ssr_active.astype(bool).mean()),
             mean_cost=float(z.cost.mean()))
    out.append(r)
    return r


def main():
    sp, nbb, edges, med = cost_model()
    S = prep(load_S(sp, nbb, edges, med), sp, edges, med)
    P1 = pd.read_csv(f'{D}/p120.csv', dtype={'day': str, 'symbol': str})
    assert P1.day.max() < TEST_FROM
    P1['ssr_void'] = False
    P1 = prep(P1, sp, edges, med)
    print(f'[pop] S {len(S):,} rows | P1x5 {len(P1):,} rows '
          f'(frames19 P1x3 was 868)', flush=True)

    rows, diffs, sec = [], [], []
    for w in WS:
        print(f'\n{"="*98}\n== WINDOW w = {w} min ==\n{"="*98}', flush=True)
        fS = filled(S, w, True)
        fP1 = filled(P1, w, False)
        for split in ('TRAIN', 'VAL'):
            print(f'\n-- {split} --', flush=True)
            rS = report('S', fS, S, split, rows)
            r1 = report('P1x5', fP1, P1, split, rows)
            for r in (rS, r1):
                if r is None or 'net' not in r:
                    continue
                print(f"  {r['pop']:6s} n={r['n']:5d}/{r['n_rows']:6d} fill {r['fill']*100:5.1f}% "
                      f"gross {r['gross']:+.3f} net {r['net']:+.3f} (tc {r['net_t']:+.2f}, "
                      f"ti {r['net_t_iid']:+.2f}) ex5 {r['ex5']:+.3f} cost {r['mean_cost']:.3f} "
                      f"meas {r['meas_cov']*100:4.0f}% ssr {r['ssr_act']*100:4.1f}%", flush=True)
            zS, zP = fS[fS.split == split], fP1[fP1.split == split]

            dmu, dt = diff_clust(zS.net.values, zS.day.values, zP.net.values, zP.day.values)
            se_i, t_i = diff_iid(zS.net.values, zP.net.values)
            mde = 2.8 * abs(dmu / dt) if (np.isfinite(dt) and dt != 0) else np.nan
            # ex-top-5 % OF THE DIFFERENCE (PREREG §3: trim each population separately, recompute)
            tS, tP = trim_top5(zS), trim_top5(zP)
            emu, et = diff_clust(tS.net.values, tS.day.values, tP.net.values, tP.day.values)
            # touch-conditioned (every row scored, unfilled = 0)
            aS = S[S.split == split].assign(
                y=np.where((S.touch_off >= 0) & (S.touch_off < w) & ~S.ssr_void.astype(bool),
                           S.net, 0.0)[S.split == split])
            aP = P1[P1.split == split].assign(
                y=np.where((P1.touch_off >= 0) & (P1.touch_off < w), P1.net, 0.0)[P1.split == split])
            tmu, tt = diff_clust(aS.y.values, aS.day.values, aP.y.values, aP.day.values)
            fr_s = len(zS) / max(len(S[S.split == split]), 1)
            fr_p = len(zP) / max(len(P1[P1.split == split]), 1)
            void = abs(fr_s - fr_p) > 0.10
            print(f"  S - P1x5: net diff {dmu:+.3f} R (tc {dt:+.2f}, ti {t_i:+.2f})", flush=True)
            print(f"            ex-top-5% of the difference {emu:+.3f} R (t {et:+.2f})", flush=True)
            print(f"            touch-conditioned {tmu:+.4f} R (t {tt:+.2f})", flush=True)
            print(f"            fill {fr_s*100:.1f}% (S) vs {fr_p*100:.1f}% (P1) -> "
                  f"{'VOID (>10pp)' if void else 'comparable'}", flush=True)
            print(f"            80%-power MDE (day-clustered): {mde:+.3f} R", flush=True)
            diffs.append(dict(w=w, split=split, n_S=len(zS), n_P=len(zP), diff=dmu, diff_t=dt,
                              diff_t_iid=t_i, mde80=mde, ex5_diff=emu, ex5_t=et,
                              touch_diff=tmu, touch_t=tt, fill_S=fr_s, fill_P=fr_p, void=void,
                              pass_bar=bool(np.isfinite(dmu) and dmu >= 0.10 and dt >= 2.0
                                            and np.isfinite(emu) and emu >= 0 and not void)))

        # ------------------------------------------------ SECONDARY: P1x5 alone (PREREG §5)
        print(f'\n-- SECONDARY (report-only): P1x5 ALONE as a book, w={w} --', flush=True)
        for split in ('TRAIN', 'VAL'):
            z = fP1[fP1.split == split]
            if len(z) < 5:
                print(f'  {split}: n={len(z)} too few', flush=True)
                continue
            mn, tn = clust_t1(z.net.values, z.day.values)
            wk = week_shape(z.assign(rr=z.net), split)
            nl = null_green(z.assign(rr=z.net), split)
            h1 = z[z.day < '2025-07-01'] if split == 'TRAIN' else z.iloc[:0]
            h2 = z[z.day >= '2025-07-01'] if split == 'TRAIN' else z.iloc[:0]
            m1 = clust_t1(h1.net.values, h1.day.values)[0] if len(h1) >= 3 else np.nan
            m2 = clust_t1(h2.net.values, h2.day.values)[0] if len(h2) >= 3 else np.nan
            print(f"  {split}: n={len(z)} net {mn:+.3f} (t {tn:+.2f}) ex5 "
                  f"{ex5(z.rr.values, z.net.values):+.3f} | halves {m1:+.3f}/{m2:+.3f} | "
                  f"green {wk['green']:.0f}% vs null50 {nl[1]:.0f}% (null95 {nl[2]:.0f}%) | "
                  f"{wk['per_wk']:.2f} fills/wk | wk mean ${wk['wk_mean']:+.0f} worst "
                  f"${wk['worst']:+.0f}", flush=True)
            sec.append(dict(w=w, split=split, n=len(z), net=mn, net_t=tn,
                            ex5=ex5(z.rr.values, z.net.values), half1=m1, half2=m2,
                            green=wk['green'], null50=nl[1], null95=nl[2],
                            per_wk=wk['per_wk'], wk_mean=wk['wk_mean'], worst=wk['worst']))

    pd.DataFrame(rows).to_csv(f'{D}/pops20.csv', index=False)
    dd = pd.DataFrame(diffs)
    dd.to_csv(f'{D}/diffs20.csv', index=False)
    pd.DataFrame(sec).to_csv(f'{D}/secondary20.csv', index=False)
    print(f'\n{"="*98}\n== PRE-COMMITTED VERDICT (PREREG §4) ==', flush=True)
    for w in WS:
        z = dd[dd.w == w]
        ok = bool(len(z) == 2 and z.pass_bar.all())
        print(f'  w={w:2d}: S-P1x5 >= +0.10 R, t >= 2, ex-top-5% of the diff >= 0, non-VOID, on '
              f'BOTH splits -> {"PASS" if ok else "FAIL"}', flush=True)
        for r in z.itertuples():
            print(f'      {r.split:5s}: {r.diff:+.3f} R t {r.diff_t:+.2f} ex5 {r.ex5_diff:+.3f} '
                  f'MDE {r.mde80:.3f} {"VOID" if r.void else ""} -> '
                  f'{"ok" if r.pass_bar else "fail"}', flush=True)
    print('\n[score20] wrote pops20.csv, diffs20.csv, secondary20.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
