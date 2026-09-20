#!/usr/bin/env python3
"""frames19 F56 — score the three-population placebo decomposition of the passive mirror short.

PREREG frames19/PREREG.md (frozen; FREEZE.md carries the git hash).  S = frames18/grid18.csv at
k=0.010 (unchanged).  P3 / P1 = frames19/p319.csv, p119.csv.  TEST sealed.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import clust_t1                                          # noqa: E402

D = f'{ROOT}/research/mature_method/frames19'
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
    # measured price-decile median of sp/mid, fitted ONCE, applied to all three populations
    q = q.assign(rel=q.sp_med / q.mid_med)
    edges = np.unique(q.mid_med.quantile(np.linspace(0, 1, 11)).values)
    lab = pd.cut(q.mid_med, edges, include_lowest=True)
    med = q.groupby(lab, observed=True).rel.median()
    print(f'[cost] pooled measured quotes {len(q):,}; decile median sp/mid '
          f'{[round(float(v)*1e4, 1) for v in med.values]} bps', flush=True)
    return sp, nbb, edges, med.values


def charge(d, sp, edges, med):
    """exit-leg cost in R = 0.5 * sp_hat / entry / rpct; measured if we have the quote, else the
    measured price-decile median of sp/mid applied to that row's entry."""
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
    """Two-sample difference of means (S - P) with a day-cluster-robust t (a day contributes ONE
    residual across both populations)."""
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


def ex5(rr, net):
    rr = np.asarray(rr, float)
    net = np.asarray(net, float)
    if len(rr) < 5:
        return np.nan
    q = np.nanquantile(rr, 0.95)
    return float(np.nanmean(net[rr <= q]))


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


def table(d, w, use_ssr, n_denom=None):
    """filled book for window w (+ the touch-conditioned denominator)."""
    t = (d.touch_off >= 0) & (d.touch_off < w)
    f = t & (~d.ssr_void.astype(bool) if use_ssr else True)
    return d[f].copy(), (len(d) if n_denom is None else n_denom), int(t.sum())


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
    P3 = pd.read_csv(f'{D}/p319.csv', dtype={'day': str, 'symbol': str})
    P1 = pd.read_csv(f'{D}/p119.csv', dtype={'day': str, 'symbol': str})
    for z in (P3, P1):
        assert z.day.max() < TEST_FROM
        z['ssr_void'] = False
    P3 = prep(P3, sp, edges, med)
    P1 = prep(P1, sp, edges, med)
    print(f'[pop] S {len(S):,} rows | P3 {len(P3):,} | P1 {len(P1):,}', flush=True)

    rows, diffs = [], []
    for w in WS:
        print(f'\n{"="*96}\n== WINDOW w = {w} min ==\n{"="*96}', flush=True)
        fS, nS, tS = table(S, w, True)
        fS_nossr, _, _ = table(S, w, False)
        fP3, nP3, tP3 = table(P3, w, False)
        fP1, nP1, tP1 = table(P1, w, False)
        # hour-stratified P3 (weights = S's hour distribution, per split)
        for split in ('TRAIN', 'VAL'):
            print(f'\n-- {split} --', flush=True)
            rS = report('S', fS, S, split, rows)
            report('S(no SSR rail)', fS_nossr, S, split, rows)
            r3 = report('P3', fP3, P3, split, rows)
            r1 = report('P1', fP1, P1, split, rows)
            for r in (rS, r3, r1):
                if r is None or 'net' not in r:
                    continue
                print(f"  {r['pop']:16s} n={r['n']:5d}/{r['n_rows']:6d} fill {r['fill']*100:5.1f}% "
                      f"gross {r['gross']:+.3f} net {r['net']:+.3f} (tc {r['net_t']:+.2f}, "
                      f"ti {r['net_t_iid']:+.2f}) ex5 {r['ex5']:+.3f} cost {r['mean_cost']:.3f} "
                      f"meas {r['meas_cov']*100:4.0f}% ssr {r['ssr_act']*100:4.1f}%", flush=True)
            zS = fS[fS.split == split]
            # hour-stratified P3
            wgt = zS.hour.value_counts(normalize=True)
            z3 = fP3[fP3.split == split]
            hm = z3.groupby('hour').net.mean()
            common = [h for h in wgt.index if h in hm.index]
            p3_strat = float(sum(wgt[h] * hm[h] for h in common) / sum(wgt[h] for h in common)) \
                if common else np.nan
            print(f"  P3 hour-stratified to S's hour mix: {p3_strat:+.3f} R "
                  f"(hours {sorted(common)})", flush=True)
            for nm, z in (('P3', z3), ('P1', fP1[fP1.split == split])):
                dmu, dt = diff_clust(zS.net.values, zS.day.values, z.net.values, z.day.values)
                # touch-conditioned: every row scored, unfilled = 0
                aS = S[(S.split == split)].assign(
                    y=np.where((S.touch_off >= 0) & (S.touch_off < w) & ~S.ssr_void.astype(bool),
                               S.net, 0.0)[S.split == split])
                P = P3 if nm == 'P3' else P1
                aP = P[(P.split == split)].assign(
                    y=np.where((P.touch_off >= 0) & (P.touch_off < w), P.net, 0.0)[P.split == split])
                tmu, tt = diff_clust(aS.y.values, aS.day.values, aP.y.values, aP.day.values)
                fr_s = len(zS) / max(len(S[S.split == split]), 1)
                fr_p = len(z) / max(len(P[P.split == split]), 1)
                void = abs(fr_s - fr_p) > 0.10
                print(f"  S - {nm}: net diff {dmu:+.3f} R (t {dt:+.2f})  |  touch-conditioned "
                      f"{tmu:+.4f} R (t {tt:+.2f})  |  fill {fr_s*100:.1f}% vs {fr_p*100:.1f}% "
                      f"-> {'VOID (>10pp)' if void else 'comparable'}", flush=True)
                mde = 2.8 * abs(dmu / dt) if (np.isfinite(dt) and dt != 0) else np.nan
                print(f"        80%-power MDE on that difference (day-clustered): {mde:+.3f} R",
                      flush=True)
                diffs.append(dict(w=w, split=split, placebo=nm, diff=dmu, diff_t=dt, mde80=mde,
                                  touch_diff=tmu, touch_t=tt, fill_S=fr_s, fill_P=fr_p,
                                  void=void,
                                  pass_bar=bool(np.isfinite(dmu) and dmu >= 0.10 and dt >= 2.0
                                                and not void)))
    pd.DataFrame(rows).to_csv(f'{D}/pops19.csv', index=False)
    dd = pd.DataFrame(diffs)
    dd.to_csv(f'{D}/diffs19.csv', index=False)
    print(f'\n{"="*96}\n== PRE-COMMITTED VERDICT (PREREG §4) ==', flush=True)
    for w in WS:
        z = dd[dd.w == w]
        ok = bool(len(z) == 4 and z.pass_bar.all())
        print(f'  w={w:2d}: both differences >= +0.10 R with t >= 2 on BOTH splits and both '
              f'comparisons non-VOID -> {"PASS" if ok else "FAIL"}', flush=True)
        for r in z.itertuples():
            print(f'      {r.split:5s} S-{r.placebo}: {r.diff:+.3f} R t {r.diff_t:+.2f} '
                  f'{"VOID" if r.void else ""} -> {"ok" if r.pass_bar else "fail"}', flush=True)
    print('\n[score19] wrote pops19.csv, diffs19.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
