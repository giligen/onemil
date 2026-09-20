#!/usr/bin/env python3
"""frames17 F52 — score the mirror short on a RESTING limit entry (PREREG frames17/PREREG.md).

Entry charged ZERO (a resting limit is paid the spread, not charged it). Exit charged the measured
NBBO half-spread (`frames16/nbbo16.csv` first, `frames17/nbbo17.csv` for the gap), unchanged in KIND
from frames16 arm3's half+half contract, applied to the exit leg only.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import clust_t1, mde_pct, null_green, week_shape        # noqa: E402
from scoreB_intra import book                                          # noqa: E402

D = f'{ROOT}/research/mature_method/frames17'
D16 = f'{ROOT}/research/mature_method/frames16'
TEST_FROM = '2026-06-01'
RISK = 100.0


def load_nbbo():
    parts = []
    for p in (f'{D16}/nbbo16.csv', f'{D}/nbbo17.csv'):
        if os.path.exists(p):
            parts.append(pd.read_csv(p, dtype={'day': str, 'symbol': str}))
    q = pd.concat(parts, ignore_index=True)
    q = q[q.n_q > 0].drop_duplicates(['day', 'symbol', 'm'])
    return {(a, b, int(c)): v for a, b, c, v in zip(q.day, q.symbol, q.m, q.sp_med.astype(float))}


def main():
    d = pd.read_csv(f'{D}/passive17.csv', dtype={'day': str, 'symbol': str})
    assert d.day.max() < TEST_FROM, f'TEST leaked: max day {d.day.max()}'
    print(f'[freeze] max day scored: {d.day.max()} (TEST_FROM {TEST_FROM})', flush=True)
    nb = load_nbbo()

    rows = []
    for k in sorted(d.k.unique()):
        z = d[d.k == k].copy()
        n_all = len(z)
        f = z[z.filled].copy()
        u = z[~z.filled].copy()
        f['exit_m'] = f.exit_m.astype(int)
        f['entry_m'] = f.fill_m.astype(int)          # for `book()` concurrency bookkeeping
        f['sp_exit'] = [nb.get((a, b, c), np.nan) for a, b, c in zip(f.day, f.symbol, f.exit_m)]
        cov = float(np.isfinite(f.sp_exit).mean())
        f['ext_pct'] = f.sp_exit / (f.entry)          # entry = the limit fill price
        f['cost_R'] = (0.5 * f.ext_pct) / f.rpct
        f['net'] = f.rr - f.cost_R.fillna(f.cost_R.mean())

        # -- unfilled counterfactual: frames16 arm3's OWN reacting fill, NET of ITS OWN measured
        # half+half cost (entry half + exit half, exactly score3b.py's S1 contract) -- a like-for-
        # like NET-vs-NET comparison, not gross-vs-net.
        u = u.copy()
        u['sp_re'] = [nb.get((a, b, c), np.nan) for a, b, c in zip(u.day, u.symbol, u.react_entry_m)]
        u['sp_rx'] = [nb.get((a, b, c), np.nan) for a, b, c in zip(u.day, u.symbol, u.react_exit_m)]
        u_cov = float(np.isfinite(u.sp_re).mean()), float(np.isfinite(u.sp_rx).mean())
        u['react_cost_R'] = (0.5 * u.sp_re / u.react_entry + 0.5 * u.sp_rx / u.react_entry) / \
            u.react_rpct
        u['react_net'] = u.rr_reacting - u.react_cost_R

        st = dict(k=k, n_signal=n_all, n_filled=len(f), n_unfilled=len(u),
                  fill_rate=len(f) / n_all if n_all else np.nan, exit_cov=cov,
                  unfilled_entry_cov=u_cov[0], unfilled_exit_cov=u_cov[1])
        print(f'\n== k={k*100:.1f}% : n={n_all:,} filled={len(f):,} '
              f'({100*len(f)/n_all:.1f}%) exit-leg coverage {cov*100:.1f}% | unfilled reacting-cost '
              f'coverage entry {u_cov[0]*100:.1f}% exit {u_cov[1]*100:.1f}% ==', flush=True)

        for sp in ('TRAIN', 'VAL'):
            fx = f[f.split == sp]
            ux = u[u.split == sp]
            if len(fx) < 10:
                continue
            mu_g, tc_g = clust_t1(fx.rr.values, fx.day.values)
            mu_n, tc_n = clust_t1(fx.net.values, fx.day.values)
            uf_mean, uf_tc = clust_t1(ux.react_net.values, ux.day.values)
            uf_gross, _ = clust_t1(ux.rr_reacting.values, ux.day.values)
            bk = book(fx.assign(entry_m=fx.entry_m, exit_m=fx.exit_m), 'exit_m')
            w = week_shape(bk.assign(rr=bk.net), sp)
            nl = null_green(bk.assign(rr=bk.net), sp)
            q95 = fx.rr.quantile(0.95)
            ex5 = float(fx.net[fx.rr <= q95].mean())
            adverse = uf_mean > mu_n
            st.update({
                f'{sp}_n': len(fx), f'{sp}_gross': mu_g, f'{sp}_gross_t': tc_g,
                f'{sp}_net': mu_n, f'{sp}_net_t': tc_n, f'{sp}_ex5': ex5,
                f'{sp}_green': w['green'], f'{sp}_null50': nl[1], f'{sp}_null95': nl[2],
                f'{sp}_wkmean': w['wk_mean'], f'{sp}_total': w['total'],
                f'{sp}_unfilled_n': len(ux), f'{sp}_unfilled_net': uf_mean,
                f'{sp}_unfilled_gross': uf_gross, f'{sp}_adverse_selection': adverse,
                f'{sp}_mde': mde_pct(fx.rr.values, fx.day.values)})
            print(f'  {sp:5s} filled n={len(fx):4,} gross {mu_g:+.3f} (t {tc_g:+.2f}) | '
                  f'net {mu_n:+.3f} (t {tc_n:+.2f}) ex5 {ex5:+.3f} | green {w["green"]:5.1f}% '
                  f'(null p50 {nl[1]:.1f} p95 {nl[2]:.1f}) wk ${w["wk_mean"]:+,.0f}', flush=True)
            print(f'        DECIDING TABLE: unfilled n={len(ux):4,} reacting NET '
                  f'{uf_mean:+.3f} (t {uf_tc:+.2f}, gross {uf_gross:+.3f})  vs  filled net '
                  f'{mu_n:+.3f}  -> {"ADVERSE SELECTION" if adverse else "no adverse selection"}',
                  flush=True)

        tr = f[f.split == 'TRAIN']
        st['h1'] = float(tr[tr.day < '2025-07-01'].net.mean()) if len(tr) else np.nan
        st['h2'] = float(tr[tr.day >= '2025-07-01'].net.mean()) if len(tr) else np.nan
        print(f'  TRAIN halves net: {st["h1"]:+.3f} / {st["h2"]:+.3f}', flush=True)
        rows.append(st)

    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/cells17.csv', index=False)
    print(f'\n[score17] wrote cells17.csv ({len(out)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
