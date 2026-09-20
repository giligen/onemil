#!/usr/bin/env python3
"""frames16 ARM 3 — re-score every cell at the MEASURED per-trade NBBO (RUNBOOK step 3).

`score3.py` charged the frames14 F45 minute-of-day median, a declared PROXY measured on the
HOD-break population. `cost_check.py` measured this population's own minutes on a 350-trade sample
and found the proxy is **1.80x too narrow in the mean**; `nbbo.py` then measured every leg of every
trade. This file re-scores on those measurements and is the number that decides arm 3.

Two cost contracts are reported side by side:
  HALF+HALF   0.5 x measured spread at the entry minute + 0.5 x at the exit minute (what score3.py
              charged, and the conservative reading)
  PER-OUTCOME the score4 contract of PLAN §2: entry half always; exit x {stop 0.875, eod 0.412,
              target 0.0} (a resting limit pays no quoted spread) -- the favourable reading
A cell that is negative under BOTH is dead; a cell positive under only the favourable one is
reported as contract-dependent, never as a pass.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
sys.path.insert(0, f'{ROOT}/research/mature_method/frames16')
from common15 import clust_t1, mde_pct, null_green, week_shape        # noqa: E402
from scoreB_intra import book, cost_at, cost_table                    # noqa: E402
from score3 import CELLS, add_borrow, add_ssr, load, universe, repro  # noqa: E402
from common15 import attach_instrument                                # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
OUTMULT = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}


def attach_nbbo(d):
    q = pd.read_csv(f'{D}/nbbo16.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    q = q[q.n_q > 0].drop_duplicates(['day', 'symbol', 'm'])
    m = {(a, b, int(c)): v for a, b, c, v in
         zip(q.day, q.symbol, q.m, q.sp_med.astype(float))}
    for leg in ('entry_m', 'exitm_a_bare', 'exitm_a_tgt', 'exitm_b_bare', 'exitm_b_tgt'):
        if leg not in d.columns:
            continue
        d[f'sp_{leg}'] = [m.get((a, b, int(c)), np.nan)
                          for a, b, c in zip(d.day, d.symbol, d[leg])]
    return d


def score(tag, name, z, spec, mode, ct, rows):
    z = z.copy()
    z['rr'] = z[f'rr_{spec}_{mode}']
    z['rpct'] = z[f'rpct_{spec}']
    z['exit_m'] = z[f'exitm_{spec}_{mode}']
    z['why'] = z[f'why_{spec}_{mode}']
    z = z[z.etb & z.fill]
    ent = z[f'sp_entry_m'] / z.price
    ext = z[f'sp_exitm_{spec}_{mode}'] / z.price
    imp_e = cost_at(ct, z.entry_m.values)
    imp_x = cost_at(ct, z.exit_m.values)
    cov = float(np.isfinite(ent).mean()), float(np.isfinite(ext).mean())
    ent = ent.fillna(pd.Series(imp_e, index=z.index))
    ext = ext.fillna(pd.Series(imp_x, index=z.index))
    z['cost_hh'] = (0.5 * ent + 0.5 * ext) / z.rpct
    z['cost_po'] = (0.5 * ent + 0.5 * ext * z.why.map(OUTMULT).fillna(0.412)) / z.rpct
    z['cost_imp'] = (0.5 * imp_e + 0.5 * imp_x) / z.rpct
    st = dict(cell=tag, name=name, cov_entry=cov[0], cov_exit=cov[1])
    for sp in ('TRAIN', 'VAL'):
        x = z[z.split == sp]
        if len(x) < 10:
            continue
        mu, tc = clust_t1(x.rr.values, x.day.values)
        out = {f'{sp}_n': len(x), f'{sp}_g': mu, f'{sp}_tc': tc,
               f'{sp}_mde': mde_pct(x.rr.values, x.day.values),
               f'{sp}_cimp': float(x.cost_imp.mean()), f'{sp}_chh': float(x.cost_hh.mean()),
               f'{sp}_cpo': float(x.cost_po.mean())}
        for lbl, col in (('hh', 'cost_hh'), ('po', 'cost_po')):
            x2 = x.assign(net=x.rr - x[col])
            mun, tcn = clust_t1(x2.net.values, x2.day.values)
            bk = book(x2.assign(entry_m=x2.entry_m, exit_m=x2.exit_m), 'exit_m')
            w = week_shape(bk.assign(rr=bk.net), sp)
            nl = null_green(bk.assign(rr=bk.net), sp)
            q95 = x2.rr.quantile(0.95)
            out.update({f'{sp}_{lbl}_net': mun, f'{sp}_{lbl}_t': tcn,
                        f'{sp}_{lbl}_wk': w['per_wk'], f'{sp}_{lbl}_green': w['green'],
                        f'{sp}_{lbl}_null95': nl[2], f'{sp}_{lbl}_total': w['total'],
                        f'{sp}_{lbl}_wkmean': w['wk_mean'], f'{sp}_{lbl}_worst': w['worst'],
                        f'{sp}_{lbl}_ex5': float(x2.net[x2.rr <= q95].mean())})
        st.update(out)
        print(f'  {tag} {sp:5s} n={len(x):5,} gross {mu:+.3f} (t {tc:+.2f}) | cost imputed '
              f'{x.cost_imp.mean():.3f} -> MEASURED half+half {x.cost_hh.mean():.3f} '
              f'({x.cost_hh.mean()/x.cost_imp.mean():.2f}x) / per-outcome {x.cost_po.mean():.3f}',
              flush=True)
        print(f'        half+half   net {out[f"{sp}_hh_net"]:+.3f} (t {out[f"{sp}_hh_t"]:+.2f}) '
              f'{out[f"{sp}_hh_wk"]:4.1f}/wk green {out[f"{sp}_hh_green"]:5.1f}% '
              f'(null {out[f"{sp}_hh_null95"]:.1f}) ${out[f"{sp}_hh_total"]:+,.0f} '
              f'wk ${out[f"{sp}_hh_wkmean"]:+,.0f} ex5 {out[f"{sp}_hh_ex5"]:+.3f}', flush=True)
        print(f'        per-outcome net {out[f"{sp}_po_net"]:+.3f} (t {out[f"{sp}_po_t"]:+.2f}) '
              f'{out[f"{sp}_po_wk"]:4.1f}/wk green {out[f"{sp}_po_green"]:5.1f}% '
              f'(null {out[f"{sp}_po_null95"]:.1f}) ${out[f"{sp}_po_total"]:+,.0f} '
              f'wk ${out[f"{sp}_po_wkmean"]:+,.0f} ex5 {out[f"{sp}_po_ex5"]:+.3f}', flush=True)
    tr = z[z.split == 'TRAIN']
    for lbl, col in (('hh', 'cost_hh'), ('po', 'cost_po')):
        st[f'h1_{lbl}'] = float((tr[tr.day < '2025-07-01'].rr
                                 - tr[tr.day < '2025-07-01'][col]).mean())
        st[f'h2_{lbl}'] = float((tr[tr.day >= '2025-07-01'].rr
                                 - tr[tr.day >= '2025-07-01'][col]).mean())
    print(f'        halves net  half+half {st["h1_hh"]:+.3f} / {st["h2_hh"]:+.3f}   '
          f'per-outcome {st["h1_po"]:+.3f} / {st["h2_po"]:+.3f}', flush=True)
    rows.append(st)


def main():
    repro()
    ct = cost_table()
    d = load()
    d = attach_instrument(d)
    d = add_borrow(d)
    d = add_ssr(d)
    d = universe(d)
    d = attach_nbbo(d)
    print(f'[nbbo] entry-leg coverage {float(np.isfinite(d.sp_entry_m).mean())*100:.1f} %',
          flush=True)
    d['win'] = d.rr_a_bare > 0
    f = ~np.isfinite(d.sp_entry_m)
    print(f'[avail] measured entry spread missing: winners {float(f[d.win].mean())*100:.1f} % vs '
          f'losers {float(f[~d.win].mean())*100:.1f} % (gap '
          f'{abs(float(f[d.win].mean())-float(f[~d.win].mean()))*100:.1f} pp)', flush=True)
    rows = []
    print('\n== ARM 3 at the MEASURED per-trade NBBO ==', flush=True)
    for tag, name, col, spec, mode in CELLS:
        score(tag, name, d[d[col].astype(bool)], spec, mode, ct, rows)
    pd.DataFrame(rows).to_csv(f'{D}/cells3_measured.csv', index=False)
    print(f'\n[arm3] wrote cells3_measured.csv ({len(rows)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
