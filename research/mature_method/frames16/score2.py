#!/usr/bin/env python3
"""frames16 ARM 2 — score the lambda / residual terciles on B2 (PREREG §3, §4).

Cells L1..L4: terciles cut on TRAIN edges, keep the best TRAIN tercile, two exits.
Rails: coverage >= 80 % with a winner/loser missingness gap <= 5 pp, or the arm is VOID.
Declared diagnostic (never a cell): the forward-mid decay structure by tercile.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import clust_t1, mde_pct, null_green, week_shape       # noqa: E402
from scoreB_intra import book                                        # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
FWD = [10, 30, 60, 300]


def main():
    if not os.path.exists(f'{D}/lam.csv'):
        print('ARM 2: no lam.csv — the arm did not run. Nothing scored.', flush=True)
        return 0
    b2 = pd.read_pickle(f'{ROOT}/research/mature_method/hod_filter_stack/b2.pkl')
    b2 = b2[b2.split.isin(('TRAIN', 'VAL'))].copy()
    lam = pd.read_csv(f'{D}/lam.csv', dtype={'symbol': str, 'day': str},
                      keep_default_na=False, na_values=[''])
    lam = lam.drop_duplicates(['day', 'symbol', 'break_m'])
    d = b2.merge(lam, on=['day', 'symbol', 'break_m'], how='left')
    print(f'[arm2] B2 {len(b2):,} · lambda rows {len(lam):,} · merged '
          f'{int(d.lam.notna().sum()):,} ({d.lam.notna().mean()*100:.1f} %)', flush=True)

    d['win'] = d.rr > 0
    for f, nm in ((d.lam.isna(), 'lambda'), (d.resid.isna(), 'residual')):
        gw, gl = float(f[d.win].mean()), float(f[~d.win].mean())
        print(f'[avail] {nm}: coverage {100*(1-float(f.mean())):.1f} % · miss winners '
              f'{gw*100:.1f} % vs losers {gl*100:.1f} % (gap {abs(gw-gl)*100:.1f} pp)', flush=True)
    cov = float(d.lam.notna().mean())
    gap = abs(float(d.lam.isna()[d.win].mean()) - float(d.lam.isna()[~d.win].mean()))
    verdict = 'OK' if (cov >= 0.80 and gap <= 0.05) else 'VOID BY THE AVAILABILITY RAIL'
    print(f'[avail] arm 2 rail: {verdict}', flush=True)

    z = d[d.lam.notna()].copy()
    print('\n== decay structure (DECLARED DIAGNOSTIC — forward mid %, not a money horizon) ==',
          flush=True)
    for fld in ('lam', 'resid'):
        q = z[z.split == 'TRAIN'][fld].quantile([1 / 3, 2 / 3]).values
        z[f'{fld}_T'] = np.digitize(z[fld].values, q)
        for sp in ('TRAIN', 'VAL'):
            x = z[z.split == sp]
            line = f'  {fld:6s} {sp:5s} '
            for t in (0, 1, 2):
                y = x[x[f'{fld}_T'] == t]
                line += f'| T{t} n={len(y):5,} ' + ' '.join(
                    f'{s}s {np.nanmean(y[f"fwd{s}"]):+.3f}' for s in FWD)
            print(line, flush=True)

    rows = []
    print('\n== ARM 2 cells (terciles cut on TRAIN, keep the best TRAIN tercile) ==', flush=True)
    for fld in ('lam', 'resid'):
        tr = z[z.split == 'TRAIN']
        means = {t: float(tr[tr[f'{fld}_T'] == t].rr.mean()) for t in (0, 1, 2)}
        best = max(means, key=means.get)
        spread = means[best] - min(means.values())
        print(f'  {fld}: TRAIN terciles gross R {means[0]:+.3f} / {means[1]:+.3f} / '
              f'{means[2]:+.3f}  best T{best}  spread {spread:+.3f} '
              f'(selection floor 0.20)', flush=True)
        for exit_tag, sfx, lbl in (('bare', '', 'shipped +2R bracket, consol-low stop'),
                                   ('tgt', '_b', '+2R bracket, tight stop')):
            rr_col, xm_col, rp_col, wy_col = f'rr{sfx}', f'exit_m{sfx}', f'r_pct{sfx}', f'why{sfx}'
            tag = {'lam': {'bare': 'L1', 'tgt': 'L2'},
                   'resid': {'bare': 'L3', 'tgt': 'L4'}}[fld][exit_tag]
            st = dict(cell=tag, field=fld, exit=lbl, best_tercile=int(best),
                      train_spread=spread, cov=cov, gap=gap, rail=verdict)
            for sp in ('TRAIN', 'VAL'):
                x = z[(z.split == sp) & (z[f'{fld}_T'] == best)].copy()
                if len(x) < 10:
                    continue
                # the score4 per-outcome cost contract (PLAN §1), applied to this exit variant
                half = 0.5 * x.sp_pct / np.maximum(x[rp_col], 0.05)
                mult = x[wy_col].map({'stop': 0.875, 'eod': 0.412, 'target': 0.0}).fillna(0.412)
                x['rr'] = x[rr_col]
                x['net'] = x[rr_col] - half - half * mult
                mu, tc = clust_t1(x.rr.values, x.day.values)
                mun, tcn = clust_t1(x.net.values, x.day.values)
                bk = book(x.assign(entry_m=x.entry_m, exit_m=x[xm_col]), 'exit_m')
                w = week_shape(bk.assign(rr=bk.net), sp)
                nl = null_green(bk.assign(rr=bk.net), sp)
                st.update({f'{sp}_n': len(x), f'{sp}_g': mu, f'{sp}_t': tc, f'{sp}_net': mun,
                           f'{sp}_tnet': tcn, f'{sp}_mde': mde_pct(x.rr.values, x.day.values),
                           f'{sp}_wk': w['per_wk'], f'{sp}_green': w['green'],
                           f'{sp}_null95': nl[2], f'{sp}_total': w['total'],
                           f'{sp}_worst': w['worst']})
                print(f'    {tag} {sp:5s} n={len(x):5,} gross {mu:+.3f} (t {tc:+.2f}) net '
                      f'{mun:+.3f} | book {w["n"]:4d} ({w["per_wk"]:4.1f}/wk) green '
                      f'{w["green"]:5.1f}% (null {nl[2]:.1f}) ${w["total"]:+,.0f} worst '
                      f'${w["worst"]:+,.0f}', flush=True)
            h1 = z[(z.split == 'TRAIN') & (z.day < '2025-07-01') & (z[f'{fld}_T'] == best)]
            h2 = z[(z.split == 'TRAIN') & (z.day >= '2025-07-01') & (z[f'{fld}_T'] == best)]
            st['h1'] = float(h1[rr_col].mean()); st['h2'] = float(h2[rr_col].mean())
            print(f'         halves {st["h1"]:+.3f} / {st["h2"]:+.3f}', flush=True)
            rows.append(st)
    pd.DataFrame(rows).to_csv(f'{D}/cells2.csv', index=False)
    print(f'\n[arm2] wrote cells2.csv ({len(rows)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
