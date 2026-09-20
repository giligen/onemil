#!/usr/bin/env python3
"""frames15 ARM B (intraday) — scoring B9..B14 and the placebo decomposition D1/D3.

Cost is a declared PROXY, not a per-trade measurement: the frames14 F45 minute-of-day NBBO MEDIAN
(measured on the HOD population, the same names at the same clocks), charged as half the entry
minute's spread plus half the exit minute's spread, in % of price, converted to R at the declared
2 % stop. Gross is reported first and the RUNBOOK's gross-before-net rule applies.
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/frames15')
from common15 import (D, ROOT, RISK, STOP_PCT, attach_instrument, clust_t1,   # noqa: E402
                      mde_pct, null_green, week_shape)

CELLS = [('B9', 'V1 hrv >= 3', 'f_hrv3', 'bare'),
         ('B10', 'V2 sustained hrv>=2 over 3 hours', 'f_sus', 'bare'),
         ('B11', 'V4 absorption hrv>=3 & |hour ret|<=1%', 'f_abs', 'bare'),
         ('B12', 'CONTROL mirror hrv>=3 & |hour ret|>2%', 'f_mir', 'bare')]


def cost_table():
    t = pd.read_csv(f'{ROOT}/research/mature_method/frames14/f45_minute_table.csv')
    return dict(zip(t.clock_m.astype(int), t.med.astype(float)))


def cost_at(ct, m):
    ks = np.array(sorted(ct))
    i = np.clip(np.searchsorted(ks, m), 0, len(ks) - 1)
    return np.array([ct[k] for k in ks[i]]) / 100.0


def book(d, exit_col, nday=12, nconc=4):
    z = d.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort')
    keep = np.zeros(len(z), dtype=bool)
    day = z.day.values
    em = z.entry_m.values
    xm = z[exit_col].values
    cur, cnt, free = None, 0, []
    for j in range(len(z)):
        if day[j] != cur:
            cur, cnt, free = day[j], 0, []
        free = [x for x in free if x > em[j]]
        if cnt >= nday or len(free) >= nconc:
            continue
        cnt += 1
        free.append(xm[j])
        keep[j] = True
    return z[keep]


def load(pat):
    fs = sorted(glob.glob(f'{D}/{pat}'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str}) for f in fs],
                  ignore_index=True)
    d = d[d.day < '2026-06-01']
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    return d


def report(tag, name, d, rr_col, exit_col, ct, rows):
    d = d.copy()
    cpct = 0.5 * cost_at(ct, d.entry_m.values) + 0.5 * cost_at(ct, d[exit_col].values)
    d['cost_R'] = cpct / STOP_PCT
    d['rr'] = d[rr_col]
    d['net'] = d.rr - d.cost_R
    st = {'cell': tag, 'name': name, 'n': len(d)}
    for sp in ('TRAIN', 'VAL'):
        z = d[d.split == sp]
        if not len(z):
            continue
        mu, t = clust_t1(z.rr.values, z.day.values)
        mun, tn = clust_t1(z.net.values, z.day.values)
        bk = book(z, exit_col)
        w = week_shape(bk.assign(rr=bk.net), sp)
        nl = null_green(bk.assign(rr=bk.net), sp)
        st.update({f'{sp}_n': len(z), f'{sp}_gross': mu, f'{sp}_pct': mu * STOP_PCT * 100,
                   f'{sp}_t': t, f'{sp}_cost': float(z.cost_R.mean()), f'{sp}_net': mun,
                   f'{sp}_tnet': tn, f'{sp}_mde': mde_pct(z.rr.values, z.day.values),
                   f'{sp}_bk_n': w['n'], f'{sp}_bk_wk': w['per_wk'], f'{sp}_green': w['green'],
                   f'{sp}_total': w['total'], f'{sp}_worst': w['worst'],
                   f'{sp}_streak': w['redstreak'], f'{sp}_null95': nl[2],
                   f'{sp}_wrap': float((z.asset_class == 'wrapper').mean()),
                   f'{sp}_ex5': float(np.nanmean(z.rr[z.rr <= z.rr.quantile(0.95)]))})
        print(f'  {tag} {sp:5s} n={len(z):6,} gross {mu:+.3f} R ({mu*STOP_PCT*100:+.3f}% of price, '
              f't {t:+.2f}, mde {mde_pct(z.rr.values, z.day.values):.3f}) cost {z.cost_R.mean():.3f} '
              f'net {mun:+.3f}  book {w["n"]:4d} ({w["per_wk"]:4.1f}/wk) green {w["green"]:5.1f}% '
              f'(null {nl[2]:5.1f}) ${w["total"]:+,.0f} wk ${w["wk_mean"]:+,.0f} '
              f'worst ${w["worst"]:+,.0f} wrap {float((z.asset_class=="wrapper").mean())*100:.0f}%',
              flush=True)
    tr = d[d.split == 'TRAIN']
    st['h1'] = float(tr[tr.day < '2025-07-01'].rr.mean())
    st['h2'] = float(tr[tr.day >= '2025-07-01'].rr.mean())
    rows.append(st)
    return st


def main():
    ct = cost_table()
    d = load('intra_*.csv')
    d = attach_instrument(d)
    print(f'[intra] {len(d):,} walked signals; gate5 (causal >=5% above open at the decision) '
          f'{float(d.gate5.mean())*100:.1f}%', flush=True)
    g = d[d.gate5.astype(bool)]

    # ------------------------------------------------------------------ the decomposition
    c = load('ctrl_*.csv')
    c = c[c.gate5.astype(bool)]
    print('\n== decomposition (bare exit, gate5 population, gross R) ==', flush=True)
    for sp in ('TRAIN', 'VAL'):
        z = c[(c.split == sp)]
        d1 = z[~z.sig_day.astype(bool)]
        d3 = z[z.sig_day.astype(bool) & ~z.is_sig.astype(bool)]
        sg = z[z.is_sig.astype(bool)]
        for nm, x in (('D1 universe bound (non-signal name)', d1),
                      ('D3 same name-day, another hour', d3),
                      ('the SIGNAL hour itself', sg)):
            mu, t = clust_t1(x.rr_bare.values, x.day.values)
            print(f'  {sp:5s} {nm:36s} n={len(x):7,}  {mu:+.3f} R ({mu*STOP_PCT*100:+.3f}% of '
                  f'price)  t={t:+.2f}', flush=True)

    rows = []
    print('\n== B intraday cells (gate5 = causal membership) ==', flush=True)
    for tag, name, col, _ in CELLS:
        z = g[g[col].astype(bool)]
        report(tag, name, z, 'rr_bare', 'exitm_bare', ct, rows)

    best = max(rows, key=lambda r: r.get('TRAIN_net', -9))
    print(f'\n  B13/B14 take the TRAIN-best signal: {best["cell"]} ({best["name"]})', flush=True)
    col = dict((r[0], r[2]) for r in [(c[0], c[1], c[2]) for c in CELLS])[best['cell']]
    z = g[g[col].astype(bool)]
    report('B13', f'{best["cell"]} + 2R target', z, 'rr_tgt', 'exitm_tgt', ct, rows)
    report('B14', f'{best["cell"]} + ORB static lock', z, 'rr_lock', 'exitm_lock', ct, rows)

    print('\n== the ungated diagnostic (UNIVERSE LOOK-AHEAD — not a cell) ==', flush=True)
    for tag, name, col, _ in CELLS:
        z = d[d[col].astype(bool)]
        for sp in ('TRAIN', 'VAL'):
            x = z[z.split == sp]
            mu, t = clust_t1(x.rr_bare.values, x.day.values)
            print(f'  {tag} {sp:5s} ungated n={len(x):6,} gross {mu:+.3f} R t={t:+.2f}', flush=True)

    pd.DataFrame(rows).to_csv(f'{D}/cellsB_intra.csv', index=False)
    print(f'\n[intra] wrote cellsB_intra.csv ({len(rows)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
