#!/usr/bin/env python3
"""F22 — THE BARE GEOMETRY.

The identical 2:1 bracket (1 R stop at the trade's own `r_pct`, +2 R target, 15:55 flat) on pass 6's
**288,174 detector-free control trades** — arm b (a matched non-signal name at a mover's clock) and
arm d (a matched non-signal name at a random non-break minute).  No admission rule is applied
anywhere in the construction.  Mapped by entry-minute band x stop-distance band x ADV$ band x
wrapper/common, with a 2,000-draw label permutation.

  python3 f22.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7 import D7, ROOT, asset_class, clustered_t, half_of, mde, split_of, universe   # noqa: E402

D6 = f'{ROOT}/research/mature_method/hod_frames6'
NDRAW = 2000
SEED = 20260920
MBANDS = [(577, 630, '09:37-10:30'), (630, 690, '10:30-11:30'),
          (690, 780, '11:30-13:00'), (780, 842, '13:00-14:01')]
RBANDS = [(0.0, 1.5, '<1.5%'), (1.5, 3.0, '1.5-3%'), (3.0, 1e9, '>=3%')]
ABANDS = [(0.0, 25e6, '<$25M'), (25e6, 150e6, '$25-150M'), (150e6, 1e18, '>=$150M')]
WKS = {'H1': 26, 'H2': 27, 'VAL': 23}          # market weeks per half (pass 4/5 week counts)


def band(v, bands):
    out = np.full(len(v), '', dtype=object)
    for lo, hi, lab in bands:
        out[(v >= lo) & (v < hi)] = lab
    return out


def load():
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str},
                     usecols=['day', 'symbol', 'entry_m', 'split', 'r_pct'])
    b = pd.read_csv(f'{D6}/pb6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    b['ctrl_entry_m'] = b.entry_m
    b['arm'] = 'b'
    d = pd.read_csv(f'{D6}/pd6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    d['arm'] = 'd'
    p = pd.concat([b, d], ignore_index=True)
    p = p.merge(bk, on=['day', 'symbol', 'entry_m'], how='left')
    print(f'  control trades {len(p):,} (arm b {int((p.arm=="b").sum()):,} / '
          f'arm d {int((p.arm=="d").sum()):,}) | r_pct join {p.r_pct.notna().mean()*100:.1f} %',
          flush=True)
    u = universe()[['day', 'symbol', 'advd']].rename(columns={'symbol': 'ctrl'})
    p = p.merge(u, on=['day', 'ctrl'], how='left')
    cls = asset_class(p.ctrl.unique())
    p['cls'] = p.ctrl.map(cls)
    p['h'] = np.where(p.split == 'VAL', 'VAL', p.day.map(half_of))
    p['mb'] = band(p.ctrl_entry_m.values.astype(float), MBANDS)
    p['rb'] = band(p.r_pct.values.astype(float), RBANDS)
    p['ab'] = band(p.advd.fillna(-1).values.astype(float), ABANDS)
    print(f'  ADV$ join {p.advd.notna().mean()*100:.1f} % | class stock '
          f'{(p.cls=="stock").mean()*100:.1f} % wrapper {(p.cls=="wrapper").mean()*100:.1f} % '
          f'unknown {(p.cls=="unknown").mean()*100:.1f} %', flush=True)
    return p[p.r_pct.notna() & p.rr.notna()]


def row(p, mask, name, rng, pool, out, key_n=True):
    a = p[mask]
    if len(a) < 50:
        return
    g = a.groupby('h').rr.agg(['mean', 'size'])
    pos = all(g['mean'].get(k, -1) > 0 for k in ('H1', 'H2', 'VAL'))
    nb = len(a)
    draws = rng.choice(pool, size=(NDRAW, min(nb, 20000)), replace=True).mean(axis=1)
    p95 = float(np.percentile(draws, 95))
    obs = float(a.rr.mean())
    # book-sized opportunity rate: distinct booked slots whose control lands in this bucket
    slots = a.drop_duplicates(['day', 'symbol', 'entry_m']).groupby(
        a.drop_duplicates(['day', 'symbol', 'entry_m']).h).size()
    rates = {k: slots.get(k, 0) / WKS[k] for k in WKS}
    ok = pos and obs > p95 and min(rates.values()) >= 10
    out.append(dict(cell=name, n=nb, H1=g['mean'].get('H1', np.nan), H2=g['mean'].get('H2', np.nan),
                    VAL=g['mean'].get('VAL', np.nan), all=obs, null_p95=p95,
                    tr_wk_H1=rates['H1'], tr_wk_H2=rates['H2'], tr_wk_VAL=rates['VAL'],
                    clust_t=clustered_t(a.rr.values, a.day.values), mde=mde(a.rr.values),
                    era_pos=pos, above_null=obs > p95, FINDING=ok))
    print(f'| {name:<28s} | {nb:7d} | {g["mean"].get("H1", np.nan):+.4f} | '
          f'{g["mean"].get("H2", np.nan):+.4f} | {g["mean"].get("VAL", np.nan):+.4f} | '
          f'{obs:+.4f} | {p95:+.4f} | {min(rates.values()):5.1f} | '
          f'{clustered_t(a.rr.values, a.day.values):+5.2f} | {"YES" if ok else "-"} |',
          flush=True)


def main():
    print('== F22 — the bare geometry on 288K detector-free controls ==', flush=True)
    p = load()
    rng = np.random.default_rng(SEED)
    pool = p.rr.values
    print(f'\n  POPULATION mean gross R {pool.mean():+.4f} on {len(pool):,} control trades '
          f'(H1 {p[p.h=="H1"].rr.mean():+.4f} / H2 {p[p.h=="H2"].rr.mean():+.4f} / '
          f'VAL {p[p.h=="VAL"].rr.mean():+.4f})', flush=True)
    out = []
    print('\n== the 12 declared marginal cells ==')
    print('| cell | n | H1 | H2 | VAL | all | null p95 | min tr/wk | clust t | FINDING |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for _, _, lab in MBANDS:
        row(p, p.mb == lab, f'minute {lab}', rng, pool, out)
    for _, _, lab in RBANDS:
        row(p, p.rb == lab, f'r_pct {lab}', rng, pool, out)
    for _, _, lab in ABANDS:
        row(p, p.ab == lab, f'ADV$ {lab}', rng, pool, out)
    for lab in ('wrapper', 'stock'):
        row(p, p.cls == lab, f'class {lab}', rng, pool, out)
    n_marg = len(out)
    print('\n== the 72-bucket cross-map (screen; multiplicity counted) ==')
    print('| cell | n | H1 | H2 | VAL | all | null p95 | min tr/wk | clust t | FINDING |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    nbuck = 0
    for _, _, mb in MBANDS:
        for _, _, rb in RBANDS:
            for _, _, ab in ABANDS:
                for cl in ('wrapper', 'stock'):
                    nbuck += 1
                    m = (p.mb == mb) & (p.rb == rb) & (p.ab == ab) & (p.cls == cl)
                    row(p, m, f'{mb}|{rb}|{ab}|{cl}', rng, pool, out)
    c = pd.DataFrame(out)
    c.to_csv(f'{D7}/cells22.csv', index=False)
    fin = c[c.FINDING]
    print(f'\n  marginal cells {n_marg} | cross-map buckets declared {nbuck}, scored '
          f'{len(c) - n_marg} (>=50 trades) | FINDINGS {len(fin)}', flush=True)
    print(f'  positive-in-all-three-eras cells: {int(c.era_pos.sum())} / {len(c)}; '
          f'above their own null p95: {int(c.above_null.sum())} / {len(c)}', flush=True)
    if len(fin):
        print(fin.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
