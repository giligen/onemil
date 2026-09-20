#!/usr/bin/env python3
"""F34 stage 2 — THE FLOOR MAP.  13 declared cells (PREREG §F34).

Scores `w34.csv` — the unconditional long bracket at a FIXED 2/3/4 % stop on pass 6's 288,174
detector-free arm-d controls — in **% of entry price** (F31's unit), net of the programme's own
measured cost, by entry hour x price band x ADV$ band x wrapper/common as MARGINALS.

  python3 s34.py            # writes cells34.csv

Reads only; TEST never loaded.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames11', 'frames9', 'frames7', 'hod_frames6', 'hod_frames5', 'hod_frames4',
           'hod_frames3', 'hod_frames2'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

import score as S                                                  # noqa: E402
from common4 import load_breaks4                                   # noqa: E402
from common3 import clustered_t                                    # noqa: E402
import c7                                                          # noqa: E402

D6 = f'{ROOT}/research/mature_method/hod_frames6'
D11 = f'{ROOT}/research/mature_method/frames11'
STOPS = (2, 3, 4)
NDRAW = 2000
SEED = 20260925
WEEKS = {'TRAIN': 53, 'VAL': 22}          # market weeks per split (the programme's counts)

MB = [(577, 630, '09:37-10:30'), (630, 690, '10:30-11:30'),
      (690, 780, '11:30-13:00'), (780, 842, '13:00-14:01')]
PB = [(0, 30, '$17-30'), (30, 100, '$30-100'), (100, 1e18, '>=$100')]
AB = [(0, 25e6, '<$25M'), (25e6, 150e6, '$25-150M'), (150e6, 1e18, '>=$150M')]


def band(v, bands):
    out = np.full(len(v), '', dtype=object)
    for lo, hi, lab in bands:
        out[(v >= lo) & (v < hi)] = lab
    return out


def load():
    w = pd.read_csv(f'{D11}/w34.csv', dtype={'day': str, 'ctrl': str})
    n0 = len(w)
    w = w.drop_duplicates(['day', 'ctrl', 'ctrl_entry_m'])
    d = pd.read_csv(f'{D6}/pd6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    d = d[d.day < '2026-06-01']
    print(f'  walked rows {n0:,} -> {len(w):,} unique keys of {len(d):,} arm-d controls '
          f'= {len(w)/len(d)*100:.1f} % availability', flush=True)
    assert len(w) / len(d) >= 0.80, 'bar-join availability below the 80 % rail'
    w['split'] = S.split_of(w.day.values)
    w = w[w.split.isin(('TRAIN', 'VAL'))].copy()
    w['half'] = np.where(w.split == 'VAL', 'VAL',
                         np.where(w.day < '2025-07-01', 'H1', 'H2'))
    # the cost model, built from the SAME measured NBBO sample the book uses
    S.build_impute(load_breaks4(verbose=False))
    pbx = pd.cut(w.entry, S.PB_EDGES, labels=S.PB_LAB)
    hbx = pd.cut(w.ctrl_entry_m, S.HB_EDGES, labels=S.HB_LAB)
    w['sp_pct'] = [S.IMPUTE.get((p, h), S.IMPUTE_GLOBAL) for p, h in zip(pbx, hbx)]
    w['sp_pct'] = w.sp_pct.fillna(S.IMPUTE_GLOBAL)
    # ADV$ and asset class of the CONTROL name
    u = c7.universe()[['day', 'symbol', 'advd']].rename(columns={'symbol': 'ctrl'})
    w = w.merge(u, on=['day', 'ctrl'], how='left')
    w['cls'] = w.ctrl.map(c7.asset_class(w.ctrl.unique()))
    print(f'  ADV$ join {w.advd.notna().mean()*100:.1f} % | class stock '
          f'{(w.cls=="stock").mean()*100:.1f} % wrapper {(w.cls=="wrapper").mean()*100:.1f} %',
          flush=True)
    w['mb'] = band(w.ctrl_entry_m.values.astype(float), MB)
    w['pb'] = band(w.entry.values.astype(float), PB)
    w['ab'] = band(w.advd.fillna(-1).values.astype(float), AB)
    for s in STOPS:
        # % OF PRICE, the F31 unit: R = s % of price, so gross_pct = rr x s
        w[f'g{s}'] = w[f'rr_{s}'] * s
        ratio = w[f'why_{s}'].map(S.RATIO).fillna(0.875)
        # the programme's cost, expressed in % of price: 0.5 x spread% x (1 + ratio).
        # NOTE it does NOT depend on the stop width — F31's point, made in reverse.
        w[f'c{s}'] = 0.5 * w.sp_pct * (1.0 + ratio)
        w[f'n{s}'] = w[f'g{s}'] - w[f'c{s}']
    return w


def cell_row(d, cid, name, s):
    col = f'n{s}'
    r = dict(cell=cid, name=name, stop_pct=s)
    for sp in ('TRAIN', 'VAL'):
        q = d[d.split == sp]
        r[f'n_{sp}'] = len(q)
        r[f'perwk_{sp}'] = len(q) / WEEKS[sp]
        r[f'gross_{sp}'] = float(q[f'g{s}'].mean()) if len(q) else np.nan
        r[f'cost_{sp}'] = float(q[f'c{s}'].mean()) if len(q) else np.nan
        r[f'net_{sp}'] = float(q[col].mean()) if len(q) else np.nan
        r[f't_{sp}'] = clustered_t(q.assign(net=q[col])) if len(q) > 3 else np.nan
        r[f'mde_{sp}'] = (2.8 * float(q[col].std(ddof=1)) / np.sqrt(max(q.day.nunique(), 1))
                          if len(q) > 5 else np.nan)
    for h in ('H1', 'H2'):
        q = d[d.half == h]
        r[f'net_{h}'] = float(q[col].mean()) if len(q) else np.nan
    return r


def main() -> int:
    rng = np.random.default_rng(SEED)
    print('F34 — the floor map: the bare instrument on the PIT HOD universe (13 cells)', flush=True)
    w = load()
    cells = [('U0', 'unconditional', None)]
    cells += [(f'U{i+1}', f'hour {lab}', ('mb', lab)) for i, (_, _, lab) in enumerate(MB)]
    cells += [(f'U{i+5}', f'price {lab}', ('pb', lab)) for i, (_, _, lab) in enumerate(PB)]
    cells += [(f'U{i+8}', f'ADV$ {lab}', ('ab', lab)) for i, (_, _, lab) in enumerate(AB)]
    cells += [('U11', 'wrapper', ('cls', 'wrapper')), ('U12', 'common', ('cls', 'stock'))]
    assert len(cells) == 13, len(cells)

    rows = []
    print(f'\n  {"cell":5s} {"name":16s} {"s%":>3s} {"n_TR":>7s} {"/wk":>6s} '
          f'{"grossTR":>8s} {"costTR":>7s} {"netTR":>7s} {"t":>6s} {"netVAL":>7s} {"t":>6s} '
          f'{"H1":>7s} {"H2":>7s} {"MDE":>6s}', flush=True)
    for cid, name, sel in cells:
        d = w if sel is None else w[w[sel[0]] == sel[1]]
        for s in STOPS:
            r = cell_row(d, cid, name, s)
            rows.append(r)
            print(f'  {cid:5s} {name:16s} {s:3d} {r["n_TRAIN"]:7,d} {r["perwk_TRAIN"]:6.0f} '
                  f'{r["gross_TRAIN"]:+8.3f} {r["cost_TRAIN"]:7.3f} {r["net_TRAIN"]:+7.3f} '
                  f'{r["t_TRAIN"]:+6.2f} {r["net_VAL"]:+7.3f} {r["t_VAL"]:+6.2f} '
                  f'{r["net_H1"]:+7.3f} {r["net_H2"]:+7.3f} {r["mde_TRAIN"]:6.3f}', flush=True)

    out = pd.DataFrame(rows)
    # the pre-committed bar
    out['positive'] = ((out.net_TRAIN > 0) & (out.net_VAL > 0) & (out.net_H1 > 0)
                       & (out.net_H2 > 0) & (out.perwk_TRAIN >= 10) & (out.perwk_VAL >= 10)
                       & (out.t_TRAIN.abs() >= 2.0))

    # permutation ACROSS ALL 13 cells: shuffle the cell labels within day, 2,000 draws, take the
    # max |net| any cell reaches — the multiplicity of the whole map is paid once.
    print('\n  permutation across all 13 cells (labels shuffled within day, 2,000 draws):',
          flush=True)
    for s in STOPS:
        col = f'n{s}'
        tr = w[w.split == 'TRAIN']
        obs = []
        for cid, name, sel in cells:
            if sel is None:
                continue
            q = tr[tr[sel[0]] == sel[1]]
            if len(q) > 5:
                obs.append((cid, float(q[col].mean()), len(q)))
        best = max(obs, key=lambda x: x[1])
        v = tr[col].values
        days = tr.day.values
        order = np.argsort(days, kind='mergesort')
        v_s = v[order]
        sizes = [n for _, _, n in obs]
        draws = np.empty(NDRAW)
        for k in range(NDRAW):
            p = rng.permutation(v_s)
            draws[k] = max(p[:n].mean() for n in sizes)
        p95 = float(np.percentile(draws, 95))
        print(f'  stop {s}%: best cell {best[0]} at {best[1]:+.3f} % of price vs the map\'s own '
              f'permutation p95 {p95:+.3f} -> {"OUTSIDE" if best[1] > p95 else "inside"}',
              flush=True)
        out.loc[out.stop_pct == s, 'perm_p95_train'] = p95

    out.to_csv(f'{D11}/cells34.csv', index=False)
    n_pos = int(out.positive.sum())
    print(f'\n  {len(out)} scored rows over 13 declared cells x 3 stop widths; '
          f'{n_pos} clear the pre-committed bar', flush=True)
    if n_pos:
        print(out[out.positive][['cell', 'name', 'stop_pct', 'net_TRAIN', 'net_VAL',
                                 't_TRAIN']].to_string(index=False), flush=True)
    # the diagnostic cross-tab (never promoted)
    print('\n  DIAGNOSTIC cross-tab, net % of price at the 3 % stop, TRAIN:', flush=True)
    tr = w[w.split == 'TRAIN']
    print(tr.pivot_table(index='mb', columns='ab', values='n3', aggfunc='mean').round(3)
          .to_string(), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
