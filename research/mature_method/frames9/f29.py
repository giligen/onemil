#!/usr/bin/env python3
"""frames9 / F29 — THE POND MAP: which attribute owns the +0.23 R.

The object is the PAIRED SELECTION MARGIN on HOD-break's OWN population:

    m(trade) = rr_signal - mean(rr over that trade's own matched non-signal controls)

under G3 (the bare stop, where the margin is largest: +0.183 / +0.245 R) and X0 (the shipped +2 R
cap: +0.123 / +0.240).  Both sides are already walked (`frames8/w_sig.csv`, `frames8/w_b.csv`), so
this frame needs no bars.

Ten DECLARED splits of that margin (PREREG §F29 S1..S10).  Every level is printed whatever it
reads; nothing is selected after the fact.  TEST is never loaded.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9                                                              # noqa: E402
from common4 import load_breaks4, admit, clustered_t                   # noqa: E402
from common5 import sigset5, S, S2                                     # noqa: E402

D8, D9 = c9.D8, c9.D9
GE = ('G3', 'X0')


def margin_frame():
    """One row per BOOKED B2 trade: its margin under each geometry + every declared attribute."""
    br = load_breaks4(verbose=False)
    S.build_impute(S2.load_pop())
    sig = sigset5(admit(br, pd.Series(True, index=br.index)), min_price=20.0)
    sig = sig[sig.split.isin(c9.SPLITS)]
    from common6 import base_book
    b, _ = base_book(br, verbose=False)
    bset = set(zip(b.day, b.symbol, b.entry_m))

    W = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''],
                    usecols=['day', 'symbol', 'entry_m'] + [f'rr_{g}' for g in GE])
    B = pd.read_csv(f'{D8}/w_b.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                    keep_default_na=False, na_values=[''],
                    usecols=['day', 'symbol', 'entry_m', 'ctrl'] + [f'rr_{g}' for g in GE])
    keys = ['day', 'symbol', 'entry_m']
    cb = B.groupby(keys)[[f'rr_{g}' for g in GE]].mean().rename(
        columns={f'rr_{g}': f'cb_{g}' for g in GE})
    nb = B.groupby(keys).size().rename('n_ctrl')
    d = sig.merge(W, on=keys, how='inner').merge(cb, on=keys, how='inner').merge(
        nb, on=keys, how='inner')
    d = d[[k in bset for k in zip(d.day, d.symbol, d.entry_m)]].copy()
    for g in GE:
        d[f'm_{g}'] = d[f'rr_{g}'] - d[f'cb_{g}']
    d['half'] = np.where(d.day < '2025-07-01', 'H1', 'H2')

    # ------------------------------------------------------------------ the declared attributes
    u = c9.panel()[['day', 'symbol', 'gap_pct', 'advd', 'age', 'open']]
    d = d.merge(u, on=['day', 'symbol'], how='left')
    cls = c9.asset_class(list(d.symbol.unique()))
    d['cls'] = d.symbol.map(cls)

    from trading.orb_asset_class import underlying_anchor, load_class_map
    cmap = load_class_map()
    nmap = c9._CLS['nm']
    anc = {s: (underlying_anchor(s, nmap.get(s), cmap) or s) for s in set(br.symbol.astype(str))}
    br2 = br.copy()
    br2['anchor'] = br2.symbol.map(anc)
    # CAUSAL sibling rule: the EARLIEST admitted-signal minute per (day, anchor) excluding the
    # trade's own symbol must be strictly before this trade's entry minute.
    a_all = sigset5(admit(br, pd.Series(True, index=br.index)), min_price=20.0)
    a_all = a_all[a_all.split.isin(c9.SPLITS)].copy()
    a_all['anchor'] = a_all.symbol.map(anc)
    sib = {}
    for (day, ank), g in a_all.groupby(['day', 'anchor']):
        for r in g.itertuples():
            others = g[g.symbol != r.symbol]
            sib[(day, r.symbol, int(r.entry_m))] = bool(
                len(others) and others.entry_m.min() < r.entry_m)
    d['sibling'] = [sib.get(k, False) for k in zip(d.day, d.symbol, d.entry_m)]
    d['anchor'] = d.symbol.map(anc)
    return d


def band(v, edges, labels):
    return pd.cut(v, edges, labels=labels, right=False)


def splits_of(d):
    """The ten declared splits, as (id, name, Series of level labels)."""
    out = []
    out.append(('S1', 'asset class', d.cls.fillna('unknown')))
    out.append(('S2', 'price band', band(d.price, [0, 30, 60, 1e9], ['<$30', '$30-60', '>=$60'])))
    out.append(('S3', 'ADV$ band', band(d.advd, [0, 25e6, 150e6, 1e15],
                                        ['<$25M', '$25-150M', '>=$150M'])))
    out.append(('S4', 'gap at the open', band(d.gap_pct, [-1e9, -2, 2, 1e9],
                                              ['gap<=-2%', 'flat', 'gap>=+2%'])))
    out.append(('S5', 'rv_profile', np.where(d.rv_profile >= 5, 'rv>=5', 'rv<5')))
    out.append(('S6', 'sibling moved (causal)', np.where(d.sibling, 'sibling', 'alone')))
    out.append(('S7', 'listing age', np.where(d.age < 60, 'new (<60 sessions)', 'old')))
    out.append(('S8', 'entry-minute band', band(d.entry_m, [577, 630, 690, 780, 842],
                                                ['09:37-10:30', '10:30-11:30', '11:30-13:00',
                                                 '13:00-14:01'])))
    out.append(('S9', 'dist_open_pct', band(d.dist_open_pct, [5, 10, 20, 1e9],
                                            ['5-10%', '10-20%', '>=20%'])))
    q = d.dollar_frac.quantile([1 / 3, 2 / 3]).values
    out.append(('S10', 'dollar_frac terciles',
                band(d.dollar_frac, [-1e9, q[0], q[1], 1e9], ['T1 low', 'T2', 'T3 high'])))
    return out


def main():
    d = margin_frame()
    print(f'booked trades with a matched control: {len(d)} '
          f'(TRAIN {int((d.split=="TRAIN").sum())} / VAL {int((d.split=="VAL").sum())}), '
          f'median controls/trade {d.n_ctrl.median():.0f}', flush=True)
    for g in GE:
        for sp in c9.SPLITS:
            x = d[d.split == sp]
            print(f'  BASE {g} {sp}: margin {x[f"m_{g}"].mean():+.4f} '
                  f't={clustered_t(x.assign(net=x[f"m_{g}"])):+.2f}', flush=True)
    rows = []
    for sid, nm, lab in splits_of(d):
        d['_lab'] = pd.Series(lab, index=d.index).astype(str)
        print(f'\n--- {sid} {nm} ---', flush=True)
        print('| level | n | margin G3 | H1 | H2 | VAL | t(G3) | MDE | margin X0 | share of trades |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for lv, g in d.groupby('_lab'):
            if lv == 'nan':
                continue
            tr = g[g.split == 'TRAIN']
            va = g[g.split == 'VAL']
            r = dict(split_id=sid, split_name=nm, level=lv, n=len(g),
                     m_g3=float(g.m_G3.mean()), m_x0=float(g.m_X0.mean()),
                     h1=float(g[g.half == 'H1'].m_G3.mean()) if len(g[g.half == 'H1']) else np.nan,
                     h2=float(g[(g.half == 'H2') & (g.split == 'TRAIN')].m_G3.mean())
                     if len(g[(g.half == 'H2') & (g.split == 'TRAIN')]) else np.nan,
                     val=float(va.m_G3.mean()) if len(va) else np.nan,
                     train=float(tr.m_G3.mean()) if len(tr) else np.nan,
                     t_g3=clustered_t(g.assign(net=g.m_G3)),
                     mde=2.80 * float(g.m_G3.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan,
                     share=len(g) / len(d))
            rows.append(r)
            print(f'| {lv} | {r["n"]} | {r["m_g3"]:+.4f} | {r["h1"]:+.4f} | {r["h2"]:+.4f} | '
                  f'{r["val"]:+.4f} | {r["t_g3"]:+.2f} | {r["mde"]:.3f} | {r["m_x0"]:+.4f} | '
                  f'{100*r["share"]:.1f} % |')
    pd.DataFrame(rows).to_csv(f'{D9}/cells29.csv', index=False)
    print(f'\ncells29.csv written ({len(rows)} levels)', flush=True)


if __name__ == '__main__':
    main()
