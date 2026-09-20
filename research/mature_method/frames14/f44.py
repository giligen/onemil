#!/usr/bin/env python3
"""F44 — IS THE OVERNIGHT TAIL SELECTABLE AT 15:55?  The eight declared cells.

  python3 f44.py

Reproduction gate: frames13's A13 cell (n = 3,310,743, gross +0.045 %, net +0.0306 %,
ex-top-5 % -0.1385 %) must reproduce before a conditioner is read.  Stores READ-ONLY.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames13')
import f40                                                       # noqa: E402

D14 = f'{ROOT}/research/mature_method/frames14'
APR = 7.0
RT_EXEC = 0.368          # frames13 §1.4: the MEASURED 15:55 -> 09:31 marketable round trip, % of px
DRAWS = 2000
SEED = 20260920
FIELDS = ['V1_day_ret', 'V2_day_range', 'V3_close_pos', 'V4_rv', 'V5_dollar_frac',
          'V6_spy_day', 'V7_gap', 'V8_hi20_prox']
SPLITS = ('H1-25', 'H2-25', 'VAL')


def load():
    X = f40.load()
    kinds = pd.Index(['common', 'fund', 'wrapper'])
    m = ((X['kind_i'] == int(kinds.get_loc('common'))) & (X['close'] >= 20.0) &
         (X['adv20'] >= 1e7))
    A = f40.sub(X, m)
    del X
    key = A['sid'].astype(np.int64) * 100000 + A['d'].astype(np.int64)
    net = A['on_pct'] - f40.margin_pct(A['nights'], APR)
    d = A['d']
    # splits: PRE = 2016-2024, then the programme's own halves
    def _dn(s):
        return int(np.datetime64(s).astype('datetime64[D]').astype(np.int32))
    sp = np.full(len(d), 'PRE  ', dtype='<U5')
    sp[d >= _dn('2025-01-01')] = 'H1-25'
    sp[d >= _dn('2025-07-01')] = 'H2-25'
    sp[d >= _dn('2026-01-01')] = 'VAL  '
    sp = np.char.strip(sp)
    F = pd.concat([pd.read_parquet(f'{D14}/feat_{y}.parquet') for y in range(2016, 2027)],
                  ignore_index=True)
    F = F.drop_duplicates('key').set_index('key')
    idx = F.index.get_indexer(key)
    out = dict(key=key, d=d, gross=A['on_pct'], net=net, split=sp)
    for f in FIELDS:
        v = np.full(len(key), np.nan)
        ok = idx >= 0
        v[ok] = F[f].values[idx[ok]]
        out[f] = v
    print(f'  panel {len(key):,} A13 name-nights | feature join {float((idx>=0).mean()):.1%}',
          flush=True)
    return out


def gate(P):
    n = len(P['net'])
    print(f'\nGATE A13: n={n:,} (frames13: 3,310,743) | gross {P["gross"].mean():+.4f} % '
          f'(frames13 +0.045) | net {P["net"].mean():+.4f} % (frames13 +0.0306)', flush=True)
    assert n == 3310743, 'A13 n does not reproduce'
    assert abs(P['net'].mean() - 0.0306) < 0.002, 'A13 net does not reproduce'
    cut = np.quantile(P['net'], 0.95)
    e5 = P['net'][P['net'] <= cut].mean()
    print(f'  ex-top-5 % {e5:+.4f} % (frames13 -0.1385)', flush=True)
    assert abs(e5 - (-0.1385)) < 0.01, 'A13 ex-top-5 % does not reproduce'
    print('  GATE PASSES', flush=True)


def within_day_quintile(v, d, top=True):
    """Boolean: is the row in the TOP (or BOTTOM) quintile of `v` on its own day?

    A within-day cut is causal at 15:55 (the cross-section is known) and cannot drift with the
    market, which a global threshold would.
    """
    ok = np.isfinite(v)
    sel = np.zeros(len(v), bool)
    order = np.lexsort((v, d))
    dd = d[order]
    oo = ok[order]
    start = np.concatenate([[0], np.flatnonzero(dd[1:] != dd[:-1]) + 1])
    ends = np.concatenate([start[1:], [len(dd)]])
    take = np.zeros(len(dd), bool)
    for s, e in zip(start, ends):
        seg = oo[s:e]
        n = int(seg.sum())
        if n < 10:
            continue
        # the finite rows are the LAST n of the segment only if NaNs sort first; np.lexsort puts
        # NaN last, so take from the finite block explicitly
        fin = np.flatnonzero(seg)
        k = max(1, n // 5)
        take[s + (fin[-k:] if top else fin[:k])] = True
    sel[order] = take
    return sel


def tail_label(net, d, split):
    """Top 5 % of the night distribution INSIDE each split."""
    lab = np.zeros(len(net), bool)
    for sp in np.unique(split):
        m = split == sp
        lab[m] = net[m] >= np.quantile(net[m], 0.95)
    return lab


def hyper_null(d, sel, lab, rng, draws=DRAWS):
    """The count-matched permutation null on the tail LABEL, drawn exactly.

    Permuting the labels within a day and counting the selected tails is, per day, a draw from
    Hypergeometric(N_d, n_tail_d, n_sel_d).  Sampling that directly IS the permutation and costs
    O(days x draws) instead of O(rows x draws).
    """
    df = pd.DataFrame(dict(d=d, s=sel.astype(int), t=lab.astype(int)))
    g = df.groupby('d').agg(N=('s', 'size'), ns=('s', 'sum'), nt=('t', 'sum'))
    g = g[(g.ns > 0) & (g.nt > 0) & (g.nt < g.N)]
    if not len(g):
        return np.nan, np.nan, np.nan
    N, ns, nt = g.N.values, g.ns.values, g.nt.values
    tot_sel = int(ns.sum())
    draws_ = np.empty(draws)
    for i in range(draws):
        draws_[i] = rng.hypergeometric(nt, N - nt, ns).sum() / tot_sel
    return float(np.mean(draws_)), float(np.quantile(draws_, 0.05)), float(np.quantile(draws_, 0.95))


def ex5(v):
    if len(v) < 40:
        return np.nan
    return float(v[v <= np.quantile(v, 0.95)].mean())


def main():
    P = load()
    gate(P)
    lab = tail_label(P['net'], P['d'], P['split'])
    rng = np.random.default_rng(SEED)
    rows = []
    print(f'\n### THE 8 SCORED CELLS — top within-day quintile of each field, A13, % of price\n',
          flush=True)
    print(f'{"cell":>16} {"side":>6} {"n":>9} {"tail rate":>10} {"null p50":>9} {"null p95":>9} '
          f'{"mean":>8} {"ex5 H1":>8} {"ex5 H2":>8} {"ex5 VAL":>9} {"ex5+exec VAL":>13}',
          flush=True)
    for f in FIELDS:
        for side, top in (('top', True), ('bot', False)):
            sel = within_day_quintile(P[f], P['d'], top=top)
            if sel.sum() < 1000:
                continue
            tr = float(lab[sel].mean())
            mu, p5, p95 = hyper_null(P['d'], sel, lab, rng)
            e = {}
            for sp in SPLITS:
                m = sel & (P['split'] == sp)
                e[sp] = ex5(P['net'][m])
            e_exec = {sp: (e[sp] - RT_EXEC if e[sp] == e[sp] else np.nan) for sp in SPLITS}
            allpos = all((e[sp] - RT_EXEC) > 0 for sp in SPLITS if e[sp] == e[sp])
            rows.append(dict(field=f, side=side, n=int(sel.sum()), tail_rate=tr, null_p50=mu,
                             null_p95=p95, mean=float(P['net'][sel].mean()),
                             **{f'ex5_{k}': v for k, v in e.items()},
                             **{f'ex5exec_{k}': v for k, v in e_exec.items()},
                             above_null=bool(tr > p95), passes=bool(allpos)))
            print(f'{f:>16} {side:>6} {int(sel.sum()):9,d} {tr:10.4f} {mu:9.4f} {p95:9.4f} '
                  f'{P["net"][sel].mean():+8.3f} {e["H1-25"]:+8.3f} {e["H2-25"]:+8.3f} '
                  f'{e["VAL"]:+9.3f} {e_exec["VAL"]:+13.3f}', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{D14}/cells44.csv', index=False)
    npass = int(R.passes.sum())
    nabove = int(R.above_null.sum())
    print(f'\n  cells whose TAIL RATE clears its count-matched null p95: {nabove} of {len(R)}',
          flush=True)
    print(f'  cells whose EX-TOP-5 % is positive on both TRAIN halves AND VAL net of the measured '
          f'execution ({RT_EXEC} % of price): {npass} of {len(R)}', flush=True)
    # the era check, printed for every cell regardless (it costs nothing and is the honest read)
    print('\n  DIAGNOSTIC — PRE (2016-2024) ex-top-5 %, margin only, by cell:', flush=True)
    for f in FIELDS:
        for side, top in (('top', True), ('bot', False)):
            sel = within_day_quintile(P[f], P['d'], top=top) & (P['split'] == 'PRE')
            if sel.sum() < 1000:
                continue
            print(f'    {f:>16} {side:>4} n={int(sel.sum()):8,d}  ex5 {ex5(P["net"][sel]):+.4f} %  '
                  f'mean {P["net"][sel].mean():+.4f} %', flush=True)
    # the wrapper/common diagnostic the PREREG promised (degenerate inside A13)
    print('\n  DIAGNOSTIC — wrapper vs common on the LIQUID panel (A13 with the common filter '
          'removed): see frames13 A11/A12 — wrapper +0.0515 %, common +0.0749 %, both negative '
          'ex-top-5 % (d4 -0.1863 %).', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
