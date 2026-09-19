#!/usr/bin/env python3
"""hod_filter_stack — supplementary: MDE, ex-tail, the weekly dollar series, and the
TRAIN-vs-VAL anatomy of the two features that looked strongest in-sample."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S       # noqa: E402
import score2 as S2     # noqa: E402

D = f'{ROOT}/research/mature_method/hod_filter_stack'


def main():
    p = S2.load_pop(); S.build_impute(p)
    pre = {nm: S2.sig_set(p, **kw) for nm, kw in S2.BASES.items()}
    print('== MDE (80% power, two-sided 5%) ==')
    print('| population | split | n | per-trade MDE80 (R) | green-week SE | green-week MDE80 (pp) |')
    for nm, s in pre.items():
        for sp in S.SPLITS:
            d = s[s.split == sp]
            b = S.apply_book(d, 12, 4)
            db = b[b.split == sp]
            mde = 2.8 * d.rr.std(ddof=1) / np.sqrt(len(d))
            w = db.groupby('wk').pnl.sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
            q = float((w > 0).mean()); se = np.sqrt(q * (1 - q) / S.NW[sp]) * 100
            print(f'| {nm} pre-book | {sp} | {len(d)} | {mde:.3f} | {se:.1f} | {2.8*se:.1f} |')

    B2 = pre['B2']
    print('\n== ex-tail diagnostics (never a rejection reason) ==')
    for nm in ('B0', 'B2'):
        b = S.apply_book(pre[nm], 12, 4)
        for sp in S.SPLITS:
            d = b[b.split == sp]
            q99, q95 = d.net.quantile(0.99), d.net.quantile(0.95)
            print(f'  {nm} {sp}: net {d.net.mean():+.3f} | ex-top-1% {d.net[d.net<=q99].mean():+.3f} '
                  f'| ex-top-5% {d.net[d.net<=q95].mean():+.3f}')

    print('\n== entry_m deciles on B2, TRAIN edges, read in BOTH splits (gross R) ==')
    tr = B2[B2.split == 'TRAIN']
    ed = np.unique(tr.entry_m.quantile(np.arange(0, 1.001, 0.1)).values)
    lab = [f'[{int(ed[i])},{int(ed[i+1])})' for i in range(len(ed) - 1)]
    for sp in S.SPLITS:
        d = B2[B2.split == sp]
        dec = pd.cut(d.entry_m, ed, labels=False, include_lowest=True)
        mu = d.rr.groupby(dec).mean(); n = d.rr.groupby(dec).size()
        print(f'  {sp:5s} ' + ' '.join(f'{lab[int(k)]}:{v:+.2f}(n{n[k]})' for k, v in mu.items()))

    print('\n== the day filter that looked best on VAL (D-c: SPY 09:30->10:00 up) ==')
    dctx = B2.drop_duplicates('day')[['day', 'split', 'spy_ret_0930_1000']].set_index('day')
    up = B2.day.map(dctx.spy_ret_0930_1000) > 0
    for sp in S.SPLITS:
        for tag, mk in (('SPY-up days', up), ('SPY-down days', ~up)):
            d = B2[(B2.split == sp) & mk.fillna(False)]
            b = S.apply_book(d, 12, 4)
            bb = b[b.split == sp]
            nd = d.day.nunique()
            print(f'  {sp:5s} {tag:14s} days {nd:3d}  pre-book n {len(d):5d} gross {d.rr.mean():+.3f} '
                  f'| booked n {len(bb):4d} gross {bb.rr.mean():+.3f} net {bb.net.mean():+.3f} '
                  f'$ {bb.pnl.sum():+,.0f}')

    print('\n== weekly dollars at the live $100 risk ==')
    for nm, mk in (('B2', pd.Series(True, index=B2.index)), ('D-g SPY-up & breadth>=mid', None)):
        if mk is None:
            bq = B2[B2.split == 'TRAIN'].drop_duplicates('day').breadth_by_1000.quantile(1 / 3)
            bcol = B2.day.map(B2.drop_duplicates('day').set_index('day').breadth_by_1000)
            mk = (up & (bcol > bq)).fillna(False)
        b = S.apply_book(B2[mk], 12, 4)
        for sp in S.SPLITS:
            d = b[b.split == sp]
            w = d.groupby('wk').pnl.sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
            t = d.groupby('wk').size().reindex(S.ALL_WEEKS[sp]).fillna(0).astype(int)
            print(f'  {nm} {sp}: ' + ' '.join(f'{int(x):+d}({int(c)})' for x, c in zip(w.values, t.values)))
            print(f'    green {int((w>0).sum())}/{S.NW[sp]}  total {w.sum():+,.0f}  worst {w.min():+,.0f}')


if __name__ == '__main__':
    main()
