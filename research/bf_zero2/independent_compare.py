#!/usr/bin/env python3
"""Diff my independent red-to-green book (variant A) against research/bf_zero2/f6_2r_book.csv.

Reference columns: day, em, xm, symbol, net, wk, split, mo, why_e1c, price, r_pct
  price  = entry price        -> compare to my `entry`
  r_pct  = R/entry*100        -> stop = price*(1 - r_pct/100), compare to my `stop`
  net    = net R per trade    -> compare to my `rr`
"""
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
MINE = f'{ROOT}/research/bf_zero2/independent_trades_A_recon.csv'
REF = f'{ROOT}/research/bf_zero2/f6_2r_book.csv'
TOL = 1e-6

pd.set_option('display.width', 200)


def main():
    ref = pd.read_csv(REF, keep_default_na=False, na_values=[''])
    ref = ref.rename(columns={'day': 'bar_date'})
    ref['bar_date'] = ref.bar_date.astype(str).str[:10]
    ref['ref_stop'] = ref.price * (1 - ref.r_pct / 100.0)

    mine = pd.read_csv(MINE, keep_default_na=False, na_values=[''])
    mine = mine[mine.in_book == 1].copy()

    mk = set(zip(mine.bar_date, mine.symbol))
    rk = set(zip(ref.bar_date, ref.symbol))
    both, mo, ro = mk & rk, mk - rk, rk - mk
    print(f'mine={len(mine)} rows / {len(mk)} keys   ref={len(ref)} rows / {len(rk)} keys')
    print(f'matched={len(both)}  mine_only={len(mo)}  ref_only={len(ro)}')

    m = mine.set_index(['bar_date', 'symbol']).sort_index()
    r = ref.set_index(['bar_date', 'symbol']).sort_index()
    print(f'dup keys: mine={m.index.duplicated().sum()} ref={r.index.duplicated().sum()}')
    idx = pd.MultiIndex.from_tuples(sorted(both), names=['bar_date', 'symbol'])
    j = m.loc[idx].join(r.loc[idx], rsuffix='_r')

    classes = []
    for name, a, b in (('entry', j.entry, j.price),
                       ('stop', j.stop, j.ref_stop),
                       ('rr(net)', j.rr, j.net),
                       ('entry_min', j.entry_m.astype(float), j.em.astype(float)),
                       ('exit_min', j.exit_m.astype(float), j.xm.astype(float))):
        d = (a - b).abs()
        bad = j[d > TOL]
        print(f'\n[{name}] agree={int((d <= TOL).sum())}/{len(j)}  disagree={len(bad)}  '
              f'max|diff|={d.max():.3g}  mean|diff|={d.mean():.3g}')
        if len(bad):
            ex = bad.assign(mine=a[d > TOL], ref=b[d > TOL],
                            diff=(a - b)[d > TOL])
            print(ex[['mine', 'ref', 'diff', 'reason', 'why_e1c', 'entry_m', 'em',
                      'exit_m', 'xm', 'r_pct', 'r_pct_r' if 'r_pct_r' in ex else 'r_pct']]
                  .head(3).to_string())
            classes.append((name, len(bad)))

    # exit-reason agreement
    xt = pd.crosstab(j.reason, j.why_e1c)
    print('\nexit reason crosstab (rows=mine, cols=ref):')
    print(xt.to_string())

    # aggregate numbers on the matched set only
    print('\nsplit totals, matched set:')
    g = j.groupby(j.why_e1c.notna())  # dummy
    for sp in ('TRAIN', 'VAL', 'TEST'):
        s = j[j.split == sp]
        print(f'  {sp:6s} n={len(s):4d}  mineR={s.rr.mean():+.4f}  refR={s.net.mean():+.4f}  '
              f'sum mine={s.rr.sum():+.2f} ref={s.net.sum():+.2f}')

    if mo:
        mine[[t in mo for t in zip(mine.bar_date, mine.symbol)]].to_csv(
            f'{ROOT}/research/bf_zero2/independent_mine_only.csv', index=False)
        print('\nMINE ONLY (3):')
        print(mine[[t in mo for t in zip(mine.bar_date, mine.symbol)]]
              [['bar_date', 'symbol', 'entry_m', 'entry', 'stop', 'r_pct', 'exit_m',
                'reason', 'rr']].head(3).to_string(index=False))
    if ro:
        ref[[t in ro for t in zip(ref.bar_date, ref.symbol)]].to_csv(
            f'{ROOT}/research/bf_zero2/independent_ref_only.csv', index=False)
        print('\nREF ONLY (3):')
        print(ref[[t in ro for t in zip(ref.bar_date, ref.symbol)]].head(3).to_string(index=False))


if __name__ == '__main__':
    main()
