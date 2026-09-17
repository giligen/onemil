#!/usr/bin/env python3
"""Stage L step 5 — TEST, read ONCE, only for the cells that passed on both TRAIN and VAL,
plus the full-grid permutation (all 45 run cells, both book families) and the assembled report tables.

The passing set is read off L/cells.csv + L/orb_cells.csv as produced by l3/l4; it is not re-decided here.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcore as C                                                             # noqa: E402
import l3_score as S                                                          # noqa: E402
import l4_orb as O                                                            # noqa: E402

L = C.L


def cell_tids(cell):
    if cell.startswith('STACK'):
        return cell.split(' ', 1)[1].split('+')
    return [cell]


def full_stats(bk, base, p, split):
    wk = C.weeks_of(p, split)
    st = C.stats(bk[bk.split == split], wk, C.months_in(split))
    bs = C.stats(base[base.split == split], wk, C.months_in(split))
    st['imp'] = st['meanR'] - bs['meanR']
    st['base'] = bs['meanR']
    return st


def main():
    cells = pd.read_csv(f'{L}/cells.csv', keep_default_na=False, na_values=[''])
    orb = pd.read_csv(f'{L}/orb_cells.csv', keep_default_na=False, na_values=[''])
    spy = C.spy_minute_ret()
    rows, perm_sets = [], []

    # ---------------- run_book books: every RUN cell gets TRAIN/VAL detail; TEST only for passers
    for bid in S.BOOKS5:
        p = S.load(bid, spy)
        base = C.book(p)
        sub = cells[(cells.book == bid) & (cells.status == 'run')]
        for _, r in sub.iterrows():
            tids = cell_tids(r.cell)
            bk, npop, n_fire, n_bad = S.cell_book(p, tids, bid)
            o = dict(book=bid, cell=r.cell, pop_n=npop, fired=n_fire, bad_fill=n_bad,
                     passed=int(r['pass']) if r['pass'] == r['pass'] else 0)
            for sp in ('TRAIN', 'VAL'):
                st = full_stats(bk, base, p, sp)
                for k in ('n', 'tpw', 'meanR', 't', 'imp', 'base', 'ex5', 'cap3', 'green', 'permo', 'mde'):
                    o[f'{sp}_{k}'] = st[k]
            if o['passed']:
                st = full_stats(bk, base, p, 'TEST')
                for k in ('n', 'tpw', 'meanR', 't', 'imp', 'base', 'ex5', 'cap3', 'green', 'permo'):
                    o[f'TEST_{k}'] = st[k]
            rows.append(o)
            a, f = base[base.split == 'TRAIN'], bk[bk.split == 'TRAIN']
            if len(a) >= 20 and len(f) >= 20:
                perm_sets.append((a.day.values, a.net.values.astype(float),
                                  f.day.values, f.net.values.astype(float)))

    # ---------------- B5: the books are already on disk, one per cell
    def tag_of(cell):
        return cell if not cell.startswith('STACK') else cell
    bb = O.rd(f'{L}/orb/base_book.csv')
    bb['R'] = bb.pnl_pct / bb.range_size_pct
    for _, r in orb.iterrows():
        if r.cell == 'base':
            continue
        b = O.rd(f'{L}/orb/{tag_of(r.cell)}_book.csv')
        b['R'] = b.pnl_pct / b.range_size_pct
        o = dict(book='B5', cell=r.cell, pop_n=len(b), fired=np.nan, bad_fill=np.nan,
                 passed=int(r['pass']))
        for sp in ('TRAIN', 'VAL') + (('TEST',) if r['pass'] else ()):
            x, y = b[C.split_of(b.date) == sp], bb[C.split_of(bb.date) == sp]
            v = np.sort(x.R.values)
            n = len(v)
            se = x.R.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            o[f'{sp}_n'] = n
            o[f'{sp}_meanR'] = float(x.R.mean()) if n else np.nan
            o[f'{sp}_base'] = float(y.R.mean()) if len(y) else np.nan
            o[f'{sp}_imp'] = o[f'{sp}_meanR'] - o[f'{sp}_base']
            o[f'{sp}_t'] = float(x.R.mean() / se) if se and se == se and se > 0 else np.nan
            o[f'{sp}_mde'] = float(2.8 * se) if se == se else np.nan
            o[f'{sp}_ex5'] = float(v[:max(n - max(int(n * 0.05), 1), 1)].mean()) if n else np.nan
            o[f'{sp}_cap3'] = float(np.minimum(v, 3.0).mean()) if n else np.nan
            o[f'{sp}_permo'] = float(x._sized_pnl.sum() / C.months_in(sp))
            o[f'{sp}_usd'] = float(x._sized_pnl.sum())
            o[f'{sp}_tpw'] = np.nan
            o[f'{sp}_green'] = np.nan
        rows.append(o)
        a = bb[C.split_of(bb.date) == 'TRAIN']
        f = b[C.split_of(b.date) == 'TRAIN']
        if len(a) >= 20 and len(f) >= 20:
            perm_sets.append((a.date.values, a.R.values.astype(float),
                              f.date.values, f.R.values.astype(float)))

    d = pd.DataFrame(rows)
    d.to_csv(f'{L}/cells_full.csv', index=False)

    # ---------------- permutation over the WHOLE grid (45 run cells, both book families)
    rng = np.random.default_rng(20260917)
    days = sorted({x for ad, _an, _fd, _fn in perm_sets for x in ad})
    dix = {x: i for i, x in enumerate(days)}
    idx = [(np.array([dix[x] for x in ad]), an, np.array([dix[x] for x in fd]), fn)
           for ad, an, fd, fn in perm_sets]
    obs = max(abs(fn.mean() - an.mean()) for _ai, an, _fi, fn in idx)
    draws = np.empty(500)
    for k in range(500):
        s = rng.choice([-1.0, 1.0], size=len(days))
        draws[k] = max(abs((fn * s[fi]).mean() - (an * s[ai]).mean()) for ai, an, fi, fn in idx)
    txt = (f'grid cells in the null: {len(idx)}\n'
           f'observed max |TRAIN improvement|: {obs:.4f}\n'
           f'null 95th percentile: {np.percentile(draws, 95):.4f}\n'
           f'null mean: {draws.mean():.4f}\n'
           f'p = {(draws >= obs).mean():.3f}\n')
    open(f'{L}/perm_summary.txt', 'w').write(txt)
    pd.DataFrame(dict(draw=draws)).to_csv(f'{L}/perm.csv', index=False)
    C.log(txt)
    C.log(d[['book', 'cell', 'passed', 'TRAIN_imp', 'VAL_imp']
            + [c for c in d.columns if c.startswith('TEST_')]].to_string(index=False))


if __name__ == '__main__':
    main()
