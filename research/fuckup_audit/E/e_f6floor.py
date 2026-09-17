#!/usr/bin/env python3
"""Stage E — the one cell that cleared the gate arithmetic, characterised BEFORE TEST is opened.

The cell is POST-HOC: it came out of follow-up A (the H6 diagnostic that splits the causal
population by whether a row would have passed the old `range_so_far_pct >= 5` floor), not out of
the 60 pre-registered cells.  It is characterised here, frozen in writing in E/REPORT.md, and only
then read on TEST (E_READ_TEST=1).

Cell: family F6 {}, fill `next`, exit hold-to-close, population `all` RESTRICTED to
range_so_far_pct >= 5, contract (c), run_book(12, 4), causal universe U1 u U2, all-day.
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/E')
import e_score as S

E = f'{ROOT}/research/fuckup_audit/E'
READ_TEST = os.environ.get('E_READ_TEST') == '1'
pd.set_option('display.width', 250)


def main():
    _, data = S.load()
    allr = pd.concat([data['next'], data['rest']], ignore_index=True)
    weeks = {s: sorted(allr[allr.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    d = data['next']
    x = d[(d.key == 'F6 {}') & (d.range_so_far_pct >= 5)]

    L = ['# Stage E — the post-hoc `F6 floor-passing` cell', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Cell: `F6 {}` x fill `next` x exit hold-to-close x population `all` AND '
         '`range_so_far_pct >= 5`, contract (c), run_book(12,4), causal universe U1 u U2, all-day. '
         'NOT one of the 60 pre-registered cells — it is the floor-passing half of follow-up A.', '']
    splits = ['TRAIN', 'VAL'] + (['TEST'] if READ_TEST else [])
    for sp in splits:
        st, tr = S.book_stats(x, sp, weeks, 'hold')
        if st is None:
            L += [f'## {sp}: no book', '']
            continue
        L += [f'## {sp}', '', str(st), '']
        n, v = len(tr), tr.net.values
        s = np.sort(v)
        L += [f'tail: top-1% removed **{s[:n-max(int(n*0.01),1)].mean():+.4f}**, '
              f'top-5% removed **{s[:n-max(int(n*0.05),1)].mean():+.4f}**, '
              f'winners capped at +3R **{np.minimum(v,3.0).mean():+.4f}**', '',
              '**per month**', '',
              tr.assign(mon=tr.day.str[:7]).groupby('mon').net.agg(['size', 'sum', 'mean']).round(3).to_string(), '',
              '**per week**', '']
        w = tr.groupby('wk').net.sum().reindex(weeks[sp]).fillna(0.0)
        L += [f'weeks {len(w)}, green {float((w>0).mean()):.2f}, mean {w.mean():+.2f}R, '
              f'min {w.min():+.2f}R, max {w.max():+.2f}R', '']
        if sp == 'TRAIN':
            # search-adjusted permutation over the 60 primary cells PLUS this one
            T, cells, _, _ = S.score_grid(data, weeks, 'U1uU2')
            g = tr.groupby('day').net
            cells[('posthoc', 'F6 floor', 'next', 'hold', 'all')] = dict(
                t=st['t'], trades=tr.net.values, bydayN=g.size().values)
            p, obs, q95 = S.perm_pvalue(cells, 500)
            L += [f'**permutation, search-adjusted over the {len(cells)} cells of this stage '
                  f'(500 day-label sign-flip draws): observed max TRAIN t {obs:.2f}, '
                  f'null 95th pct {q95:.2f}, p = {p:.3f}**', '']
    open(f'{E}/score_e_f6floor{"_TEST" if READ_TEST else ""}.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
