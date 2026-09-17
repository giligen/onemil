#!/usr/bin/env python3
"""Stage H / F6 — the SECONDARY exit variant (2R on a close, stop at level-1%) under the frozen
stack, TRAIN and VAL, both universes.  Reported, never selected on.  TEST is not read."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C, h2_filters as F
from h3_val_test import FROZEN

H, V = C.H, 'stopm1'
by = {c['name']: c for c in F.CUTS}
KEYS = ['cell', 'n', 'tpw', 'meanR', 't', 'WR', 'stopP', 'wkR', 'green', 'worstWk', 'mdd', 'ex5', 'cap3']
HDR = ['cell', 'n', 'tr/wk', 'net R', 't', 'WR%', 'stop%', 'wk R', 'green', 'worst wk', 'MDD', 'ex-top5%', 'cap +3R']


def st(tag, d, wk, sp):
    s, t = C.book_stats(d, sp, wk, V)
    return None if s is None else dict(cell=tag, **s)


def main():
    L = ['# Stage H / F6 — the secondary exit (2R close, stop -1%) under the frozen stack', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Frozen stack: `' + '` + `'.join(FROZEN) + '`.  Reported, never selected on.', '']
    rows = []
    for pop in ('Q', 'P'):
        d, wk = F.prep(pop)
        x = C.scoreable(d, V, floor=True)
        m = pd.Series(True, index=x.index)
        for k in FROZEN:
            m &= by[k]['fn'](x)
        L += [f'## universe {pop}', '', '| split | ' + ' | '.join(HDR) + ' |',
              '|---|' + '|'.join(['---'] * len(HDR)) + '|']
        for sp in ('TRAIN', 'VAL'):
            for tag, xx in (('baseline', x), ('FROZEN stack', x[m]), ('vetoed cohort', x[~m])):
                s = st(tag, xx, wk, sp)
                if s is None:
                    L.append(f'| {sp} | {tag} | (no book) |')
                    continue
                rows.append(dict(pop=pop, split=sp, **s))
                L.append(f'| {sp} | ' + ' | '.join(
                    (f'{s[k]:+.4f}' if k in ('meanR', 'ex5', 'cap3') else str(s[k])) for k in KEYS) + ' |')
        L.append('')
    pd.DataFrame(rows).to_csv(f'{H}/h4_secondary.csv', index=False)
    open(f'{H}/h4_secondary.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
