#!/usr/bin/env python3
"""Stage H / F6 — step 3 continued: the candidate STACKS on TRAIN, before anything is frozen.

Step 2 produced the single-cut table.  This script scores the small number of 3-filter stacks that
step 2's mechanisms allow, on BOTH universes, so the freeze in REPORT.md is made on a table and not
on a hunch.  Every stack scored here is counted in the cell tally.  VAL and TEST are NOT read.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h2b_stacks.py
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C
import h2_filters as F

H = C.H
V, NET = 'hold', 'net_hold'

STACKS = {
    'S1  pdr8 + vwap + col1':    ['pdr>=8', 'vwap>0', 'col<=1'],
    'S2  pdr8 + vwap + body0.5': ['pdr>=8', 'vwap>0', 'body<0.5'],
    'S3  pdr8 + vwap + dopen':   ['pdr>=8', 'vwap>0', 'dopen>=2.2'],
    'S4  pdr8 + vwap + col0.25': ['pdr>=8', 'vwap>0', 'col<=0.25'],
    'S5  pdr8 + vwap + noconf':  ['pdr>=8', 'vwap>0', 'noconfirm'],
    'S6  vwap + dopen':          ['vwap>0', 'dopen>=2.2'],
    'S7  pdr8 + vwap':           ['pdr>=8', 'vwap>0'],
    'S8  pdr8 + dopen':          ['pdr>=8', 'dopen>=2.2'],
}


def tables(T):
    """Render h2b_stacks.md from the scored table (also re-runnable on the CSV alone)."""
    L = ['# Stage H / F6 — step 3 continued: candidate stacks on TRAIN', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Each stack is applied BEFORE `run_book(12,4)`. Both TRAIN halves are shown; a stack is only '
         'eligible for the freeze if it raises the book mean net R AND both TRAIN halves in BOTH '
         'universes.', '']
    for pop in ('Q', 'P'):
        L += [f'## universe {pop}', '', '| ' + ' | '.join(['stack', 'kept'] + F.HDR[1:]) + ' |',
              '|' + '|'.join(['---'] * (len(F.HDR) + 1)) + '|']
        for r in T[T['pop'] == pop].to_dict('records'):
            k = '' if pd.isna(r.get('kept')) else f'{r["kept"]:.0%}'
            L.append(f'| {r["stack"]} | {k} | ' + F.fmt(r, F.KEYS)[2:])
        L.append('')
    L += ['## eligibility', '', '| stack | Q net R | Q H1 | Q H2 | P net R | P H1 | P H2 | eligible |',
          '|---|---:|---:|---:|---:|---:|---:|---|']
    b = {p: T[(T['pop'] == p) & (T['stack'] == 'BASELINE')].iloc[0] for p in ('Q', 'P')}
    for nm in STACKS:
        q = T[(T['pop'] == 'Q') & (T['stack'] == nm)]
        p = T[(T['pop'] == 'P') & (T['stack'] == nm)]
        if not len(q) or not len(p):
            continue
        q, p = q.iloc[0], p.iloc[0]
        ok = (q.H1 > b['Q'].H1 and q.H2 > b['Q'].H2 and p.H1 > b['P'].H1 and p.H2 > b['P'].H2
              and q.meanR > b['Q'].meanR and p.meanR > b['P'].meanR)
        L.append(f'| {nm} | {q.meanR:+.4f} | {q.H1:+.4f} | {q.H2:+.4f} | {p.meanR:+.4f} | '
                 f'{p.H1:+.4f} | {p.H2:+.4f} | {"**YES**" if ok else "no"} |')
    return L + ['']


def main():
    if '--from-csv' in sys.argv:
        L = tables(pd.read_csv(f'{H}/h2b_stacks.csv'))
        open(f'{H}/h2b_stacks.md', 'w').write('\n'.join(L))
        print('\n'.join(L))
        return
    by = {c['name']: c for c in F.CUTS}
    L = ['# Stage H / F6 — step 3 continued: candidate stacks on TRAIN', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         'Each stack is applied BEFORE `run_book(12,4)`. Both TRAIN halves are shown; a stack is only '
         'eligible for the freeze if it raises the book mean net R in BOTH halves of BOTH universes.', '']
    rows = []
    for pop in ('Q', 'P'):
        d, wk = F.prep(pop)
        x = C.scoreable(d, V, floor=True)
        base = F.row('BASELINE', x, wk)[0]
        L += [f'## universe {pop}', '', '| ' + ' | '.join(F.HDR) + ' |',
              '|' + '|'.join(['---'] * len(F.HDR)) + '|', F.fmt(base, F.KEYS)]
        rows.append(dict(pop=pop, stack='BASELINE', **base))
        for nm, names in STACKS.items():
            m = pd.Series(True, index=x.index)
            for k in names:
                m &= by[k]['fn'](x)
            r = F.row(f'{nm}  ({m.mean():.0%})', x[m], wk)
            if r is None:
                continue
            L.append(F.fmt(r[0], F.KEYS))
            rows.append(dict(pop=pop, stack=nm, kept=round(float(m.mean()), 3), **r[0]))
        L.append('')
    T = pd.DataFrame(rows)
    T.to_csv(f'{H}/h2b_stacks.csv', index=False)
    L = tables(T)
    open(f'{H}/h2b_stacks.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
