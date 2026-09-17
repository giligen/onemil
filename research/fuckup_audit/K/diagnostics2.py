#!/usr/bin/env python3
"""Stage K diagnostic D5 — where the declared cells lose: the raw family, the slot limit, or the
declared ranking key?

D3 showed K2's WHOLE signal set at TRAIN net +23.8 bps while its declared 10-slot cell is -78.8.
Only two layers sit between them, and both are pre-registered: the slot constraint (10 positions,
one per name, first-come) and the ranking key PREREG declares for each family (K1 gap size, K2
volume ratio, K3 reversal depth, K4 the mean, K5 pullback depth).  For every family at its declared
hold this compares four books on the SAME signals:

    all      — no book at all (every signal taken)
    declared — 10 slots, ranked by the declared strength key
    random   — 10 slots, ranked by a coin flip (isolates the slot constraint alone)
    reverse  — 10 slots, ranked by the NEGATED strength key (the direct test of "is the key
               pointing the wrong way?")

DIAGNOSTIC ONLY.  The gate is decided by the 20 declared cells; nothing here can promote anything.
Every one of these cells is counted in REPORT.md's multiplicity count.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_k as B                                                    # noqa: E402

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
K = f'{ROOT}/research/fuckup_audit/K'
SEED = 91


def stat(t, days, split, col='net'):
    x = t[[B.split_of(days[i]) == split for i in t.sig_day]]
    if len(x) < 30:
        return f'{len(x)} | — | — | — | —'
    return (f'{len(x):,} | {x.gross.mean() * 1e4:+.1f} | {x[col].mean() * 1e4:+.1f} | '
            f'{x.net_auction.mean() * 1e4:+.1f} | {B.tstat(x[col]):+.2f}')


def main():
    f = B.build_features()
    days = f['days']
    n_days = len(days)
    u_prim, _s, _c = B.build_universe(f)
    early = np.array([d in B.EARLY_CLOSES for d in days])
    u_prim = u_prim & (~early[f['day']])
    rng = np.random.default_rng(SEED)

    L = ['# Stage K diagnostic D5 — raw family vs slot limit vs declared ranking key', '',
         'Diagnostic only; cannot promote anything. Counted in the report\'s cell count.', '',
         '| family | book | split | n | gross bps | net bps | net bps (auction) | t |',
         '|---|---|---|---:|---:|---:|---:|---:|']
    rows = []
    for fam in B.FAMILIES:
        hold = B.HOLDS[fam][0]
        cfg = B.FAMILIES[fam]
        m = B.family_mask(fam, f, u_prim)
        t0, _a, _b = B.simulate(fam, f, m, hold, n_days)
        if not len(t0):
            continue
        t0 = t0.reset_index(drop=True)
        key = cfg['strength']
        books = {
            'all (no book)': None,
            'declared key, 10 slots': (key, cfg['asc']),
            'random key, 10 slots': ('_rand', False),
            'reversed key, 10 slots': (key, not cfg['asc']),
        }
        t0['_rand'] = rng.random(len(t0))
        for nm, spec in books.items():
            if spec is None:
                tb = t0
            else:
                k, asc = spec
                tt = t0.copy()
                tt['fam'] = fam
                # run_book sorts on FAMILIES[fam] -- drive it directly instead
                tt = tt.sort_values(['entry_day', k], ascending=[True, asc], kind='mergesort')
                open_pos = []
                booked = np.zeros(len(tt), dtype=bool)
                sy, ed, xd = tt.sym.to_numpy(), tt.entry_day.to_numpy(), tt.exit_day.to_numpy()
                for p, day in enumerate(ed):
                    open_pos = [q for q in open_pos if q[0] >= day]
                    if len(open_pos) >= 10 or any(q[1] == sy[p] for q in open_pos):
                        continue
                    booked[p] = True
                    open_pos.append((xd[p], sy[p]))
                tb = tt[booked]
            for sp in ('TRAIN', 'VAL'):
                L.append(f'| {fam} h{hold} | {nm} | {sp} | {stat(tb, days, sp)} |')
                rows.append(dict(fam=fam, hold=hold, book=nm, split=sp, n=len(tb)))
        B.log(f'{fam} done')
    L.append('')
    L.append('The declared ranking key is the pre-registered tie-break; where "declared" is far '
             'below "random" on BOTH splits, the key itself is adverse — the strongest signal by '
             'that measure is the worst trade, and the book is spending its ten slots on it.')
    with open(f'{K}/diagnostics_d5.md', 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    pd.DataFrame(rows).to_csv(f'{K}/diagnostics_d5.csv', index=False)


if __name__ == '__main__':
    main()
