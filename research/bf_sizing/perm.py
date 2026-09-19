#!/usr/bin/env python3
"""POST-HOC diagnostic (declared as post-hoc in REPORT §4c; counted in the cell
budget): is a sizing cell's advantage the CONVICTION ORDERING, or merely the
change in size dispersion?

Null: the (conviction, macd_zone) pair carries no information about R. Realised
by SHUFFLING the multiplier pair across picks within a split, 2,000 times, and
re-scoring the cell. If the real cell sits inside the null band, its advantage
is dispersion, not signal.

TEST is never touched. Read-only. Writes only into research/bf_sizing/.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/bf_sizing')
import part2 as P                                    # reuses the exact cells

RNG = np.random.default_rng(20260919)
NPERM = 2000
OUT = '/home/ec2-user/onemil/research/bf_sizing'

rows = []
for ps in ('P1', 'F7'):
    m = P.build(ps)
    tr = m['split'] == 'TRAIN'
    s0 = m.copy()
    s0['shares_c'] = s0['shares'].astype(float)
    s0['risk_c'] = s0['shares_c'] * s0['rps']
    BASE = float(s0.loc[tr, 'risk_c'].mean())

    def green(mult, split, frame=None):
        f = P.apply_cell(frame if frame is not None else m, mult, 'x', BASE)
        s = P.score(f, split)
        return (float('nan'), float('nan')) if s is None else (s['green_wk'], s['pnl'])

    thr = float(np.median(m.loc[tr, 'conv']))
    C = float(np.median(m.loc[tr, 'conv'])) ** 2
    real = {
        'S0': (m['conv'] * m['macd']).values,
        'S1': np.ones(len(m)),
        'S4': np.where(m['conv'] >= thr, 1.4, 0.7),
        'S2': np.clip(C / m['conv'], 0.25, 3.0).values,
    }
    for split in ('TRAIN', 'VAL'):
        sel = (m['split'] == split).values
        for cell in ('S0', 'S4', 'S2'):
            g_real, p_real = green(real[cell], split)
            g1, p1 = green(real['S1'], split)
            gs, ps_ = np.empty(NPERM), np.empty(NPERM)
            for i in range(NPERM):
                mu = real[cell].copy()
                idx = np.where(sel)[0]
                perm = RNG.permutation(idx)
                mu[idx] = mu[perm]          # shuffle the multiplier WITHIN split
                gs[i], ps_[i] = green(mu, split)
            pg = float((gs >= g_real).mean())
            pp = float((ps_ >= p_real).mean())
            print(f'{ps:3s} {split:5s} {cell}: green {g_real:5.1f}% '
                  f'(S1 {g1:5.1f}%, null mean {np.nanmean(gs):5.1f}% '
                  f'sd {np.nanstd(gs):4.1f}, perm p {pg:.3f})   '
                  f'$ {p_real:9,.0f} (S1 {p1:9,.0f}, null mean '
                  f'{np.nanmean(ps_):9,.0f} sd {np.nanstd(ps_):8,.0f}, '
                  f'perm p {pp:.3f})', flush=True)
            rows.append(dict(pickset=ps, split=split, cell=cell,
                             green=g_real, green_S1=g1,
                             null_green_mean=float(np.nanmean(gs)),
                             null_green_sd=float(np.nanstd(gs)), p_green=pg,
                             pnl=p_real, pnl_S1=p1,
                             null_pnl_mean=float(np.nanmean(ps_)),
                             null_pnl_sd=float(np.nanstd(ps_)), p_pnl=pp))

pd.DataFrame(rows).to_csv(f'{OUT}/perm_null.csv', index=False)
print('\nwrote research/bf_sizing/perm_null.csv', flush=True)
