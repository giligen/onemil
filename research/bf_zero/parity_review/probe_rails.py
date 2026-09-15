#!/usr/bin/env python3
"""Parity review probe 4: kill rails (-6R day / -15R week at $100 risk) on the reconstructed $20 gated book; days hitting the rail and the R truncated."""
import numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/bf_zero'
T = pd.read_csv(f'{D}/spec_trades.csv', usecols=['day', 'symbol', 'entry_m', 'exit_m', 'rr', 'why', 'price'], dtype={'symbol': str})
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST')); T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
P20 = T[(T.price >= 20) & (T.entry_m <= 840)].copy()
res = []
for seed in range(10):
    rng = np.random.default_rng(seed); P = P20[rng.random(len(P20)) < 0.42].copy(); P['net'] = P.rr - 0.08
    out = []
    for day, g in P.sort_values(['day', 'entry_m']).groupby('day'):
        open_exits = []; taken = 0
        for r in g.itertuples():
            open_exits = [e for e in open_exits if e > r.entry_m]
            if taken >= 12 or len(open_exits) >= 4: continue
            out.append(r); open_exits.append(r.exit_m); taken += 1
    bk = pd.DataFrame(out)
    # realized-P&L rail: a trade counts as realized at its exit minute; new entries blocked once realized day P&L <= -6R
    hit_days = 0; trunc = 0.0; n_dropped = 0
    for day, g in bk.groupby('day'):
        g = g.sort_values('entry_m'); realized = 0.0; blocked = False
        for r in g.itertuples():
            done = g[(g.exit_m <= r.entry_m)]
            realized = done.net.sum()
            if realized <= -6.0 and not blocked: blocked = True; hit_days += 1
            if blocked: trunc += r.net; n_dropped += 1
    wk = bk.groupby('wk').net.sum(); res.append((hit_days, bk.day.nunique(), n_dropped, trunc, int((wk <= -15).sum()), len(wk), bk.groupby('day').net.sum().min(), (bk.groupby('day').net.sum() <= -6).mean()))
a = np.array(res)
print(f'day rail -6R hit on {a[:,0].mean():.1f} of {a[0,1]:.0f} days (mean over seeds); trades dropped after the rail {a[:,2].mean():.1f}, their R (would-have-been) {a[:,3].mean():+.1f}; weeks <= -15R (no rail) {a[:,4].mean():.1f}/{a[0,5]:.0f}; worst day {a[:,6].mean():+.1f}; share days closing <= -6R {a[:,7].mean():.3f}')
