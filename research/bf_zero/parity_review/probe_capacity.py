#!/usr/bin/env python3
"""Parity review probe 2: rebuild REPORT §8a's capacity table (no script exists for it) from spec_trades.csv:
price >= 20, entry_m <= 840, 12/day, 4 concurrent, gate as a 42% random pass with cost 8% of R when passing."""
import sys, numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/bf_zero'
T = pd.read_csv(f'{D}/spec_trades.csv', usecols=['day', 'symbol', 'entry_m', 'exit_m', 'entry', 'stop', 'r_pct', 'rr', 'why', 'price'], dtype={'symbol': str})
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST')); T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
NW = {s: len(pd.period_range(a, b, freq='W-FRI')) for s, (a, b) in {'TRAIN': ('2025-01-02', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31'), 'TEST': ('2026-06-01', '2026-09-11')}.items()}
all_days = sorted(T.day.unique()); ND = {s: len(set(T[T.split == s].day)) for s in NW}
print('spec signals', len(T), 'days with any signal', len(all_days), ND)


def run_book(F, n_day, n_conc):
    out = []
    for day, g in F.sort_values(['day', 'entry_m']).groupby('day'):
        open_exits = []; taken = 0
        for r in g.itertuples():
            open_exits = [e for e in open_exits if e > r.entry_m]
            if taken >= n_day: continue
            if len(open_exits) >= n_conc: continue
            out.append(r); open_exits.append(r.exit_m); taken += 1
    return pd.DataFrame(out)


def report(bk, title, col='net'):
    print(f'## {title}')
    for s in ('TRAIN', 'VAL', 'TEST'):
        d = bk[bk.split == s]; nw = NW[s]
        if not len(d): print(f'  {s}: none'); continue
        w = d.groupby('wk')[col].sum().reindex(sorted(T[T.split == s].wk.unique())).fillna(0)
        days = d.day.nunique()
        print(f"  {s:5s} n {len(d):5d} {len(d) / nw:4.1f}/wk {len(d) / ND[s]:.2f}/signal-day  mean {d[col].mean():+.3f} WR {(d[col] > 0).mean() * 100:4.1f} | weekly {w.sum() / nw:+.1f} sd {w.std():.1f} green {(w > 0).sum()}/{nw} worst {w.min():+.1f} | days traded {days}/{ND[s]} | exits {d.why.value_counts(normalize=True).round(2).to_dict()}")


P20 = T[(T.price >= 20) & (T.entry_m <= 840)].copy()
print('\nprice>=20 & entry_m<=840 signals', len(P20), P20.split.value_counts().to_dict())
sig_per_day = P20.groupby('day').size().reindex(all_days).fillna(0)
for s in NW:
    dd = [d for d in all_days if (d < '2026-01-01') == (s == 'TRAIN') and ((s != 'VAL') or ('2026-01-01' <= d < '2026-06-01')) and ((s != 'TEST') or d >= '2026-06-01')]
    x = sig_per_day.reindex(dd)
    print(f'  {s}: signals/day mean {x.mean():.1f} median {x.median():.0f} p10/p90 {x.quantile(.1):.0f}/{x.quantile(.9):.0f} share days 0 signals {(x == 0).mean():.2f} share days <4 {(x < 4).mean():.2f}')
P20['net'] = P20.rr
report(run_book(P20, 12, 4), '$20 floor, no gate, raw rr, 12/day 4 conc', 'net')
report(run_book(P20, 12, 4), 'same, RAW', 'rr')
report(run_book(P20, 1000, 1000), '$20 floor, no caps (population)', 'rr')
report(run_book(P20, 12, 1000), '$20, 12/day, NO concurrency cap', 'rr')
report(run_book(P20, 1000, 4), '$20, NO day cap, 4 concurrent', 'rr')
report(run_book(P20, 8, 4), '$20, 8/day, 4 concurrent', 'rr')
# the §8a model: 42% random pass (independent of R); cost 8% of R if passing; the failing ones are not traded
rng = np.random.default_rng(7)
res = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    P = P20.copy(); P['pass'] = rng.random(len(P)) < 0.42
    P = P[P['pass']].copy(); P['net'] = P.rr - 0.08
    bk = run_book(P, 12, 4)
    row = {}
    for s in ('TRAIN', 'VAL', 'TEST'):
        d = bk[bk.split == s]; nw = NW[s]; w = d.groupby('wk').net.sum().reindex(sorted(T[T.split == s].wk.unique())).fillna(0)
        row[s] = (len(d) / nw, d.net.mean(), w.sum() / nw, int((w > 0).sum()), w.min())
    res.append(row)
print('\n## §8a model: 42% random pass, net = rr - 0.08, 12/day 4 conc — 20 seeds (mean [min..max])')
for s in ('TRAIN', 'VAL', 'TEST'):
    a = np.array([r[s] for r in res])
    print(f"  {s}: /wk {a[:, 0].mean():.1f} [{a[:, 0].min():.1f}..{a[:, 0].max():.1f}]  net R/trade {a[:, 1].mean():+.3f} [{a[:, 1].min():+.3f}..{a[:, 1].max():+.3f}]  weekly R {a[:, 2].mean():+.1f} [{a[:, 2].min():+.1f}..{a[:, 2].max():+.1f}]  green {a[:, 3].mean():.1f}/{NW[s]} [{a[:, 3].min():.0f}..{a[:, 3].max():.0f}]  worst {a[:, 4].mean():+.1f} [{a[:, 4].min():+.1f}..{a[:, 4].max():+.1f}]")
print('  published §8a: 24–36/wk, +0.25/+0.28/+0.34, weekly +6.1/+10.1/+10.8, green 40/53 · 20/22 · 14/14, worst −10/−8/+0.4')
# seed 0 detail: is 4-concurrent binding? count signals rejected by each rule in the gated book
rng = np.random.default_rng(0); P = P20.copy(); P = P[rng.random(len(P)) < 0.42].copy()
rej = {'day_cap': 0, 'conc': 0, 'taken': 0}
for day, g in P.sort_values(['day', 'entry_m']).groupby('day'):
    open_exits = []; taken = 0
    for r in g.itertuples():
        open_exits = [e for e in open_exits if e > r.entry_m]
        if taken >= 12: rej['day_cap'] += 1; continue
        if len(open_exits) >= 4: rej['conc'] += 1; continue
        open_exits.append(r.exit_m); taken += 1; rej['taken'] += 1
print('\n  seed-0 gated $20 book rejections:', rej)
# hold time in the $20 book: exit_m - entry_m
h = P20.exit_m - P20.entry_m
print('  hold minutes $20 signals: median', h.median(), 'mean', round(h.mean(), 1), 'p75', h.quantile(.75), 'share eod', round((P20.why == 'eod').mean(), 3))
# weekly R at $100 risk in dollars for the seeds: 1R = $100
print('\n  $ per week at $100 risk = weekly R x 100. TEST weekly-R sd (seed 0):')
P['net'] = P.rr - 0.08; bk = run_book(P, 12, 4); d = bk[bk.split == 'TEST']; w = d.groupby('wk').net.sum()
print('  ', w.round(1).tolist())
