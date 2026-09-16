#!/usr/bin/env python3
"""Book-rule sensitivity on the F6 eligible population: the claimed cap (4/day) vs the pre-registered
cap (12/day), cost model, and the entry-minute concentration. No bars needed. Read-only."""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_data'
e = pd.read_csv(f'{A}/f6_eligible.csv', dtype={'day': str, 'symbol': str, 'why_e1c': str}, keep_default_na=False, na_values=[''])
e['split'] = np.where(e.day < '2026-01-01', 'TRAIN', np.where(e.day < '2026-06-01', 'VAL', 'TEST'))
e['wk'] = pd.to_datetime(e.day).dt.to_period('W-FRI').astype(str)
print('eligible', len(e), flush=True)


def net(d, entry_bps, exit_bps):
    """cost in R: entry_bps/2 charged always, exit_bps/2 charged on non-target exits."""
    half_in = 0.5 * entry_bps / 100.0 / d.r_pct.clip(lower=0.05)
    half_out = 0.5 * exit_bps / 100.0 / d.r_pct.clip(lower=0.05)
    return d.rr_e1c - half_in - np.where(d.why_e1c == 'target', 0.0, half_out)


def run(d, per_day, conc, tag):
    rows = [(r.day, int(r.entry_m), int(r.exit_m_e1c), r.symbol, r.netR, r.wk, r.split) for r in d.itertuples()]
    t = pd.DataFrame(run_book(rows, per_day, conc), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk', 'split'])
    out = {}
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = t[t.split == sp]
        nw = e[e.split == sp].wk.nunique()
        w = x.groupby('wk').net.sum().reindex(sorted(e[e.split == sp].wk.unique())).fillna(0)
        out[sp] = (len(x), round(x.net.mean(), 3), round(float(w.mean()), 1),
                   round(x.net.mean() / (x.net.std() / np.sqrt(len(x))), 2), round(len(x) / x.day.nunique(), 1))
    print(f'{tag:52s} ' + ' | '.join(f'{sp} n{out[sp][0]:5d} {out[sp][1]:+.3f}R {out[sp][2]:+5.1f}/wk t{out[sp][3]:+5.2f} {out[sp][4]}/d' for sp in ('TRAIN', 'VAL', 'TEST')), flush=True)
    return t


print('\n=== A. book cap sensitivity, score3 cost model (0 in, 40 bps half out on non-target)')
e['netR'] = net(e, 0, 40)
for pd_cap, conc in ((4, 4), (6, 4), (8, 4), (12, 4), (20, 4), (12, 8), (999, 4)):
    run(e, pd_cap, conc, f'cap {pd_cap}/day, {conc} concurrent')

print('\n=== B. cost model sensitivity at the CLAIMED cap (4/day, 4 concurrent)')
for ein, eout, tag in ((0, 40, 'score3: 0 in / 40 bps out (the claim)'),
                       (35, 35, 'pre-registered: half 35 bps in + half out (median measured)'),
                       (40, 40, 'pre-registered at 40 bps'),
                       (57, 57, 'pre-registered at the MEAN measured spread 57 bps'),
                       (0, 80, '0 in / 80 bps out')):
    e['netR'] = net(e, ein, eout)
    run(e, 4, 4, tag)

print('\n=== C. where the edge lives: entry-minute buckets, population (no book)')
e['netR'] = net(e, 0, 40)
b = pd.cut(e.entry_m, [569, 572, 575, 580, 600, 660, 720, 842])
print(e.groupby(b, observed=True).netR.agg(['count', 'mean', 'sum']).round(3).to_string(), flush=True)

print('\n=== D. the claimed book, split by entry minute')
t = run(e, 4, 4, 'as claimed')
t['early'] = t.em <= 572
print(t.groupby(['split', 'early']).net.agg(['count', 'mean', 'sum']).round(3).to_string(), flush=True)
print('\nshare of book P&L from entry_m <= 572:', flush=True)
for sp in ('TRAIN', 'VAL', 'TEST'):
    x = t[t.split == sp]
    print(f'  {sp}: total {x.net.sum():+.1f}R  early {x[x.early].net.sum():+.1f}R ({(x.early).sum()} tr)  rest {x[~x.early].net.sum():+.1f}R ({(~x.early).sum()} tr)', flush=True)

print('\n=== E. drop the first two minutes entirely (entry_m >= 573), same book')
e2 = e[e.entry_m >= 573].copy()
e2['netR'] = net(e2, 0, 40)
rows = [(r.day, int(r.entry_m), int(r.exit_m_e1c), r.symbol, r.netR, r.wk, r.split) for r in e2.itertuples()]
t2 = pd.DataFrame(run_book(rows, 4, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk', 'split'])
for sp in ('TRAIN', 'VAL', 'TEST'):
    x = t2[t2.split == sp]
    print(f'  {sp}: n {len(x)} meanR {x.net.mean():+.3f} t {x.net.mean()/(x.net.std()/np.sqrt(len(x))):+.2f}', flush=True)
print('DONE', flush=True)
