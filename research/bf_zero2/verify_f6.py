#!/usr/bin/env python3
"""Focused verification of the one configuration that is positive on all three splits with t > 2.5 everywhere:
F6 'red to green', hold to close with a -1R stop. Rule (all causal): the stock OPENS BELOW the prior close; entry on the
first 1-min bar whose high reaches prev_close x 1.003 (the 30 bps slip is the fill); stop = the lowest low before entry;
exit = the stop or the 15:55 close. Universe membership guaranteed causally by range-so-far >= 5% at the signal bar.
Checks: cost sensitivity (40/60/80/120 bps spread), book size, trades per day, R distribution, weekly path, drawdown,
per-month table, and the sub-samples (price band, time of day, gap size) so the result is not one bucket."""
import os, sys, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
D = 'research/bf_zero2'
c = pd.read_csv(f'{D}/candidates_full.csv', usecols=['day', 'symbol', 'fam', 'cfg', 'entry_m', 'price', 'r_pct', 'rr_e4', 'range_so_far_pct',
                                                     'dist_open_pct', 'gap_pct', 'rv_profile', 'adv20', 'is_wrapper'],
                dtype={'symbol': str, 'day': str, 'fam': 'category'}, keep_default_na=False, na_values=[''], low_memory=True)
c = c[(c.fam == 'F6') & (c.price >= 5) & (c.entry_m <= 841) & (c.r_pct >= 1) & (c.range_so_far_pct >= 5)].copy()
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str); c['mo'] = c.day.str[:7]
WEEKS = {s: c[c.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}
L = [f'# F6 red-to-green, hold to close with a -1R stop — verification | rows {len(c):,} | weeks {WEEKS}', '']

def run(spread_bps, N, extra=None):
    d = c if extra is None else c[extra]
    d = d.assign(net=d.rr_e4 - 0.5 * spread_bps / 100.0 / d.r_pct.clip(lower=0.05))
    out = {}
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = d[d.split == sp]
        if len(x) < 40: out[sp] = None; continue
        rows = [(r.day, int(r.entry_m), 955, r.symbol, r.net, r.wk) for r in x.itertuples()]
        t = pd.DataFrame(run_book(rows, N, N), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
        w = t.groupby('wk').net.sum().reindex(sorted(c[c.split == sp].wk.unique())).fillna(0)
        cum = w.cumsum()
        out[sp] = dict(n=len(t), tpd=round(len(t) / t.day.nunique(), 1), meanR=round(t.net.mean(), 3),
                       t=round(t.net.mean() / (t.net.std() / np.sqrt(len(t))), 2), WR=round((t.net > 0).mean() * 100, 1),
                       wkR=round(float(w.mean()), 1), green=f'{int((w>0).sum())}/{len(w)}', worst=round(float(w.min()), 1),
                       mddR=round(float((cum - cum.cummax()).min()), 1))
    return out

L.append('## cost sensitivity and book size (net R per trade / R per week)')
for N in (4, 10, 20):
    for sb in (40, 60, 80, 120):
        o = run(sb, N)
        L.append(f'  slots {N:2d} spread {sb:3d} bps | ' + ' | '.join(
            f"{sp} {o[sp]['meanR']:+.3f}R {o[sp]['wkR']:+5.1f}/wk t {o[sp]['t']:+4.1f} green {o[sp]['green']} mdd {o[sp]['mddR']:+.0f}" if o[sp] else f'{sp} n/a' for sp in ('TRAIN', 'VAL', 'TEST')))
o = run(40, 4)
L += ['', '## the 4-slot book at 40 bps, in detail']
for sp in ('TRAIN', 'VAL', 'TEST'):
    if o[sp]: L.append(f'  {sp}: n {o[sp]["n"]} ({o[sp]["tpd"]}/day) meanR {o[sp]["meanR"]:+.3f} t {o[sp]["t"]} WR {o[sp]["WR"]}% weekly {o[sp]["wkR"]:+.1f}R green {o[sp]["green"]} worst {o[sp]["worst"]}R maxDD {o[sp]["mddR"]}R')
d = c.assign(net=c.rr_e4 - 0.5 * 0.40 / c.r_pct.clip(lower=0.05))
rows = [(r.day, int(r.entry_m), 955, r.symbol, r.net, r.wk) for r in d.itertuples()]
bk = pd.DataFrame(run_book(rows, 4, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
bk = bk.merge(d[['day', 'symbol', 'price', 'r_pct', 'gap_pct', 'entry_m', 'rv_profile', 'split', 'mo', 'adv20', 'is_wrapper']], on=['day', 'symbol'], how='left').drop_duplicates(['day', 'symbol'])
bk.to_csv(f'{D}/f6_book_trades.csv', index=False)
L += ['', '## monthly R (4 slots, 40 bps)', bk.groupby('mo').net.agg(['sum', 'count']).round(1).to_string(),
      '', '## R distribution', bk.net.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_string(),
      '', '## sub-samples (mean net R, n) — the result must not be one bucket']
for col, cuts in (('price', [5, 10, 20, 50, 1e9]), ('entry_m', [570, 600, 660, 720, 841]), ('gap_pct', [-1e9, -10, -5, -2, 0]),
                  ('r_pct', [1, 2, 4, 8, 1e9]), ('rv_profile', [0, 1, 2, 5, 1e9]), ('adv20', [0, 5e5, 2e6, 1e13])):
    b = pd.cut(bk[col], cuts)
    L.append(f'  {col}: ' + ' | '.join(f'{str(i)} {g.net.mean():+.2f} (n {len(g)})' for i, g in bk.groupby(b, observed=True)))
L.append(f"  wrapper: " + ' | '.join(f'{k} {g.net.mean():+.2f} (n {len(g)})' for k, g in bk.groupby('is_wrapper')))
open(f'{D}/f6_verify.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
