#!/usr/bin/env python3
"""Parity review probe 1: reproduce REPORT §8 net-of-cost table from spread_study_clean.csv under several
cost-model variants; report which one (if any) lands within ±0.02R of the published numbers."""
import numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/bf_zero'
cols = ['day', 'symbol', 'split', 'n_quotes', 'spread_last', 'spread_med', 'ask_last', 'bid_last', 'entry', 'stop', 'r_pct', 'rr', 'why', 'price']
d = pd.read_csv(f'{D}/spread_study_clean.csv', usecols=cols, dtype={'symbol': str})
print('rows', len(d), 'n_quotes>0', (d.n_quotes > 0).sum(), 'dup keys', d.duplicated(['day', 'symbol']).sum())
d = d[d.n_quotes > 0].copy()
d['R'] = d.entry - d.stop; d['sfr'] = d.spread_last / d.R; d['sbps'] = d.spread_last / d.entry * 1e4
nt = (d.why != 'target').astype(float)
# exit price implied by the spec's rr (rr is on R = entry - stop, entry = next open)
d['exit_px'] = d.entry + d.rr * d.R
sp = d.spread_last
variants = {
    'A: rr - 0.5 sfr - 0.5 sfr*(non-target)  [no re-base]': d.rr - 0.5 * d.sfr - 0.5 * d.sfr * nt,
    'B: re-base: fill=entry+s/2, R2=fill-stop, exit-s/2 if non-target, target unchanged (spec exit px)':
        ((d.exit_px - 0.5 * sp * nt) - (d.entry + 0.5 * sp)) / (d.entry + 0.5 * sp - d.stop),
    'C: re-base, target moved to fill+2R2 (as live): target exits pay 2R2 exactly':
        np.where(d.why == 'target', 2.0, ((d.exit_px - 0.5 * sp * nt) - (d.entry + 0.5 * sp)) / (d.entry + 0.5 * sp - d.stop)),
    'D: rr - sfr (full spread every trade; spread_score.py)': d.rr - d.sfr,
    'E: rr - 0.5 sfr (entry only)': d.rr - 0.5 * d.sfr,
    'F: rr - sfr*(non-target) - 0.5 sfr*(target) ': d.rr - d.sfr * nt - 0.5 * d.sfr * (1 - nt),
    'G: A but spread_med instead of spread_last': d.rr - 0.5 * (d.spread_med / d.R) - 0.5 * (d.spread_med / d.R) * nt,
}
books = {'no gate': np.ones(len(d), bool), 'gate<=15%': (d.sfr <= 0.15).values, 'gate<=15% & price>=20': ((d.sfr <= 0.15) & (d.price >= 20)).values}
pub = {'no gate': (0.049, -0.003, 0.010), 'gate<=15%': (0.283, 0.195, 0.260), 'gate<=15% & price>=20': (0.341, 0.350, 0.444)}
for name, net in variants.items():
    print('\n==', name)
    for b, mask in books.items():
        vals = [float(np.nanmean(np.asarray(net)[mask & (d.split == s).values])) for s in ('TRAIN', 'VAL', 'TEST')]
        share = mask.mean() * 100
        diff = max(abs(v - p) for v, p in zip(vals, pub[b]))
        print(f'  {b:24s} T/V/T {vals[0]:+.3f} {vals[1]:+.3f} {vals[2]:+.3f} share {share:4.1f}%  pub {pub[b]}  maxdiff {diff:.3f} {"OK" if diff <= 0.02 else "X"}')
# raw and structure of the $20 gated book
g = d[(d.sfr <= 0.15) & (d.price >= 20)]
print('\n$20 gated book: n', len(g), g.split.value_counts().to_dict())
print(' raw rr T/V/T', [round(g[g.split == s].rr.mean(), 3) for s in ('TRAIN', 'VAL', 'TEST')])
print(' exits', g.why.value_counts(normalize=True).round(3).to_dict())
print(' median sfr', round(g.sfr.median(), 3), 'mean sfr', round(g.sfr.mean(), 3), 'median spread bps', round(g.sbps.median(), 1), 'mean bps', round(g.sbps.mean(), 1))
print(' median r_pct', round(g.r_pct.median(), 2), 'mean r_pct', round(g.r_pct.mean(), 2), 'p25/p75', g.r_pct.quantile([.25, .75]).round(2).tolist())
print(' 10 bps in R (median):', round(0.10 / g.r_pct.median(), 3), ' 20 bps in R:', round(0.20 / g.r_pct.median(), 3))
print(' price bands (gated):', g.groupby(pd.cut(g.price, [20, 50, 1e6]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))
# gate pass rate among price>=20 (what live will see)
p20 = d[d.price >= 20]
print('\nprice>=20 signals', len(p20), 'gate pass', round((p20.sfr <= 0.15).mean(), 3), 'by split', p20.groupby('split').sfr.apply(lambda s: round((s <= 0.15).mean(), 3)).to_dict())
print(' median sfr passing', round(p20[p20.sfr <= 0.15].sfr.median(), 3), 'median sfr failing', round(p20[p20.sfr > 0.15].sfr.median(), 3))
print(' all >=5: pass', round((d.sfr <= 0.15).mean(), 3), 'median pass sfr', round(d[d.sfr <= 0.15].sfr.median(), 3), 'median fail', round(d[d.sfr > 0.15].sfr.median(), 3))
# ask_last vs entry (next open): is the ask at the end of the signal minute above the next open?
d['ask_vs_open_bps'] = (d.ask_last / d.entry - 1) * 1e4; d['mid_vs_open_bps'] = ((d.ask_last + d.bid_last) / 2 / d.entry - 1) * 1e4
g = d[(d.sfr <= 0.15) & (d.price >= 20)]
# raw R in the $20 band by gate pass/fail (is the gate a cost filter or a selector here?)
for s in ('TRAIN', 'VAL', 'TEST', 'ALL'):
    x = p20 if s == 'ALL' else p20[p20.split == s]
    print(f' $20 band {s}: raw rr pass {x[x.sfr <= 0.15].rr.mean():+.3f} (n {int((x.sfr <= 0.15).sum())}) fail {x[x.sfr > 0.15].rr.mean():+.3f} (n {int((x.sfr > 0.15).sum())}) all {x.rr.mean():+.3f}')
# per-split SE of net R in the gated $20 book (for the multiple-comparisons haircut)
for s in ('TRAIN', 'VAL', 'TEST'):
    x = g[g.split == s]; net = (x.rr - 0.5 * x.sfr - 0.5 * x.sfr * (x.why != 'target'))
    print(f' gated $20 {s}: n {len(x)} net mean {net.mean():+.3f} sd {net.std():.3f} SE {net.std() / np.sqrt(len(x)):.3f}')
for lab, x in (('all>=5', d), ('gated $20', g)):
    print(f' {lab}: ask_last vs next-open bps median {x.ask_vs_open_bps.median():+.1f} mean {x.ask_vs_open_bps.mean():+.1f} | mid vs open median {x.mid_vs_open_bps.median():+.1f} | share open>ask {(x.entry > x.ask_last).mean():.2f} open<bid {(x.entry < x.bid_last).mean():.2f}')
# by-price-band with gate (REPORT says $5–10 +0.28/+0.17/+0.13 · $10–20 +0.20/+0.04/+0.20 · $20–50 +0.34/+0.33/+0.38 · $50+ +0.35/+0.40/+0.59)
netA = variants['A: rr - 0.5 sfr - 0.5 sfr*(non-target)  [no re-base]']
gg = d[d.sfr <= 0.15].assign(net=netA[d.sfr <= 0.15])
print('\nby price band gated (variant A):')
print(gg.groupby([pd.cut(gg.price, [5, 10, 20, 50, 1e6]), 'split'], observed=True).net.mean().round(3).unstack()[['TRAIN', 'VAL', 'TEST']])
# grid of cuts x price floors: how many combos, how many pass a "excluded worse on TRAIN and VAL" rule
print('\ngrid: gate x floor -> net (A) T/V/T and n')
for frac in (0.10, 0.15, 0.20, 0.30):
    for fl in (5, 10, 20, 50):
        m = (d.sfr <= frac) & (d.price >= fl)
        vals = [round(float(netA[m & (d.split == s)].mean()), 3) for s in ('TRAIN', 'VAL', 'TEST')]
        print(f'  gate {frac:.2f} floor {fl:3d}: {vals} n {int(m.sum())}')
