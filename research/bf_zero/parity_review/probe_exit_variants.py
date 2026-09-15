#!/usr/bin/env python3
"""Parity review probe 3: the gated $20 book inside spread_exit_variants_rows.csv (V0 = live's fill/target geometry),
plus stop-slippage sensitivity: extra X bps through the stop beyond the spec's 10 bps + half spread."""
import numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/bf_zero'
T = pd.read_csv(f'{D}/spread_exit_variants_rows.csv')
q = pd.read_csv(f'{D}/spread_study_clean.csv', usecols=['day', 'symbol', 'price', 'spread_last', 'entry', 'stop'], dtype={'symbol': str})
T = T.merge(q, on=['day', 'symbol'], how='left')
v0 = T[T.variant == 'V0'].copy()
print('V0 rows', len(v0), 'symbols', v0.symbol.nunique(), 'splits', v0.split.value_counts().to_dict())
g = v0[(v0.spread_frac_r <= 0.15) & (v0.price >= 20)].copy()
print('gated $20 V0: n', len(g), g.split.value_counts().to_dict())
for s in ('TRAIN', 'VAL', 'TEST'):
    x = g[g.split == s]
    print(f'  {s}: meanR {x.rr.mean():+.3f} usd/100 {x.usd_per_100.mean():+.1f} WR {(x.rr > 0).mean() * 100:.1f} target {(x.why == "target").mean() * 100:.1f}% stop {(x.why == "stop").mean() * 100:.1f}% eod {(x.why == "eod").mean() * 100:.1f}%')
# sensitivity: extra slippage on stop exits (bps of price) and on eod exits
g['r_pct'] = g.R / (g.entry + g.spread_last / 2) * 100
print('  gated $20 V0 median r_pct (re-based)', round(g.r_pct.median(), 2))
for extra_bps in (0, 10, 20, 30, 50, 100):
    out = []
    for s in ('TRAIN', 'VAL', 'TEST'):
        x = g[g.split == s]
        pen = np.where(x.why == 'stop', extra_bps / 1e4 / (x.r_pct / 100), 0.0)
        out.append(x.rr.mean() - pen.mean())
    print(f'  extra stop slippage {extra_bps:3d} bps → net T/V/T {out[0]:+.3f} {out[1]:+.3f} {out[2]:+.3f}')
# what if the target leg fills on a wick (live) instead of a close — count trades where a bar's high touched the target
# (cannot from this CSV: needs bars) -> skip; report share of 'eod' exits that were above entry (partial winners)
e = g[g.why == 'eod']; print('  eod exits: n', len(e), 'mean rr', round(e.rr.mean(), 3), 'share > 0', round((e.rr > 0).mean(), 2))
# breakdown: mean rr by exit type for the gated $20 book
print('  mean rr by exit:', g.groupby('why').rr.agg(['mean', 'count']).round(3).to_dict('index'))
# entry slippage sensitivity: extra X bps on the fill (ask lifts between the bar close and the order) — charged to every trade,
# and it also shrinks the target payoff (target is fixed from the ask so 2R payoff stays 2R of the smaller R... approximated as a flat charge)
for extra_bps in (0, 5, 10, 20, 30):
    out = []
    for s in ('TRAIN', 'VAL', 'TEST'):
        x = g[g.split == s]; out.append(x.rr.mean() - (extra_bps / 1e4 / (x.r_pct / 100)).mean())
    print(f'  extra entry slippage {extra_bps:3d} bps (all trades) → net T/V/T {out[0]:+.3f} {out[1]:+.3f} {out[2]:+.3f}')
