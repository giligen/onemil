#!/usr/bin/env python3
"""Scoring for the spread study on the batch-safe clean CSV (see spread_study.py for the design)."""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); D = 'research/bf_zero'
OUT = f'{D}/spread_study_clean.csv'
# ---------------- scoring ----------------
d = pd.read_csv(OUT, dtype={'symbol': str}); d = d[d.n_quotes > 0].copy()
d['R'] = d.entry - d.stop; d['spread_frac_r'] = d.spread_last / d.R; d['spread_bps'] = d.spread_last / d.entry * 1e4
d['rr_after_cost'] = d.rr - d.spread_frac_r                    # pessimistic: one full spread charged to every trade
edges = d[d.split == 'TRAIN'].spread_frac_r.quantile([.2, .4, .6, .8]).values
d['q'] = np.searchsorted(edges, d.spread_frac_r) + 1
lines = [f'# HOD-break spread study — {len(d):,} signals with quotes ({d.split.value_counts().to_dict()}); TRAIN quintile edges of spread/R: {np.round(edges, 3).tolist()}', '']
for s in ('TRAIN', 'VAL', 'TEST'):
    x = d[d.split == s]
    g = x.groupby('q').agg(n=('rr', 'size'), meanR=('rr', 'mean'), meanR_after_cost=('rr_after_cost', 'mean'), WR=('rr', lambda r: (r > 0).mean() * 100), spread_bps=('spread_bps', 'median'), sfr=('spread_frac_r', 'median')).round(3)
    lines += [f'## {s}', g.to_markdown(), '']
for frac in (0.10, 0.15, 0.20, 0.30):
    lines.append(f'gate spread/R <= {frac:.0%}: ' + ' | '.join(f"{s}: keep {int((d[(d.split == s)].spread_frac_r <= frac).sum())}/{int((d.split == s).sum())} meanR kept {d[(d.split == s) & (d.spread_frac_r <= frac)].rr.mean():+.3f} dropped {d[(d.split == s) & (d.spread_frac_r > frac)].rr.mean():+.3f} | after-cost kept {d[(d.split == s) & (d.spread_frac_r <= frac)].rr_after_cost.mean():+.3f}" for s in ('TRAIN', 'VAL', 'TEST')))
lines += ['', 'by price band (all splits): ' + d.groupby(pd.cut(d.price, [5, 10, 20, 50, 1e6]), observed=True).agg(n=('rr', 'size'), spread_bps=('spread_bps', 'median'), sfr=('spread_frac_r', 'median'), meanR=('rr', 'mean'), after=('rr_after_cost', 'mean')).round(3).to_string().replace('\n', '\n  ')]
open(f'{D}/spread_study.md', 'w').write('\n'.join(lines)); print('\n'.join(lines)); print('DONE', flush=True)
