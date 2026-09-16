#!/usr/bin/env python3
"""a0 — extract the score3 candidate pool (all 27 family-configs) into a compact parquet.

Applies EXACTLY the score3.py filters so the audit reproduces the search that produced the claim:
price >= 5, entry_m <= 841, r_pct >= 1.0, and for F5-F10 the causal floor range_so_far_pct >= 5.
Writes research/bf_zero2/audit_stats/pool.parquet (~860K rows).
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
D = 'research/bf_zero2'
OUT = f'{D}/audit_stats'

COLS = ['day', 'symbol', 'fam', 'cfg', 'entry_m', 'exit_m_e1c', 'why_e1c', 'rr_e1c', 'rr_e4',
        'r_pct', 'price', 'range_so_far_pct', 'is_wrapper', 'adv20', 'gap_pct', 'rv_profile',
        'dist_open_pct', 'prev_range_pct', 'pm_vol', 'bar_vol_x']

parts = []
n_in = 0
reader = pd.read_csv(f'{D}/candidates_full.csv', usecols=COLS, chunksize=400_000,
                     dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str, 'why_e1c': str},
                     keep_default_na=False, na_values=[''], low_memory=True)
for i, ch in enumerate(reader):
    n_in += len(ch)
    ch = ch[(ch.price >= 5.0) & (ch.entry_m <= 841) & (ch.r_pct >= 1.0)]
    need = ~ch.fam.isin(['F1', 'F2', 'F3', 'F4'])
    ch = ch[~(need & ~(ch.range_so_far_pct >= 5))]
    parts.append(ch)
    if i % 5 == 0:
        print(f'chunk {i} in={n_in:,} kept={sum(len(p) for p in parts):,}', flush=True)

c = pd.concat(parts, ignore_index=True)
del parts
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c['mo'] = c.day.str[:7]
c['key'] = c.fam + ' ' + c.cfg
half = 0.5 * 0.40 / c.r_pct.clip(lower=0.05)
c['net_e1c'] = c.rr_e1c - np.where(c.why_e1c == 'target', 0.0, half)
c['net_e4'] = c.rr_e4 - half
c['exit_m_e4'] = 955
for col in ('fam', 'cfg', 'key', 'why_e1c', 'split', 'wk', 'mo'):
    c[col] = c[col].astype('category')
print(f'rows in {n_in:,} kept {len(c):,} keys {c.key.nunique()}', flush=True)
print(c.groupby('split', observed=True).size(), flush=True)
c.to_parquet(f'{OUT}/pool.parquet', index=False)
print('WROTE pool.parquet', flush=True)

f6 = c[c.fam == 'F6'].reset_index(drop=True)
f6.to_parquet(f'{OUT}/pool_f6.parquet', index=False)
print(f'F6 rows {len(f6):,} days {f6.day.nunique()}', flush=True)
print('DONE', flush=True)
