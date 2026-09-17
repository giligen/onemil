#!/usr/bin/env python3
"""R3 - join A's and B's PRE-BOOK candidate sets on (day, symbol) so book-slot effects are separated
from detection effects."""
import os
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit'; OUT = f'{D}/H/F6_reconcile'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)

a = RD(f'{OUT}/a_cands.csv')[['day', 'symbol', 'sig_m', 'level', 'stop', 'price', 'prev_close',
                              'prev_day_range_pct', 'range_so_far_pct', 'next_entry', 'next_entry_m', 'next_r_pct']]
a.columns = ['day', 'symbol'] + ['A_' + c for c in a.columns[2:]]
a['in_A'] = 1
b = RD(f'{D}/H/F6_rebuild/scan_ai.csv')
# B's scan rows are already fully gated (cap, price, R, 14:01, floor) -- see scan.py
b = b[['day', 'symbol', 'src', 'sig_min', 'entry_min', 'entry', 'stop', 'R', 'level']]
b.columns = ['day', 'symbol', 'B_src'] + ['B_' + c for c in ['sig_m', 'entry_m', 'entry', 'stop', 'R', 'level']]
b['in_B'] = 1
j = a.merge(b, on=['day', 'symbol'], how='outer')
j['in_A'] = j.in_A.fillna(0).astype(int); j['in_B'] = j.in_B.fillna(0).astype(int)
j['split'] = np.where(j.day < '2026-01-01', 'TRAIN', np.where(j.day < '2026-06-01', 'VAL', 'TEST'))
j.to_csv(f'{OUT}/cand_join.csv', index=False)
t = j.groupby('split').apply(lambda g: pd.Series({
    'A_cands': int((g.in_A == 1).sum()), 'B_cands': int((g.in_B == 1).sum()),
    'A_only': int(((g.in_A == 1) & (g.in_B == 0)).sum()),
    'B_only': int(((g.in_A == 0) & (g.in_B == 1)).sum()),
    'both': int(((g.in_A == 1) & (g.in_B == 1)).sum())}))
print(t.to_string(), flush=True)
print('\nTOTAL', t.sum().to_dict(), flush=True)
open(f'{OUT}/r3_cand_join.md', 'w').write('# R3 pre-book candidate join\n\n' + t.to_string() + '\n')
