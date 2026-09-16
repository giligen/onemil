#!/usr/bin/env python3
"""Extract the F6 eligible candidate set (the scoring population the 4-slot book draws from) from
candidates_full.csv, plus an adv20 causality probe. Read-only."""
import os
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
A = 'research/bf_zero2/audit_data'
KEEP = ['day', 'symbol', 'fam', 'entry_m', 'entry', 'stop', 'price', 'r_pct', 'range_so_far_pct', 'dist_open_pct',
        'gap_pct', 'adv20', 'rr_e1c', 'why_e1c', 'exit_m_e1c', 'rr_e4', 'is_wrapper', 'rv_profile', 'n_bars_at_entry']
parts = []
for ch in pd.read_csv('research/bf_zero2/candidates_full.csv', usecols=KEEP,
                      dtype={'day': str, 'symbol': str, 'fam': str, 'why_e1c': str},
                      keep_default_na=False, na_values=[''], chunksize=250_000, low_memory=True):
    ch = ch[ch.fam == 'F6']
    parts.append(ch)
    print('chunk F6', len(ch), flush=True)
f = pd.concat(parts, ignore_index=True); del parts
print('all F6 rows', len(f), flush=True)
for k in KEEP:
    if k not in ('day', 'symbol', 'fam', 'why_e1c'): f[k] = pd.to_numeric(f[k], errors='coerce')
e = f[(f.price >= 5) & (f.entry_m <= 841) & (f.r_pct >= 1.0) & (f.range_so_far_pct >= 5)].copy()
print('F6 ELIGIBLE (book population)', len(e), 'days', e.day.nunique(), flush=True)
e.to_csv(f'{A}/f6_eligible.csv', index=False)
n = e.groupby('day').size()
print('candidates per day: mean %.1f median %d min %d max %d  |  days with <4: %d' % (
    n.mean(), n.median(), n.min(), n.max(), int((n < 4).sum())), flush=True)
print('mean rr_e1c over ALL eligible (no book cap): %.4f  n=%d' % (e.rr_e1c.mean(), len(e)), flush=True)
e['split'] = np.where(e.day < '2026-01-01', 'TRAIN', np.where(e.day < '2026-06-01', 'VAL', 'TEST'))
half = 0.5 * 0.40 / e.r_pct.clip(lower=0.05)
e['net'] = e.rr_e1c - np.where(e.why_e1c == 'target', 0.0, half)
print('POPULATION net mean R by split (no book):')
print(e.groupby('split').net.agg(['count', 'mean']).round(4), flush=True)
print('by entry minute decile:')
print(e.groupby(pd.qcut(e.entry_m, 10, duplicates='drop')).net.agg(['count', 'mean']).round(3), flush=True)

# ---- adv20 causality probe ----
daily = pd.read_parquet('data/research/databento/equs_daily_2025_2026.parquet',
                        columns=['bar_date', 'symbol', 'volume', 'close'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna()].sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = daily.groupby('symbol')
daily['adv20_excl'] = g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
daily['adv20_incl'] = g.volume.transform(lambda s: s.rolling(20, min_periods=5).mean())
u = pd.read_csv('research/bf_zero/universe.csv', dtype={'symbol': str, 'bar_date': str}, keep_default_na=False)
u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
rng = np.random.default_rng(3)
s = u.iloc[rng.choice(len(u), 20000, replace=False)].merge(
    daily[['symbol', 'bar_date', 'adv20_excl', 'adv20_incl']], on=['symbol', 'bar_date'], how='left')
for k in ('adv20_excl', 'adv20_incl'):
    d = (s.adv20 / s[k]).dropna()
    print(f'adv20 vs {k}: median ratio {d.median():.6f}  within 0.1%: {(abs(d-1)<0.001).mean()*100:.1f}%', flush=True)
print('DONE', flush=True)
