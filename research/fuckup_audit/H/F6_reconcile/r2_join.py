#!/usr/bin/env python3
"""R2 - join A's and B's BOOKED trades on (day, symbol), hold exit, all three splits.

A: research/fuckup_audit/H/F6/f6_pdr_trades_hold_{TRAIN,VAL,TEST}.csv (booked) + a_cands.csv (pre-book fields)
B: research/fuckup_audit/H/F6_rebuild/trades_hold_ai.csv (booked, carries entry/stop/sig_min/src)
Writes join_hold.csv (full outer join, all splits) and prints the count/difference tables.
"""
import os
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit'
OUT = f'{D}/H/F6_reconcile'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)


def split_of(d):
    return np.where(d < '2026-01-01', 'TRAIN', np.where(d < '2026-06-01', 'VAL', 'TEST'))


# ---- A ----
a = pd.concat([RD(f'{D}/H/F6/f6_pdr_trades_hold_{s}.csv') for s in ('TRAIN', 'VAL', 'TEST')], ignore_index=True)
a = a.rename(columns={'em': 'A_entry_m', 'xm': 'A_exit_m', 'net': 'A_net', 'gross': 'A_gross', 'why': 'A_why'})
ac = RD(f'{OUT}/a_cands.csv')
ac = ac.rename(columns={'next_entry': 'A_entry', 'stop': 'A_stop', 'level': 'A_level', 'sig_m': 'A_sig_m',
                        'next_r_pct': 'A_r_pct', 'prev_close': 'A_prev_close',
                        'prev_day_range_pct': 'A_pdr', 'range_so_far_pct': 'A_rsf', 'spread_cc_bps': 'A_cc_bps'})
a = a.merge(ac[['day', 'symbol', 'A_entry', 'A_stop', 'A_level', 'A_sig_m', 'A_r_pct', 'A_prev_close', 'A_pdr',
                'A_rsf', 'A_cc_bps']], on=['day', 'symbol'], how='left')
a['in_A'] = 1

# ---- B ----
b = RD(f'{D}/H/F6_rebuild/trades_hold_ai.csv')
b = b.rename(columns={'entry_min': 'B_entry_m', 'exit_min': 'B_exit_m', 'entry': 'B_entry', 'stop': 'B_stop',
                      'R': 'B_R', 'exit_type': 'B_why', 'gross_R': 'B_gross', 'net_R': 'B_net',
                      'half': 'B_half', 'sig_min': 'B_sig_m', 'src': 'B_src'})
b['in_B'] = 1

j = a.merge(b, on=['day', 'symbol'], how='outer')
j['in_A'] = j.in_A.fillna(0).astype(int); j['in_B'] = j.in_B.fillna(0).astype(int)
j['split'] = split_of(j.day.values)
j['B_level'] = j.B_entry * np.nan
j.to_csv(f'{OUT}/join_hold.csv', index=False)

L = []


def p(*a_):
    s = ' '.join(str(x) for x in a_); print(s, flush=True); L.append(s)


p('# R2 join - booked HOLD trades, A vs B, on (day, symbol)\n')
tb = j.groupby(['split', 'in_A', 'in_B']).size().unstack(fill_value=0)
p('counts by split (rows: split x in_A; cols in_B)')
p(j.groupby('split').apply(lambda g: pd.Series({
    'A_only': int(((g.in_A == 1) & (g.in_B == 0)).sum()),
    'B_only': int(((g.in_A == 0) & (g.in_B == 1)).sum()),
    'both': int(((g.in_A == 1) & (g.in_B == 1)).sum()),
    'A_n': int((g.in_A == 1).sum()), 'B_n': int((g.in_B == 1).sum()),
    'A_netR': round(float(g.loc[g.in_A == 1, 'A_net'].sum()), 2),
    'B_netR': round(float(g.loc[g.in_B == 1, 'B_net'].sum()), 2),
    'A_only_netR': round(float(g.loc[(g.in_A == 1) & (g.in_B == 0), 'A_net'].sum()), 2),
    'B_only_netR': round(float(g.loc[(g.in_A == 0) & (g.in_B == 1), 'B_net'].sum()), 2),
    'both_A_netR': round(float(g.loc[(g.in_A == 1) & (g.in_B == 1), 'A_net'].sum()), 2),
    'both_B_netR': round(float(g.loc[(g.in_A == 1) & (g.in_B == 1), 'B_net'].sum()), 2),
})).to_string())

s = j[(j.in_A == 1) & (j.in_B == 1)].copy()
s['d_entry_m'] = s.A_entry_m - s.B_entry_m
s['d_sig_m'] = s.A_sig_m - s.B_sig_m
s['d_entry'] = s.A_entry - s.B_entry
s['d_stop'] = s.A_stop - s.B_stop
s['d_exit_m'] = s.A_exit_m - s.B_exit_m
s['d_net'] = s.A_net - s.B_net
s['d_gross'] = s.A_gross - s.B_gross
s['why_same'] = s.A_why == s.B_why
s.to_csv(f'{OUT}/join_hold_shared.csv', index=False)
p('\nshared trades: field differences (all splits, n=%d)' % len(s))
rows = []
for col, tol in (('d_sig_m', 0), ('d_entry_m', 0), ('d_exit_m', 0), ('d_entry', 1e-9), ('d_stop', 1e-9),
                 ('d_net', 1e-9), ('d_gross', 1e-9)):
    v = s[col].astype(float)
    rows.append(dict(field=col, n_diff=int((v.abs() > tol).sum()), pct=round((v.abs() > tol).mean() * 100, 1),
                     mean_abs=round(float(v.abs().mean()), 4), p50=round(float(v.median()), 4),
                     min=round(float(v.min()), 3), max=round(float(v.max()), 3)))
rows.append(dict(field='why_differs', n_diff=int((~s.why_same).sum()), pct=round((~s.why_same).mean() * 100, 1),
                 mean_abs=np.nan, p50=np.nan, min=np.nan, max=np.nan))
p(pd.DataFrame(rows).to_string(index=False))

p('\nhistogram of d_entry_m (A minus B, shared trades)')
p(s.d_entry_m.value_counts().sort_index().to_string())
p('\nhistogram of d_exit_m bucketed')
p(pd.cut(s.d_exit_m.astype(float), [-1e9, -60, -10, -1, -1e-9, 1e-9, 1, 10, 60, 1e9]).value_counts().sort_index().to_string())
p('\nhistogram of d_net (A minus B)')
p(pd.cut(s.d_net.astype(float), [-1e9, -2, -1, -.25, -.01, .01, .25, 1, 2, 1e9]).value_counts().sort_index().to_string())
p('\nexit-reason cross-tab (shared)')
p(pd.crosstab(s.A_why, s.B_why).to_string())

for sp in ('TRAIN', 'VAL', 'TEST'):
    g = s[s.split == sp]
    p(f'\n{sp}: shared n={len(g)}  A_net={g.A_net.sum():.1f}  B_net={g.B_net.sum():.1f}  '
      f'identical(entry,stop,exit_m,net)={int(((g.d_entry.abs()<1e-9)&(g.d_stop.abs()<1e-9)&(g.d_exit_m==0)&(g.d_net.abs()<1e-9)).sum())}')

open(f'{OUT}/r2_join.md', 'w').write('\n'.join(L) + '\n')
