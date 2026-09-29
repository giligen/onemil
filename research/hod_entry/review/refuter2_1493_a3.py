"""Refuter-2: real-SIP-only surface; mirror M vs the 1,480 short reconciliation."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_entry')
from cell_1445 import day_clustered_t
H = '/home/ec2-user/onemil/research/hod_entry/'
f = pd.read_csv(H+'cell_1493_fills.csv', dtype={'day':str,'symbol':str})
feat = pd.read_csv(H+'features_1478_A.csv', usecols=['day','symbol','store_served_1438'], dtype={'day':str,'symbol':str}).drop_duplicates(['day','symbol'])
f = f.merge(feat, on=['day','symbol'], how='left')
s0 = f[f.store_served_1438==0].groupby(['cell','split']).net_pct.mean().unstack()
print('real-SIP only: positive cells', (s0>0).sum().to_dict(), 'best', s0.drop(index='M').idxmax().to_dict(), s0.drop(index='M').max().round(3).to_dict())
s1 = f[f.store_served_1438==1].groupby(['cell','split']).net_pct.mean().unstack()
print('cache-only: positive cells', (s1>0).sum().to_dict())
# Short 1480
sh = pd.read_csv(H+'cell_1480_fills.csv', dtype={'day':str,'symbol':str})
sh = sh[sh.short_net_R.notna()].copy()
sh['R_pct'] = (sh.short_stop - sh.short_entry)/sh.short_entry*100
sh['raw_pct'] = (sh.short_entry - sh.short_exit_px)/sh.short_entry*100
sh['net_pct'] = sh.short_net_R * sh.R_pct
sh['cost_pct'] = sh.raw_pct - sh.net_pct
sh['print_to_bid_bps'] = (sh.print_px - sh.short_entry)/sh.print_px*1e4
for s, g in sh.groupby('split'):
    print(f'short {s} n{len(g)} netR {g.short_net_R.mean():+.3f} rawR {g.short_raw_R.mean():+.3f} R% med {g.R_pct.median():.2f} mean {g.R_pct.mean():.2f} | raw% {g.raw_pct.mean():+.3f} net% {g.net_pct.mean():+.3f} cost% {g.cost_pct.mean():+.3f} print->bid bps {g.print_to_bid_bps.mean():.1f} | why {g.short_why.value_counts(normalize=True).round(3).to_dict()}')
# paired on same (day,symbol): short raw% + M raw% ; and short vs long with an exactly mirrored price path
m = f[f.cell=='M'][['day','symbol','split','raw_pct','net_pct','why']].merge(sh[['day','symbol','raw_pct','net_pct','short_entry','print_px','level','short_why']], on=['day','symbol'], suffixes=('_M','_S'))
m['entry_gap_bps'] = ( (m.level-0.01) - m.short_entry)/m.short_entry*1e4
for s, g in m.groupby('split'):
    print(f'paired {s} n{len(g)} M raw {g.raw_pct_M.mean():+.3f} net {g.net_pct_M.mean():+.3f} | S raw {g.raw_pct_S.mean():+.3f} net {g.net_pct_S.mean():+.3f} | sum raw {(g.raw_pct_M+g.raw_pct_S).mean():+.3f} sum net {(g.net_pct_M+g.net_pct_S).mean():+.3f} | long-entry minus short-entry {g.entry_gap_bps.mean():.1f} bps')
