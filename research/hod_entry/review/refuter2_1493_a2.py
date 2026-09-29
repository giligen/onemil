"""Refuter-2 placebo decomposition: pre-break vs post-retest placebo minutes, by store flag."""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_entry')
from cell_1445 import day_clustered_t
H = '/home/ec2-user/onemil/research/hod_entry/'
SLIP = {'TRAIN': 0.88*2.9+0.12*94.0, 'VAL': 0.88*3.2+0.12*76.0}; EODB = {'TRAIN': 11.5, 'VAL': 9.7}
fl = pd.read_csv(H+'rebuild_1481_fills.csv', dtype={'day':str,'symbol':str}); fl = fl[fl.status=='fill'].reset_index(drop=True)
feat = pd.read_csv(H+'features_1478_A.csv', usecols=['day','symbol','store_served_1438'], dtype={'day':str,'symbol':str}).drop_duplicates(['day','symbol'])
fl = fl.merge(feat, on=['day','symbol'], how='left')
con = sqlite3.connect(H+'bars_fills_1478.db')
b = pd.read_sql_query('SELECT symbol, day, t, o, h, l, c FROM bars', con); con.close()
ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York'); b['m'] = ts.dt.hour*60+ts.dt.minute
b = b.sort_values(['symbol','day','m'])
G = {k: g[['m','o','h','l','c']].to_numpy(float) for k, g in b.groupby(['symbol','day'], sort=False)}
print('bars loaded', len(G), flush=True)
real = pd.read_csv(H+'cell_1493_fills.csv', dtype={'day':str,'symbol':str}, usecols=['day','symbol','cell','net_pct'])
real = real[real.cell.isin(['3.0%|NONE','1.0%|NONE'])].pivot_table(index=['day','symbol'], columns='cell', values='net_pct').reset_index()
def walk(path, entry, stoppct, split):
    stop = entry*(1-stoppct)
    for m,o,h,l,c in path:
        if m >= 955: return (o-entry)/entry*100 - EODB[split]/100
        if l <= stop:
            px = o if o <= stop else stop; return (px-entry)/entry*100 - SLIP[split]/100
    return (path[-1][4]-entry)/entry*100 - EODB[split]/100 if len(path) else np.nan
rng = np.random.RandomState(1493); out = []
nbars = []
for i, r in fl.iterrows():
    g = G.get((r.symbol, r.day))
    if g is None: continue
    rth = g[(g[:,0]>=570)&(g[:,0]<960)]; nbars.append((r.store_served_1438, len(rth)))
    lo = int(r.fill_min)
    # draw 3 placebo minutes: any, pre-break only, post-window only
    for kind, (a, z) in {'pre': (585, lo-1), 'post': (lo+15, 900)}.items():
        if z < a: continue
        pm = rng.randint(a, z+1)
        at = g[g[:,0]==pm]
        if not len(at): continue
        path = g[g[:,0]>=pm+1]
        out.append(dict(day=r.day, symbol=r.symbol, split=r.split, store=r.store_served_1438, kind=kind, pm=pm,
                        p3=walk(path, at[0,1], .03, r.split), p1=walk(path, at[0,1], .01, r.split)))
P = pd.DataFrame(out).merge(real, on=['day','symbol'])
nb = pd.DataFrame(nbars, columns=['store','n']); print('RTH bars per symbol-day by store:', nb.groupby('store').n.describe().round(0).to_string())
for (s, k), g in P.groupby(['split','kind']):
    d = g['3.0%|NONE'] - g.p3
    print(f'{s:5s} {k:4s} n{len(g)} real3 {g["3.0%|NONE"].mean():+.3f} plac3 {g.p3.mean():+.3f} margin {d.mean():+.3f} t {day_clustered_t(d.reset_index(drop=True), g.day.reset_index(drop=True)):+.2f} | real1 {g["1.0%|NONE"].mean():+.3f} plac1 {g.p1.mean():+.3f}')
for (s, k, st), g in P.groupby(['split','kind','store']):
    print(f'  {s:5s} {k:4s} store{st} n{len(g)} real3 {g["3.0%|NONE"].mean():+.3f} plac3 {g.p3.mean():+.3f}')
P.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/r1493/placebo_decomp.csv', index=False)
