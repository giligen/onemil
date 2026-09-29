"""Refuter-2 of cell 1,493: surface recompute, tails, day concentration, cache-only, gross/net."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_entry')
from cell_1445 import day_clustered_t
H = '/home/ec2-user/onemil/research/hod_entry/'
f = pd.read_csv(H + 'cell_1493_fills.csv', dtype={'day': str, 'symbol': str})
print('rows', len(f), 'cells', f.cell.nunique(), 'fills', f.groupby('split')[['day','symbol']].apply(lambda g: len(g.drop_duplicates())).to_dict(), flush=True)
rows = []
for (c, s), g in f.groupby(['cell', 'split']):
    rows.append(dict(cell=c, split=s, n=len(g), mean=g.net_pct.mean(), gross=g.raw_pct.mean(),
                     t=day_clustered_t(g.net_pct.reset_index(drop=True), g.day.reset_index(drop=True))))
S = pd.DataFrame(rows)
P = S.pivot(index='cell', columns='split', values=['mean', 't', 'gross'])
print(P.round(3).sort_values(('mean', 'TRAIN'), ascending=False).head(12).to_string())
print('positive net cells TRAIN', (S[S.split=='TRAIN']['mean']>0).sum(), 'VAL', (S[S.split=='VAL']['mean']>0).sum())
print('positive gross cells TRAIN', (S[S.split=='TRAIN']['gross']>0).sum(), 'VAL', (S[S.split=='VAL']['gross']>0).sum())
print('max t TRAIN', S[S.split=='TRAIN'].t.max(), 'max t VAL', S[S.split=='VAL'].t.max())
sel = S[(S.split=='TRAIN') & ~S.cell.str.startswith('0.5%') & (S.cell!='M')].sort_values('mean').iloc[-1]
vb = S[(S.split=='VAL') & ~S.cell.str.startswith('0.5%') & (S.cell!='M')].sort_values('mean').iloc[-1]
print('TRAIN-best', sel.cell, 'VAL of it', S[(S.cell==sel.cell)&(S.split=='VAL')]['mean'].item(), '| VAL-best', vb.cell, vb['mean'])
# Spearman TRAIN vs VAL ranking
m = P['mean'].drop('M'); print('spearman TRAIN~VAL', m['TRAIN'].rank().corr(m['VAL'].rank()))
# tails on 3.0%|NONE and 2.0%|NONE, CL|NONE
for c in ['3.0%|NONE', 'CL|NONE', '3.0%|T60', '2.0%|tgt1.0', 'M']:
    for s in ['TRAIN', 'VAL']:
        g = f[(f.cell==c)&(f.split==s)]
        y = g.net_pct.sort_values(ascending=False); n=len(y)
        dd = g.groupby('day').net_pct.sum().sort_values(ascending=False)
        drop2 = g[~g.day.isin(dd.index[:2])].net_pct.mean()
        top5 = y.iloc[:int(round(.05*n))].sum()/y.sum() if y.sum()!=0 else np.nan
        print(f'{c:12s} {s:5s} n{n} mean {y.mean():+.3f} ex1% {y.iloc[int(round(.01*n)):].mean():+.3f} ex5% {y.iloc[int(round(.05*n)):].mean():+.3f} '
              f'cap3 {np.minimum(y,3).mean():+.3f} drop2d {drop2:+.3f} bestday {dd.iloc[0]/n:+.3f}pp worstday {dd.iloc[-1]/n:+.3f}pp days {g.day.nunique()} '
              f'median {y.median():+.3f} max {y.max():+.1f} min {y.min():+.1f} why {g.why.value_counts(normalize=True).round(3).to_dict()}')
# cache-only
feat = pd.read_csv(H + 'features_1478_A.csv', usecols=['day','symbol','store_served_1438'], dtype={'day':str,'symbol':str}).drop_duplicates(['day','symbol'])
print('store_served values', feat.store_served_1438.value_counts().to_dict())
g = f[f.cell=='3.0%|NONE'].merge(feat, on=['day','symbol'], how='left')
print('join miss', g.store_served_1438.isna().sum())
print(g.groupby(['split','store_served_1438']).net_pct.agg(['size','mean']).round(3))
g2 = f[f.cell=='CL|NONE'].merge(feat, on=['day','symbol'], how='left')
print('CL|NONE', g2.groupby(['split','store_served_1438']).net_pct.agg(['size','mean']).round(3))
S.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/r1493/surface_recomputed.csv', index=False)
