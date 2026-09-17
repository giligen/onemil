"""B+ rank (post threshold+Q1+dedup, pre no-refill vetoes) of each live ORB trade."""
import sys
import pandas as pd
import yaml
sys.path.insert(0, '/home/ec2-user/onemil')
import os
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv
from study_orb_filter import FILTER_FEATURES, composite_score
from study_orb_sizing import assign_quintile
from study_orb_correlation_filter import symbol_family, symbol_super_group

cfg = yaml.safe_load(open('orb.yaml'))
filt = cfg['filter']
Z = {f: {'mean': float(filt['features'][f]['mean']),
         'std': float(filt['features'][f]['std']),
         'sign': int(filt['features'][f]['sign'])} for f, _ in FILTER_FEATURES}
THR = float(filt['threshold'])
CUT = [float(x) for x in cfg['quintile_cutoffs']]
QO = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}

f = read_orb_csv('analysis_results/orb_features_20260916_2053.csv')
f['date'] = pd.to_datetime(f['date']).dt.strftime('%Y-%m-%d')
f = f.dropna(subset=[x for x, _ in FILTER_FEATURES]).copy()
f['_c'] = composite_score(f, Z)
f['_q'] = assign_quintile(f['_c'], CUT)
k = f[(f['_c'] >= THR) & (f['_q'] != 'Q1')].copy()
k['_qr'] = k['_q'].map(QO)
ranks = {}
pool = {}
for d, g in k.groupby('date'):
    g = g.sort_values(['_qr', '_c'], ascending=[True, False])
    sf, ss, r = set(), set(), 0
    for _, row in g.iterrows():
        fam = symbol_family(row['symbol'])
        sup = symbol_super_group(row['symbol'])
        if fam and fam in sf:
            continue
        if sup and sup in ss:
            continue
        if fam:
            sf.add(fam)
        if sup:
            ss.add(sup)
        r += 1
        ranks[(row['symbol'], d)] = r
    pool[d] = r
print(f"post-filter post-dedup daily pool: n_days={len(pool)} "
      f"mean {sum(pool.values())/len(pool):.1f} median "
      f"{pd.Series(list(pool.values())).median():.0f} "
      f"p90 {pd.Series(list(pool.values())).quantile(.9):.0f} "
      f"max {max(pool.values())}")
for n in (3, 4, 6, 8, 12):
    cap = sum(min(v, n) for v in pool.values())
    print(f"  slots {n}: {cap} pick-slots fillable "
          f"({cap/sum(pool.values())*100:.0f}% of the deduped pool)")

live = pd.read_csv('research/fuckup_audit/D1_orb/taskA_live_retro_veto.csv',
                   keep_default_na=False)
live['pnl'] = pd.to_numeric(live['pnl'])
live['R'] = pd.to_numeric(live['R'])
live['blocked'] = live['blocked'].astype(str).isin(['True', 'true'])
live['bplus_rank'] = [ranks.get((s, d)) for s, d in zip(live['symbol'], live['date'])]
print("\nB+ rank of each live ORB trade:")
print(live['bplus_rank'].value_counts(dropna=False).sort_index().to_string())
for n in (3, 4, 6, 8, 12):
    m = live['bplus_rank'].notna() & (live['bplus_rank'] <= n)
    mk = m & ~live['blocked']
    print(f"  live trades inside B+ top-{n}: {int(m.sum())}/{len(live)}  "
          f"surviving vetoes {int(mk.sum())}  ${live.loc[mk, 'pnl'].sum():,.0f}")
live.to_csv('research/fuckup_audit/D1_orb/taskA_live_retro_veto.csv', index=False)
print("\nsurvivors with their B+ rank:")
s = live[~live['blocked']].sort_values('bplus_rank', na_position='last')
print(s[['symbol', 'date', 'bplus_rank', 'quintile', 'pnl', 'R',
         'exit_reason']].to_string(index=False))
