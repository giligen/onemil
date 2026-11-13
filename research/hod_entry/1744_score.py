"""Cell 1,744 scorer (PREREG_1744): no-stop hold-to-close (V1F = v1_net_R*2, V2F = v2_net_R*2) on the cheapest half_spread tercile.
Reads TRAIN/VAL only (1735_per_signal.csv has no TEST; signals.parquet TEST rows are dropped on load). Usage: python3 1744_score.py"""
import math, numpy as np, pandas as pd
R = '/home/ec2-user/onemil/research/hod_entry'
ps = pd.read_csv(f'{R}/1735_per_signal.csv', dtype={'symbol': str}, usecols=['day', 'symbol', 'split', 'half', 'b0_net_R', 'v1_net_R', 'v2_net_R'], keep_default_na=False, na_values=[''])
ps = ps[ps.split != 'TEST']
sg = pd.read_parquet('/home/ec2-user/onemil/research/hod_exit_lab/signals.parquet')
sg = sg[sg.split != 'TEST'][['day', 'symbol', 'entry_m', 'r_pct', 'price', 'half_spread', 'wk']]
fl = pd.read_csv(f'{R}/1741_flow.csv', dtype={'symbol': str}, usecols=['day', 'symbol', 'covered', 'n_prints'], keep_default_na=False, na_values=[''])
n0 = len(ps)
d = ps.merge(sg, on=['day', 'symbol'], how='inner'); n1 = len(d)
d = d.merge(fl, on=['day', 'symbol'], how='left'); n2 = int(d.covered.notna().sum())
d['n_prints'] = d.n_prints.fillna(0); d['hs_bps'] = d.half_spread / d.price * 1e4
d['V1F'] = d.v1_net_R * 2; d['V2F'] = d.v2_net_R * 2
print(f'join: per_signal {n0} -> with signals {n1} (loss {n0-n1}); flow matched {n2}/{n1} (unmatched treated n_prints=0)', flush=True)
tr = d[d.split == 'TRAIN']
c1, c2 = tr.hs_bps.quantile([1/3, 2/3]).values
d['terc'] = np.where(d.hs_bps <= c1, 'cheap', np.where(d.hs_bps <= c2, 'mid', 'dear'))
print(f'TRAIN tercile cuts half_spread bps: {c1:.2f} {c2:.2f}', flush=True)
d.to_csv(f'{R}/1744_rows.csv', index=False)

def capped(g):
    """First-come book by entry minute: <=8 fills/day, <=4 concurrent; all positions held to the close so concurrency = first 4/day."""
    return g.sort_values(['day', 'entry_m']).groupby('day').head(4)

def stats(g, col, nweeks):
    """Per-spec statistics for the frame g on net-R column col."""
    r = g[col].values; n = len(r)
    if n < 3: return dict(n=n)
    m = r.mean(); iid = m / (r.std(ddof=1) / math.sqrt(n))
    dsum = g.assign(_r=r - m).groupby('day')._r.sum(); dt = m / (math.sqrt((dsum ** 2).sum()) / n)
    s = np.sort(r); k5 = int(math.ceil(.05 * n)); k1 = max(int(math.ceil(.01 * n)), 1)
    wk = g.assign(_w=pd.to_datetime(g.day).dt.strftime('%G-%V'), _r2=r).groupby('_w')._r2.sum()
    wkf = np.concatenate([wk.values, np.zeros(max(nweeks - len(wk), 0))])
    cum = g.assign(_r2=r).sort_values(['day', 'entry_m'])._r2.cumsum().values
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum).max())
    cp = capped(g); cm = cp[col].mean(); cdsum = cp.assign(_r=cp[col] - cm).groupby('day')._r.sum()
    return dict(n=n, mean_R=m, day_t=dt, iid_t=iid, ex_top5=s[:-k5].mean(), ex_top1=s[:-k1].mean(), cap3=np.minimum(r, 3).mean(),
                share_lt1=(r < -1).mean(), share_lt2=(r < -2).mean(), green_wk=(wk > 0).sum() / nweeks, wk_p10=np.percentile(wkf, 10),
                max_dd_R=dd, fills_wk=n / nweeks, cap_n=len(cp), cap_mean=cm, cap_day_t=cm / (math.sqrt((cdsum ** 2).sum()) / len(cp)),
                b0_mean=g.b0_net_R.mean())

def null_green(whole, col, n, nweeks, rng):
    """Count-matched null: mean green-week share and mean R of 300 same-n random draws from the whole split."""
    gs, ms = [], []
    wkk = pd.to_datetime(whole.day).dt.strftime('%G-%V').values; v = whole[col].values
    for _ in range(300):
        ix = rng.choice(len(whole), n, replace=False)
        w = pd.Series(v[ix]).groupby(wkk[ix]).sum(); gs.append((w > 0).sum() / nweeks); ms.append(v[ix].mean())
    return np.mean(gs), np.mean(ms)

reads = [('PRIMARY V1F cheap', 'V1F', lambda x: x.terc == 'cheap'), ('X1 V1F cheap & print', 'V1F', lambda x: (x.terc == 'cheap') & (x.n_prints > 0)),
         ('X2 V1F cheap & r_pct>=2', 'V1F', lambda x: (x.terc == 'cheap') & (x.r_pct >= 2)), ('X3 V2F cheap', 'V2F', lambda x: x.terc == 'cheap'),
         ('X4 V1F mid (dose)', 'V1F', lambda x: x.terc == 'mid'), ('ref V1F dear', 'V1F', lambda x: x.terc == 'dear'), ('ref V1F all', 'V1F', lambda x: x.terc != 'zz')]
rng = np.random.default_rng(1744); out = []
for name, col, f in reads:
    for gn, gsel in [('TRAIN_H1', (d.split == 'TRAIN') & (d.half == 'H1')), ('TRAIN_H2', (d.split == 'TRAIN') & (d.half == 'H2')), ('TRAIN', d.split == 'TRAIN'), ('VAL', d.split == 'VAL')]:
        whole = d[gsel]; nweeks = pd.to_datetime(whole.day).dt.strftime('%G-%V').nunique(); g = whole[f(whole)]
        st = stats(g, col, nweeks)
        if st['n'] >= 3: st['null_green_wk'], st['null_mean_R'] = null_green(whole, col, st['n'], nweeks, rng)
        st.update(read=name, group=gn, nweeks=nweeks); out.append(st)
        print(f"{name:26s} {gn:9s} n={st['n']:5d} mean={st.get('mean_R', float('nan')):+.3f} dayt={st.get('day_t', float('nan')):+.2f} iidt={st.get('iid_t', float('nan')):+.2f} ex5={st.get('ex_top5', float('nan')):+.3f} ex1={st.get('ex_top1', float('nan')):+.3f} cap3={st.get('cap3', float('nan')):+.3f} <-1={st.get('share_lt1', float('nan')):.2f} <-2={st.get('share_lt2', float('nan')):.2f} gw={st.get('green_wk', float('nan')):.2f}/{st.get('null_green_wk', float('nan')):.2f} p10={st.get('wk_p10', float('nan')):+.2f} dd={st.get('max_dd_R', float('nan')):.0f} fpw={st.get('fills_wk', float('nan')):.2f} cap={st.get('cap_mean', float('nan')):+.3f}(n{st.get('cap_n')}) b0={st.get('b0_mean', float('nan')):+.3f}", flush=True)
pd.DataFrame(out).to_csv(f'{R}/1744_summary.csv', index=False)
P = {x['group']: x for x in out if x['read'] == 'PRIMARY V1F cheap'}
bar = [('TRAIN mean>=.10 & t>=2', P['TRAIN']['mean_R'] >= .10 and P['TRAIN']['day_t'] >= 2), ('VAL mean>=.10 & t>=2', P['VAL']['mean_R'] >= .10 and P['VAL']['day_t'] >= 2),
       ('halves same sign', P['TRAIN_H1']['mean_R'] * P['TRAIN_H2']['mean_R'] > 0), ('VAL ex5>=0', P['VAL']['ex_top5'] >= 0), ('VAL wkP10>=-2', P['VAL']['wk_p10'] >= -2),
       ('VAL fills/wk>=3', P['VAL']['fills_wk'] >= 3), ('VAL capped>=+.05', P['VAL']['cap_mean'] >= .05)]
for k, v in bar: print(('PASS ' if v else 'FAIL ') + k)
print('VERDICT', 'PASS' if all(v for _, v in bar) else 'FAIL')
